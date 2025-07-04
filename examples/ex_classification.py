import argparse
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import torch
import wandb
from ex_helper import (
    CSVLogger,
    LimitedLoader,
    fix_random_seed,
    generate_experiment_name,
    load_model_checkpoint,
    load_with_pickle,
    optimize_prior_prec_gradient,
    save_model_checkpoint,
    save_with_pickle,
    split_model,
    train_map_model,
)
from flax import nnx
from loguru import logger
from plotting import create_proportion_diagram, create_reliability_diagram
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from laplax.api import (
    GGN,
    CalibrationObjective,
    Predictive,
    Pushforward,
    calibration,
    evaluation,
    nll_gaussian_classification,
)
from laplax.curv import estimate_curvature, set_posterior_fn
from laplax.curv.cov import LowRankTerms
from laplax.enums import CurvApprox, LossFn
from laplax.eval import apply_fns
from laplax.eval.metrics import (
    calculate_bin_metrics,
    correctness,
    expected_calibration_error,
)
from laplax.register import register_calibration_method

RESET_CSV_LOG = False

# ------------------------------------------------------------------------------
# CIFAR-10 Data
# ------------------------------------------------------------------------------


def cifar10_collate_fn(batch):
    images, labels = zip(*batch, strict=False)
    images = torch.stack(images, dim=0).permute(0, 2, 3, 1).detach().cpu().numpy()
    labels = torch.tensor(labels, dtype=torch.int32).detach().cpu().numpy()
    return images, labels


def setup_cifar10_data():
    """Setup CIFAR-10 data.

    Returns:
        A triple of train, val, and test loaders.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Download (or Load) datasets
    train_set = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transforms
    )
    test_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transforms
    )

    # Split train → train/valid
    train_size = int(0.9 * len(train_set))  # 45,000
    valid_size = len(train_set) - train_size  # 5,000
    train_set, valid_set = random_split(
        train_set, [train_size, valid_size], generator=torch.Generator().manual_seed(42)
    )

    # Create DataLoaders
    batch_size = 128
    num_workers = 4  # adjust if you like

    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": cifar10_collate_fn,
        "multiprocessing_context": "spawn",
    }
    train_loader = DataLoader(train_set, shuffle=True, **dataloader_kwargs)
    valid_loader = DataLoader(valid_set, shuffle=False, **dataloader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **dataloader_kwargs)

    return train_loader, valid_loader, test_loader


# ------------------------------------------------------------------------------
# CIFAR-10 CNN
# ------------------------------------------------------------------------------


class CIFAR10CNN(nnx.Module):
    """A simple 3-conv / 2-FC CNN for CIFAR-10."""

    def __init__(self, *, rngs: nnx.Rngs):
        # Conv blocks
        self.conv1 = nnx.Conv(3, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv3 = nnx.Conv(64, 128, kernel_size=(3, 3), padding="SAME", rngs=rngs)

        # Pooling helper
        self.pool = partial(
            nnx.max_pool, window_shape=(2, 2), strides=(2, 2), padding="VALID"
        )

        # Fully-connected layers
        # After 3xpool on 32x32 input -> 4x4 feature maps of 128 channels:
        #   32 -> 16 -> 8 -> 4
        flat_dim = 4 * 4 * 128
        self.fc1 = nnx.Linear(flat_dim, 256, rngs=rngs)
        self.final_layer = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: [32, 32, 3] - No batch dimension assumed
        x = nnx.relu(self.conv1(x))
        x = self.pool(x)  # [16, 16, 32]

        x = nnx.relu(self.conv2(x))
        x = self.pool(x)  # [8, 8, 64]

        x = nnx.relu(self.conv3(x))
        x = self.pool(x)  # [4, 4, 128]

        x = x.reshape(-1)  # [4*4*128]
        x = nnx.relu(self.fc1(x))
        x = self.final_layer(x)  # [10]
        return x


# ------------------------------------------------------------------------------
# Laplace Approximation
# ------------------------------------------------------------------------------


def compute_ggn(
    model,
    train_loader,
    curv_type: CurvApprox = CurvApprox.LANCZOS,
    *,
    num_batches: int = 10,
    max_rank: int = 10,
    last_layer_only=False,
    checkpoint_dir="./checkpoints/",
):
    """Compute the GGN of the model.

    Returns:
        A dict containing the curvature estimate.
    """
    # Split model
    model_fn, params = split_model(model, last_layer_only=last_layer_only)

    # Create GGN
    ggn_name = f"ggn_c={curv_type}_n={num_batches}_r={max_rank}_ll={last_layer_only}"
    train_loader = LimitedLoader(train_loader, num_batches)

    ggn_mv = GGN(
        model_fn,
        params,
        train_loader,
        loss_fn=LossFn.CROSS_ENTROPY,
        factor=num_batches / len(train_loader),
        has_batch=True,
        verbose_logging=True,
    )

    # Curvature estimation
    curv_estimate = estimate_curvature(
        curv_type=curv_type,
        mv=ggn_mv,
        layout=params,
        rank=max_rank,
        key=jax.random.key(4910),
        has_batch=True,
    )

    # # Save GGN
    ggn_path = Path(checkpoint_dir) / (ggn_name + ".pkl")
    save_with_pickle(curv_estimate, ggn_path)

    logger.debug("Curvature estimated: {}", ggn_name)
    return {
        "curv_estimate": curv_estimate,
    }


# ------------------------------------------------------------------------------
# CIFAR-10 Classification
# ------------------------------------------------------------------------------


def train_cifar10_model(
    lr=1e-3,
    n_epochs=2,
    *,
    verbose=True,
    model_seed=42,
    checkpoint_dir="./checkpoints/",
    use_wandb: bool = True,
):
    # Generate experiment name for the checkpoint
    experiment_name = f"cifar10_model_seed{model_seed}"

    # WANDB Logging
    if use_wandb:
        wandb.init(
            project="laplax_train",
            name=experiment_name,
            config={"n_epochs": n_epochs, "lr": lr, "model_seed": model_seed},
        )

    # Prepare data
    train_loader, valid_loader, test_loader = setup_cifar10_data()
    logger.info(f"Starting model training: {experiment_name}")

    # Set model
    model = CIFAR10CNN(rngs=nnx.Rngs(model_seed))
    model = train_map_model(
        model,
        train_loader,
        n_epochs,
        lr=lr,
        verbose=verbose,
        log_every_n_epochs=1,
        loss_type="cross_entropy",
        test_loader=test_loader,
    )

    # Save checkpoint
    ckpt_path = Path(checkpoint_dir) / experiment_name
    save_model_checkpoint(model, ckpt_path)

    if use_wandb:
        wandb.finish()

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "model": model,
        "checkpoint_path": ckpt_path,
    }


def curvature_cifar10_model(
    model_name: str = "cifar10_model_seed42",
    curv_type: CurvApprox = CurvApprox.LANCZOS,
    *,
    num_batches: int = 10,
    max_rank: int = 100,
    last_layer_only=False,
    checkpoint_dir="./checkpoints/",
):
    """Compute the GGN of the model.

    Returns:
        The GGN.
    """
    ckpt_path = Path(checkpoint_dir) / model_name
    model, _, _ = load_model_checkpoint(CIFAR10CNN, {}, ckpt_path)
    train_loader, _, _ = setup_cifar10_data()

    res = compute_ggn(
        model,
        train_loader,
        curv_type=curv_type,
        num_batches=num_batches,
        max_rank=max_rank,
        last_layer_only=last_layer_only,
        checkpoint_dir=checkpoint_dir,
    )
    return res


EMPTY_DICT = {}


def evaluate_cifar10_model(
    # Checkpoint information
    ckpt_dir: str = "./checkpoints/",
    model_name: str = "cifar10_model_seed42",
    *,
    # Laplace settings
    laplace_kwargs: dict,
    # Pushforward
    pushforward_kwargs: dict,
    # Calibration settings (None to Skip)
    clbr_kwargs: dict = EMPTY_DICT,
    # Evaluation settings
    eval_kwargs: dict = EMPTY_DICT,
    # Output settings
    csv_logger: CSVLogger | None = None,
    use_wandb: bool = False,
):
    """Evaluate the CIFAR-10 model."""
    # Load map model
    ckpt_path = Path(ckpt_dir) / model_name
    model, _, _ = load_model_checkpoint(CIFAR10CNN, {}, ckpt_path)

    # Load data
    _, valid_loader, test_loader = setup_cifar10_data()

    # Start evaluation
    results = {}
    csv_logger = (
        CSVLogger(file_name=f"{model_name}_results.csv", force=RESET_CSV_LOG)
        if csv_logger is None
        else csv_logger
    )

    # Extract parameters
    last_layer_only = laplace_kwargs.get("last_layer_only", False)
    curv_type = laplace_kwargs.get("curv_type")
    low_rank_rank = laplace_kwargs.get("low_rank_rank", 100)
    sample_seed = pushforward_kwargs.get("sample_seed", 21904)
    pushforward_type = pushforward_kwargs.get("pushforward_type", Pushforward.LINEAR)
    predictive_type = pushforward_kwargs.get("predictive_type", Predictive.MC_BRIDGE)
    clbr_obj = clbr_kwargs.get("calibration_objective")
    clbr_mthd = clbr_kwargs.get("calibration_method")
    max_rank = laplace_kwargs.get("max_rank", 10)
    num_batches = laplace_kwargs.get("num_batches", 2)

    experiment_name = generate_experiment_name(
        ct=curv_type,
        ll=last_layer_only,
        co=clbr_obj,
        cm=clbr_mthd,
        pt=pushforward_type,
        prt=predictive_type,
        rk=low_rank_rank,
    )

    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project="laplax-classification",
            name=experiment_name,
            config={
                "curv_type": curv_type,
                "last_layer_only": last_layer_only,
                "pushforward_type": pushforward_type,
                "predictive_type": predictive_type,
                "calibration_objective": clbr_obj,
                "calibration_method": clbr_mthd,
                "low_rank_rank": low_rank_rank,
                "num_batches": num_batches,
            },
        )

    model_fn, params = split_model(model, last_layer_only=last_layer_only)

    # Load curvature and set posterior_fn
    ggn_name = f"ggn_c={curv_type}_n={num_batches}_r={max_rank}_ll={last_layer_only}"
    ggn_path = Path(ckpt_dir) / (ggn_name + ".pkl")
    curv_estimate = load_with_pickle(ggn_path)

    # Adjust low rank
    if curv_type is CurvApprox.LANCZOS:
        current_rank = curv_estimate.U.shape[1]
        logger.info(f"Current rank: {current_rank}, requested rank: {low_rank_rank}")
        curv_estimate = LowRankTerms(
            U=curv_estimate.U[:, :low_rank_rank],
            S=curv_estimate.S[:low_rank_rank],
            scalar=curv_estimate.scalar,
        )

    # Set posterior_fn
    posterior_fn = set_posterior_fn(
        curv_type=curv_type,
        curv_estimate=curv_estimate,
        layout=params,
    )

    # Calibration
    prior_args = {"prior_prec": 1.0}
    if len(clbr_kwargs) > 0:
        prior_args, _ = calibration(
            posterior_fn=posterior_fn,
            model_fn=model_fn,
            params=params,
            data=next(iter(valid_loader)),
            curv_estimate=curv_estimate,
            curv_type=curv_type,
            loss_fn=LossFn.CROSS_ENTROPY,
            predictive_type=predictive_type,
            pushforward_type=pushforward_type,
            num_samples=100,
            **clbr_kwargs,
        )

    # Evaluation
    results, _ = evaluation(
        posterior_fn=posterior_fn,
        model_fn=model_fn,
        params=params,
        arguments=prior_args,
        data=test_loader,
        metrics=eval_kwargs.get("eval_metrics"),
        predictive_type=predictive_type,
        pushforward_type=pushforward_type,
        sample_key=jax.random.key(sample_seed),
        num_samples=100,
    )

    # Compute expected calibration error
    ece_pred = expected_calibration_error(
        confidence=results["confidences_pred"],
        correctness=results["correctness_pred"],
        num_bins=15,
    )
    ece_map = expected_calibration_error(
        confidence=results["confidences_map"],
        correctness=results["correctness_map"],
        num_bins=15,
    )
    results_avg = jax.tree.map(jnp.mean, results)
    results_avg["ece_pred"] = ece_pred
    results_avg["ece_map"] = ece_map

    # Log to wandb if enabled
    if use_wandb:
        # Log metrics
        wandb.log(results_avg)

        # Create and log reliability diagram
        bin_prop_pred, bin_conf_pred, bin_acc_pred = calculate_bin_metrics(
            confidence=results["confidences_pred"],
            correctness=results["correctness_pred"],
            num_bins=15,
        )
        fig_rel = create_reliability_diagram(
            bin_confidences=bin_conf_pred,
            bin_accuracies=bin_acc_pred,
            num_bins=15,
        )
        wandb.log({"reliability_pred": wandb.Image(fig_rel)})
        plt.close(fig_rel)

        # Create and log proportion diagram
        fig_prop = create_proportion_diagram(
            bin_proportions=bin_prop_pred,
            num_bins=15,
        )
        wandb.log({"proportion_pred": wandb.Image(fig_prop)})
        plt.close(fig_prop)

        # Create and log reliability diagram
        bin_prop_map, bin_conf_map, bin_acc_map = calculate_bin_metrics(
            confidence=results["confidences_map"],
            correctness=results["correctness_map"],
            num_bins=15,
        )
        fig_rel_map = create_reliability_diagram(
            bin_confidences=bin_conf_map,
            bin_accuracies=bin_acc_map,
            num_bins=15,
        )
        wandb.log({"reliability_map": wandb.Image(fig_rel_map)})
        plt.close(fig_rel_map)

        # Create and log proportion diagram
        fig_prop_map = create_proportion_diagram(
            bin_proportions=bin_prop_map,
            num_bins=15,
        )
        wandb.log({"proportion_map": wandb.Image(fig_prop_map)})
        plt.close(fig_prop_map)

    csv_logger.log(results, experiment_name)

    logger.info(f"Eval: {results}")
    logger.info(f"ECE (Pred): {ece_pred}")
    logger.info(f"ECE (MAP): {ece_map}")


if __name__ == "__main__":
    """Different scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "ggn", "evaluate"],
    )

    parser.add_argument(
        "--data_seed",
        type=int,
        default=42,
    )

    # --------------------------
    # Train args
    # --------------------------
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=2,
    )

    # --------------------------
    # Curvature args
    # --------------------------
    parser.add_argument(
        "--curv_type",
        type=CurvApprox,
        default=CurvApprox.LANCZOS,
    )

    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--last_layer_only",
        type=lambda x: x.lower() == "true",
        default=False,
    )

    parser.add_argument(
        "--max_rank",
        type=int,
        default=200,
    )

    # --------------------------
    # Evaluation args
    # --------------------------

    parser.add_argument(
        "--calibrate",
        type=lambda x: x.lower() == "true",
        default=True,
    )

    parser.add_argument(
        "--calibration_method",
        type=str,
        choices=["gradient_descent", "grid_search"],
        default="gradient_descent",
    )

    parser.add_argument(
        "--calibration_objective",
        type=CalibrationObjective,
        default=CalibrationObjective.ECE,
    )

    parser.add_argument(
        "--pushforward_type", type=Pushforward, default=Pushforward.LINEAR
    )

    parser.add_argument(
        "--predictive_type", type=Predictive, default=Predictive.MC_BRIDGE
    )

    parser.add_argument("--low_rank_rank", type=int, default=100)

    # --------------------------
    # Wandb args
    # --------------------------
    parser.add_argument(
        "--wandb",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Enable Weights & Biases logging",
    )

    args = parser.parse_args()

    fix_random_seed(args.data_seed + 2103)

    # -------------------------------------------
    # Train model
    # -------------------------------------------

    if args.task == "train":
        train_cifar10_model(n_epochs=args.n_epochs, use_wandb=args.wandb)

    # -------------------------------------------
    # GGN estimation
    # -------------------------------------------

    if args.task == "ggn":
        curvature_cifar10_model(
            curv_type=args.curv_type,
            num_batches=args.num_batches,
            last_layer_only=args.last_layer_only,
            max_rank=args.max_rank,
        )

    # -------------------------------------------
    # Evaluation
    # -------------------------------------------

    if args.task == "evaluate":
        register_calibration_method("gradient_descent", optimize_prior_prec_gradient)

        csv_logger = CSVLogger(file_name="cifar10_results.csv", force=RESET_CSV_LOG)

        logger.info(f"Running Laplace with curvature type: {args.curv_type}")

        evaluate_cifar10_model(
            laplace_kwargs={
                "curv_type": args.curv_type,
                "last_layer_only": args.last_layer_only,
                "num_batches": args.num_batches,
                "max_rank": args.max_rank,
                "low_rank_rank": args.low_rank_rank,
            },
            pushforward_kwargs={
                "pushforward_type": args.pushforward_type,
                "predictive_type": args.predictive_type,
            },
            clbr_kwargs={
                "calibration_objective": args.calibration_objective,
                "calibration_method": args.calibration_method,
                "log_prior_prec_min": -4.0,
                "log_prior_prec_max": 4.0,
                "grid_size": 200,
                "init_prior_prec": 1.0,
                "init_sigma_noise": 1.0,
            }
            if args.calibrate
            else {},
            eval_kwargs={
                "eval_metrics": [
                    apply_fns(
                        lambda map, **kwargs:  # noqa: ARG005
                        jnp.max(jax.nn.softmax(map, axis=-1), axis=-1),
                        lambda pred_act, **kwargs:  # noqa: ARG005
                        jnp.max(pred_act, axis=-1),
                        lambda map, target, **kwargs:  # noqa: ARG005
                        correctness(map, target) * 1,
                        lambda pred_mean, target, **kwargs:  # noqa: ARG005
                        correctness(pred_mean, target) * 1,
                        names=[
                            "confidences_map",
                            "confidences_pred",
                            "correctness_map",
                            "correctness_pred",
                        ],
                    ),
                    apply_fns(nll_gaussian_classification, names=["nll_gaussian_pred"]),
                ]
            },
            csv_logger=csv_logger,
            use_wandb=args.wandb,
        )
