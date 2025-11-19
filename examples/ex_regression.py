import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb
from ex_helper import (
    CSVLogger,
    fix_random_seed,
    generate_experiment_name,
    load_model_checkpoint,
    optimize_prior_prec_gradient,
    save_model_checkpoint,
    split_model,
    train_map_model,
)
from flax import nnx
from helper import DataLoader, get_sinusoid_example
from loguru import logger
from plotting import plot_regression_with_uncertainty

from laplax.api import (
    DEFAULT_REGRESSION_METRICS,
    CalibrationObjective,
    Predictive,
    Pushforward,
    calibration,
    evaluation,
    laplace,
    lin_pred_mean,
    lin_pred_std,
    lin_setup,
    nonlin_pred_mean,
    nonlin_pred_std,
    nonlin_setup,
)
from laplax.enums import CurvApprox, LossFn
from laplax.eval.pushforward import lin_samples, nonlin_samples
from laplax.eval.utils import transfer_entry
from laplax.register import register_calibration_method

# ------------------------------------------------------------------------------
# TASK Setup
# ------------------------------------------------------------------------------


DEFAULT_INTERVALS = [
    (-1.0, -0.5),
    (0.5, 1.0),
]
RESET_CSV_LOG = False


def build_sinusoid_data(
    num_train=150,
    num_valid=50,
    num_test=150,
    sigma_noise=0.2,
    sinus_factor=2.0 * jnp.pi,
    intervals=DEFAULT_INTERVALS,
    batch_size=20,
    rng_seed=0,
    test_interval=(-2.0, 2.0),
):
    key = jax.random.key(rng_seed)
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_sinusoid_example(
        num_train_data=num_train,
        num_valid_data=num_valid,
        num_test_data=num_test,
        sigma_noise=sigma_noise,
        sinus_factor=sinus_factor,
        intervals=intervals,
        rng_key=key,
        test_interval=test_interval,
    )

    train_loader = DataLoader(X_train, y_train, batch_size)
    valid_loader = DataLoader(X_valid, y_valid, batch_size)
    test_loader = DataLoader(X_test, y_test, batch_size)

    return train_loader, valid_loader, test_loader


class Model(nnx.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, rngs):
        self.linear1 = nnx.Linear(in_channels, hidden_channels, rngs=rngs)
        self.final_layer = nnx.Linear(hidden_channels, out_channels, rngs=rngs)

    def __call__(self, x):
        h = nnx.tanh(self.linear1(x))
        return self.final_layer(h)


# ------------------------------------------------------------------------------
# Training experiment
# ------------------------------------------------------------------------------


def train_sinusoid_model(
    *,
    # Data settings
    num_train=150,
    num_valid=50,
    num_test=300,
    sigma_noise=0.2,
    sinus_factor=2.0 * jnp.pi,
    intervals=DEFAULT_INTERVALS,
    batch_size=20,
    model_seed=0,
    data_seed=42,
    # Model settings
    hidden_channels=64,
    # Training settings
    n_epochs=1000,
    lr=1e-3,
    # Output settings
    ckpt_dir="./checkpoints/",
):
    """Train a MAP model on sinusoid data and save the checkpoint.

    Returns:
        The NNX model.
    """
    # Set model
    model = Model(
        in_channels=1,
        hidden_channels=hidden_channels,
        out_channels=1,
        rngs=nnx.Rngs(model_seed),
    )

    # Set data
    train_loader, _, _ = build_sinusoid_data(
        num_train,
        num_valid,
        num_test,
        sigma_noise,
        sinus_factor,
        intervals,
        batch_size,
        data_seed,
    )

    # Generate experiment name for the checkpoint
    model_name = f"regression_model_h{hidden_channels}_n{num_train}_seed{model_seed}"
    logger.info(f"Starting model training: {model_name}")

    # Train the model
    model = train_map_model(model, train_loader, n_epochs=n_epochs, lr=lr)

    # Save checkpoint
    checkpoint_path = save_model_checkpoint(
        model,
        Path(ckpt_dir) / model_name,
    )

    # Return data loaders and checkpoint path for potential immediate use
    return checkpoint_path


# ------------------------------------------------------------------------------
# LAPLACE experiment
# ------------------------------------------------------------------------------


def evaluate_regression_example(
    ckpt_dir: str = "./checkpoints/",
    model_name: str = "regression_model_h64_n150_seed0",
    *,
    # Laplace settings
    laplace_kwargs: dict,
    # Pushforward
    pushforward_kwargs: dict,
    # Calibration
    clbr_kwargs: dict | None = None,
    # Output settings
    csv_logger: CSVLogger | None = None,
    use_wandb: bool = False,
    seed: int = 42,
    valid_size: int = 100,  # Size of validation set for calibration
):
    """Evaluate regression example.

    Args:
        ckpt_dir: Directory containing model checkpoints
        model_name: Name of the model checkpoint to load
        laplace_kwargs: Dictionary containing Laplace approximation settings
        pushforward_kwargs: Dictionary containing pushforward settings
        clbr_kwargs: Dictionary containing calibration settings
        csv_logger: CSVLogger instance for logging results
        use_wandb: Whether to log results to Weights & Biases
        seed: Random seed for reproducibility
        valid_size: Number of points to subsample for validation
    """
    # Set seed
    fix_random_seed(seed)
    # Load map model
    ckpt_path = Path(ckpt_dir) / model_name
    model, _, _ = load_model_checkpoint(
        Model,
        model_kwargs={"in_channels": 1, "hidden_channels": 64, "out_channels": 1},
        checkpoint_path=ckpt_path,
    )

    # Load data
    train_loader, valid_loader, test_loader = build_sinusoid_data(
        num_test=400, test_interval=(-2.0, 2.0)
    )

    # Create targeted validation set by subsampling from test distribution
    # rng = np.random.default_rng(seed)
    # valid_indices = rng.choice(
    #     len(test_loader.X),
    #     size=valid_size,
    #     replace=False
    # )
    # valid_X = jnp.take(test_loader.X, jnp.array(valid_indices), axis=0)
    # valid_y = jnp.take(test_loader.y, jnp.array(valid_indices), axis=0)
    # valid_loader = DataLoader(valid_X, valid_y, batch_size=valid_size)

    # Start evaluation
    results = {}
    csv_logger = CSVLogger(force=RESET_CSV_LOG) if csv_logger is None else csv_logger

    # Extract parameters
    last_layer_only = laplace_kwargs.get("last_layer_only", False)
    curv_type = laplace_kwargs.get("curv_type")
    low_rank_rank = laplace_kwargs.get("low_rank_rank", 100)
    low_rank_seed = laplace_kwargs.get("low_rank_seed", 1950)
    sample_seed = pushforward_kwargs.get("sample_seed", 21904)
    pushforward_type = pushforward_kwargs.get("pushforward_type", Pushforward.LINEAR)

    # Extract calibration parameters if provided
    clbr_obj = clbr_kwargs.get("calibration_objective") if clbr_kwargs else None
    clbr_mthd = clbr_kwargs.get("calibration_method") if clbr_kwargs else None

    experiment_name = generate_experiment_name(
        ct=curv_type, ll=last_layer_only, co=clbr_obj, cm=clbr_mthd, pt=pushforward_type
    )

    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project="laplax-regression",
            name=experiment_name,
            config={
                "curv_type": laplace_kwargs.get("curv_type"),
                "last_layer_only": laplace_kwargs.get("last_layer_only"),
                "pushforward_type": pushforward_kwargs.get("pushforward_type"),
                "calibration_objective": (
                    clbr_kwargs.get("calibration_objective") if clbr_kwargs else None
                ),
                "calibration_method": (
                    clbr_kwargs.get("calibration_method") if clbr_kwargs else None
                ),
                "valid_size": valid_size,
            },
        )

    # Split model
    model_fn, params = split_model(model, last_layer_only=last_layer_only)

    # Approximate curvature
    posterior_fn, curv_est = laplace(
        model_fn=model_fn,
        params=params,
        data=train_loader,
        loss_fn=LossFn.MSE,
        curv_type=curv_type,
        rank=low_rank_rank,
        key=jax.random.key(low_rank_seed),
        has_batch=True,
    )

    # Calibration using targeted validation set
    prior_args = {"prior_prec": 1.0}
    if clbr_kwargs is not None:
        prior_args, _ = calibration(
            posterior_fn=posterior_fn,
            model_fn=model_fn,
            params=params,
            data={
                "input": valid_loader.X,
                "target": valid_loader.y,
            },  # Use targeted validation set
            curv_estimate=curv_est,
            curv_type=curv_type,
            loss_fn=LossFn.MSE,
            predictive_type=Predictive.NONE,
            pushforward_type=pushforward_type,
            **clbr_kwargs,
        )

    # Evaluation on full test set
    logger.info("Starting evaluation.")
    additional_entries = ["pred_mean", "pred_std", "samples"]
    eval_metrics = [*DEFAULT_REGRESSION_METRICS, transfer_entry(additional_entries)]
    results, _ = evaluation(
        posterior_fn=posterior_fn,
        model_fn=model_fn,
        params=params,
        arguments=prior_args,
        data={"input": test_loader.X, "target": test_loader.y},
        metrics=eval_metrics,
        predictive_type=Predictive.NONE,
        pushforward_type=pushforward_type,
        pushforward_fns=[
            lin_setup,
            lin_pred_mean,
            lin_pred_std,
            lin_samples,
        ]
        if pushforward_type is Pushforward.LINEAR
        else [
            nonlin_setup,
            nonlin_pred_mean,
            nonlin_pred_std,
            nonlin_samples,
        ],
        sample_key=jax.random.key(sample_seed),
        num_samples=20,
    )

    avg_results = {
        "rmse": jnp.mean(results["rmse"][100:300]),
        "nll": jnp.mean(results["nll"][100:300]),
        "chi^2": jnp.mean(results["chi^2"][100:300]),
        "crps": jnp.mean(results["crps"][100:300]),
    }
    logger.info(f"Eval: {avg_results}")

    # Log to wandb if enabled
    if use_wandb:
        # Log metrics
        wandb.log(avg_results)

        # Create and log regression plot with uncertainty
        fig = plot_regression_with_uncertainty(
            X_train=train_loader.X,
            y_train=train_loader.y,
            X_test=test_loader.X,
            y_test=test_loader.y,
            X_pred=test_loader.X,
            y_pred=results["pred_mean"],
            y_std=results["pred_std"],
            title=f"Regression with {curv_type} Curvature",
        )
        wandb.log({"regression_plot": wandb.Image(fig)})
        plt.close(fig)

    csv_logger.log(
        avg_results,
        experiment_name=experiment_name,
        log_args={
            "curv_type": curv_type,
            "last_layer_only": last_layer_only,
            "pushforward_type": pushforward_type,
            "calibration_objective": clbr_obj,
            "calibration_method": clbr_mthd,
            "valid_size": valid_size,
        },
    )
    csv_logger.log_samples(
        {
            "input": test_loader.X,
            "target": test_loader.y,
            "pred_mean": results["pred_mean"],
            "pred_std": results["pred_std"],
            "samples": results["samples"],
        },
        experiment_name=experiment_name,
    )


if __name__ == "__main__":
    """Different scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="evaluate",
        choices=["train", "evaluate"],
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=1,
    )

    # --------------------------
    # Train args
    # --------------------------
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--model_seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=42,
    )

    # --------------------------
    # Laplace args
    # --------------------------
    parser.add_argument(
        "--curv_type",
        type=CurvApprox,
        default=CurvApprox.LANCZOS,
    )
    parser.add_argument(
        "--last_layer_only",
        type=lambda x: x.lower() == "true",
        default=False,
    )
    parser.add_argument(
        "--low_rank_rank",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--low_rank_seed",
        type=int,
        default=1950,
    )

    # --------------------------
    # Evaluation args
    # --------------------------
    parser.add_argument(
        "--calibration_method",
        type=str,
        choices=["gradient_descent", "grid_search"],
        default="gradient_descent",
    )
    parser.add_argument(
        "--calibration_objective",
        type=CalibrationObjective,
        default=CalibrationObjective.NLL,
    )
    parser.add_argument(
        "--pushforward_type", type=Pushforward, default=Pushforward.LINEAR
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=21904,
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./checkpoints/",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="regression_model_h64_n150_seed0",
    )

    parser.add_argument(
        "--valid_size",
        type=int,
        default=150,
    )

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
        train_sinusoid_model(
            n_epochs=args.n_epochs,
            hidden_channels=args.hidden_channels,
            model_seed=args.model_seed,
            data_seed=args.data_seed,
        )

    # -------------------------------------------
    # Evaluation
    # -------------------------------------------
    if args.task == "evaluate":
        register_calibration_method(
            "gradient_descent",
            optimize_prior_prec_gradient,
        )

        csv_logger = CSVLogger(file_name="regression_results.csv", force=RESET_CSV_LOG)

        logger.info(f"Running Laplace with curvature type: {args.curv_type}")

        evaluate_regression_example(
            ckpt_dir=args.ckpt_dir,
            model_name=args.model_name,
            laplace_kwargs={
                "curv_type": args.curv_type,
                "last_layer_only": args.last_layer_only,
                "low_rank_rank": args.low_rank_rank,
                "low_rank_seed": args.low_rank_seed,
            },
            pushforward_kwargs={
                "pushforward_type": args.pushforward_type,
                "sample_seed": args.sample_seed,
            },
            clbr_kwargs={
                "calibration_objective": args.calibration_objective,
                "calibration_method": args.calibration_method,
                "init_prior_prec": 1.0,
                "init_sigma_noise": 1.0,
                "grid_size": 2000,
                "log_prior_prec_min": -3,
                "log_prior_prec_max": 3,
                "patience": None,
                "num_epochs": 300,
            },
            csv_logger=csv_logger,
            use_wandb=args.wandb,
            seed=args.data_seed,
            valid_size=args.valid_size,
        )
