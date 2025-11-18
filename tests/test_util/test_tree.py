import math

import jax
import jax.numpy as jnp
import pytest
import pytest_cases

from laplax.types import KeyType, PyTree
from laplax.util.flatten import create_pytree_flattener
from laplax.util.tree import add, allclose, sub, tree_vec_get

TreeTestCase = tuple[PyTree, jax.Array]


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in [1, 42, 256]],
)
def key(request) -> KeyType:
    return jax.random.key(request.param)


@pytest_cases.case(id="vector_tree")
def case_vector_tree(key: KeyType):
    vector = jax.random.normal(key, shape=(6,))
    return {"a": vector[:2], "b": vector[2:4], "c": vector[4:]}, vector


@pytest_cases.case(id="one_vector_tree")
def case_one_vector_tree(key: KeyType):
    vector = jax.random.normal(key, shape=(6,))
    return {"a": vector[:2], "b": {"b1": vector[2:4], "b2": vector[4:]}}, vector


@pytest_cases.case(id="two_vector_tree")
def case_two_vector_tree(key: KeyType):
    keys = jax.random.split(key, 2)
    vector0 = jax.random.normal(keys[0], shape=(6,))
    pytree0 = {"a": vector0[:2], "b": {"b1": vector0[2:4], "b2": vector0[4:]}}
    vector1 = jax.random.normal(keys[1], shape=(6,))
    pytree1 = {"a": vector1[:2], "b": {"b1": vector1[2:4], "b2": vector1[4:]}}
    return (pytree0, vector0), (pytree1, vector1)


@pytest_cases.parametrize_with_cases("test_case", cases=[case_two_vector_tree])
def test_add(test_case):
    (tree1, vector1), (tree2, vector2) = test_case
    flatten, _ = create_pytree_flattener(tree1)
    allclose(flatten(add(tree1, tree2)), vector1 + vector2)


@pytest_cases.parametrize_with_cases("test_case", cases=[case_two_vector_tree])
def test_sub(test_case):
    (tree1, vector1), (tree2, vector2) = test_case
    flatten, _ = create_pytree_flattener(tree1)
    allclose(flatten(sub(tree1, tree2)), vector1 - vector2)


# TODO(2bys): finish
# @pytest_cases.parametrize_with_cases("test_case", cases=[case_one_vector_tree])
# def test_invert(test_case):
#     (tree1, vector1) = test_case
#     flatten, _ = create_pytree_flattener(tree1)
#     allclose(flatten(invert(tree1))), vector1


@pytest.mark.parametrize(
    ("shape", "indices"),
    [
        ((5,), [0, 2, 4]),
        ((2, 3), [0, 3, 5]),
        ((3, 2, 2), [0, 5, 11]),
    ],
)
def test_tree_vec_get_array_leaf(shape: tuple[int, ...], indices: list[int]):
    array = jnp.arange(math.prod(shape)).reshape(shape)
    flat = array.reshape(-1)
    for idx in indices:
        assert tree_vec_get(array, idx) == flat[idx]


def test_tree_vec_get_pytree():
    tree = {
        "a": jnp.arange(4).reshape(2, 2),
        "b": {
            "c": jnp.arange(3),
            "d": jnp.arange(6).reshape(2, 3),
        },
    }
    leaves, _ = jax.tree_util.tree_flatten(tree)
    flat = jnp.concatenate([leaf.reshape(-1) for leaf in leaves])
    for idx, expected in enumerate(flat):
        assert tree_vec_get(tree, idx) == expected
