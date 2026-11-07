import argparse
import json
import pickle

import jax
import numpy as np


def load(path, sector):
    with open(path, "rb") as handle:
        return pickle.load(handle)["sectors"][sector]


def flatten(tree):
    leaves = [np.asarray(value, dtype=np.float64).ravel()
              for value in jax.tree_util.tree_leaves(tree)]
    return np.concatenate(leaves)


parser = argparse.ArgumentParser()
parser.add_argument("reference")
parser.add_argument("first")
parser.add_argument("second")
parser.add_argument("--sector", type=int, default=0)
args = parser.parse_args()

reference = flatten(load(args.reference, args.sector))
first = flatten(load(args.first, args.sector)) - reference
second = flatten(load(args.second, args.sector)) - reference
first_norm = np.linalg.norm(first)
second_norm = np.linalg.norm(second)
result = {
    "first_norm": float(first_norm),
    "second_norm": float(second_norm),
    "cosine": float(np.dot(first, second) / (first_norm * second_norm)),
    "difference_norm": float(np.linalg.norm(first - second)),
}
print(json.dumps(result, indent=2))
