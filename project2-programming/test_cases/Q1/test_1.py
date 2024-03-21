import numpy as np
import matplotlib.pyplot as plt
import os

import sys
from pathlib import Path
# export PYTHONPATH="${PYTHONPATH}:/path/to/dir"
# Add the parent directory to the Python path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
import test_utils

import neural_network

def max_score():
    return 2

def timeout():
    return 60

def test():
    alpha = np.array([0.5] * 18).reshape(3, 6)
    alpha[:, 0] = 0
    beta = np.array([0.5] * 40).reshape(10, 4)
    beta[:, 0] = 0
    x = np.array([0,0,0,0,0,0]).reshape(6,1)

    test_params = {"x": x, "y": 2, "alpha":  alpha, "beta": beta}
    expected = {"expected": "Q1_forward00.npz"}
    return test_utils.run_and_testforward(test_params, expected, max_score())

if __name__ == "__main__":
    test()
