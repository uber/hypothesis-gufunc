# Copyright (c) 2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0
import numpy as np


def test_np_dot_golden():
    X = np.array([[1.5, 2], [3, 4]])
    Y = np.array([[5, 6], [7, 8.8]])
    Z = np.dot(X, Y)
    assert np.all(Z == np.array([[21.5, 26.6], [43.0, 53.2]]))


test_np_dot_golden()
