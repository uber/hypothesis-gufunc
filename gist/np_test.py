# Copyright (c) 2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats

easy_floats = floats(min_value=-10, max_value=10)


@given(arrays(np.float, (3, 3), elements=easy_floats), arrays(np.float, (3, 2), elements=easy_floats))
def test_np_dot(x, y):
    assert np.allclose(np.dot(x, y), np.dot(y.T, x.T).T)


test_np_dot()
