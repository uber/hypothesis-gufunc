# Copyright (c) 2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from hypothesis import given
from hypothesis.strategies import floats

from hypothesis_gufunc.gufunc import gufunc_args

easy_floats = floats(min_value=-10, max_value=10)


@given(gufunc_args("(m,n),(n,p)->(m,p)", dtype=np.float_, elements=easy_floats))
def test_np_dot(args):
    x, y = args
    assert np.allclose(np.dot(x, y), np.dot(y.T, x.T).T)


test_np_dot()
