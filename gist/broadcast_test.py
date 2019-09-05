# Copyright (c) 2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from hypothesis import given
from hypothesis.strategies import floats

from hypothesis_gufunc.gufunc import gufunc_args

easy_floats = floats(min_value=-10, max_value=10)


@given(gufunc_args("(m,n),(n,p)->(m,p)", dtype=np.float_, elements=easy_floats, max_dims_extra=3))
def test_np_matmul(args):
    x, y = args
    f_vec = np.vectorize(np.matmul, signature="(m,n),(n,p)->(m,p)", otypes=[np.float_])
    assert np.allclose(np.matmul(x, y), f_vec(x, y))


test_np_matmul()
