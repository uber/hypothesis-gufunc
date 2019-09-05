# Copyright (c) 2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
from hypothesis import given
from hypothesis.strategies import floats

from hypothesis_gufunc.gufunc import gufunc_args

easy_floats = floats(min_value=-10, max_value=10)


def torchify(args):
    args = tuple(torch.tensor(X) for X in args)
    return args


@given(gufunc_args("(m,n),(n,p)->(m,p)", dtype=np.float_, elements=easy_floats, min_side=1).map(torchify))
def test_torch_matmul(args):
    x, y = args
    assert torch.allclose(torch.matmul(x, y), torch.matmul(y.T, x.T).T)


test_torch_matmul()
