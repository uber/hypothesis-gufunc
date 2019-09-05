# Copyright (c) 2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0
from hypothesis import given
from hypothesis.strategies import floats, integers


def foo(x: int, y: float):
    return x * y


@given(integers(min_value=-10, max_value=10), floats(min_value=-10, max_value=10))
def test_foo_rescale(x: int, y: float):
    z = foo(x, y)
    z2 = foo(2 * x, y)
    assert 2 * z == z2


test_foo_rescale()  # Call the test without specifying inputs!
