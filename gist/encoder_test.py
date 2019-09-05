# Copyright (c) 2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0
import urllib

from hypothesis import given
from hypothesis.strategies import text

encode = urllib.parse.quote
decode = urllib.parse.unquote


@given(text())
def test_decode_inverts_encode(s):
    assert decode(encode(s)) == s


test_decode_inverts_encode()
