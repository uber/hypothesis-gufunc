# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import xarray as xr
from hypothesis import assume, given
from hypothesis.extra.numpy import scalar_dtypes
from hypothesis.strategies import booleans, data, fixed_dictionaries, integers, just, lists, one_of, tuples

import hypothesis_gufunc.extra.xr as hxr
from hypothesis_gufunc.extra.xr import _hashable

MAX_DIM_LEN = 32


def sizes(allow_none=True):
    def mapper(T):
        if T[1] is not None:
            T = tuple(sorted(T))
        return T

    if allow_none:
        S = tuples(integers(0, 5), integers(0, 5) | just(None)).map(mapper)
    else:
        S = tuples(integers(0, 5), integers(0, 5)).map(mapper)
    return S


def dtypes():
    def to_native(dtype):
        tt = dtype.type
        # Only keep if invertible
        tt = tt if np.dtype(tt) == dtype else dtype
        return tt

    def cast_it(args):
        return args[0](args[1])

    dtypes = scalar_dtypes().filter(lambda x: x.kind in "biuf")
    return one_of(dtypes, dtypes.map(str), dtypes.map(to_native))


@given(hxr.xr_dim_lists(), data())
def test_get_all_dims(dims, data):
    vars_to_dims = data.draw(hxr.xr_coords_dicts(dims))

    all_dims = hxr._get_all_dims(vars_to_dims)

    assert all_dims == sorted(set(all_dims))

    all_dims2 = []
    for dd in vars_to_dims.values():
        all_dims2.extend(dd)
    assert all_dims == sorted(set(all_dims2))


@given(lists(_hashable()), sizes(), data())
def test_subset_lists(L, sizes, data):
    min_size, max_size = sizes

    assume(min_size <= len(set(L)))

    S = hxr.subset_lists(L, min_size, max_size)
    L2 = data.draw(S)

    n = len(L2)

    assert n >= min_size
    assert (max_size is None) or (n <= max_size)
    assert set(L2).issubset(set(L))


@given(lists(_hashable()), sizes(), data())
def test_subset_lists_empty(L, sizes, data):
    min_size, max_size = sizes

    S = hxr.subset_lists([], min_size=0, max_size=max_size)
    L2 = data.draw(S)

    assert len(L2) == 0
    assert L2 == []


@given(sizes(), data())
def test_xr_dim_lists(sizes, data):
    min_dims, max_dims = sizes

    S = hxr.xr_dim_lists(min_dims, max_dims)

    L = data.draw(S)

    n = len(L)
    assert n >= min_dims
    assert (max_dims is None) or (n <= max_dims)
    assert all(isinstance(ss, str) for ss in L)

    if n <= MAX_DIM_LEN:
        da = xr.DataArray(np.zeros((1,) * n), dims=L)
        assert da.dims == tuple(L)


@given(sizes(), data())
def test_xr_var_lists(sizes, data):
    min_vars, max_vars = sizes

    S = hxr.xr_var_lists(min_vars, max_vars)

    L = data.draw(S)

    n = len(L)
    assert n >= min_vars
    assert (max_vars is None) or (n <= max_vars)
    assert all(isinstance(hash(ss), int) for ss in L)

    ds = xr.Dataset({vv: xr.DataArray(0) for vv in L})
    assert set(ds) == set(L)


@given(sizes(), sizes(), data())
def test_vars_to_dims_dicts(var_sizes, dim_sizes, data):
    min_vars, max_vars = var_sizes
    min_dims, max_dims = dim_sizes

    S = hxr.vars_to_dims_dicts(min_vars, max_vars, min_dims, max_dims)

    D = data.draw(S)

    n = len(D)
    assert n >= min_vars
    assert (max_vars is None) or (n <= max_vars)
    assert all(len(dd) >= min_dims for _, dd in D.items())
    assert (max_dims is None) or all(len(dd) <= max_dims for _, dd in D.items())
    assert all(all(isinstance(ss, str) for ss in dd) for _, dd in D.items())

    if all(len(dd) <= MAX_DIM_LEN for _, dd in D.items()):
        ds = xr.Dataset({vv: xr.DataArray(np.zeros((1,) * len(dd)), dims=dd) for vv, dd in D.items()})
        assert set(ds) == set(D.keys())
        assert all(ds[vv].dims == tuple(dd) for vv, dd in D.items())


@given(sizes(), booleans(), data())
def test_xr_coords(sizes, unique, data):
    elements = None
    min_side, max_side = sizes

    S = hxr.xr_coords(elements, min_side, max_side, unique)

    L = data.draw(S)

    n = len(L)
    assert n >= min_side
    assert (max_side is None) or (n <= max_side)
    assert all(isinstance(ss, int) for ss in L)

    if unique:
        assert len(set(L)) == len(L)


@given(sizes(allow_none=False), data())
def test_simple_coords(sizes, data):
    min_side, max_side = sizes

    S = hxr.simple_coords(min_side, max_side)

    L = data.draw(S)

    n = len(L)
    assert n >= min_side
    assert (max_side is None) or (n <= max_side)
    assert all(isinstance(ss, int) for ss in L)
    assert L == list(range(len(L)))


@given(hxr.xr_dim_lists(), sizes(), booleans(), data())
def test_xr_coords_dicts(dims, sizes, unique_coords, data):
    elements = None
    min_side, max_side = sizes

    # special dims just filled with dim name on coords as test case
    special_dims = data.draw(hxr.subset_lists(dims))
    coords_st = {dd: lists(just(dd), min_side, max_side) for dd in special_dims}

    S = hxr.xr_coords_dicts(dims, elements, min_side, max_side, unique_coords, coords_st)

    D = data.draw(S)

    assert list(D.keys()) == dims
    assert all(min_side <= len(cc) for cc in D.values())
    assert (max_side is None) or all(max_side >= len(cc) for cc in D.values())
    assert all((dd in special_dims) or all(isinstance(ss, int) for ss in L) for dd, L in D.items())
    assert all(all(ss == dd for ss in D[dd]) for dd in special_dims)

    if unique_coords:
        for dd, cc in D.items():
            assert (dd in special_dims) or (len(set(cc)) == len(cc))


@given(hxr.xr_dim_lists(), dtypes(), data())
def test_fixed_coords_dataarrays(dims, dtype, data):
    elements = None

    coords = data.draw(hxr.xr_coords_dicts(dims))

    S = hxr.fixed_coords_dataarrays(dims, coords, dtype, elements)

    da = data.draw(S)

    assert da.dims == tuple(dims)
    assert da.dtype == np.dtype(dtype)
    for dd in dims:
        assert da.coords[dd].values.tolist() == coords[dd]


@given(hxr.xr_dim_lists(), dtypes(), sizes(), data())
def test_fixed_dataarrays(dims, dtype, sizes, data):
    elements = None
    coords_elements = None
    min_side, max_side = sizes

    # special dims just filled with dim name on coords as test case
    special_dims = data.draw(hxr.subset_lists(dims))
    coords_st = {dd: lists(just(dd), min_side, max_side) for dd in special_dims}

    S = hxr.fixed_dataarrays(dims, dtype, elements, coords_elements, min_side, max_side, coords_st)

    da = data.draw(S)

    assert da.dims == tuple(dims)
    assert all(ss >= min_side for ss in da.sizes.values())
    assert (max_side is None) or all(ss <= max_side for ss in da.sizes.values())
    assert da.dtype == np.dtype(dtype)
    assert all(all(ss == dd for ss in da.coords[dd].values.tolist()) for dd in special_dims)
    for dd in dims:
        L = da.coords[dd].values.tolist()
        assert (dd in special_dims) or all(isinstance(ss, int) for ss in L)
        assert (dd in special_dims) or (len(set(L)) == len(L))


@given(hxr.xr_dim_lists(), dtypes(), sizes(allow_none=False), data())
def test_simple_dataarrays(dims, dtype, sizes, data):
    elements = None
    min_side, max_side = sizes

    S = hxr.simple_dataarrays(dims, dtype, elements, min_side, max_side)

    da = data.draw(S)

    assert da.dims == tuple(dims)
    assert all(ss >= min_side for ss in da.sizes.values())
    assert (max_side is None) or all(ss <= max_side for ss in da.sizes.values())
    assert da.dtype == np.dtype(dtype)
    for dd in dims:
        L = da.coords[dd].values.tolist()
        assert all(isinstance(ss, int) for ss in L)
        assert L == list(range(len(L)))


@given(dtypes(), sizes(allow_none=False), sizes(allow_none=False), data())
def test_dataarrays(dtype, size_dims, size_sides, data):
    elements = None
    coords_elements = None

    min_dims, max_dims = size_dims
    min_side, max_side = size_sides

    S = hxr.dataarrays(dtype, elements, coords_elements, min_side, max_side, min_dims, max_dims)

    da = data.draw(S)

    assert len(da.dims) >= min_dims
    assert len(da.dims) <= max_dims
    assert all(ss >= min_side for ss in da.sizes.values())
    assert (max_side is None) or all(ss <= max_side for ss in da.sizes.values())
    assert da.dtype == np.dtype(dtype)
    for dd in da.dims:
        L = da.coords[dd].values.tolist()
        assert all(isinstance(ss, int) for ss in L)
        assert len(set(L)) == len(L)


@given(hxr.vars_to_dims_dicts(), data())
def test_fixed_coords_datasets(vars_to_dims, data):
    elements = None

    all_dims = sorted(set(sum((list(dd) for dd in vars_to_dims.values()), [])))
    coords = data.draw(hxr.xr_coords_dicts(all_dims))

    dtype_d = data.draw(fixed_dictionaries({vv: dtypes() for vv in vars_to_dims}))

    S = hxr.fixed_coords_datasets(vars_to_dims, coords, dtype_d, elements)

    ds = data.draw(S)

    assert list(ds) == list(vars_to_dims.keys())
    assert all(ds[vv].dims == tuple(vars_to_dims[vv]) for vv in vars_to_dims)
    assert all(ds[vv].dtype == np.dtype(dtype_d[vv]) for vv in vars_to_dims)
    for dd in all_dims:
        L = ds.coords[dd].values.tolist()
        assert L == coords[dd]


@given(hxr.vars_to_dims_dicts(), data())
def test_fixed_coords_datasets_no_dtype(vars_to_dims, data):
    elements = None

    all_dims = sorted(set(sum((list(dd) for dd in vars_to_dims.values()), [])))
    coords = data.draw(hxr.xr_coords_dicts(all_dims))

    S = hxr.fixed_coords_datasets(vars_to_dims, coords, dtype=None, elements=elements)

    ds = data.draw(S)

    assert list(ds) == list(vars_to_dims.keys())
    assert all(ds[vv].dims == tuple(vars_to_dims[vv]) for vv in vars_to_dims)
    assert all(ds[vv].dtype == np.dtype(hxr.DEFAULT_DTYPE) for vv in vars_to_dims)
    for dd in all_dims:
        L = ds.coords[dd].values.tolist()
        assert L == coords[dd]


@given(hxr.vars_to_dims_dicts(), sizes(), data())
def test_fixed_datasets(vars_to_dims, sizes, data):
    elements = None
    coords_elements = None
    min_side, max_side = sizes

    # special dims just filled with dim name on coords as test case
    all_dims = sorted(set(sum((list(dd) for dd in vars_to_dims.values()), [])))
    special_dims = data.draw(hxr.subset_lists(all_dims))
    coords_st = {dd: lists(just(dd), min_side, max_side) for dd in special_dims}

    dtype_d = data.draw(fixed_dictionaries({vv: dtypes() for vv in vars_to_dims}))

    S = hxr.fixed_datasets(vars_to_dims, dtype_d, elements, coords_elements, min_side, max_side, coords_st)

    ds = data.draw(S)

    assert list(ds) == list(vars_to_dims.keys())
    assert all(ds[vv].dims == tuple(vars_to_dims[vv]) for vv in vars_to_dims)
    assert all(ds[vv].dtype == np.dtype(dtype_d[vv]) for vv in vars_to_dims)
    assert all(all(ss == dd for ss in ds.coords[dd].values.tolist()) for dd in special_dims)
    for dd in all_dims:
        L = ds.coords[dd].values.tolist()
        assert len(L) >= min_side
        assert (max_side is None) or (len(L) <= max_side)
        assert (dd in special_dims) or all(isinstance(ss, int) for ss in L)
        assert (dd in special_dims) or len(set(L)) == len(L)


@given(hxr.vars_to_dims_dicts(), sizes(allow_none=False), data())
def test_simple_datasets(vars_to_dims, sizes, data):
    elements = None
    min_side, max_side = sizes

    dtype_d = data.draw(fixed_dictionaries({vv: dtypes() for vv in vars_to_dims}))

    S = hxr.simple_datasets(vars_to_dims, dtype_d, elements, min_side, max_side)

    ds = data.draw(S)

    all_dims = sorted(set(sum((list(dd) for dd in vars_to_dims.values()), [])))

    assert list(ds) == list(vars_to_dims.keys())
    assert all(ds[vv].dims == tuple(vars_to_dims[vv]) for vv in vars_to_dims)
    assert all(ds[vv].dtype == np.dtype(dtype_d[vv]) for vv in vars_to_dims)
    for dd in all_dims:
        L = ds.coords[dd].values.tolist()
        assert len(L) >= min_side
        assert (max_side is None) or (len(L) <= max_side)
        assert all(isinstance(ss, int) for ss in L)
        assert L == list(range(len(L)))


@given(dtypes(), sizes(allow_none=False), sizes(allow_none=False), sizes(allow_none=False), data())
def test_datasets(dtype, size_sides, size_vars, size_dims, data):
    elements = None
    coords_elements = None
    min_dims, max_dims = size_dims
    min_side, max_side = size_sides
    min_vars, max_vars = size_vars

    S = hxr.datasets(dtype, elements, coords_elements, min_side, max_side, min_vars, max_vars, min_dims, max_dims)

    ds = data.draw(S)

    all_dims = sorted(set(sum((list(ds[vv].dims) for vv in ds), [])))

    n = len(list(ds))
    assert n >= min_vars
    assert (max_vars is None) or (n <= max_vars)
    assert all(len(ds[vv].dims) >= min_dims for vv in ds)
    assert (max_dims is None) or all(len(ds[vv].dims) <= max_dims for vv in ds)
    assert all(ds[vv].dtype == np.dtype(dtype) for vv in ds)
    for dd in all_dims:
        L = ds.coords[dd].values.tolist()
        assert len(L) >= min_side
        assert (max_side is None) or (len(L) <= max_side)
        assert all(isinstance(ss, int) for ss in L)
        assert len(set(L)) == len(L)
