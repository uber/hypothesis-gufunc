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
"""This module implements strategies for creating :class:`xarray:xarray.DataArray` and
:class:`xarray:xarray.Dataset` objects.
"""
import string
from collections import OrderedDict, defaultdict

import xarray as xr
from hypothesis.extra.numpy import arrays, order_check
from hypothesis.internal.validation import check_valid_bound
from hypothesis.strategies import fixed_dictionaries, floats, integers, lists, nothing, sampled_from, text, tuples

DEFAULT_DTYPE = int
DEFAULT_SIDE = 5
DEFAULT_DIMS = 5
DEFAULT_VARS = 5


def _check_valid_size_interval(min_size, max_size, name, floor=0):
    """Check valid for integers strategy and array shapes."""
    # same checks as done in integers
    check_valid_bound(min_size, name)
    check_valid_bound(max_size, name)
    if max_size is None:
        order_check(name, floor, min_size, min_size)
    else:
        order_check(name, floor, min_size, max_size)


def _easy_text():
    return text(alphabet=string.ascii_lowercase, min_size=0, max_size=5)


def _hashable():
    S = floats() | integers() | _easy_text()
    return S


def _get_all_dims(vars_to_dims):
    all_dims = sorted(set(sum((list(dd) for dd in vars_to_dims.values()), [])))
    return all_dims


xr_dims = _easy_text
xr_vars = _hashable


def subset_lists(L, min_size=0, max_size=None):
    """Strategy to generate a subset of a `list`.

    This should be built in to hypothesis (see hypothesis issue #1115), but was rejected.

    Parameters
    ----------
    L : list
        List of elements we want to get a subset of.
    min_size : int
        Minimum size of the resulting subset list.
    max_size : int or None
        Maximum size of the resulting subset list.

    Returns
    -------
    L : list
        List that is subset of `L` with all unique elements.
    """
    _check_valid_size_interval(min_size, max_size, "subset list size")
    uniq_len = len(set(L))
    order_check("input list size", 0, min_size, uniq_len)

    max_size = uniq_len if max_size is None else min(uniq_len, max_size)

    # Avoid deprecation warning HypothesisDeprecationWarning: sampled_from()
    elements_st = nothing() if uniq_len == 0 else sampled_from(L)

    S = lists(elements=elements_st, min_size=min_size, max_size=max_size, unique=True)
    return S


def xr_dim_lists(min_dims=0, max_dims=DEFAULT_DIMS):
    """Generate `list` of dimension names for a :class:`xarray:xarray.DataArray`.

    Parameters
    ----------
    min_dims : int
        Minimum size of the resulting dimension list.
    max_dims : int or None
        Maximum size of the resulting dimension list.

    Returns
    -------
    L : list(str)
        List of dimension names.
    """
    _check_valid_size_interval(min_dims, max_dims, "dimensions")
    S = lists(elements=xr_dims(), min_size=min_dims, max_size=max_dims, unique=True)
    return S


def xr_var_lists(min_vars=0, max_vars=DEFAULT_VARS):
    """Generate `list` of variable names for a :class:`xarray:xarray.Dataset`.

    Parameters
    ----------
    min_vars : int
        Minimum size of the resulting variable list.
    max_vars : int or None
        Maximum size of the resulting variable list.

    Returns
    -------
    L : list(typing.Hashable)
        List of variable names.
    """
    _check_valid_size_interval(min_vars, max_vars, "variables")
    S = lists(elements=xr_vars(), min_size=min_vars, max_size=max_vars, unique=True)
    return S


def _vars_and_dims_pairs(min_vars=0, max_vars=DEFAULT_VARS, min_dims=0, max_dims=DEFAULT_DIMS):
    """Generate both variable and dimension names.

    xarray requires that there are no name collisions between the two.
    """

    def no_overlap(args):
        vars_, dims = args
        # Dataset does not allow the same names for variable and dimensions, so we filter by looking at intersection
        ok = len(set(dims).intersection(vars_)) == 0
        return ok

    S = tuples(xr_var_lists(min_vars, max_vars), xr_dim_lists(min_dims, max_dims)).filter(no_overlap)
    return S


def vars_to_dims_dicts(min_vars=0, max_vars=DEFAULT_VARS, min_dims=0, max_dims=DEFAULT_DIMS):
    """Generate mapping of variable name to `list` of dimensions, which is compatible with building a
    :class:`xarray:xarray.Dataset`.

    Parameters
    ----------
    min_vars : int
        Minimum size of the resulting variable list.
    max_vars : int or None
        Maximum size of the resulting variable list.
    min_dims : int
        Minimum size of the resulting dimension list.
    max_dims : int or None
        Maximum size of the resulting dimension list.

    Returns
    -------
    D : dict(typing.Hashable, list(str))
        Mapping of variable names to `list` of dimensions, which can be fed to constructor for a
        :class:`xarray:xarray.Dataset`.
    """
    _check_valid_size_interval(min_vars, max_vars, "variables")
    _check_valid_size_interval(min_dims, max_dims, "dimensions")

    def map_dict(args):
        vars_, dims = args
        dim_st = subset_lists(dims, min_size=min_dims, max_size=max_dims)
        S = fixed_dictionaries(OrderedDict([(vv, dim_st) for vv in vars_]))
        return S

    S = _vars_and_dims_pairs(min_vars, max_vars, min_dims, max_dims).flatmap(map_dict)
    return S


def xr_coords(elements=None, min_side=0, max_side=DEFAULT_SIDE, unique=True):
    """Generate values for the coordinates in a :class:`xarray:xarray.DataArray`.

    Non-unique coords do not make much sense, but xarray allows it. So we should be able to generate it.

    Parameters
    ----------
    elements : SearchStrategy or None
        Strategy to fill the elements of coordinates. Uses :func:`hypothesis:hypothesis.strategies.integers` by default.
    min_side : int
        Minimum length of coordinates array.
    max_side : int or None
        Maximum length of coordinates array.
    unique : bool
        If all coordinate values should be unique. `xarray` allows non-unique values, but it makes no sense.

    Returns
    -------
    L : list
        The coordinates filled with samples from `elements`.
    """
    _check_valid_size_interval(min_side, max_side, "side")

    if elements is None:
        elements = integers()

    S = lists(elements=elements, min_size=min_side, max_size=max_side, unique=unique)
    return S


def simple_coords(min_side=0, max_side=DEFAULT_SIDE):
    """Generate a simple coordinate for a :class:`xarray:xarray.DataArray`.

    A simple coordinate is one in which the values go: 0, 1, ..., n.

    Parameters
    ----------
    min_side : int
        Minimum length of coordinates array.
    max_side : int or None
        Maximum length of coordinates array.

    Returns
    -------
    L : list(int)
        The coordinates filled with values of: ``list(range(len(L)))``.
    """
    _check_valid_size_interval(min_side, max_side, "side")

    n = integers(min_value=min_side, max_value=max_side)
    S = n.map(range).map(list)  # Always make list to be consistent with xr_coords
    return S


def xr_coords_dicts(dims, elements=None, min_side=0, max_side=DEFAULT_SIDE, unique_coords=True, coords_st={}):
    """Build a dictionary of coordinates for the purpose of building a :class:`xarray:xarray.DataArray`.

    `xarray` allows some dims to not have any specified coordinate. This strategy assigns a coord to every dimension. If
    we really want to test those possibilities we need to take a subset of the `dict` that is sampled from this
    strategy.

    Parameters
    ----------
    dims : list(str)
        Dimensions we need to generate coordinates for.
    elements : SearchStrategy or None
        Strategy to fill the elements of coordinates. Uses `integers` by default.
    min_side : int
        Minimum length of coordinates array.
    max_side : int or None
        Maximum length of coordinates array.
    unique_coords : bool
        If all coordinate values should be unique. `xarray` allows non-unique values, but it makes no sense.
    coords_st : dict(str, SearchStrategy)
        Special strategies for filling specific dimensions. Use the dimension name as the key and the strategy for
        generating the coordinate as the value.

    Returns
    -------
    coords : dict(str, list)
        Dictionary mapping dimension name to its coordinate values (a list with elements from the `elements` strategy).
    """
    _check_valid_size_interval(min_side, max_side, "side")

    default_st = xr_coords(elements=elements, min_side=min_side, max_side=max_side, unique=unique_coords)
    C = OrderedDict([(dd, coords_st.get(dd, default_st)) for dd in dims])
    S = fixed_dictionaries(C)
    return S


def fixed_coords_dataarrays(dims, coords, dtype=DEFAULT_DTYPE, elements=None):
    """Generate a :class:`xarray:xarray.DataArray` with coordinates that are fixed a-priori.

    Parameters
    ----------
    dims : list(str)
        Dimensions we need to generate coordinates for.
    coords : dict(str, list)
        Dictionary mapping dimension name to its coordinate values.
    dtype : type
        Data type for values in the :class:`xarray:xarray.DataArray`. This can be anything understood by
        :func:`hypothesis:hypothesis.extra.numpy.arrays`.
    elements : SearchStrategy or None
        Strategy to fill the elements of the :class:`xarray:xarray.DataArray`. If `None`, a default is selected based
        on `dtype`.

    Returns
    -------
    da : :class:`xarray:xarray.DataArray`
        :class:`xarray:xarray.DataArray` generated with the specified coordinates and elements from the specified
        strategy.
    """
    shape = [len(coords[dd]) for dd in dims]
    data_st = arrays(dtype, shape, elements=elements)
    coords = {dd: cc for dd, cc in coords.items() if dd in dims}
    S = data_st.map(lambda data: xr.DataArray(data, coords=coords, dims=dims))
    return S


def fixed_dataarrays(
    dims, dtype=DEFAULT_DTYPE, elements=None, coords_elements=None, min_side=0, max_side=DEFAULT_SIDE, coords_st={}
):
    """Generate :class:`xarray:xarray.DataArray` with dimensions (but not coordinates) that are fixed a-priori.

    Parameters
    ----------
    dims : list(str)
        Dimensions we need to generate coordinates for.
    dtype : type
        Data type for values in the :class:`xarray:xarray.DataArray`. This can be anything understood by
        :func:`hypothesis:hypothesis.extra.numpy.arrays`.
    elements : SearchStrategy or None
        Strategy to fill the elements of the :class:`xarray:xarray.DataArray`. If `None`, a default is selected based
        on `dtype`.
    coords_elements : SearchStrategy or None
        Strategy to fill the elements of coordinates.
    min_side : int
        Minimum side length of the :class:`xarray:xarray.DataArray`.
    max_side : int or None
        Maximum side length of the :class:`xarray:xarray.DataArray`.
    coords_st : dict(str, SearchStrategy)
        Special strategies for filling specific dimensions. Use the dimension name as the key and the strategy for
        generating the coordinate as the value.

    Returns
    -------
    da : :class:`xarray:xarray.DataArray`
        :class:`xarray:xarray.DataArray` generated with the dimensions and elements from the specified strategy.
    """
    _check_valid_size_interval(min_side, max_side, "side")

    coords_st = xr_coords_dicts(
        dims, elements=coords_elements, min_side=min_side, max_side=max_side, coords_st=coords_st
    )
    S = coords_st.flatmap(lambda C: fixed_coords_dataarrays(dims, C, dtype=dtype, elements=elements))
    return S


def simple_dataarrays(dims, dtype=DEFAULT_DTYPE, elements=None, min_side=0, max_side=DEFAULT_SIDE):
    """Generate a :class:`xarray:xarray.DataArray` with dimensions that are fixed a-priori and simple coordinates.

    Parameters
    ----------
    dims : list(str)
        Dimensions we need to generate coordinates for.
    dtype : type
        Data type for values in the :class:`xarray:xarray.DataArray`. This can be anything understood by
        :func:`hypothesis:hypothesis.extra.numpy.arrays`.
    elements : SearchStrategy or None
        Strategy to fill the elements of the :class:`xarray:xarray.DataArray`. If `None`, a default is selected based on
        `dtype`.
    min_side : int
        Minimum side length of the :class:`xarray:xarray.DataArray`.
    max_side : int or None
        Maximum side length of the :class:`xarray:xarray.DataArray`.

    Returns
    -------
    da : :class:`xarray:xarray.DataArray`
        :class:`xarray:xarray.DataArray` generated with the dimensions, simple coordinates, and elements from the
        specified strategy.
    """
    _check_valid_size_interval(min_side, max_side, "side")

    default_st = simple_coords(min_side=min_side, max_side=max_side)
    coords_st = OrderedDict([(dd, default_st) for dd in dims])
    S = fixed_dataarrays(dims, dtype=dtype, elements=elements, coords_st=coords_st)
    return S


def dataarrays(
    dtype=DEFAULT_DTYPE,
    elements=None,
    coords_elements=None,
    min_side=0,
    max_side=DEFAULT_SIDE,
    min_dims=0,
    max_dims=DEFAULT_DIMS,
):
    """Generate a :class:`xarray:xarray.DataArray` with no dimensions or coordinates fixed a-priori.

    Parameters
    ----------
    dtype : type
        Data type for values in the :class:`xarray:xarray.DataArray`. This can be anything understood by
        :func:`hypothesis:hypothesis.extra.numpy.arrays`.
    elements : SearchStrategy or None
        Strategy to fill the elements of the :class:`xarray:xarray.DataArray`. If `None`, a default is selected based on
        `dtype`.
    coords_elements : SearchStrategy or None
        Strategy to fill the elements of coordinates.
    min_side : int
        Minimum side length of the :class:`xarray:xarray.DataArray`.
    max_side : int or None
        Maximum side length of the :class:`xarray:xarray.DataArray`.
    min_dims : int
        Minimum number of dimensions.
    max_dims : int or None
        Maximum number of dimensions.

    Returns
    -------
    da : :class:`xarray:xarray.DataArray`
        :class:`xarray:xarray.DataArray` generated with the dimensions, simple coordinates, and elements from the
        specified strategies.
    """
    _check_valid_size_interval(min_side, max_side, "side")
    _check_valid_size_interval(min_dims, max_dims, "dimensions")

    def mapper(D):
        S = fixed_dataarrays(
            D, dtype=dtype, elements=elements, coords_elements=coords_elements, min_side=min_side, max_side=max_side
        )
        return S

    dims_st = xr_dim_lists(min_dims, max_dims)
    S = dims_st.flatmap(mapper)
    return S


def fixed_coords_datasets(vars_to_dims, coords, dtype=None, elements=None):
    """Generate a :class:`xarray:xarray.Dataset` where the variables, dimensions, and coordinates are specified a-priori.

    Parameters
    ----------
    vars_to_dims : dict(typing.Hashable, list(str))
        Mapping of variable names to list of dimensions, which can be fed to constructor for a
        :class:`xarray:xarray.Dataset`.
    coords : dict(str, list)
        Dictionary mapping dimension name to its coordinate values.
    dtype : dict(typing.Hashable, type) or None
        Dictionary mapping variables names to the data type for that variable's elements.
    elements : SearchStrategy or None
        Strategy to fill the elements of the :class:`xarray:xarray.Dataset`. If `None`, a default is selected based on
        `dtype`.

    Returns
    -------
    ds : :class:`xarray:xarray.Dataset`
        :class:`xarray:xarray.Dataset` with the specified variables, dimensions, and coordinates.
    """
    if dtype is None:
        dtype = defaultdict(lambda: DEFAULT_DTYPE)

    C = OrderedDict([(vv, fixed_coords_dataarrays(dd, coords, dtype[vv], elements)) for vv, dd in vars_to_dims.items()])
    data_st = fixed_dictionaries(C)
    S = data_st.map(lambda data: xr.Dataset(data, coords=coords))
    return S


def fixed_datasets(
    vars_to_dims, dtype=None, elements=None, coords_elements=None, min_side=0, max_side=DEFAULT_SIDE, coords_st={}
):
    """Generate :class:`xarray:xarray.Dataset` where the variables and dimensions (but not coordinates) are specified
    a-priori.

    Parameters
    ----------
    vars_to_dims : dict(typing.Hashable, list(str))
        Mapping of variable names to list of dimensions, which can be fed to constructor for a
        :class:`xarray:xarray.Dataset`.
    dtype : dict(typing.Hashable, type) or None
        Dictionary mapping variables names to the data type for that variable's elements.
    elements : SearchStrategy or None
        Strategy to fill the elements of the :class:`xarray:xarray.Dataset`. If `None`, a default is selected based on
        `dtype`.
    coords_elements : SearchStrategy or None
        Strategy to fill the elements of coordinates.
    min_side : int
        Minimum side length of the :class:`xarray:xarray.Dataset`.
    max_side : int or None
        Maximum side length of the :class:`xarray:xarray.Dataset`.
    coords_st : dict(str, SearchStrategy)
        Special strategies for filling specific dimensions. Use the dimension name as the key and the strategy for
        generating the coordinate as the value.

    Returns
    -------
    ds: :class:`xarray:xarray.Dataset`
        :class:`xarray:xarray.Dataset` with the specified variables and dimensions.
    """
    _check_valid_size_interval(min_side, max_side, "side")

    all_dims = _get_all_dims(vars_to_dims)
    coords_st = xr_coords_dicts(
        all_dims, elements=coords_elements, min_side=min_side, max_side=max_side, coords_st=coords_st
    )
    S = coords_st.flatmap(lambda C: fixed_coords_datasets(vars_to_dims, C, dtype=dtype, elements=elements))
    return S


def simple_datasets(vars_to_dims, dtype=None, elements=None, min_side=0, max_side=DEFAULT_SIDE):
    """Generate :class:`xarray:xarray.Dataset` with variables and dimensions that are fixed a-priori and simple
    coordinates.

    Parameters
    ----------
    vars_to_dims : dict(typing.Hashable, list(str))
        Mapping of variable names to list of dimensions, which can be fed to constructor for a
        :class:`xarray:xarray.Dataset`.
    dtype : dict(typing.Hashable, type) or None
        Dictionary mapping variables names to the data type for that variable's elements.
    elements : SearchStrategy or None
        Strategy to fill the elements of the :class:`xarray:xarray.Dataset`. If `None`, a default is selected based on
        `dtype`.
    min_side : int
        Minimum side length of the :class:`xarray:xarray.Dataset`.
    max_side : int or None
        Maximum side length of the :class:`xarray:xarray.Dataset`.

    Returns
    -------
    ds: :class:`xarray:xarray.Dataset`
        A :class:`xarray:xarray.Dataset` with the specified variables and dimensions, and simple coordinates.
    """
    _check_valid_size_interval(min_side, max_side, "side")

    all_dims = _get_all_dims(vars_to_dims)
    default_st = simple_coords(min_side=min_side, max_side=max_side)
    coords_st = OrderedDict([(dd, default_st) for dd in all_dims])
    S = fixed_datasets(vars_to_dims, dtype=dtype, elements=elements, coords_st=coords_st)
    return S


def datasets(
    dtype=DEFAULT_DTYPE,
    elements=None,
    coords_elements=None,
    min_side=0,
    max_side=DEFAULT_SIDE,
    min_vars=0,
    max_vars=DEFAULT_VARS,
    min_dims=0,
    max_dims=DEFAULT_DIMS,
):
    """Generate a :class:`xarray:xarray.Dataset` with no variables, dimensions, or coordinates fixed a-priori.

    We could also allow a strategy with a different data type per variable, but until there is a use case for that, we
    will leave `dtype` as a scalar input.

    Parameters
    ----------
    dtype : type
        Data type used to fill the elements of the :class:`xarray:xarray.Dataset`.
    elements : SearchStrategy or None
        Strategy to fill the elements of the :class:`xarray:xarray.Dataset`. If `None`, a default is selected based on
        `dtype`.
    coords_elements : SearchStrategy or None
        Strategy to fill the elements of coordinates.
    min_side : int
        Minimum side length of the :class:`xarray:xarray.Dataset`.
    max_side : int or None
        Maximum side length of the :class:`xarray:xarray.Dataset`.
    min_vars : int
        Minimum number of variables.
    max_vars : int or None
        Maximum number of variables.
    min_dims : int
        Minimum number of dimensions.
    max_dims : int or None
        Maximum number of dimensions.

    Returns
    -------
    ds : :class:`xarray:xarray.Dataset`
        :class:`xarray:xarray.Dataset` generated with the variables, dimensions, coordinates, and elements from the
        specified strategies.
    """
    _check_valid_size_interval(min_side, max_side, "side")
    _check_valid_size_interval(min_vars, max_vars, "variables")
    _check_valid_size_interval(min_dims, max_dims, "dimensions")

    dtype_d = defaultdict(lambda: dtype)
    vars_to_dims = vars_to_dims_dicts(min_vars, max_vars, min_dims, max_dims)

    def mapper(V):
        S = fixed_datasets(
            V, dtype=dtype_d, elements=elements, coords_elements=coords_elements, min_side=min_side, max_side=max_side
        )
        return S

    S = vars_to_dims.flatmap(mapper)
    return S
