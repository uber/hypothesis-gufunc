===================
Hypothesis GU Funcs
===================

.. image:: https://api.travis-ci.com/uber/hypothesis-gufunc.png?token=RSemjpisB7uiZv78DVwd&branch=master
    :target: https://travis-ci.com/uber/hypothesis-gufunc

This project is experimental and the APIs are not considered stable.

Only `Python>=3.6` is officially supported, but older versions of Python likely work as well.

This package includes support for strategies which generate arguments to
functions that follow the numpy general universal function API. So, it can
automatically generate the matrices with shapes that follow the shape
constraints. For example, to generate test inputs for `np.dot`, one can use,

.. code-block:: python

  import numpy as np
  from hypothesis import given
  from hypothesis.strategies import floats
  from hypothesis_gufunc.gufunc import gufunc_args

  easy_floats = floats(min_value=-10, max_value=10)

  @given(gufunc_args('(m,n),(n,p)->(m,p)', dtype=np.float_, elements=easy_floats))
  def test_np_dot(args):
      x, y = args
      assert np.allclose(np.dot(x, y), np.dot(y.T, x.T).T)

  test_np_dot()  # Run the test

We also allow for adding extra dimensions that follow the numpy broadcasting
conventions. This allows one to test that the broadcasting follows the
vectorization conventions:

.. code-block:: python

  @given(gufunc_args('(m,n),(n,p)->(m,p)', dtype=np.float_, elements=easy_floats, max_dims_extra=3))
  def test_np_matmul(args):
      x, y = args
      f_vec = np.vectorize(np.matmul, signature='(m,n),(n,p)->(m,p)', otypes=[np.float_])
      assert np.allclose(np.matmul(x, y), f_vec(x, y))

  test_np_matmul()  # Run the test

Providing `max_dims_extra=3` gives up to 3 broadcast compatible dimensions on each of the arguments.

------------------------
Quick Start/Installation
------------------------

Simply install with pip:

.. code-block::

  pip install hypothesis-gufunc

If one would like the same pinned requirements we use during testing, then one can install from the repo with:

.. code-block::

  git clone git@github.com:uber/hypothesis-gufunc.git
  cd hypothesis-gufunc
  pip install -r requirements/base.txt
  pip install -e .

-----------------
Running the Tests
-----------------

The tests for this package can be run by first doing a `cd` to its root directory and then

.. code-block::

  ./test.sh

The script creates a conda environment using the requirements found in `requirements/test.txt`.

---------------------
Hypothesis for Xarray
---------------------

This package also contains an extension to hypothesis to generate xarray data structures.

To install the package with the xarray dependencies install it with pip as

.. code-block::

  pip install hypothesis-gufunc[xarray]

Once installed, one can generate a data array as follows:

.. code-block:: python

  from hypothesis.strategies import integers, lists
  from hypothesis_gufunc.extra.xr import fixed_dataarrays

  S = fixed_dataarrays(("a", "b"), coords_st={"a": lists(integers(0, 3))})
  S.example()

Here, `coords_st` allows one to specify a custom strategy for the coordinates on a per-dimension basis. Likewise, if
one has known coordinates one can call `fixed_coords_dataarrays`; or
`dataarrays` if one wants both the dimensions and coordinates determined by the
strategy.

The package also has the ability to generate a dataset:

.. code-block:: python

  from hypothesis_gufunc.extra.xr import fixed_datasets

  S = fixed_datasets({5: ("a", "b"), "bar": ("b"), "baz": ()}, coords_st={"a": lists(integers(0, 3))})
  S.example()

One can use `fixed_coords_datasets` when the coordinates are determined; or simply
`datasets` to have both the dimensions and coordinates generated.

-----
Links
-----

The `source <https://github.com/uber/hypothesis-gufunc>`_ is hosted on GitHub.

The `documentation <https://hypothesis-gufunc.readthedocs.io/en/latest/>`_ is hosted at Read the Docs.

The main `hypothesis project <https://hypothesis.readthedocs.io/en/latest/>`_.

A description of the numpy
`Generalized Universal Function API <https://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html>`_.

Likewise, the numpy broadcasting rules are described
`here <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_.

The `xarray <http://xarray.pydata.org/en/stable/index.html>`_ project describes data arrays and datasets.

-------
License
-------

This project is licensed under the Apache 2 License - see the LICENSE file for details.
