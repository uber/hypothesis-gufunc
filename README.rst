===================
Hypothesis GU Funcs
===================

.. image:: https://api.travis-ci.com/uber/hypothesis-gufunc.png?token=RSemjpisB7uiZv78DVwd&branch=master
    :target: https://travis-ci.com/uber/hypothesis-gufunc

This project is experimental and the APIs are not considered stable.

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

Checkout this repo, and `cd` to its root directory. Then install with

.. code-block::

  pip install -e .

If one would like the same pinned requirements we use during testing, then install with

.. code-block::

  pip install requirements/base.txt
  pip install -e .

-----------------
Running the Tests
-----------------

The tests for this package can be run by first doing a `cd` to its root directory and then

.. code-block::

  ./test.sh

The requirements used in testing can be found in `requirements/test.txt`.

-----
Links
-----

The main `hypothesis project <https://hypothesis.readthedocs.io/en/latest/>`_.

A description of the numpy
`Generalized Universal Function API <https://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html>`_.

Likewise, the numpy broadcasting rules are described
`here <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_.

-------
License
-------

This project is licensed under the Apache 2 License - see the LICENSE file for details.
