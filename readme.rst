===================
Hypothesis GU Funcs
===================

This package includes support for strategies which generate arguments to
functions that follow the numpy general universal function API. So, it can
automatically generate the matrices with shapes that follow the shape
constraints. For example, to generate test inputs for `np.dot`, one can use,

.. code-block:: python

  @gufunc_args('(m,n),(n,p)->(m,p)', dtype=np.float_, elements=floats())

We also allow for adding extra dimensions that follow the numpy broadcasting
conventions via

.. code-block:: python
  @gufunc_args('(m,n),(n,p)->(m,p)', dtype=np.float_, elements=floats(), max_dims_extra=3)

This can be used when checking if a function follows the correct numpy
broadcasting semantics.

------------------------
Quick Start/Installation
------------------------
Checkout this repo, and cd to its root directory. Then install with

.. code-block::

  pip install -e .

If one would like the same pinned requirements we use during testing, then install with

.. code-block::

  pip install requirements/base.txt
  pip install -e .

-----------------
Running the Tests
-----------------

The tests for this package can be run with:

.. code-block::

  pip install requirements/test.txt
  pip install -e .
  ./test.sh

-----
Links
-----

The main `hypothesis project <https://hypothesis.readthedocs.io/en/latest/>`_.

A description of the numpy
`Generalized Universal Function API <https://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html>`_.

Likewise, the numpy broadcasting rules are described
`here <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_.
