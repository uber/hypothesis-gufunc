[//]: # (pandoc -t beamer slides.md -o slides.pdf --highlight-style=espresso -H make-code-footnotesize.tex)

# Introducing Hypothesis GU Funcs

`pip install hypothesis-gufunc`

# Standard testing in ML

* Standard unit testing difficult in ML models

. . .

* Golden test is most common type

```python
def test_np_dot_golden():
    X = np.array([[1.5, 2], [3, 4]])
    Y = np.array([[5, 6], [7, 8.8]])
    Z = np.dot(X, Y)
    assert np.all(Z == np.array([[21.5, 26.6], [43.0, 53.2]]))
```

* Appears to be few alternatives
* Hard to specify what the correct output is

# Property-based testing

* Surprisingly effective
* Also known as auto-generated tests or fuzz testing
* Generate cases to cover the whole space, then test properties

# Hypothesis

* Popular Python package for property-based testing
* Decorator that lets a strategy build the test cases

```python
def foo(x: int, y: float):
    return x * y

@given(integers(min_value=-10, max_value=10),
       floats(min_value=-10, max_value=10))
def test_foo_rescale(x: int, y: float):
    z = foo(x, y)
    z2 = foo(2 * x, y)
    assert 2 * z == z2

test_foo_rescale()  # Call the test without specifying inputs!
```

# Calls to foo

```
foo(0, 0.000000)
foo(-6, -9.716510)
foo(-8, -8.820218)
foo(-10, -9.436477)
foo(-4, 0.668380)
foo(-6, -5.872543)
foo(7, 2.795894)
foo(9, 3.500834)
foo(-9, 7.423713)
foo(9, 0.839371)
foo(4, 2.357988)
foo(-1, 1.407822)
foo(0, -5.529375)
foo(-10, -0.000000)
foo(5, -0.999203)
foo(8, 1.836821)
foo(4, -0.000000)
foo(1, -10.000000)
foo(-8, 10.000000)
foo(-9, -10.000000)
foo(-2, -4.890470)
foo(1, -1.107240)
foo(-2, 2.793453)
foo(8, -8.820218)
foo(-5, -7.838072)
foo(5, -9.654960)
foo(-6, 1.407822)
foo(-2, -3.425022)
foo(-8, 8.632516)
foo(-5, -10.000000)
foo(-10, 7.215488)
foo(7, -9.970599)
foo(-3, 3.026636)
foo(1, 9.828312)
foo(0, -1.321247)
foo(5, -3.337223)
foo(9, -9.147613)
foo(0, -7.014612)
foo(5, 7.193583)
foo(7, -6.156863)
foo(-4, -0.069953)
foo(0, 0.655373)
foo(-1, -2.824547)
foo(10, 7.042401)
foo(1, 0.000000)
foo(0, 7.478717)
foo(-4, 8.094905)
foo(4, -0.516329)
foo(3, -2.548281)
foo(-1, 1.228075)
foo(7, -1.214610)
foo(-5, 5.314360)
foo(5, 4.251199)
foo(-6, 4.772800)
foo(7, -8.980209)
foo(-7, -5.682882)
foo(4, 2.009149)
foo(-7, 10.000000)
foo(9, -0.000000)
foo(-8, -0.129252)
foo(0, -10.000000)
foo(3, 0.685860)
foo(-9, 1.008407)
foo(-2, -6.667133)
foo(7, -2.570148)
foo(-7, -4.965941)
foo(10, 3.246200)
foo(6, 0.000000)
foo(-2, -6.384194)
foo(-6, 1.140648)
foo(8, 5.271106)
foo(8, -9.010244)
foo(-5, 3.246200)
foo(-9, 0.301835)
foo(-2, 7.521622)
foo(-7, 1.212029)
foo(-9, 7.366118)
foo(-9, 8.723718)
foo(0, 6.533770)
foo(6, -3.443954)
foo(4, 8.220348)
foo(0, -1.065632)
foo(0, 7.803156)
foo(-6, -2.788803)
foo(8, 4.420570)
foo(5, 8.419269)
foo(-7, -9.752219)
foo(6, -3.068181)
foo(7, 8.932106)
foo(-8, -8.955573)
foo(3, 9.581830)
foo(8, 7.772419)
foo(-5, -0.000000)
foo(0, 6.213662)
foo(-8, 6.884335)
foo(3, -6.691430)
foo(-3, -0.000000)
foo(0, 1.651323)
foo(0, -3.251268)
foo(-1, -9.256589)
```

# Example properties

* Check output against a slower implementation
* Other properties such as "z is sorted"
* Call an inverse function

Classic example is encoder-decoder test:

```python
encode = urllib.parse.quote
decode = urllib.parse.unquote

@given(text())
def test_decode_inverts_encode(s):
    assert decode(encode(s)) == s
```

# NumPy support

* Unlike encoder-decoder, ML relies on NumPy
    * or PyTorch/TensorFlow
* Hypothesis has basic NumPy support (thanks to Stripe)

```python
easy_floats = floats(min_value=-10, max_value=10)

@given(arrays(np.float, (3, 3), elements=easy_floats),
       arrays(np.float, (3, 2), elements=easy_floats))
def test_np_dot(x, y):
    assert np.allclose(np.dot(x, y), np.dot(y.T, x.T).T)
```

# Calls to dot

```
dot([[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]],[[0. 0.]
 [0. 0.]
 [0. 0.]])
dot([[ 8.2492118  -6.92579821 -8.68665109]
 [ 1.94804632 -2.86664605 -6.87985808]
 [-6.92579821 -6.92579821 -8.41916786]],[[ 0.53746735  0.53746735]
 [-5.38702148 -8.63738731]
 [ 0.53746735 -5.53231572]])
dot([[ 0.41548838  0.41548838  0.41548838]
 [-1.14967951  0.41548838  0.41548838]
 [ 0.41548838  0.41548838  0.41548838]],[[5.01139295 5.01139295]
 [5.01139295 5.01139295]
 [5.01139295 5.01139295]])
dot([[ 3.30924356  3.30924356 -0.15061277]
 [ 3.30924356  3.30924356  3.30924356]
 [ 3.30924356  3.30924356  3.30924356]],[[-3.81918844 -3.81918844]
 [-8.28837652  3.7540776 ]
 [-3.81918844 -3.81918844]])
dot([[-5.60616965  3.78955301  7.16699955]
 [ 3.78955301  3.78955301  4.08845095]
 [ 0.27653055  3.78955301  3.78955301]],[[6.76177236 6.44185449]
 [2.41010892 2.41010892]
 [2.41010892 2.41010892]])
dot([[-6.5624438 -6.5624438 -6.5624438]
 [-6.5624438 -6.5624438 -6.5624438]
 [-6.5624438 -6.5624438 -6.5624438]],[[ 8.08836247  4.377592  ]
 [ 8.08836247  8.08836247]
 [ 8.08836247 -7.45708831]])
dot([[-8.54774824 -8.54774824 -8.54774824]
 [-8.54774824 -8.54774824 -8.54774824]
 [-8.54774824 -8.54774824  8.68646006]],[[2.18224439 1.03799656]
 [3.07381712 1.03799656]
 [1.03799656 1.03799656]])
dot([[9.01940666 9.01940666 1.64506765]
 [9.01940666 9.01940666 9.01940666]
 [9.01940666 9.01940666 9.01940666]],[[-4.65569538  1.46824761]
 [ 5.5061455   5.5061455 ]
 [ 5.5061455   5.5061455 ]])
 ```

# Hypothesis GU functions extension

* Hypothesis' NumPy support insufficient
* Want variable size inputs with mutual size constraints
* Can specify via NumPy general universal (GU) function API
* `np.dot` has signature `'(m,n),(n,p)->(m,p)'`

Consider:

```python
easy_floats = floats(min_value=-10, max_value=10)

@given(gufunc_args("(m,n),(n,p)->(m,p)", dtype=np.float_,
                   elements=easy_floats))
def test_np_dot(args):
    x, y = args
    assert np.allclose(np.dot(x, y), np.dot(y.T, x.T).T)
```

# Testing broadcasting

* Difficult functionality to test is broadcasting
* Pyro team: `>50%` of bugs in Pyro are due to broadcasting errors

* NumPy defines a convention for vectorization
* For example, `(3,2,m,n)` instead of `(m,n)`
    * Should be equivalent to looping over the extra dimensions
* `np.vectorize` defines the correct way to vectorize

# Testing broadcasting II

* Hypothesis-gufuncs can generate cases with extra dimensions that are broadcast-compatible

```python
easy_floats = floats(min_value=-10, max_value=10)

@given(gufunc_args("(m,n),(n,p)->(m,p)", dtype=np.float_,
                   elements=easy_floats, max_dims_extra=3))
def test_np_matmul(args):
    x, y = args
    f_vec = np.vectorize(np.matmul,
                         signature="(m,n),(n,p)->(m,p)",
                         otypes=[np.float_])
    assert np.allclose(np.matmul(x, y), f_vec(x, y))
```

* `max_dims_extra=3` gives up to three broadcast compatible dimensions on each of the arguments

# Finding bugs in released code

Rediscovered issues in NumPy that have now been fixed:

* NumPy issue #7014: inconsistent broadcasting in `np.isclose(0, np.inf)`
* NumPy issue #9884: problems with the case `np.unravel_index(0, ())`

# Torch support

Generate Torch variables for testing with the following recipe:

```python
easy_floats = floats(min_value=-10, max_value=10)

def torchify(args):
    args = tuple(torch.tensor(X) for X in args)
    return args

@given(gufunc_args("(m,n),(n,p)->(m,p)", dtype=np.float_,
                   elements=easy_floats,
                   min_side=1).map(torchify))
def test_torch_matmul(args):
    x, y = args
    assert torch.allclose(torch.matmul(x, y),
                          torch.matmul(y.T, x.T).T)
```

# Get started

* Use `pip install hypothesis-gufunc`
* Docs: `https://hypothesis-gufunc.readthedocs.io`
* GitHub: `https://github.com/uber/hypothesis-gufunc`
* PyPI: `https://pypi.org/project/hypothesis-gufunc`
