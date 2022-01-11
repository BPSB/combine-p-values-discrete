Combine discrete *p* values (in Python)
=======================================

This module provides a toolbox for applying `Fisher’s method`_ to combine *p* values of rank tests and other tests with a discrete null distribution.

When do you need this?
----------------------

This module has a scope similar to SciPy’s `combine_pvalues`_:

* You have a dataset consisting of **independent** sub-datasets. (So this is not about multiple testing or pseudo-replication.)
* For each sub-dataset, you have performed a test investigating the **same** null hypothesis. (Often, this is the same test and the sub-datasets only differ in size.)
* There is no straightforward test to apply to the entire dataset.
* You want a single *p* value for the null hypothesis taking into account the entire dataset, i.e., you want to combine your test results for the sub-datasets.

**However,** `combine_pvalues` assumes that the individual tests are continuous instead of discrete (see below what these are).
If you apply `combine_pvalues` to *p* values from a discrete test, it will systematically overestimate the combined *p* value, i.e., you may falsely accept the null hypothesis (false negative).
This module addresses this and thus you should consider it if:

* At least one of the sub-tests is *discrete* with a low number of possible *p* values. What is a “low number” depends on the details, but 30 almost always is.
* The combined *p* value returned by `combine_pvalues` (with Fisher’s method) is not very low already.

Note that this module is restricted to `Fisher’s method`_ for combining *p* values.
While `combine_pvalues` also provides alternatives methods, those also assume continuous tests and thus suffer from a similar problem, as there is no way to address this without knowing more about the employed tests.
However, some of the alternative methods may underestimate the *p* value, whereas Fisher’s method consistently overestimates.

Discrete and continuous tests
`````````````````````````````

If the null hypothesis of a given test holds, its *p* values are uniformly distributed on the interval :math:`(0,1]` in the sense that :math:`\text{CDF}(p) = p`.
However, for some tests, there is a limited number of possible outcomes for a given sample size.
For example, the only possible outcomes (*p* values) of the one-sided sign test for a sample size of 5 are
:math:`\frac{ 1}{32}`,
:math:`\frac{ 3}{16}`,
:math:`\frac{ 1}{ 2}`,
:math:`\frac{13}{16}`,
:math:`\frac{31}{32}`, and
:math:`1`,
simply because five numbers can only have so many different (unordered) combinations of signs.
For the purposes of this module, I call these tests *discrete.*
By contrast, for a *continous* test, all values on the interval :math:`(0,1]` are possible outcomes (for any given sample size).

Discrete tests include all `rank tests <https://en.wikipedia.org/wiki/Rank_test>`_, since there is only a finite number of ways to rank a given number of samples.
Moreover, they contain tests of bound integer data.
The most relevant **discrete tests** are:

* the sign test,
* the Mann-Whitney *U* test,
* Wilcoxon’s signed rank test,
* any test based on a ranked correlation such as Kendall’s *τ* and Spearman’s *ρ*,
* the Kruskal–Wallis test,
* Fisher’s exact test and any of its alternatives.

Tests whose result continuously depends on the samples are continuous.
The most relevant **continuous tests** are:

* all flavours of the *t* test,
* the Kolmogorov–Smirnov test,
* the test for significance of Pearson’s *r*,
* ANOVA.


How this module works
---------------------

This modules uses the discrete analog of `Fisher’s method`_.
Like Fisher’s method, it uses the sum of logarithms of individual *p* values as a test statistics:

.. math::

	\sum_{i=1}^n \log(p_i) = \log\left( \prod_{i=1}^n p_i \right )

For continuous tests, we know that the null distribution for each single :math:`p_i` is identical, namely the standard uniform distribution.
All we need to do is convolve it with itself :math:`n` times (in logarithmic space) to obtain the null distribution of the compound statistics.
This is analytically tractable, which is what Fisher did.
For discrete tests, we have different null distributions for each test, which this module convolves numerically.

This module allows you to obtain and store the null distribution for a single test, using either provided functions for common tests or building such functions yourself.
The framework then allows you to combine test results with each other, convolving the respective null distributions, and finally obtain the combined *p* value.
If you have continuous tests in the mix, you can also include them.
At the core is the class `CTR` ("Combined Test Result”), instances of which stores single tests result with their null distributions or combinations thereof.
Two instances can be combined with simple Python multiplication (i.e., what the `*` operator does).


A simple example
----------------

.. automodule:: simple_example
..

Command reference
-----------------

.. automodule:: combine_pvalues_discrete
	:members: CTR, LogPDist, tree_prod, sign_test



.. _combine_pvalues: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html

.. _Fisher’s method: https://en.wikipedia.org/wiki/Fisher%27s_method

