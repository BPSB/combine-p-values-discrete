Combine discrete *p* values (in Python)
=======================================

This module provides a toolbox for combining *p* values of rank tests and other tests with a discrete null distribution.

When do you need this?
----------------------

This module has a scope similar to SciPy’s `combine_pvalues`_:

* You have a dataset consisting of **independent** sub-datasets. (So this is not about multiple testing or pseudo-replication.)
* For each sub-dataset, you have performed a test investigating the **same** null hypothesis. (Often, this is the same test and the sub-datasets only differ in size.)
* There is no straightforward test to apply to the entire dataset.
* You want a single *p* value for the null hypothesis taking into account the entire dataset, i.e., you want to combine your test results for the sub-datasets.

**However,** `combine_pvalues` assumes that the individual tests are continuous (see below what this means), while applying it to discrete tests will yield a systematically wrong combined *p* value.
For example, for `Fisher’s method`_ it systematically overestimates the *p* value, i.e., you may falsely accept the null hypothesis (false negative).
This module addresses this and thus you should consider it if:

* At least one of the sub-tests is *discrete* with a low number of possible *p* values. What is a “low number” depends on the details, but 30 almost always is.
* The combined *p* value returned by `combine_pvalues` is not very low already.

Also see `comparison`, for a hands-on example, where only combining *p* values with accounting for the discreteness of tests yield the correct result.

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
* Fisher’s exact test and any other test for integer contingency tables.

Tests whose result continuously depends on the samples are continuous.
The most relevant **continuous tests** are:

* all flavours of the *t* test,
* the Kolmogorov–Smirnov test,
* the test for significance of Pearson’s *r*,
* ANOVA.


How this module works
---------------------

To correctly compute the combined *p* value, we need to take into account the null distributions of the individual tests, i.e., what *p* values are possible.
This module determines these values for popular tests or lets you specify them yourself.
Of course, if you have continuous tests in the mix, you can also include them.
Either way, the relevant information is stored in a `CTR` object (“combinable test result”).
These objects can then be combined using the `combine` function.

The difficulty for determining the combined *p* value is convolving the respective null distributions.
While this is analytically possible for continuous tests or a small number of discrete tests, it is requires numerical approximations otherwise.
To perform these approximations, we use a Monte Carlo simulation sampling combinations of individual *p* values.
Thanks to modern computing and NumPy, it is easy to make the number of samples very high and the result very accurate.

A simple example
----------------

.. automodule:: simple_example


.. _comparison:

An extensive example
--------------------

.. automodule:: comparison

What needs to be done
---------------------

This module is work in progress:

* The core structures and two tests are finished.
* Everything you *can* use is thoroughly tested.
* So far, only the sign test and Mann–Whitney *U* test are supported.
* An instruction for implementing your own tests is planned.

Command reference
-----------------

.. automodule:: combine_pvalues_discrete
	:members: CTR, combine, sign_test



.. _combine_pvalues: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html

.. _Fisher’s method: https://en.wikipedia.org/wiki/Fisher%27s_method

