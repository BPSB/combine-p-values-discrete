Combine discrete *p* values (in Python)
=======================================

This module provides a toolbox for combining *p* values of rank tests and other tests with a discrete null distribution.
The focus lies on the analysis of segmented datasets as opposed to a meta-analysis of separate studies, although the latter is also possible.

Installation
------------

To install from PyPI use something along the lines of:

	.. code-block:: bash
	
		pip install combine-pvalues-discrete

Or use the following to directly install from GitHub:

	.. code-block:: bash
	
		pip install git+git://github.com/BPSB/combine-p-values-discrete

When do you need this?
----------------------

*If you want a more hands-on introduction on what kind of problems this module handles and that it can make a difference, feel free to first read:* `comparison`.

This module has a scope similar to SciPy’s `combine_pvalues`_:

* You have a dataset consisting of **independent** sub-datasets. (So this is not about multiple testing or pseudo-replication.)
* For each sub-dataset, you have performed a test investigating the **same or similar** research hypothesis. (Often, this is the same one-sided test and the sub-datasets only differ in size.)
* There is no straightforward test to apply to the entire dataset.
* You want a single *p* value for the null hypothesis taking into account the entire dataset, i.e., you want to combine your test results for the sub-datasets.

**However,** SciPy’s `combine_pvalues` assumes that the individual tests are continuous (see below for a definition), and applying it to discrete tests will yield a systematically wrong combined *p* value [Kincaid1962]_, [Mielke2004]_.
For example, for `Fisher’s method`_ it systematically overestimates the *p* value, i.e., you may falsely accept the null hypothesis (false negative).
This module addresses this and thus you should consider it if:

* At least one of the sub-tests is *discrete* with a low number of possible *p* values. What is a “low number” depends on the details, but 30 almost always is.
* The combined *p* value returned by `combine_pvalues` is not very low already.

See `comparison` for an example, where combining *p* values only yields the correct result when we account for the discreteness of tests.

**Also,** as a side product, this module also implements Monte Carlo-based **weighted** variants of methods other than Stouffer’s, which `combine_pvalues` does not provide.

Discrete and continuous tests
`````````````````````````````

If the null hypothesis of a given test holds, its *p* values are uniformly distributed on the interval :math:`(0,1]` in the sense that :math:`\text{CDF}(p_0) ≡ P(p≤p_0) = p_0`.
However, for some tests, there is a limited number of possible outcomes for a given sample size.
For the purposes of this module, I call such tests *discrete.*
By contrast, for a *continuous* test, all values on the interval :math:`(0,1]` are possible outcomes (for any given sample size).

For example, the `sign test <https://en.wikipedia.org/wiki/Sign_test>`_ is discrete:
Its one-sample variant evaluates how often each sign occurs in the dataset.
Therefore, for a sample size of five, every dataset can be boiled down to one of six possible scenarios:
:math:`[++++++]`,
:math:`[+++++-]`,
…,
:math:`[------]`.
For the one-sided test, these correspond to the *p* values
:math:`\frac{ 1}{32}`,
:math:`\frac{ 3}{16}`,
:math:`\frac{ 1}{ 2}`,
:math:`\frac{13}{16}`,
:math:`\frac{31}{32}`, and
:math:`1`.
These six *p* values are the only possible outcomes of the test.

By contrast, the *t* test is continuous:
Even for a sample size of two, every value in the interval :math:`(0,1]` can be obtained with the right data.
For example, for the dataset :math:`[98,99]`, the one-sided, one-sample variant of the test yields :math:`p=0.0016`.

Discrete tests include all `rank tests <https://en.wikipedia.org/wiki/Rank_test>`_, since there is only a finite number of ways to rank a given number of samples.
Moreover, they contain tests of bound integer data.
The most relevant **discrete tests** are:

* the sign test,
* Wilcoxon’s signed rank test,
* the Mann–Whitney *U* test,
* any test based on a ranked correlation such as Kendall’s *τ* and Spearman’s *ρ*,
* Boschloo’s exact test and any other test for integer contingency tables.

Tests whose result continuously depends on the samples are continuous.
The most relevant **continuous tests** are:

* all flavours of the *t* test,
* the test for significance of Pearson’s *r*,

Tests such as the Kruskal–Wallis test, ANOVA or the Kolmogorov–Smirnov test are not listed  above because I cannot imagine a scenario where combining their *p* values makes sense.

How this module works
---------------------

To correctly compute the combined *p* value, we need to take into account the null distributions of the individual tests, i.e., what *p* values are possible.
This module determines these values for popular tests or lets you specify them yourself.
Of course, if you have continuous tests in the mix, you can also include them.
Either way, the relevant information is stored in a `CTR` object (“combinable test result”).
These objects can then be combined using the `combine` function.

The difficulty for determining the combined *p* value is convolving the respective null distributions.
While this is analytically possible for continuous tests or a small number of discrete tests, it requires numerical approximations otherwise due to a combinatorial explosion (e.g., even for the small dataset in `comparison`, we we would have to handle 53508000 combinations).
To perform these approximations, we use a Monte Carlo sampling of combinations of individual *p* values (from the respective null distributions).
Thanks to modern computing and NumPy, it is easy to make the number of samples very high and the result very accurate.

.. _complements:

Complements
```````````

In several cases, this module uses the complement *q* of a *p* value.
For example, combining methods such as Pearson’s or Mudholkar’s and George’s use it as part of their statistics.
For continuous tests, this complement is straightforwardly computed as :math:`q = 1-p`.
However, for discrete tests this leads to implausible results, in particular if :math:`p=1`.
To avoid this, this module uses for *q* the probability to observe such a *p* value or a higher one.
In analogy to :math:`\text{CDF}(p_0) = P(p≤p_0) = p_0`, we have :math:`\text{CCDF}(p_0) = P(p≥0) = q` (both under the null hypothesis).
This applies whenever the complement of a *p* value is relevant.

A simple example
----------------

.. automodule:: simple_example


.. _comparison:

An extensive example
--------------------

.. automodule:: comparison

Implementing your own test
--------------------------

If you want to analyse a given dataset with a test that this module does not provide, you need to determine two things:

* The *p* value of the test applied to your dataset.
* A list of all possible *p* values that you test can yield for datasets with the same sample size(s).

You can use these as arguments of `CTR`’s default constructor.

The best way to find all possible *p* values is to get a rough understanding of the test statistics and look into an existing implementation of the test, so you don’t have to fully re-invent the wheel.

If your test is continuous, you do not need to implement anything, but can just use `CTR(p)`, where `p` is the *p* value of your individual test.

Note that individual tests should always be one-sided for the following reason:
If you have two equally significant, but opposing sub-results, they should not add in effect, but cancel each other.
This is not possible when you use two-sided sub-tests, since all information on the directionality of a result gets lost.
You can obtain a two-sided result with `combine` though, which accounts for trends in either direction as long as consistent over all datasets.

Example: Mann–Whitney *U* test
``````````````````````````````

.. automodule:: mwu


Supported combining methods
---------------------------

This module supports the following combining methods [Heard2018]_.
They are listed together with their test statistics – with :math:`p_i` being the *p* values of the individual tests and :math:`q_i` being their complements (see `complements`):

* Fisher: :math:`\prod\limits_i p_i`
* Pearson: :math:`\prod\limits_i q_i^{-1}`
* Mudholkar and George (default): :math:`\prod\limits_i \frac{p_i}{q_i}`
* Edgington: :math:`\sum\limits_i p_i` and Edgington symmetrised: :math:`\sum\limits_i p_i+1-q_i` (see below as to why you may want this)
* Stouffer: :math:`\sum\limits_i \Phi^{-1} (p_i)`, where :math:`\Phi` is the CDF of the standard normal distribution.
* Simes: :math:`\min\limits_i \left( \frac{p_i}{R(p_i)} \right)`, where :math:`R(p_i)` is the rank of :math:`p_i` amongst all combined *p* values [Simes1986]_..
* Tippett: :math:`\min\limits_i(p_i)`

Weighted variants exist for all but the latter two (for which they do not make sense).

Note that we use different, but equivalent statistics internally for numerical reasons.
In particular, we transform products to sums in logarithmic space to avoid numerical underflows.

.. _method_choice:

Choosing a combining method
```````````````````````````

Mudholkar’s and George’s method being the default is based on the assumption that the research hypothesis is that **all datasets are subject to the same trend**.
In `comparison`, this corresponds to the drug being beneficial to dogs in general.

The trend may manifest more clearly in some of the datasets (and you don’t know which a priori), but it should not be inverted (other than by chance).
In this case, you would perform one-sided sub-tests.
(If you would consider a trend in either direction a finding, the combination needs to be two-sided, not the sub-tests.)

If the *p* value of such a sub-test is small, the sub-dataset exhibits the trend you hypothesised.
Conversely, if the complement :math:`q ≈ 1-p` of a sub-test is small, the sub-dataset exhibits a trend opposite to what you hypothesised – with a *p* value *q*.
(See `complements` on how *q* is defined for the purposes of this module.)
I think that the combined *p* values should reflect this, i.e., the complement *q* should indicate the significance of the opposite one-sided hypothesis (not to be confused with the null hypothesis) just like the *p* value indicates the significance of the research hypothesis.

To achieve this, the combining method must treat *p* and *q* in a symmetrical fashion.
This also means that the following results exactly negate each other:

* a sub-test with :math:`p=p_0`.
* a sub-test with :math:`q=p_0`, i.e., :math:`p≈1-p_0`.

Of the supported methods, only three fulfil this:

* Stouffer’s method. However, its statistics becomes infinite if :math:`p=1` for any sub-test and thus the method cannot distinguish between this happening for one or almost all tests.
* Edgington’s symmetrised method. However, this does not give extreme *p* values the emphasis they deserve (in my humble opinion), e.g., a *p* value changing from 0.1 to 0.001 has the same effect as one changing from 0.5 to 0.401.
* Mudholkar’s and George’s method. This one puts emphasis on extreme *p* values, i.e., close to 0 or 1.

I therefore prefer the latter in this case.

This changes if your research hypothesis is that **some datasets exhibit a given trend**.
In the dog example, this corresponds to the research hypothesis that the drug is beneficial to some dog breeds, while it may be harmful to others.
In this case, a method emphasising low *p* values is more appropriate, e.g., Fisher’s.

For other research hypotheses, you have yet other considerations and appropriate methods.
Also see [Adcock1960]_ for a discussion of this.

Supported Tests
---------------

Currently, this module supports:

* the sign test and Wilocoxon’s signed rank test,
* the Mann–Whitney *U* test,
* Fisher’s and Boschloo’s exact tests,
* Spearman’s ρ and Kendall’s τ,
* all continuous tests (just use `CTR(p)`).

Ties are not supported in every case. If you require any further test or support for ties, please `tell me <https://github.com/BPSB/combine-p-values-discrete/issues/new>`_.
Two-sided tests are not supported because I cannot imagine a combination scenario where they make sense.


Command reference
-----------------

.. automodule:: combine_pvalues_discrete
	:members: CTR, combine, direction, sign_test

References
----------

.. _combine_pvalues: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html

.. _Fisher’s method: https://en.wikipedia.org/wiki/Fisher%27s_method

.. [Adcock1960] C. J. Adcock: *A note on combining probabilities,* Psychometrika 25, pp. 303–305 (1960), `10.1007/BF02289734 <https://dx.doi.org/10.1007/BF02289734>`_

.. [Kincaid1962] W. M. Kincaid: *The combination of tests based on discrete distributions,*  Journal of the American Statistical Association 57 (297), pp. 10–19 (1962), `10.1080/01621459.1962.10482147 <https://dx.doi.org/10.1080/01621459.1962.10482147>`_

.. [Simes1986] R. J. Simes: *An improved Bonferroni procedure for multiple tests of significance,* Biometrika 73 (3), pp. 751–754 (1986), `10.1093/biomet/73.3.751 <https://dx.doi.org/10.1093/biomet/73.3.751>`_

.. [Mielke2004] P. W. Mielke and J. E. Johnston and K. J. Berry: *Combining probability values from independent permutation tests: A discrete analog of Fisher’s classical method,* Psychological Reports 95 (2), pp. 449–458 (2004), `10.2466/pr0.95.2.449-458 <https://dx.doi.org/10.2466/pr0.95.2.449-458>`_

.. [Heard2018] N. A. Heard and P. Rubin-Delanchy: *Choosing between methods of combining p-values,* Biometrika 105 (1), pp. 239–246 (2018) `10.1093/biomet/asx076 <https://dx.doi.org/10.1093/biomet/asx076>`_

