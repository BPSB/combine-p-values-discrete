---
title: 'A Python module to combine $p$ values arising from discrete tests.'
tags:
  - Python
  - statistics
  - hypothesis tests
authors:
  - name: Gerrit Ansmann
    orcid: 0000-0002-5472-7067
affiliations:
 - name: Institute for Biological Physics, University of Cologne, Germany
bibliography: paper.bib

---

# Summary

Most popular hypothesis tests are designed for datasets with a simple structure.
For example, in a simple medical trial, the dataset would be a collection of health score for members of a treatment and control group, and the test would answer whether there is a significant difference between these.
However, many datasets are naturally segmented or benefit from being analysed in segments.
Continuing our example, if we know a priori that females score higher than males on average, we should see the effect of the treatment more clearly when analysing each sex separately.
To draw an overall conclusion in such a case, we need to combine the $p$ values for each sub-dataset.

Starting with Fisher’s method, several ways have been proposed to combine $p$ values [@Heard2018].
Their usual implementations assume *continuous tests,* i.e., tests whose $p$ values under the null hypothesis follow the uniform distribution on the unit interval.
However, rank tests and many others are *discrete tests,* i.e., they can only yield $p$ values from a finite selection (for a given sample size).
Assuming continuous tests when combining $p$ values from discrete tests can lead to considerably misleading outcomes [@Kincaid1962; @Mielke2004], in particular when the sub-datasets are small.

We here present the Python module `combine_pvalues_discrete`, which provides a toolbox for combining $p$ values from discrete tests.
Results from individual tests are stored with the respective null distribution of $p$ values and the combined $p$ value is accurately estimated using a Monte Carlo simulation based on these null distributions.

# Statement of need

Combining $p$ values is a standard problem in statistical data analysis.
Before the advent of modern computing, many methods have been devoted to analytically solving this problem for continuous tests [@Heard2018], and implementations of these simple methods are a staple of many statistical software suits [@SciPy; @Cinar2022].
However, applying these implementations to discrete tests can result in considerable errors [@Kincaid1962; @Mielke2004] – a problem of which there appears to be little awareness.

Thanks to modern computing, we can solve this problem using simple Monte Carlo simulations:
For each test, we sample one $p$ value from each of the respective discrete null distributions, apply the combining statistics to these, and repeat this until we obtain a good estimate of the null distribution of the combining statistics.
Finally we compare the combining statistics of the actual data to estimate the combined $p$ value.
`combine_pvalues_discrete` implements this approach in a fast, thorough, and tested manner, taking care of pitfalls such as correctly handling complements, sidedness, and empirical $p$ values as well as handling tedious and error-prone tasks such as determining the null distribution $p$ values for a given test.
This approach is considerably faster than a permutation test starting at the level of individual datasets.

Note that this module de-emphasises some typical applications of combining $p$ values as it can be rarely applied to them.
For example, when performing meta analyses, the $p$ values to be combined often originate from continuous tests or tests with a discrete but rather dense supports, and one rarely has the luxury to know the null distribution of $p$ values.
However, as a side product, our module contains weighted versions of popular combining methods that may be of interest to researchers combining continuous tests.

# Acknowledgements

# References

