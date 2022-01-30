#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
In this example, we illustrate the benefits of this module by comparing it with other approaches to the same dataset, all of which yield wrong results or consume a lot of time.

Suppose we want to to explore the effect on a drug on dogs.
We expect our observable (by which we measure the drug) be more affected by the breed than by the drug.
Therefore we group the dogs by breed a priori, creating sub-datasets.
The dogs from each group randomly get administered the drug or a placebo.
Since not all dogs complete the study, our sub-datasets become very inhomogeneous in sample size.

Our data looks like this:

.. literalinclude:: ../examples/comparison.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 1-12

Each pair represents one breed, with the first half being the control group and the second the treatment group.
If our drug works as desired, the second half should exhibit higher values.
Finally, due to the nature of our observable, we only want to use a ranked statistics.

Thus, we want to investigate the null hypothesis that for each sub-dataset, both samples are from distributions with the same median (the null hypothesis of the Mann–Whitney *U* test).
The alternative hypothesis is that that the first pair of samples is from a distribution with a lower median.

First, suppose we discard our information on breeds and pool the control and treatment groups.
We then apply the Mann–Whitney *U* test to the pooled samples.
This way, we do not need to combine tests, but we lose statistical power.

.. literalinclude:: ../examples/comparison.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 15-19

Alternatively, we can summarise the samples in each pair by their median and use the sign test to compare the groups.
Again, we discard information and lose statistical power:

.. literalinclude:: ../examples/comparison.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 22-27

To properly take into account all information, we have to apply the Mann–Whitney *U* test to each pair (breed) and then combine the *p* values.
SciPy’s `combine_pvalues` allows us to do this, but it requires continuous tests.
Since the Mann–Whitney *U* test does not fulfil this requirement, we will overestimate the combined *p* value:

.. literalinclude:: ../examples/comparison.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 30-33

By using this module, you can take into account the discreteness of tests, obtaining a correct combined *p* value:

.. literalinclude:: ../examples/comparison.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 36-39

Checking the Result
```````````````````

Finally, let’s convince ourselves that this result is actually correct.
To this end, we first implement the statistic of Fisher’s method for combining tests.
Note how the result agrees with that of applying `combine_pvalues` above:

.. literalinclude:: ../examples/comparison.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 42-48


Next we implement a function that samples analogous datasets corresponding to our null hypothesis (surrogates).
(Since we only care about the order of samples, we do not have to recreate the magnitude of values.)

.. literalinclude:: ../examples/comparison.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 50-55

Finally we sample `n=10000` times from our null model and estimate the *p* value by comparing the values of Fisher’s statistic for the null model and the original data.

.. literalinclude:: ../examples/comparison.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 57-61

This confirms the low *p* value we obtained with `combine` above and that the *p* values obtained with the other methods were too high.
Note that this value is several standard deviations away from the result of `combine`.
This is not due the former result or the standard deviation being incorrect, but due to `n` being so low.
Obtaining a comparable precision with the null-model approach would require an excessive amount of time.
"""

if __name__ == "__main__":
	# example-start
	data = [
		( [8,13,37]      , [43,51]       ),
		( [60,68,46,45]  , [30]          ),
		( [92,97,98]     , [84,89]       ),
		( [14]           , [21,45,31,23] ),
		( [24,58,0,24,33], [65,51,61]    ),
		( [93,76,70,83]  , [84]          ),
		( [10,2]         , [28,36,11]    ),
		( [27]           , [38,58]       ),
		( [18]           , [12]          ),
		( [20,44,14,68]  , [73,22,80]    ),
	]
	
	# Pooling data and MWU test
	from scipy.stats import mannwhitneyu
	pooled_Xs = [ x for X,Y in data for x in X ]
	pooled_Ys = [ y for X,Y in data for y in Y ]
	print( mannwhitneyu(pooled_Xs,pooled_Ys,alternative="less") )
	# MannwhitneyuResult(statistic=282.0, pvalue=0.30908071682819527)
	
	# Summarizing data and sign test
	import numpy as np
	from combine_pvalues_discrete import sign_test
	reduced_Xs = [ np.median(X) for X,Y in data ]
	reduced_Ys = [ np.median(Y) for X,Y in data ]
	print( sign_test( reduced_Xs, reduced_Ys, alternative="less" ) )
	# (0.171875, 10)
	
	# Combining MWU results without respecting discreteness
	from scipy.stats import combine_pvalues, mannwhitneyu
	pvalues = [ mannwhitneyu(X,Y,alternative="less").pvalue for X,Y in data ]
	print( combine_pvalues(pvalues,method="fisher") )
	# (27.447712265267114, 0.123131292229715)
	
	# Combining MWU results with respecting discreteness
	from combine_pvalues_discrete import CTR, combine
	ctrs = [CTR.mann_whitney_u(X,Y,alternative="less") for X,Y in data ]
	print( combine(ctrs,method="fisher") )
	# Combined_P_Value(pvalue=0.0014229998577000142, std=1.1920046440408576e-05)
	
	# Demonstrating correctness with null-model simulation
	def fisher_statistic(dataset):
		pvalues = [ mannwhitneyu(X,Y,alternative="less").pvalue for X,Y in dataset ]
		return -2*np.sum(np.log(pvalues))
	
	data_statistic = fisher_statistic(data)
	print(data_statistic)
	# 27.447712265267114
	
	rng = np.random.default_rng()
	def null_sample(data):
		return [
			( rng.random(len(X)), rng.random(len(Y)) )
			for X,Y in data
		]
	
	n = 10000
	null_statistic = [ fisher_statistic(null_sample(data)) for _ in range(n) ]
	count = np.sum( null_statistic >= data_statistic )
	print( (count+1)/(n+1) )
	# 0.0016998300169983002

