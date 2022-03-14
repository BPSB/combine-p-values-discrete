#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
In this example, we illustrate the benefits of this module by comparing it with other approaches to the same dataset, all of which yield wrong results or consume a lot of time.

Suppose we want to to explore the effect on a drug on dogs.
We expect our observable (by which we measure the effect of the drug) be more affected by the dog breed than by the drug, e.g., because a poodle is generally weaker than a mastiff.
Therefore we group the dogs by breed a priori, creating sub-datasets.
Each breed group gets further randomly split into a treatment (T) and control group (C).
Since not all dogs complete the study, our sub-datasets become very inhomogeneous in sample size.

Our data looks like this:

.. literalinclude:: ../examples/comparison.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 1-12

Each pair represents one breed, with the first half being the control group and the second the treatment group.
If our drug works as desired, the second half should exhibit higher values.
Finally, due to the nature of our observable, we only want to use a ranked statistics.

Thus, we want to investigate the null hypothesis that for each sub-dataset, both samples are from the same distribution (or more precisely, the null hypothesis of the Mann–Whitney *U* test).
The alternative hypothesis is that that the first pair of samples are from a distribution with a lower median.

First, suppose we discard our information on breeds and pool the control and treatment groups.
We then apply the Mann–Whitney *U* test to the pooled samples.
This way, we do not need to combine tests, but we lose statistical power, i.e., we increase the false-negative rate.

.. literalinclude:: ../examples/comparison.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 15-20

Alternatively, we can summarise the samples in each pair by their median and use the sign test to compare the groups.
Again, we discard information and lose statistical power:

.. literalinclude:: ../examples/comparison.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 22-29

To properly take into account all information, we have to apply the Mann–Whitney *U* test to each pair (breed) and then combine the *p* values.
SciPy’s `combine_pvalues` allows us to do this, but it requires continuous tests.
Since the Mann–Whitney *U* test does not fulfil this requirement, we will overestimate the combined *p* value:

.. literalinclude:: ../examples/comparison.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 32-37

Finally, by using this module, we can take into account the discreteness of tests, obtaining a correct combined *p* value:

.. literalinclude:: ../examples/comparison.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 40-44

(We here use Fisher’s method, since it simplifies the following demonstration. In general, I don’t recommend it.)

Checking the result
```````````````````

Let’s convince ourselves that the result of `combine` is actually correct.
To this end, we first implement the statistic of Fisher’s method for combining Mann–Whitney *U* tests.
Note how the result agrees with that of applying `combine_pvalues` above:

.. literalinclude:: ../examples/comparison.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 47-53


Next we implement a function that samples analogous datasets corresponding to our null hypothesis (surrogates).
(Since we only care about the order of samples, we do not have to recreate the magnitude of values.)

.. literalinclude:: ../examples/comparison.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 55-60

Finally we sample `n=10000` times from our null model and estimate the *p* value by comparing the values of Fisher’s statistic for the null model and the original data.

.. literalinclude:: ../examples/comparison.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 62-66

This confirms the low *p* value we obtained with `combine` above and that the *p* values obtained with the other methods were too high.
You may note that although this value is low it is not within the confidence interval of the result of `combine` from above.
The reason for this is that the confidence interval of the null-model approach is much larger (on account of `n` being low) and obtaining a precision comparable to `compare` would require an excessive amount of time.
"""

if __name__ == "__main__":
	# example-start
	data = [# control group  , treatment group
	        ( [8,13,37]      , [43,51]       ), # Breed 1
	        ( [60,68,46,45]  , [30]          ), # Breed 2
	        ( [92,97,98]     , [84,89]       ), # …
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
	
	pooled_Cs = [ c for C,T in data for c in C ]
	pooled_Ts = [ t for C,T in data for t in T ]
	print( mannwhitneyu(pooled_Cs,pooled_Ts,alternative="less") )
	# MannwhitneyuResult(statistic=282.0, pvalue=0.30908071682819527)
	
	# Summarizing data and sign test
	import numpy as np
	from combine_pvalues_discrete import sign_test
	
	reduced_Cs = [ np.median(C) for C,T in data ]
	reduced_Ts = [ np.median(T) for C,T in data ]
	print( sign_test( reduced_Cs, reduced_Ts, alternative="less" ) )
	# SignTestResult(pvalue=0.171875, not_tied=10, statistic=3)
	
	# Combining MWU results without respecting discreteness
	from scipy.stats import combine_pvalues, mannwhitneyu
	
	pvalues = [ mannwhitneyu(C,T,alternative="less").pvalue for C,T in data ]
	statistic,pvalue = combine_pvalues(pvalues,method="fisher")
	print(statistic,pvalue)
	# (27.447712265267114, 0.123131292229715)
	
	# Combining MWU results with respecting discreteness
	from combine_pvalues_discrete import CTR, combine
	
	ctrs = [ CTR.mann_whitney_u(C,T,alternative="less") for C,T in data ]
	print( combine(ctrs,method="fisher") )
	# Combined_P_Value(pvalue=0.0014229998577000142, std=1.1920046440408576e-05)
	
	# Demonstrating correctness with null-model simulation
	def fisher_statistic(dataset):
		pvalues = [ mannwhitneyu(C,T,alternative="less").pvalue for C,T in dataset ]
		return -2*np.sum(np.log(pvalues))
	
	data_statistic = fisher_statistic(data)
	print(data_statistic)
	# 27.447712265267114
	
	rng = np.random.default_rng()
	def null_sample(data):
		return [
			( rng.random(len(C)), rng.random(len(T)) )
			for C,T in data
		]
	
	n = 10000
	null_statistic = [ fisher_statistic(null_sample(data)) for _ in range(n) ]
	count = np.sum( null_statistic >= data_statistic )
	print( (count+1)/(n+1) )
	# 0.0016998300169983002

