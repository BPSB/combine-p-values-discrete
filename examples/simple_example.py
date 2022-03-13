#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
We have three pairs of independent datasets `(A_1,A_2)`, `(B_1,B_2)`, and `(C_1,C_2)`.
Our research hypothesis is that the values in the respective first dataset are lower.
However, the datasets have different properties and thus cannot simply be pooled:

* `(A_1,A_2)` is a paired dataset. Since we cannot assume any distribution, we want to apply the sign test.
* `(B_1,B_2)` is an unpaired dataset (with unequal sizes). Again, we cannot assume a distribution. Hence we want to apply the Mann–Whitney *U* test.
* `(C_1,C_2)` is a paired dataset, however we can assume that the differences to be normally distributed and thus apply the *t* test for two dependent samples.

We start with performing the sign test on `A_1` and `A_2`.
We create a `CTR` from this:

.. literalinclude:: ../examples/simple_example.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 1-2

`result_A` now contains the *p* value of the sign test as well as the null distribution of possible *p* values.
It is ready for being combined with other test results, but we have to create these first.
We do so by doing something very similar with `B_1` and `B_2` and the Mann–Whitney *U* test:

.. literalinclude:: ../examples/simple_example.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 4-4

Finally, we perform the *t* test on `C_1` and `C_2`.
Since the *t* test is a continuous test, we do not need a special constructor to create a `CTR`, but can use generic one using only the *p* value computed with an existing function, here `scipy.stats.ttest_rel`.
Leaving the second argument empty specifies that the test is continuous:

.. literalinclude:: ../examples/simple_example.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 6-8

After we have performed the individual tests, we can obtain the compound *p* value using `combine` (which we imported earlier).

.. literalinclude:: ../examples/simple_example.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 10-10
"""

if __name__ == "__main__":
	import numpy as np
	
	A_1,A_2 = np.random.random((2,10))
	B_1 = np.random.random(7)
	B_2 = np.random.random(8)
	C_1,C_2 = np.random.normal(size=(2,10))
	
	# example-start
	from combine_pvalues_discrete import CTR, combine
	result_A = CTR.sign_test(A_1,A_2,alternative="less")
	
	result_B = CTR.mann_whitney_u(B_1,B_2,alternative="less")
	
	from scipy.stats import ttest_rel
	p_C = ttest_rel(C_1,C_2).pvalue
	result_C = CTR(p_C)
	
	print( combine([result_A,result_B,result_C]) )

