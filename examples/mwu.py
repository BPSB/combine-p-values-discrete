#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Suppose, we want to implement the Mann–Whitney *U* test (with the alternative “less”) into the CTR framework.
(This test is already implemented, so we don’t actually need to do this.)

The test is named for its statistics *U* which can exhibit values between 0 and :math:`n·m`, where :math:`n` and :math:`m` are the sizes of the datasets that are compared.
These *U* values are then translated into *p* values.
If we look into SciPy’s implementation of this test, we find that it uses an implementation of the test’s null distribution (`scipy.stats._mannwhitneyu._mwu_state`), which we can exploit.
Finally, we can use SciPy’s implementation to compute the *p* value of our given datasets.

With that we have all the ingredients to write a small function to return a combinable test result for a given pair of datasets:

.. literalinclude:: ../examples/mwu.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 1-8
"""

if __name__ == "__main__":
	import numpy as np
	
	data_A = np.random.random(7)
	data_B = np.random.random(8)
	# example-start
	from combine_pvalues_discrete import CTR
	from scipy.stats._mannwhitneyu import _mwu_state, mannwhitneyu
	
	def mwu_ctr(x,y):
		p = mannwhitneyu(x,y,method="exact",alternative="less").pvalue
		n,m = len(x),len(y)
		possible_ps = [ _mwu_state.cdf(U,n,m) for U in range(n*m+1) ]
		return CTR( p, possible_ps )
	
	print(mwu_ctr(data_A,data_B))
	
