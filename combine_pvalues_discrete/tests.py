import numpy as np
from warnings import warn
from scipy.stats._mannwhitneyu import _mwu_state, mannwhitneyu

from .ctr import CombinedTestResult

def single_mwu( x, y, density=1000, uniform_threshold=30, **kwargs ):
	"""
	Performs a single Mann–Whitney U test (using SciPy’s `mannwhitneyu`) and returns the result as a `CombinedTestResult` suitable for combination.
	
	The two-sided test is not supported because it makes little sense in a combination scenario.
	Ties are not supported yet because I expect them not to occur in the scenarios that require test combinations (but I may be wrong about this) and they make things much more complicated.
	
	Parameters
	----------
	x,y
		The two arrays of samples to compare.
	
	density
		The number of sampling points per unit interval (of logarithms of p values) used for representing the null distribution.
	
	uniform_threshold
		If there are more p values than this, approximate the null distribution as a continuous uniform one instead of a discrete one.
	
	kwargs
		Further keyword arguments to be passed on to SciPy’s `mannwhitneyu`, such as `alternative` or `axis`.
	"""
	if kwargs["alternative"].lower() == "two-sided":
		raise NotImplementedError("The two-sided test is not supported (and makes little sense for combining test results).")
	n,m = len(x),len(y)
	
	if n*m+1 < uniform_threshold:
		if kwargs.pop("method","exact") != "exact":
			warn('Can only use `method="exact"` when below the uniform threshold.')
		
		p = mannwhitneyu(x,y,method="exact",**kwargs).pvalue
		possible_ps = [ _mwu_state.cdf( U,n,m ) for U in range(n*m+1) ]
		return CombinedTestResult.from_discrete_test( p, possible_ps, density=density )
	else:
		p = mannwhitneyu(x,y,**kwargs).pvalue
		return CombinedTestResult.from_continuous_test( p, density=density )

