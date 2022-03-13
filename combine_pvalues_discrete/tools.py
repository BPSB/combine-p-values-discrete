from collections import namedtuple
import numpy as np
from scipy.stats import binomtest, kstwo

def is_empty(x):
	try:
		return len(x)==0
	except TypeError:
		return x is None

def searchsorted_closest(array,values):
	"""
	Wrapper around NumPy’s `searchsorted` that returns the index of the closest value(s) – as opposed to the next lower or higher one.
	"""
	
	array = np.asarray(array)
	interval = (0,len(array)-1)
	right_idcs = np.searchsorted(array,values,side="left").clip(*interval)
	left_idcs = (right_idcs-1).clip(*interval)
	
	left_or_right = values-array[left_idcs] < array[right_idcs]-values
	return np.choose( left_or_right, (right_idcs,left_idcs) )

def has_ties(array):
	"""
	Whether any two values in the array are identical (tied).
	"""
	return np.any(np.diff(sorted(array))==0)

def find_similar(array,rtol,atol):
	"""
	Returns mask of which pairs of neighbours in an array are closer than eps but not identical.
	"""
	return np.logical_and(
		array[1:] != array[:-1],
		np.isclose(array[1:],array[:-1],rtol=rtol,atol=atol)
	)

def unify_sorted(array,rtol=1e-14,atol=0):
	"""
	Unify values in a sorted array that only differ from their predecessor by eps – in place.
	"""
	while np.any( similar:= find_similar(array,rtol,atol) ):
		array[1:][similar] = array[:-1][similar]

SignTestResult = namedtuple("SignTestResult",("pvalue","not_tied","statistic"))

def sign_test(x,y=0,alternative="less"):
	"""
	Just the sign test without any combination features, provided because it’s there.
	
	**two-sided:**
	Pass paired samples `x` and `y` as arguments. The tested null hypothesis is that `x[i]` and `y[i]` are from the same distribution (separately for each `i`).
	
	**one-sided**
	Pass a single sample `x` and a number `y`. The tested null hypothesis is that `x` is sampled from a distribution with a median larger than `y`.
	
	Returns a tuple consisting of the p value and the number of non-tied samples.
	"""
	
	x = np.asarray(x)
	y = np.asarray(y)
	greater = np.sum(x>y)
	less    = np.sum(x<y)
	non_tied = less+greater
	return SignTestResult(
			binomtest( greater, non_tied, alternative=alternative ).pvalue,
			non_tied,
			greater,
		)

def count_greater_or_close(x,y,atol=0,rtol=0):
	"Counts how often x is greater than y or close with atol or rtol."
	
	comparison = (y<=x)
	
	if atol or rtol:
		comparison |= np.isclose(x,y,atol=atol,rtol=rtol)
	
	return np.sum(comparison,axis=0)

Combined_P_Value = namedtuple("Combined_P_Value",("pvalue","std"))

def counted_p(orig_stat,null_stats,**tols):
	"""
	Estimates the p value of a statistic (`orig_stat`) by comparing with the statistic for samples of a null model (`null_stats`), with a small statistic being extreme. Returns the p value and its (estimated) standard deviation when sampling with this method.
	"""
	
	null_stats = np.asarray(null_stats)
	size = null_stats.shape[0]
	count = count_greater_or_close(orig_stat,null_stats,**tols)
		
	p = (count+1)/(size+1)
	std = np.maximum(
			np.sqrt(count*(1-count/size))/(size+1),
			1/(size+1),
		)
	return Combined_P_Value(p,std)

def std_from_true_p(true_p,size):
	"""
	Standard deviation of p value from samples, if the true p value is known.
	"""
	return np.sqrt(true_p*(1-true_p)*size)/(size+1)

def assert_matching_p_values(p,target_p,n,factor=3,compare=False):
	"""
	Asserts that `p` (estimated with `counted_p`) matches `target_p` when estimated from `n` samples of the null model.
	
	The allowed error is `factor` times the expected standard deviation.
	
	If `target_p` is not exact but estimated by sampling as well, set `compare=True`. In this case, the average of the two values is used for estimating the standard deviation (instead of `target_p`).
	"""
	p = np.atleast_1d(p)
	
	# Correction because the p value is estimated conservatively and, e.g., can never be below 1/(n+1):
	size_offset = (1-target_p)/(n+1)
	
	diffs = np.abs( target_p - p + (0 if compare else size_offset) )
	
	reference_p = (p+target_p)/2 if compare else target_p
	with np.errstate(invalid="ignore",divide="ignore"):
		ratios = diffs/std_from_true_p(reference_p,n)
	
	if np.any(ratios>factor):
		i = np.nanargmax(ratios-factor)
		
		try: target = target_p[i]
		except (IndexError,TypeError): target=target_p
		
		raise AssertionError(
			f"""
			p values don’t match. Maximum deviation:
				target: {target}
				actual: {p[i]}
				difference / std: {ratios[i]} > {factor}
			""")

def assert_discrete_uniform(data,factor=3):
	data = np.asarray(data)
	n = len(data)
	values = set(data)
	if len(values)<2:
		raise ValueError("Need at least two distinct values.")
	
	for value in values:
		assert_matching_p_values(
				count_greater_or_close(value,data,atol=1e-15,rtol=1e-15)/n,
				value,
				n = n,
				factor = factor,
				compare = False
			)

