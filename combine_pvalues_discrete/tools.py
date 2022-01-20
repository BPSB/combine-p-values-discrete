import numpy as np
from scipy.stats import binomtest

def is_unity(thing):
	"""
	Returns whether `thing` is equivalent to one, i.e., can be seen as a the neutral element of multiplication.
	"""
	
	try:
		return thing==1
	except:
		return False

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

def sign_test(x,y=0,alternative="less"):
	"""
	Sign test.
	
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
	return binomtest( greater, non_tied, alternative=alternative ).pvalue, non_tied

def counted_p(orig_stat,null_stats):
	null_stats = np.asarray(null_stats)
	total = null_stats.shape[0]+1
	count = np.sum( orig_stat>=null_stats, axis=0 )
	return (count+1)/total

def std_counted_p(p,n):
	"""
	Estimates the standard deviation of a p value obtained by sampling a statistics n times.
	"""
	
	return np.sqrt(p*(1-p)/n)

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
	with np.errstate(invalid='ignore'):
		ratios = diffs/std_counted_p(reference_p,n)
	
	if np.any(ratios>factor):
		i = np.nanargmax(ratios-factor)
		
		try: target = target_p[i]
		except IndexError: target=target_p
		
		raise AssertionError(
			f"""
			p values don’t match. Maximum deviation:
				target: {target}
				actual: {p[i]}
				difference / std: {ratios[i]} > {factor}
			""")

