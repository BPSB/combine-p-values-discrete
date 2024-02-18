from collections import namedtuple
import numpy as np
from scipy.stats import binomtest, kstwo, boschloo_exact, fisher_exact

def is_empty(x):
	"""
	Whether the argument is an empty 1D iterable or None.
	"""
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

SignTestResult = namedtuple("SignTestResult",("pvalue","not_tied","statistic"))

def sign_test(x,y=0,alternative="less"):
	"""
	Just the sign test without any combination features, provided because I have it anyway.
	
	**two-sided:**
	Pass paired samples `x` and `y` as arguments. The tested null hypothesis is that `x[i]` and `y[i]` are from the same distribution (separately for each `i`).
	
	**one-sided**
	Pass a single sample `x` and a number `y`. The tested null hypothesis is that `x` is sampled from a distribution with a median larger than `y`.
	
	Returns a named tuple consisting of the p value, the number of non-tied samples, and the statistic of the sign test (number of samples which are greater than the reference).
	
	Examples
	--------
	.. code-block:: python3
	
		dataset_A = [20,44,14,68]
		dataset_B = [73,22,80,53]
		p = sign_test(dataset_A,dataset_B,alternative="less").pvalue
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
	Estimates the p value of a statistic (`orig_stat`) by comparing with the statistic for samples of a null model (`null_stats`), with a small statistic being extreme. Returns the p value and its (estimated) standard deviation when estimating with this method.
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

def assert_matching_p_values(tested_ps,target_ps,n,threshold=1e-4,compare=False):
	"""
	Asserts that `ps` (estimated with `counted_p`) matches `target_ps` when estimated from `n` samples of the null model using the binomial test.
	
	`threshold` is the maximal *p* value of that binomial test.
	
	If `target_ps` is not exact but estimated by sampling as well (with the same `n`), set `compare=True`. In this case, Boschloo’s or Fisher’s exact test is used for comparison, with the latter being used for n>10 because it’s considerably faster, though less accurate. If the sample sizes are different, `n` needs to be a pair.
	"""
	
	tested_ps = np.atleast_1d(tested_ps)
	assert tested_ps.ndim == 1
	assert np.all(tested_ps<=1)
	assert np.all(0<=tested_ps)
	
	if np.ndim(target_ps)==0:
		target_ps = np.full_like(tested_ps,target_ps)
	
	if compare:
		for i,ps in enumerate(zip(tested_ps,target_ps)):
			ns = [n,n] if np.ndim(n) == 0 else n
			
			# If you wonder why we are reconstructing the ks here: It’s because we are mostly testing functions that return p values, not ks and ns.
			ks = [ round(p*(n+1)-1) for p,n in zip(ps,ns) ]
			comparison_test = boschloo_exact if max(ns)<=10 else fisher_exact
			comparison_p = comparison_test([
					ks,
					[n-k for k,n in zip(ks,ns)],
				]).pvalue
			if comparison_p < threshold:
				ratios = [f"{p} = ({k}+1)/({n}+1)" for p,n,k in zip(ps,ns,ks)]
				raise AssertionError(
					f"""
					p values don’t match.
						ratios[0] ≉ ratios[1]
						p value of {comparison_test.__name__}: {comparison_p}
					""")
	else:
		for i,(tested_p,target_p) in enumerate(zip(tested_ps,target_ps)):
			assert np.ndim(n) == 0
			k = round(tested_p*(n+1)-1)
			binomtest_p = binomtest(k,n,target_p).pvalue
			if binomtest_p < threshold:
				raise AssertionError(
					f"""
					p values don’t match.
						{tested_p} = ({k}+1)/({n}+1) ≉ {target_p}
						p value of binomial test: {binomtest_p}
					""")

def assert_discrete_uniform(data,threshold=1e-4):
	"""
	Checks whether `data` complies with a discrete uniform distribution on the interval [0,1]. `threshold` is the maximum *p* value of the binomial test used under the hood.
	"""
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
				threshold = threshold,
				compare = False
			)

