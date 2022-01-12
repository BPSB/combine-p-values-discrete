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

def tree_prod(factors):
	"""
	Returns the product of `factors` with sub-results being combined in a tree-like manner as opposed to a sequential one: First products of pairs are computed, then pairs of pairs, and so on. This may reduce the computing time and increase the accuracy when multiplying `CTRs`s. Returns `1` if `factors` is empty.
	
	Parameters
	----------
	factors
		iterable of objects to be multiplied
	"""
	
	factors = list(factors)
	number = len(factors)
	if number == 1:
		return factors[0]
	elif number == 0:
		return 1
	else:
		return tree_prod(factors[:number//2]) * tree_prod(factors[number//2:])

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

def std_counted_p(p,n):
	"""
	Estimates the standard deviation of a p value obtained by sampling a statistics n times.
	"""
	
	return np.sqrt(p*(1-p)/n)



