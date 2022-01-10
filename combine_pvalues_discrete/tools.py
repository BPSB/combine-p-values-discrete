import numpy as np

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
	Returns the product of `factors` with sub-results being combined in a tree-like manner as opposed to a sequential one: First products of pairs are computed, then pairs of pairs, and so on. This may reduce the computing time and increase the accuracy when multiplying `CombinedTestResults`s. Returns `1` if `factors` is empty.
	
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

