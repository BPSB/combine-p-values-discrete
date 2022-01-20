import math
import numpy as np
from warnings import warn

from .tools import is_unity, sign_test, counted_p
from .pdist import PDist

from scipy.stats._mannwhitneyu import _mwu_state, mannwhitneyu

class CTR(object):
	"""
	CTR = combined test result
	
	Represents a single test result or combination thereof. You usually do not want to use the default constructor but one of the class methods for a specific or generic test.
	
	Multiplying instances of this class (using the `*` operator or similar) combines the respective results.
	
	Parameters
	----------
	p_values
		Iterable of p_values of the individual tests.
	
	nulldists
		Iterable of null distributions of the individual tests.
	"""
	def __init__(self,p_values,nulldists):
		if len(p_values) != len(nulldists):
			raise ValueError("p_values and nulldists must have same length")
		self.p_values = list(p_values)
		self.nulldists = list(nulldists)
	
	def __mul__(self,other):
		if is_unity(other): return self
		return CTR( self.p_values+other.p_values, self.nulldists+other.nulldists )
	
	__rmul__ = __mul__
	
	def __repr__(self):
		return f"CombinedTest(\n\t p-values: {self.p_values},\n\t nulldists: {self.nulldists}\n )"
	
	def sorted(self):
		indices = np.argsort(self.p_values)
		return CTR(
				np.array(self.p_values )[indices],
				np.array(self.nulldists)[indices],
			)
	
	def __eq__(self,other):
		A = self.sorted()
		B = other.sorted()
		return A.p_values==B.p_values and A.nulldists==B.nulldists
	
	def combined_p(self,RNG=None,size=10000000):
		"""
		Return the p value of the combined tests. Usually, this result is why you are doing all this.
		So far only Fisher’s method is supported.
		
		Parameters
		----------
		RNG
			NumPy random-number generator used for the Monte Carlo simulation.
			Will be automatically generated if not specified.
		
		size
			Number of samples used for Monte Carlo simulation.
		"""
		
		statistic = lambda x: np.sum(np.log(x),axis=0)
		null_samples = [
				nulldist.sample(RNG,size)
				for nulldist in self.nulldists
			]
		return counted_p( statistic(self.p_values), statistic(null_samples) )
	
	@classmethod
	def from_test(cls,p,all_ps):
		"""
		Creates an object representing a single result of a test – which can then be combined with others using Python multiplication.
		
		Parameters
		----------
		p
			The p value yielded by the test for the investigated sub-dataset.
		
		all_ps
			An iterable containing all possible p values of the test for datasets with the same size as the investigated sub-dataset.
			If empty, all p values will be considered possible, i.e., the test will be assumed to be continuous.
		"""
		
		if all_ps and p not in all_ps:
			next_higher = min( other for other in all_ps if other > p )
			if next_higher/p > 1+1e-10:
				raise ValueError("p value must be in `all_ps`.")
			else:
				p = next_higher
		
		return cls( [p], [PDist(all_ps)] )
	
	@classmethod
	def from_mann_whitney_u( cls, x, y, **kwargs ):
		"""
		Creates an object representing the result of a single Mann–Whitney *U* test (using SciPy’s `mannwhitneyu`).
		
		The two-sided test is not supported because it makes little sense in a combination scenario.
		Ties are not supported yet because I expect them not to occur in the scenarios that require test combinations (but I may be wrong about this) and they make things much more complicated.
		
		Parameters
		----------
		x,y
			The two arrays of samples to compare.
		
		kwargs
			Further keyword arguments to be passed on to SciPy’s `mannwhitneyu`, such as `alternative` or `axis`.
		"""
		if kwargs["alternative"].lower() == "two-sided":
			raise NotImplementedError("The two-sided test is not supported (and makes little sense for combining test results).")
		n,m = len(x),len(y)
		
		if kwargs.pop("method","exact") != "exact":
			warn('Can only use `method="exact"`.')
		
		p = mannwhitneyu(x,y,method="exact",**kwargs).pvalue
		possible_ps = [ _mwu_state.cdf( U,n,m ) for U in range(n*m+1) ]
		return cls.from_test( p, possible_ps )
	
	@classmethod
	def from_sign_test( cls, x, y=0, alternative="less" ):
		"""
		Creates an object representing the result of a single sign test.
		
		Parameters
		----------
		x,y
			The two arrays of paired samples to compare. If `y` is a number, a one-sample sign test is performed with `y` as the median. With `y` as an iterable, the test is two-sided.
		
		alternative: "less" or "greater"
			The two-sided test is not supported because it makes little sense in a combination scenario.
		"""
		
		if "alternative" == "two-sided":
			raise NotImplementedError("The two-sided test is not supported (and makes little sense for combining test results).")
		
		p,m = sign_test(x,y,alternative)
		
		all_ps = list( np.cumsum([math.comb(m,i)/2**m for i in range(m)]) ) + [1]
		return cls.from_test( p, all_ps )

