import math
import numpy as np
from warnings import warn

from .tools import is_unity, sign_test
from .logpdist import LogPDist

from scipy.stats._mannwhitneyu import _mwu_state, mannwhitneyu

class CTR(object):
	"""
	CTR = combined test result
	
	Represents a single test result or combination thereof. You usually do not want to use the default constructor but one of the class methods for a specific or generic test.
	
	Multiplying instances of this class (using the `*` operator or similar) combines the respective results.
	
	Parameters
	----------
	logp
		The logarithm of product of the p values of the tests that are combined in this one.
		This is not the p value of the combined test results under the compound null hypothesis.
	
	nulldist
		The distribution of p values if the compound null hypothesis is true.
	"""
	def __init__(self,logp,nulldist):
		index = np.clip( np.searchsorted(nulldist.logps,logp,side="right"), 0, len(nulldist)-1 )
		self.logp = nulldist[ index ][0]
		self.nulldist = nulldist
	
	def __mul__(self,other):
		if is_unity(other): return self
		return CTR( self.logp+other.logp, self.nulldist*other.nulldist )
	
	__rmul__ = __mul__
	
	def __repr__(self):
		return f"CombinedTest(\n\t logp: {self.logp},\n\t null: {self.nulldist}\n )"
	
	def __eq__(self,other):
		return self.logp==other.logp and self.nulldist==other.nulldist
	
	@property
	def combined_p(self):
		"""
		The p value of the combined tests. Usually, this result is why you are doing all this.
		"""
		return self.nulldist.cdf(self.logp)
	
	@classmethod
	def from_discrete_test(cls,p,all_ps,**kwargs):
		"""
		Creates an object representing a single result of a **discrete** test – which can then be combined with others using Python multiplication.
		
		Parameters
		----------
		p
			The p value yielded by the test for the investigated sub-dataset (not logarithmised).
		
		all_ps
			An iterable containing all possible p values of the test for datasets with the same size as the investigated sub-dataset.
		
		density
		density_warning
			like the respective parameters of `LogPDist.uniform_from_ps` (for the representation of the null distribution).
		"""
		return cls( np.log10(p), LogPDist.uniform_from_ps(all_ps,**kwargs) )
	
	@classmethod
	def from_continuous_test(cls,p,min_p=None,density=1000):
		"""
		Creates an object representing a single result of a **continuous** test – which can then be combined with others using Python multiplication.
		
		Parameters
		----------
		p
			The p value yielded by the test for the investigated sub-dataset (not logarithmised).
		
		all_ps
			An iterable containing all possible p values of the test for datasets with the same size as the investigated sub-dataset.
		
		min_p
		density
			like the respective parameters of `LogPDist.uniform_continuous` (for the representation of the null distribution), except that if `min_p` is `None`, it will be dynamically chosen from `p`.
		"""
		if min_p is None:
			min_p = p/10000
		return cls( np.log10(p), LogPDist.uniform_continuous(min_p,density) )
	
	@classmethod
	def from_mann_whitney_u( cls, x, y, density=1000, uniform_threshold=30, **kwargs ):
		"""
		Creates an object representing the result of a single Mann–Whitney *U* test (using SciPy’s `mannwhitneyu`).
		
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
			return cls.from_discrete_test( p, possible_ps, density=density )
		else:
			p = mannwhitneyu(x,y,**kwargs).pvalue
			return cls.from_continuous_test( p, density=density )
	
	@classmethod
	def from_sign_test( cls, x, y=0, alternative="less", density=1000, uniform_threshold=30 ):
		"""
		Creates an object representing the result of a single sign test.
		
		Parameters
		----------
		x,y
			The two arrays of paired samples to compare. If `y` is a number, a one-sample sign test is performed with `y` as the median. With `y` as an iterable, the test is two-sided.
		
		alternative: "less" or "greater"
			The two-sided test is not supported because it makes little sense in a combination scenario.
		
		density
			The number of sampling points per unit interval (of logarithms of p values) used for representing the null distribution.
		
		uniform_threshold
			If there are more p values than this, approximate the null distribution as a continuous uniform one instead of a discrete one.
		
		kwargs
			Further keyword arguments to be passed on to SciPy’s `mannwhitneyu`, such as `alternative` or `axis`.
		"""
		
		if "alternative" == "two-sided":
			raise NotImplementedError("The two-sided test is not supported (and makes little sense for combining test results).")
		
		p,m = sign_test(x,y,alternative)
		
		if m+1 < uniform_threshold:
			all_ps = list( np.cumsum([math.comb(m,i)/2**m for i in range(m)]) ) + [1]
			assert p in all_ps
			return cls.from_discrete_test( p, all_ps, density=density, density_warning=False )
		else:
			return cls.from_continuous_test( p, density=density )

