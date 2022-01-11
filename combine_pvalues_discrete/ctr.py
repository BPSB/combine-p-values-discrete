import numpy as np

from .tools import is_unity
from .logpdist import LogPDist

class CombinedTestResult(object):
	"""
	Represents a single test result or combination thereof. You usually do not want to use the default constructor but one of the class methods `from_discrete_test` or `from_continuous_test`.
	
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
		return CombinedTestResult( self.logp+other.logp, self.nulldist*other.nulldist )
	
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

