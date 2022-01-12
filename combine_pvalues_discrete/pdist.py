import numpy as np

class PDist(object):
	"""
	Represents a uniform distribution on the unit interval with a specified support, i.e., a distribution with :math:`\\mathop{CDF}(p)=p` for any :math:`p` in the support.
	For any test, the p values of follow such a distribution under the null hypothesis.
	All you need to know are the possible p values.
	
	Parameters
	----------
	ps
		iterable containing the p values that are the support of the new distribution.
		If empty, this represents the continous uniform distribution.
	"""
	def __init__(self,ps):
		self.ps = np.atleast_1d(ps)
		self.ps.sort()
		if not self.continuous:
			if not ( ( 0 < self.ps[0] ) and ( 1-1e-10 < self.ps[-1] <= 1 ) ):
				raise ValueError(f"p values must be between 0 and 1, with the largest being 1; but they are {ps}")
			self.ps[-1] = 1
	
	@property
	def probs(self):
		return np.diff(self.ps,prepend=0)
	
	@property
	def continuous(self):
		return self.ps.size == 0
	
	def __iter__(self):
		yield from self.ps
	
	def __repr__(self):
		if self.continuous:
			return f"PDist( uniform )"
		else:
			points = ", ".join(f"{p:.3g}" for p in self)
			return f"PDist( {points} )"
	
	def __eq__(self,other):
		if self.continuous:
			return other.continuous
		else:
			return all( p1==p2 for p1,p2 in zip(self,other) )
	
	def sample(self,RNG=None,size=10000000):
		RNG = RNG or np.random.default_rng()
		
		if self.continuous:
			return 1-RNG.uniform(size=size)
		else:
			return RNG.choice( self.ps, p=self.probs, size=size, replace=True )

