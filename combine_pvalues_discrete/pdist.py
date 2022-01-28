import numpy as np
from .tools import is_empty

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
		self.ps = np.array([]) if is_empty(ps) else np.atleast_1d(ps)
		self.ps.sort()
		if not self.continuous:
			if not ( ( 0 < self.ps[0] ) and ( abs(self.ps[-1]-1) < 1e-10 ) ):
				raise ValueError(f"p values must be between 0 and 1, with the largest being 1; but they are {ps}")
			if self.ps[-1]>=1+1e-10 and len(self.ps)>1 and self.ps[-2]>=1:
				raise ValueError("Two p values slightly larger than or equal to 1.")
			
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
	
	def sample(self,RNG=None,size=10000000,method="proportional"):
		"""
		Returns `size` samples from the distribution using `RNG` as the random-number generator.
		
		If `method` is `"proportional"`, the frequency of each value will be exactly proportional to its probability – except for rounding. Only the rounding and the order of elements will be stochastic.
		
		If `method` is `"stochastic"`, the values will be randomly sampled and thus their actual frequencies are subject to stochastic fluctuations. This usually leads to slightly less accurate results.
		"""
		RNG = RNG or np.random.default_rng()
		
		if method=="stochastic":
			if self.continuous:
				return 1-RNG.uniform(size=size)
			else:
				return RNG.choice( self.ps, p=self.probs, size=size, replace=True )
		elif method=="proportional":
			if self.continuous:
				pad = 1/(2*size)
				result = np.linspace( pad, 1-pad, size )
			else:
				result = np.empty(size)
				start = 0
				combos = list(zip(self.ps,self.probs))
				RNG.shuffle(combos)
				for p,prob in combos:
					end = start + prob*size
					result[ round(start) : round(end) ] = p
					start = end
				assert round(end) == size
			RNG.shuffle(result)
			return result
		else:
			raise ValueError('Method must either be "proportional" or "stochastic"')

