import numpy as np
from .tools import is_empty

def sample_discrete(values,frequencies,RNG=None,size=10000000,method="proportional"):
	"""
	Returns `size` samples from `values` with `frequencies` using `RNG` as the random-number generator.
	
	If `method` is `"proportional"`, the frequency of each value will be exact – except for rounding. Only the rounding and the order of elements will be stochastic.
	
	If `method` is `"stochastic"`, the values will be randomly sampled and thus their actual frequencies are subject to stochastic fluctuations. This usually leads to slightly less accurate results, but independent samples.
	"""
	RNG = RNG or np.random.default_rng()
	
	if method=="stochastic":
		return RNG.choice( values, p=frequencies, size=size, replace=True )
	elif method=="proportional":
		combos = list(zip(values,frequencies))
		RNG.shuffle(combos)
		result = np.empty(size)
		start = 0
		for p,prob in combos:
			end = start + prob*size
			result[ round(start) : round(end) ] = p
			start = end
		assert round(end) == size
		RNG.shuffle(result)
		return result
	else:
		raise ValueError('Method must either be "proportional" or "stochastic"')

def sample_uniform(RNG=None,size=10000000,method="proportional"):
	"""
	Returns `size` samples from the continuous uniform distribution on [0,1) using `RNG` as the random-number generator.
	
	If `method` is `"proportional"`, the values will be evenly spaced (but random in order).
	
	If `method` is `"stochastic"`, the values will be randomly sampled, but independent.
	"""
	
	RNG = RNG or np.random.default_rng()
	
	if method=="stochastic":
		return RNG.uniform(size=size)
	elif method=="proportional":
		pad = 1/(2*size)
		result = np.linspace( pad, 1-pad, size )
		RNG.shuffle(result)
		return result
	else:
		raise ValueError('Method must either be "proportional" or "stochastic"')

class PDist(object):
	"""
	Represents a uniform distribution (of p values) on the unit interval with a specified support, i.e., a distribution with :math:`\\mathop{CDF}(p)=p` for any :math:`p` in the support.
	For any test, the p values follow such a distribution under the null hypothesis.
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
		return self.approx(other,tol=0)
	
	def approx(self,other,tol=1e-14):
		if self.continuous:
			return other.continuous
		else:
			return all( abs(p1-p2)<=tol for p1,p2 in zip(self,other) )
	
	def sample(self,which="p",**kwargs):
		"""
		Returns `size` samples from the p or q values (specified by `which`), using `RNG` as the random-number generator. See sample_discrete and sample_uniform for arguments.
		"""
		if which=="p":
			return self.sample_ps(**kwargs)
		elif which=="q":
			return self.sample_complement(**kwargs)
	
	def sample_ps(self,which="p",**kwargs):
		"""
		Returns `size` samples from the distribution using `RNG` as the random-number generator. See sample_discrete and sample_uniform for arguments.
		"""
		if self.continuous:
			return 1-sample_uniform(**kwargs)
		else:
			return sample_discrete(self.ps,self.probs,**kwargs)
	
	@property
	def complement_ps(self):
		comp_ps = 1-np.roll(self.ps,1)
		comp_ps[0] = 1
		return comp_ps

	def sample_complement(self,**kwargs):
		"""
		Returns `size` samples from the complementary values of the distribution using `RNG` as the random-number generator. See sample_discrete and sample_uniform for arguments.
		"""
		if self.continuous:
			return sample_uniform(**kwargs)
		else:
			return sample_discrete(
					self.complement_ps,
					self.probs,
					**kwargs,
				)
	
	def sample_both(self,**kwargs):
		"""
		Returns `size` samples from the values and the complementary values of the distribution using `RNG` as the random-number generator. These are paired. See sample_discrete and sample_uniform for arguments.
		"""
		if self.continuous:
			qs = sample_uniform(**kwargs)
			return 1-qs, qs
		else:
			indices = sample_discrete(
					np.arange(len(self.ps)),
					self.probs,
					**kwargs,
				).astype(int)
			return self.ps[indices], self.complement_ps[indices]
	
	def complement(self,pvalue):
		"""
		Returns the complement of a particular p value with respect to the distribution, i.e., the probability that a value is larger or equal than the given value.
		"""
		if self.continuous:
			return 1-pvalue
		else:
			pos = np.argmin(np.abs(self.ps-pvalue))
			if pos!=0:
				return 1-self.ps[pos-1]
			else:
				return 1

