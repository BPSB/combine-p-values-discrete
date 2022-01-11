from warnings import warn
import numpy as np

from scipy.signal import convolve
import math

from .tools import searchsorted_closest, is_unity

class LogPDist(object):
	"""
	Represents a distribution of logarithms of p values, whose support is an interval ending at :math:`\\log(p)=0` or :math:`p=1`, respectively. Python multiplication (using `*` or similar) convolves the distributions, i.e., what happens when one multiplies p values from the distributions.
	
	Parameters
	----------
	logps
		iterable containing the logarithms of p values (must be equidistant and end at 0)
	
	probs
		iterable containing the probabilities of the respective p values (must have the same size as `logps` and sum up to 1)
	"""
	
	def __init__(self,logps,probs):
		if len(logps) != len(probs):
			raise ValueError("Lengths of input must match.")
		if logps[-1] <= -1e-5:
			raise ValueError(f"Last log p value must be 0, but is {logps[-1]}.")
		else:
			logps[-1] = 0
		if abs(np.sum(probs)-1) > 1e-10*len(probs):
			raise ValueError(f"Sum of probabilities is not 1 but {np.sum(probs)}.")
		
		steps = np.diff(logps)
		if np.any(steps<0):
			raise ValueError("Log p values must be sorted.")
		if np.max(steps)/np.min(steps) > 1+1e-5:
			raise ValueError("Log p values must be equally spaced.")
		
		self.logps = logps
		self.probs = probs
	
	def __len__(self):
		return len(self.logps)
	
	def __getitem__(self,index):
		return (self.logps[index],self.probs[index])
	
	def __iter__(self):
		return zip(self.logps,self.probs)
	
	@property
	def density(self):
		"""
		The number of sampling points per unit interval (of logarithms of p values).
		"""
		return -len(self)/self[0][0]
	
	def _rescaled(self,density):
		"""
		Returns a LogPDist instance representing the same distribution, but with a different sampling density. Some accuracy may be lost.
		"""
		logps = np.linspace( self[0][0], 0, round(-density*self[0][0]) )
		transfer_indices = searchsorted_closest(logps,self.logps)
		probs = np.zeros_like(logps)
		np.add.at(probs,transfer_indices,self.probs)
		return LogPDist(logps,probs)
	
	def _adjust_density(self,other,rtol=0):
		"""
		Adjusts the densities of `self` and `other` to each other (taking the higher one) and returns both. Some accuracy may be lost.
		"""
		if   self.density/other.density > 1+rtol:
			return self, other._rescaled(self.density)
		elif other.density/self.density > 1+rtol:
			return self._rescaled(other.density), other
		else:
			return self, other
	
	def __mul__(self,other):
		if is_unity(other): return self
		
		self,other = self._adjust_density(other,rtol=1e-10)
			
		probs = convolve(self.probs,other.probs)
		logps = np.linspace( self[0][0]+other[0][0], 0, len(probs) )
		return LogPDist(logps,probs)
	
	__rmul__ = __mul__
	
	@classmethod
	def uniform_from_ps(cls,ps,density=1000,density_warning=True):
		"""
		Returns the discrete uniform distribution with a specified support (or more precisely, an approximation thereof).
		For any test (with discrete p values), the p values of any test follow such a distribution under the null hypothesis.
		All you need to know are the possible p values.
		
		Parameters
		----------
		ps
			iterable containing the p values that are the support of the new distribution. Mind that these are not yet logarithmised.
		
		density
			the number of sampling points per unit interval (of logarithms of p values) of the output.
		
		density_warning
			whether a warning shall be thrown when two p values are mapped to one point in the support, which indicates that the density is chosen too low or the number of p values is so high that a continuous uniform distribution is a better choice.
		"""
		ps = np.atleast_1d(ps)
		ps.sort()
		probs = np.diff(ps,prepend=0)
		
		if not ( 0 < ps[0] ) and ( 1-1e-10 < ps[-1] <= 1 ):
			print(ps)
			raise ValueError(f"p values must be between 0 and 1, with the largest being 1; but they are {ps}")
		ps[-1] = 1
		
		minimum = np.log10(ps[0])
		cont_logps = np.linspace(minimum,0,round(-density*minimum))
		cont_probs = np.zeros_like(cont_logps)
		p_indices = searchsorted_closest(cont_logps,np.log10(ps))
		
		np.add.at(cont_probs,p_indices,probs)
		if density_warning and np.any( np.diff(p_indices)==0 ):
			warn("Some ps are conflated by discretisation. Consider using a higher density or a uniform distribution.")
		
		return cls(cont_logps,cont_probs)
	
	@classmethod
	def uniform_continuous(cls,min_p=1e-5,density=1000):
		"""
		Returns an approximation of the continuous uniform distribution (for p values from [0,1]).
		For any test (with continuous p values), the p values of any test follow such a distribution under the null hypothesis.
		
		Parameters
		----------
		min_p
			the minimal p value. I recommend to choose this at least a few orders of magnitude lower than the p value of the test you want to combine.
		
		density
			the number of sampling points per unit interval (of logarithms of p values) of the output.
		"""
		if not 0 < min_p < 1:
			raise ValueError("Minimal p value must be between 0 and 1.")
		minimum = np.log10(min_p)
		logps = np.linspace( minimum, 0, math.floor(-density*minimum) )
		probs = np.diff(10**logps,prepend=0)
		return cls(logps,probs)
	
	def __repr__(self):
		points = ", ".join(f"{logp:.3g}: {prob:.3g}" for logp,prob in self if prob)
		return f"LogPDist( {points} )"
	
	@property
	def cumprobs(self):
		return np.cumsum(self.probs)
	
	def cdf(self,logp):
		return np.sum( self.probs[:self._index_of_logps(logp,clip=False)] )
	
	def _diff(self,other):
		"""
		Computes the magnitude of deviations to another distribution, allowing for off-by-one errors in discretisation.
		Mostly for testing purposes.
		"""
		
		if self[0][0] != other[0][0]:
			raise NotImplementedError("Cannot compare distributions on different intervals … yet.")
		
		self,other = self._adjust_density(other,rtol=0)
		
		return np.sum(np.min(np.vstack([
			np.abs( self.cumprobs - np.roll(other.cumprobs,shift) )
			for shift in (-1,0,1)
		])))
	
	def __eq__(self,other):
		return self._diff(other) == 0
	
	def _index_of_logps(self,logps,clip=True):
		result = np.searchsorted(self.logps,logps,side="right")
		if clip:
			return np.clip(result,0,len(self)-1)
		else:
			return result
	
	def _adjust_logps(self,logps):
		return self.logps[self._index_of_logps(logps)]

