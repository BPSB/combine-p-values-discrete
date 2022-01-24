import math
import numpy as np
from warnings import warn

from .tools import sign_test, counted_p
from .pdist import PDist

from scipy.special import erfinv
from scipy.stats import rankdata
from scipy.stats._mannwhitneyu import _mwu_state, mannwhitneyu

class CTR(object):
	"""
	CTR = combinable test result
	
	Represents a single test result. Use the default constructor to implement a test yourself or use one of the class methods for the respective result.
	
	Parameters
	----------
	p
		The p value yielded by the test for the investigated sub-dataset.
	
	all_ps
		An iterable containing all possible p values of the test for datasets with the same size as the dataset for this individual test.
		If `None` or empty, all p values will be considered possible, i.e., the test will be assumed to be continuous.
	"""
	def __init__(self,p,all_ps=None):
		if all_ps and p not in all_ps:
			next_higher = min( other for other in all_ps if other > p )
			if next_higher/p > 1+1e-10:
				raise ValueError("p value must be in `all_ps`.")
			else:
				p = next_higher
		
		self.p = p
		self.nulldist = PDist(all_ps)
	
	@classmethod
	def mann_whitney_u( cls, x, y, **kwargs ):
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
		return cls( p, possible_ps )
	
	@classmethod
	def sign_test( cls, x, y=0, alternative="less" ):
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
		return cls( p, all_ps )
	
	def __repr__(self):
		return f"CombinableTest(\n\t p-value: {self.p_value},\n\t nulldist: {self.nulldist}\n )"
	
	def __eq__(self,other):
		return self.p==other.p and self.nulldist==other.nulldist

combining_statistics = {
	("fisher"          ,"normal"  ): lambda x:  np.sum( np.log(x)       , axis=0 ),
	("pearson"         ,"normal"  ): lambda x: -np.sum( np.log(1-x)     , axis=0 ),
	("mudholkar_george","normal"  ): lambda x:  np.sum( np.log(x/(1-x)) , axis=0 ),
	("stouffer"        ,"normal"  ): lambda x:  np.sum( erfinv(2*x-1)   , axis=0 ),
	("tippett"         ,"normal"  ): lambda x:  np.min( x               , axis=0 ),
	("edgington"       ,"normal"  ): lambda x:  np.sum( x               , axis=0 ),
	("simes"           ,"normal"  ): lambda x:  np.min(x/rankdata(x,axis=0,method="ordinal"),axis=0),
	("fisher"          ,"weighted"): lambda x,w:  w.dot(np.log(x))      ,
	("pearson"         ,"weighted"): lambda x,w: -w.dot(np.log(1-x))    ,
	("mudholkar_george","weighted"): lambda x,w:  w.dot(np.log(x/(1-x))),
	("stouffer"        ,"weighted"): lambda x,w:  w.dot(erfinv(2*x-1))  ,
	("edgington"       ,"weighted"): lambda x,w:  w.dot(x)              ,
}

statistics_with_inf = {"pearson","mudholkar_george","stouffer"}

def combine(ctrs,method="fisher",weights=None,size=10000000,RNG=None):
	"""
	Estimates the combined p value of combinable test results. Usually, this result is why you are doing all this.
	
	Parameters
	----------
	ctrs: iterable of CTRs
		The test results that shall be combined.
	
	method: string or function
		One of "fisher", "pearson", "mudholkar_george", "stouffer", "tippett", "edgington", "simes", or a self-defined function that takes a two-dimensional array and computes the test statistics alongth the zero-th axis.
	
	weights: iterable of numbers
		Weights for individual results. Does not work for minimum-based methods (Tippett and Simes).
	
	size
		Number of samples used for Monte Carlo simulation.
	
	RNG
		NumPy random-number generator used for the Monte Carlo simulation.
		Will be automatically generated if not specified.
	
	Returns
	-------
	pvalue
		The estimated combined p value.
	
	std
		The estimated standard deviation of p values when repeating the sampling.
	"""
	
	null_samples = np.vstack([ctr.nulldist.sample(RNG,size) for ctr in ctrs])
	ps = np.array([ctr.p for ctr in ctrs])
	
	kwargs = {"divide":"ignore","invalid":"ignore"} if (method in statistics_with_inf) else {}
	
	if method in (method for method,_ in combining_statistics):
		if weights is None:
			statistic = combining_statistics[method,"normal"]
		else:
			try:
				statistic = combining_statistics[method,"weighted"]
			except KeyError:
				raise ValueError(f'No weighted version of "{method}" method')
	else:
		if not callable(method):
			raise ValueError(f'Method "{method}" is neither known nor callable.')
		statistic = method
	
	with np.errstate(**kwargs):
		if weights is None:
			orig_stat = statistic(ps)
			null_stats = statistic(null_samples)
		else:
			weights = np.asarray(weights)
			orig_stat = statistic(ps,weights)
			null_stats = statistic(null_samples,weights)
	
	return counted_p( orig_stat, null_stats)

