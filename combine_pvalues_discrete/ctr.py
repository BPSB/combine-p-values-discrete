import math
from inspect import signature
import numpy as np
from warnings import warn
from itertools import permutations

from .tools import sign_test, counted_p, Combined_P_Value, is_empty, searchsorted_closest, unify_sorted, has_ties
from .pdist import PDist

from scipy.special import erfinv, factorial
from scipy.stats import rankdata, spearmanr, pearsonr, kendalltau, fisher_exact
from scipy.stats._mannwhitneyu import _mwu_state, mannwhitneyu
from scipy.stats._stats_py import _ttest_finish
from scipy.stats._mstats_basic import _kendall_p_exact
from scipy.stats.distributions import hypergeom

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
		if not is_empty(all_ps) and p not in all_ps:
			all_ps = np.asarray(all_ps)
			closest = all_ps[np.argmin(np.abs(all_ps-p))]
			if (closest-p)/p > 1e-10:
				raise ValueError(f"p value {p} must be in `all_ps`.")
			else:
				p = closest
		
		self.p = p
		self.nulldist = PDist(all_ps)
		self.q = self.nulldist.complement(self.p)
	
	def __repr__(self):
		return f"CombinableTest(\n\t p-value: {self.p},\n\t nulldist: {self.nulldist}\n )"
	
	def __eq__(self,other):
		return self.approx(other,tol=0)
	
	def approx(self,other,tol=1e-14):
		return abs(self.p-other.p)<=tol and self.nulldist.approx(other.nulldist,tol)
	
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
			Further keyword arguments to be passed on to SciPy’s `mannwhitneyu`, such as `alternative`.
		"""
		if "alternative" not in kwargs:
			raise ValueError("You must specify the alternative.")
		
		if has_ties(np.hstack((x,y))):
			raise NotImplementedError("Ties are not yet implemented.")
		
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
		
		if alternative == "two-sided":
			raise NotImplementedError("The two-sided test is not supported (and makes little sense for combining test results).")
		
		p,m,_ = sign_test(x,y,alternative)
		
		all_ps = list( np.cumsum([math.comb(m,i)/2**m for i in range(m)]) ) + [1]
		return cls( p, all_ps )
	
	@classmethod
	def spearmanr( cls, x, y, alternative="greater", n_thresh=9 ):
		"""
		Creates an object representing the result of a single Spearman’s ρ test.
		If the size of arrays n! is smaller than n_thresh, p values are exactly determined using a permutation test. Otherwise p values are computed using SciPy’s `spearmanr`, but with an imposed lower limit of n! and a uniform distribution of p values is assumed.
		
		Parameters
		----------
		x,y
			The two arrays of samples to correlate.
		
		alternative: "greater" or "less"
		
		n_thresh:
			Threshold under which a permutation test is used.
		"""
		n = len(x)
		
		if n>n_thresh:
			p = spearmanr(x,y,alternative=alternative).pvalue
			p = np.clip( p, 1/factorial(n), 1 )
			return cls(p)
		
		# Working with n³·cov(2R(x),2R(y)) because it is integer. As a statistics, it is equivalent to Spearman’s ρ.
		x_r = np.fix(2*rankdata(x)).astype(int)
		y_r = np.fix(2*rankdata(y)).astype(int)
		x_normed = n*x_r - np.sum(x_r)
		y_normed = n*y_r - np.sum(y_r)
		
		orig_cov = np.sum(x_normed*y_normed)
		possible_covs = np.sort([
				np.sum(x_normed*y_permut)
				for y_permut in permutations(y_normed)
			])
		
		if alternative == "greater":
			possible_covs = np.flip(possible_covs)
		elif alternative != "less":
			raise ValueError('Alternative must be "less" or "greater". (A two-sided test is not supported and makes little sense for combining test results.)')
		
		k = len(possible_covs)
		# Using the last of duplicate covs by updating dictionary in the right order:
		cov_to_p = dict( zip( possible_covs, np.linspace(1/k,1,k) ) )
		
		orig_p = cov_to_p[orig_cov]
		return cls( orig_p, list(cov_to_p.values()) )
	
	@classmethod
	def kendalltau( cls, x, y, **kwargs ):
		"""
		Creates an object representing the result of a single Kendall’s τ test using SciPy’s `kendalltau` to compute p values.
		
		NaNs and ties are not supported.
		
		Parameters
		----------
		x,y
			The two arrays of samples to correlate.
		
		alternative: "greater" or "less"
		"""

		if kwargs["alternative"] == "two-sided":
			raise NotImplementedError("The two-sided test is not supported (and makes little sense for combining test results).")
		
		if has_ties(x) or has_ties(y):
			raise NotImplementedError("Ties are not yet implemented.")
		
		p = kendalltau(x,y,**kwargs).pvalue
		n = len(x)
		tot = math.comb(n,2)
		
		possible_ps = [
			_kendall_p_exact(n,dis,"greater")
			for dis in range(0,math.comb(n,2)+1)
		]
		
		return cls(p,possible_ps)
	
	@classmethod
	def fisher_exact( cls, C, alternative="less" ):
		"""
		Creates an object representing the result of Fisher’s exact test for a single contingency table C. This is unrelated to Fisher’s method of combining p values.
		
		Parameters
		----------
		C
			The contingency table.
		
		alternative: "less" or "greater"
		"""
		
		if alternative == "two-sided":
			raise NotImplementedError("The two-sided test is not supported (and makes little sense for combining test results).")
		elif alternative=="less":
			C = np.array(C)
		elif alternative=="greater":
			C = np.fliplr(C)
		
		p = fisher_exact(C,alternative="less")[1]
		
		n1,n2 = np.sum(C,axis=1)
		n ,_  = np.sum(C,axis=0)
		
		possible_ps = [
				hypergeom.cdf( x, n1+n2, n1, n )
				for x in range( max(0,n-n2), min(n,n1)+1 )
			]
		
		return cls( p, possible_ps )


combining_statistics = {
	("fisher"          ,"normal"  ): lambda p:  np.sum( np.log(p)     , axis=0 ),
	("pearson"         ,"normal"  ): lambda q: -np.sum( np.log(q)     , axis=0 ),
	("mudholkar_george","normal"  ): lambda p,q:  np.sum( np.log(p/q) , axis=0 ),
	("stouffer"        ,"normal"  ): lambda p:  np.sum( erfinv(2*p-1) , axis=0 ),
	("tippett"         ,"normal"  ): lambda p:  np.min( p             , axis=0 ),
	("edgington"       ,"normal"  ): lambda p:  np.sum( p             , axis=0 ),
	("simes"           ,"normal"  ): lambda p:  np.min(p/rankdata(p,axis=0,method="ordinal"),axis=0),
	("fisher"          ,"weighted"): lambda p,w:    w.dot(np.log(p))     ,
	("pearson"         ,"weighted"): lambda q,w:   -w.dot(np.log(q))     ,
	("mudholkar_george","weighted"): lambda p,q,w:  w.dot(np.log(p/q))   ,
	("stouffer"        ,"weighted"): lambda p,w:    w.dot(erfinv(2*p-1)) ,
	("edgington"       ,"weighted"): lambda p,w:    w.dot(p)             ,
}

def has_arg(statistic,*args):
	return all(
			arg in signature(statistic).parameters
			for arg in args
		)

statistics_with_inf = {"stouffer"}

def combine(
		ctrs, weights=None,
		method="mudholkar_george", alternative="less",
		n_samples=10000000, sampling_method="proportional",
		RNG=None,
	):
	"""
	Estimates the combined p value of combinable test results. Usually, this result is why you are doing all this.
	
	Parameters
	----------
	ctrs: iterable of CTRs
		The test results that shall be combined.
	
	method: string or function
		One of "fisher", "pearson", "mudholkar_george", "stouffer", "tippett", "edgington", "simes", or a self-defined function.
		
		In the latter case, the function can have the following arguments (which must be named as given):
		* A two-dimensional array `p` containing the p values.
		* A two-dimensional array `q` containing their complements.
		* A one-dimensional array `w` containing the weights.
		The function must return the statistics computed along the zero-th axis.
		For example for the weighted Mudholkar–George method, this function would be `lambda p,q,w:  w.dot(np.log(p/q))`.
	
	alternative: "less" or "two-sided"
		Whether your combined null hypothesis is one- or two-sided.
		Mind that this is not about the sidedness of the individual tests: Those should always be one-sided.
	
	weights: iterable of numbers
		Weights for individual results. Does not work for minimum-based methods (Tippett and Simes).
	
	n_samples
		Number of samples used for Monte Carlo simulation.
	
	RNG
		NumPy random-number generator used for the Monte Carlo simulation.
		Will be automatically generated if not specified.
	
	sampling_method: "proportional" or "stochastic"
		If `"proportional"`, the frequency p values for each individual result will be exactly proportional to its probability – except for rounding. Only the rounding and the order of elements will be stochastic.
		
		If `method` is `"stochastic"`, the values will be randomly sampled and thus their actual frequencies are subject to stochastic fluctuations. This usually leads to slightly less accurate results, but the simulations are statistically independent.
		
		The author of these lines cannot think of any disadvantage to the latter approach.
	
	Returns
	-------
	pvalue
		The estimated combined p value.
	
	std
		The estimated standard deviation of p values when repeating the sampling. This is accurate for stochastic sampling and overestimating for proportional sampling.
	"""
	
	if len(ctrs)==1:
		return Combined_P_Value(ctrs[0].p,0)
	
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
	
	kwargs_null = {}
	kwargs_orig = {}
	sampling_kwargs = dict(RNG=RNG,size=n_samples,method=sampling_method)
	
	if has_arg(statistic,"p","q"):
		kwargs_null["p"] = np.empty((len(ctrs),n_samples))
		kwargs_null["q"] = np.empty((len(ctrs),n_samples))
		for ctr,target_p,target_q in zip(ctrs,kwargs_null["p"],kwargs_null["q"]):
			# target[:] to overwrite the content of target instead of reassigning the variable.
			target_p[:],target_q[:] = ctr.nulldist.sample_both(**sampling_kwargs)
	else:
		if has_arg(statistic,"p"):
			kwargs_null["p"] = np.vstack([
				ctr.nulldist.sample(**sampling_kwargs)
				for ctr in ctrs
			])
		if has_arg(statistic,"q"):
			kwargs_null["q"] = np.vstack([
				ctr.nulldist.sample_complement(**sampling_kwargs)
				for ctr in ctrs
			])
	
	if has_arg(statistic,"p"):
		kwargs_orig["p"] = np.array([ctr.p for ctr in ctrs])
	if has_arg(statistic,"q"):
		kwargs_orig["q"] = np.array([ctr.q for ctr in ctrs])
	
	if weights is not None:
		for kwargs in (kwargs_null,kwargs_orig):
			kwargs["w"] = np.asarray(weights)
	
	err_kwargs = {"divide":"ignore","invalid":"ignore"} if (method in statistics_with_inf) else {}
	with np.errstate(**err_kwargs):
		orig_stat = statistic(**kwargs_orig)
		null_stats = statistic(**kwargs_null)
	
	onesided_p = counted_p( orig_stat, null_stats)
	
	if alternative=="less":
		return onesided_p
	elif alternative=="two-sided":
		return 2*onesided_p
	else:
		raise ValueError('Alternative must be "less" or "two-sided".')

