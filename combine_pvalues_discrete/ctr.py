import math
from inspect import signature
import numpy as np
from warnings import warn
from itertools import permutations

from .tools import sign_test, counted_p, Combined_P_Value, is_empty, searchsorted_closest, has_ties, unify_sorted
from .pdist import PDist

from scipy.special import erfinv, factorial
from scipy.stats import rankdata, spearmanr, pearsonr, kendalltau, fisher_exact, boschloo_exact, wilcoxon
from scipy.stats._morestats import _get_wilcoxon_distr
from scipy.stats._mannwhitneyu import _mwu_state, mannwhitneyu
from scipy.stats._stats_py import _ttest_finish
from scipy.stats._mstats_basic import _kendall_p_exact
from scipy.stats.distributions import hypergeom

def assert_one_sided(alternative):
	if alternative.lower() == "two-sided":
		raise NotImplementedError("The two-sided test is not supported (and makes little sense for combining test results).")
	elif alternative not in ["less","greater"]:
		raise ValueError('Alternative must be "less" or "greater".')

class CTR(object):
	"""
	CTR = combinable test result
	
	Represents a single test result. Use the default constructor to implement a test yourself or use one of the class methods for the respective test.
	
	Parameters
	----------
	p
		The *p* value yielded by the test for the investigated sub-dataset.
	
	all_ps
		An iterable containing all possible *p* values of the test for datasets with the same size as the dataset for this individual test.
		If `None` or empty, all *p* values will be considered possible, i.e., the test will be assumed to be continuous.
	"""
	def __init__(self,p,all_ps=None):
		if p==0: raise ValueError("p value cannot be zero.")
		if np.isnan(p): raise ValueError("p value must not be NaN.")
		
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
		
		Ties are not supported yet because I expect them not to occur in the scenarios that require test combinations (but I may be wrong about this) and they make things much more complicated.
		
		Parameters
		----------
		x,y
			The two arrays of samples to compare.
		
		kwargs
			Further keyword arguments to be passed on to SciPy’s `mannwhitneyu`, such as `alternative`.
		"""
		x = np.array(x)
		y = np.array(y)
		
		if "alternative" not in kwargs:
			raise ValueError("You must specify the alternative.")
		assert_one_sided(kwargs["alternative"])
		
		if np.any(x[:,None]==y):
			raise NotImplementedError("Ties are not yet implemented.")
		
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
			The two arrays of paired samples to compare. If `y` is a number, a one-sample sign test is performed with `y` as the median. With `y` as an iterable, the two-sample test is performed.
		
		alternative: "less" or "greater"
		"""
		
		assert_one_sided(alternative)
		
		p,m,_ = sign_test(x,y,alternative)
		
		all_ps = list( np.cumsum([math.comb(m,i)/2**m for i in range(m)]) ) + [1]
		return cls( p, all_ps )
	
	@classmethod
	def wilcoxon_signed_rank( cls, x, y=None, alternative="less" ):
		"""
		Creates an object representing the result of a single Wilcoxon signed-rank test.
		
		Parameters
		----------
		x,y
			The two arrays of paired samples to compare. If `y` is `None`, the one-sample test is performed, otherwise the two-sample one.
		
		alternative: "less" or "greater"
		"""
		
		assert_one_sided(alternative)
		
		d = np.asarray(x) - (y or 0)
		if has_ties(np.abs(d)) or np.any(d==0):
			raise NotImplementedError("Ties and zeros are not yet implemented.")
		
		n = len(x)
		all_ps = np.cumsum( _get_wilcoxon_distr(n) / 2**n )
		p = wilcoxon(x,y,alternative=alternative,mode="exact").pvalue
		
		return cls( p, all_ps )
	
	@classmethod
	def spearmanr( cls, x, y, alternative="greater", n_thresh=9 ):
		"""
		Creates an object representing the result of a single Spearman’s ρ test.
		If the size of `x` and `y`, *n,* is smaller than `n_thresh`, *p* values are exactly determined using a permutation test. Otherwise *p* values are computed using SciPy’s `spearmanr` assuming a uniform distribution of *p* values and ensuring :math:`p≥\\frac{1}{n!}`.
		
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
		assert_one_sided(alternative)
		
		k = len(possible_covs)
		# Using the last of duplicate covs by updating dictionary in the right order:
		cov_to_p = dict( zip( possible_covs, np.linspace(1/k,1,k) ) )
		
		orig_p = cov_to_p[orig_cov]
		return cls( orig_p, list(cov_to_p.values()) )
	
	@classmethod
	def kendalltau( cls, x, y, **kwargs ):
		"""
		Creates an object representing the result of a single Kendall’s τ test using SciPy’s `kendalltau` to compute *p* values.
		
		NaNs and ties are not supported.
		
		Parameters
		----------
		x,y
			The two arrays of samples to correlate.
		
		alternative: "greater" or "less"
		"""
		
		assert_one_sided(kwargs["alternative"])
		
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
		Creates an object representing the result of Fisher’s exact test for a single contingency table C. This is unrelated to Fisher’s method of combining *p* values. Note that most scientific applications do not meet the restrictive conditions of this test and Boschloo’s exact test is more appropriate.
		
		Parameters
		----------
		C: 2×2 array or nested iterable
			The contingency table.
		
		alternative: "less" or "greater"
		"""
		
		assert_one_sided(alternative)
		C = np.fliplr(C) if alternative=="greater" else np.array(C)
		
		p = fisher_exact(C,alternative="less")[1]
		
		n1,n2 = np.sum(C,axis=1)
		n ,_  = np.sum(C,axis=0)
		
		possible_ps = [
				hypergeom.cdf( x, n1+n2, n1, n )
				for x in range( max(0,n-n2), min(n,n1)+1 )
			]
		
		return cls( p, possible_ps )
	
	@classmethod
	def boschloo_exact( cls, C, alternative="less", n=32, atol=1e-10 ):
		"""
		Creates an object representing the result of Boschloo’s exact for a single contingency table C using SciPy’s implementation.
		
		Parameters
		----------
		C: 2×2 array or nested iterable
			The contingency table.
		
		alternative: "less" or "greater"
		
		n
			The same parameter of SciPy’s `boschloo_exact`.
		
		atol
			*p* values that are closer than this are treated as identical.
		"""
		
		assert_one_sided(alternative)
		C = np.fliplr(C) if alternative=="greater" else np.array(C)
		
		p = boschloo_exact(C,alternative="less",n=n).pvalue
		
		n1,n2 = np.sum(C,axis=1)
		
		possible_ps = sorted(
				boschloo_exact(
						[ [ C11, n1-C11 ], [ C21, n2-C21 ] ],
						alternative="less",
						n=n,
					).pvalue
				for C11 in range( 0, n1+1 )
				for C21 in range( C11==0, n2+(C11!=n1) )
			)
		
		# Unify close p values.
		i = 1
		while i<len(possible_ps):
			if possible_ps[i-1]+atol > possible_ps[i]:
				del possible_ps[i]
			else:
				i += 1
		
		return cls( p, possible_ps )

combining_statistics = {
	("fisher"          ,"normal"  ): lambda p:  np.sum( np.log(p)     , axis=0 ),
	("pearson"         ,"normal"  ): lambda q: -np.sum( np.log(q)     , axis=0 ),
	("mudholkar_george","normal"  ): lambda p,q:np.sum( np.log(p/q)   , axis=0 ),
	("stouffer"        ,"normal"  ): lambda p:  np.sum( erfinv(2*p-1) , axis=0 ),
	("tippett"         ,"normal"  ): lambda p:  np.min( p             , axis=0 ),
	("edgington"       ,"normal"  ): lambda p:  np.sum( p             , axis=0 ),
	("edgington_sym"   ,"normal"  ): lambda p,q:np.sum( p-q           , axis=0 ),
	("simes"           ,"normal"  ): lambda p:  np.min(p/rankdata(p,axis=0,method="ordinal"),axis=0),
	("fisher"          ,"weighted"): lambda p,w:    w.dot(np.log(p))     ,
	("pearson"         ,"weighted"): lambda q,w:   -w.dot(np.log(q))     ,
	("mudholkar_george","weighted"): lambda p,q,w:  w.dot(np.log(p/q))   ,
	("stouffer"        ,"weighted"): lambda p,w:    w.dot(erfinv(2*p-1)) ,
	("edgington"       ,"weighted"): lambda p,w:    w.dot(p)             ,
	("edgington_sym"   ,"weighted"): lambda p,q,w:  w.dot(p+1-q)           ,
}

statistics_with_inf = {"stouffer"}

def flip_pq(args):
	if isinstance(args,str) and len(args)==1:
		if args == "p":
			return "q"
		elif args == "q":
			return "p"
		else:
			return args
	else:
		return { flip_pq(arg) for arg in args }

def apply_statistics(statistic,data,alternative="less"):
	if alternative in ["less","greater"]:
		kwargs = {
			par: data[ par if alternative=="less" else flip_pq(par) ]
			for par in signature(statistic).parameters
		}
		return statistic(**kwargs)
	elif alternative == "two-sided":
		return np.minimum(
				apply_statistics(statistic,data,"less"   ),
				apply_statistics(statistic,data,"greater"),
			)
	else:
		raise ValueError('Alternative must be "less", "greater", or "two-sided".')

def combine(
		ctrs, weights=None,
		method="mudholkar_george", alternative="less",
		n_samples=10000000, sampling_method="proportional",
		rtol=1e-15, atol=1e-15,
		RNG=None,
	):
	"""
	Estimates the combined *p* value of combinable test results. Usually, this result is why you are using this module.
	
	Parameters
	----------
	ctrs: iterable of CTRs
		The test results that shall be combined.
	
	method: string or function
		One of "fisher", "pearson", "mudholkar_george", "stouffer", "tippett", "edgington", "edgington_sym", "simes", or a self-defined function.
		
		In the latter case, the function can have the following arguments (which must be named as given):
		
		* A two-dimensional array `p` containing the *p* values.
		* A two-dimensional array `q` containing their complements.
		* A one-dimensional array `w` containing the weights.
		
		The function must return the statistics computed along the zero-th axis.
		For example for the weighted Mudholkar–George method, this function would be `lambda p,q,w:  w.dot(np.log(p/q))`.
		The sign of the statistics must be such that low values indicate a high significance.
	
	alternative: "less", "greater", or "two-sided"
		The direction of the (common) trend that your compound null hypothesis is testing against.
		Mind that this is not about the sidedness of the individual tests: Those should always be one-sided.
		
		* If "less", the compound research hypothesis is that the subtests exhibit a trend towards a low *p* value.
		* If "greater", the compound research hypothesis is that the subtests exhibit a trend towards high *p* values (close to 1). In this case, the method of choice will be applied to the complements of the *p* values (see `complements`).
		* If "two-sided", the compound research hypothesis is that the subtests exhibit either of the two above trends.
	
	weights: iterable of numbers
		Weights for individual results. Does not work for minimum-based methods (Tippett and Simes).
	
	n_samples
		Number of samples used for Monte Carlo simulation. High numbers increase the accuracy, but also the runtime and memory requirements.
	
	rtol: non-negative float
	atol: non-negative float
		Values of the statistics with closer than specified by `atol` and `rtol` are regarded as identical (as in `numpy.isclose`). A small value (such as the default) may improve the results if numerical noise makes values different.
	
	RNG
		NumPy random-number generator used for the Monte Carlo simulation.
		If `None`, it will be automatically generated if not specified.
	
	sampling_method: "proportional" or "stochastic"
		If `"proportional"`, the frequency *p* values for each individual result will be exactly proportional to its probability – except for rounding. Only the rounding and the order of elements will be random.
		
		If `"stochastic"`, the values will be randomly sampled and thus their sampled frequencies are subject to stochastic fluctuations. This usually leads to slightly less accurate results, but the simulations are statistically independent.
		
		The author of these lines cannot think of any disadvantage to the first approach and has not found any in numerical experiments.
	
	Returns
	-------
	pvalue
		The estimated combined *p* value.
	
	std
		The estimated standard deviation of *p* values when repeating the sampling. This is accurate for stochastic sampling and overestimating for proportional sampling.
	"""
	
	if len(ctrs)==1:
		p = ctrs[0].p
		q = ctrs[0].q
		if alternative=="less":
			return Combined_P_Value(p,0)
		elif alternative=="greater":
			return Combined_P_Value(q,0)
		elif alternative=="two-sided":
			p = ctrs[0].p
			return Combined_P_Value( 2*min(p,q), 0 )
	
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
	
	required_args = set(signature(statistic).parameters)
	if alternative == "greater":
		required_args = flip_pq(required_args)
	elif alternative == "two-sided":
		required_args = required_args | flip_pq(required_args)
	
	sampling_kwargs = dict(RNG=RNG,size=n_samples,method=sampling_method)
	
	data_null = {}
	if {"p","q"} <= required_args:
		data_null["p"] = np.empty((len(ctrs),n_samples))
		data_null["q"] = np.empty((len(ctrs),n_samples))
		for ctr,target_p,target_q in zip(ctrs,data_null["p"],data_null["q"]):
			# target[:] to overwrite the content of target instead of reassigning the variable.
			target_p[:],target_q[:] = ctr.nulldist.sample_both(**sampling_kwargs)
	else:
		for x in ["p","q"]:
			if x in required_args:
				data_null[x] = np.empty((len(ctrs),n_samples))
				for ctr,target in zip(ctrs,data_null[x]):
					target[:] = ctr.nulldist.sample(which=x,**sampling_kwargs)
	
	data_orig = {
			x : np.array([getattr(ctr,x) for ctr in ctrs])
			for x in ["p","q"]
		}
	
	if weights is not None:
		data_null["w"] = data_orig["w"] = np.asarray(weights)
	
	err_kwargs = {"divide":"ignore","invalid":"ignore"} if (method in statistics_with_inf) else {}
	with np.errstate(**err_kwargs):
		orig_stat  = apply_statistics(statistic,data_orig,alternative=alternative)
		null_stats = apply_statistics(statistic,data_null,alternative=alternative)
	
	return counted_p( orig_stat, null_stats, rtol=rtol, atol=atol )

