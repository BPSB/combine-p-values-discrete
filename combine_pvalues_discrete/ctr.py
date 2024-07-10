import math
from inspect import signature
import numpy as np
from warnings import warn
from itertools import permutations

from .tools import sign_test, counted_p, Combined_P_Value, is_empty, searchsorted_closest, has_ties
from .pdist import PDist

from scipy.special import erfinv, factorial
from scipy.stats import rankdata, spearmanr, pearsonr, kendalltau, fisher_exact, boschloo_exact, wilcoxon
from scipy.stats._hypotests import _get_wilcoxon_distr
from scipy.stats._mannwhitneyu import _MWU, mannwhitneyu
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
	
	dof
		Number of degrees of freedom that affect the test’s outcome; only relevant for automatic weighting (see the `weights` argument of `combine`).
	
	Examples
	--------
	.. code-block:: python3
	
		p = 0.1
		possible_ps = [0.1,0.5,1]
		dof = 7
		ctr = CTR(p,possible_ps,dof)
	
	"""
	def __init__(self,p,all_ps=None,dof=None):
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
		self.dof = dof
	
	def __repr__(self):
		return f"""CombinableTestResult(
		\t p-value: {self.p},
		\t nulldist: {self.nulldist}
		\t degrees of freedom: {self.dof}
		\t )"""
	
	def __eq__(self,other):
		return self._approx(other,atol=0)
	
	def _approx(self,other,atol=1e-14):
		"""
		Whether this result is identical to another with in an absolute tolerance `atol` between *p* values.
		"""
		return (
				abs(self.p-other.p) <= atol and
				self.nulldist.approx(other.nulldist,atol) and
				self.dof == other.dof
			)
	
	@classmethod
	def mann_whitney_u( cls, x, y, **kwargs ):
		"""
		Creates an object representing the result of a single Mann–Whitney *U* test (using SciPy’s `mannwhitneyu`).
		
		Ties are not supported yet because I expect them not to occur in the scenarios that require test combinations (but I may be wrong about this) and they make things much more complicated.
		
		For automatic weighting, the number of degrees of freedom is taken to be `len(x) + len(y) - 1`. Note how this aligns with the sign test and Wilcoxon’s signed rank test when either `x` or `y` has size 1.
		
		Parameters
		----------
		x,y
			The two iterables of samples to compare.
		
		kwargs
			Further keyword arguments to be passed on to SciPy’s `mannwhitneyu`, such as `alternative`.
		
		Examples
		--------
		.. code-block:: python3
		
			dataset_A = [20,44,14,68]
			dataset_B = [73,22,80]
			ctr = CTR.mann_whitney_u(dataset_A,dataset_B,alternative="less")
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
		possible_ps = np.cumsum( _MWU(n,m).build_u_freqs_array(n*m) )
		return cls( p, possible_ps, n+m-1 )
	
	@classmethod
	def sign_test( cls, x, y=0, alternative="less" ):
		"""
		Creates an object representing the result of a single sign test.
		
		For automatic weighting, the number of pairs is taken as the number of degrees of freedom.
		
		Parameters
		----------
		x,y
			The two arrays of paired samples to compare. If `y` is a number, a one-sample sign test is performed with `y` as the median. With `y` as an iterable, the two-sample test is performed.
		
		alternative: "less" or "greater"
		
		Examples
		--------
		.. code-block:: python3
		
			dataset_A = [20,44,14,68]
			dataset_B = [73,22,80,53]
			ctr = CTR.sign_test(dataset_A,dataset_B,alternative="less")
		"""
		
		assert_one_sided(alternative)
		
		p,m,_ = sign_test(x,y,alternative)
		
		all_ps = list( np.cumsum([math.comb(m,i)/2**m for i in range(m)]) ) + [1]
		
		return cls( p, all_ps, m )
	
	@classmethod
	def wilcoxon_signed_rank( cls, x, y=None, alternative="less" ):
		"""
		Creates an object representing the result of a single Wilcoxon signed-rank test.
		
		For automatic weighting, the number of pairs is taken as the number of degrees of freedom.
		
		Parameters
		----------
		x,y
			The two arrays of paired samples to compare. If `y` is `None`, the one-sample test is performed, otherwise the two-sample one.
		
		alternative: "less" or "greater"
		
		Examples
		--------
		.. code-block:: python3
		
			dataset_A = [20,44,14,68]
			dataset_B = [73,22,80,53]
			ctr = CTR.wilcoxon_signed_rank(dataset_A,dataset_B,alternative="less")
		"""
		
		assert_one_sided(alternative)
		
		d = np.asarray(x) - (y or 0)
		if has_ties(np.abs(d)) or np.any(d==0):
			raise NotImplementedError("Ties and zeros are not yet implemented.")
		
		n = len(x)
		all_ps = np.cumsum( _get_wilcoxon_distr(n) )
		p = wilcoxon(x,y,alternative=alternative,mode="exact").pvalue
		
		return cls( p, all_ps, n )
	
	@classmethod
	def spearmanr( cls, x, y, alternative="greater", n_thresh=9 ):
		"""
		Creates an object representing the result of a single Spearman’s ρ test.
		If the size of `x` and `y`, *n,* is smaller than `n_thresh`, *p* values are exactly determined using a permutation test. Otherwise *p* values are computed using SciPy’s `spearmanr` assuming a uniform distribution of *p* values and ensuring :math:`p≥\\frac{1}{n!}`.
		
		For automatic weighting, the number of degrees of freedom is taken to be one less than the number of pairs.
		
		Parameters
		----------
		x,y
			The two arrays of samples to correlate.
		
		alternative: "greater" or "less"
		
		n_thresh:
			Threshold under which a permutation test is used.
		
		Examples
		--------
		.. code-block:: python3
		
			dataset_A = [1,3,4,2,5,6]
			dataset_B = [3,1,2,5,6,4]
			ctr = CTR.spearmanr(dataset_A,dataset_B,alternative="greater")
		"""
		n = len(x)
		
		if n>n_thresh:
			p = spearmanr(x,y,alternative=alternative).pvalue
			p = np.clip( p, 1/factorial(n), 1 )
			return cls(p,dof=n-1)
		
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
		return cls( orig_p, list(cov_to_p.values()), n-1 )
	
	@classmethod
	def kendalltau( cls, x, y, **kwargs ):
		"""
		Creates an object representing the result of a single Kendall’s τ test using SciPy’s `kendalltau` to compute *p* values.
		
		NaNs and ties are not supported yet.
		
		For automatic weighting, the number of degrees of freedom is taken to be one less than the number of pairs.
		
		Parameters
		----------
		x,y
			The two arrays of samples to correlate.
		
		alternative: "greater" or "less"
		
		Examples
		--------
		.. code-block:: python3
		
			dataset_A = [1,3,4,2,5,6]
			dataset_B = [3,1,2,5,6,4]
			ctr = CTR.kendalltau(dataset_A,dataset_B,alternative="greater")
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
		
		return cls(p,possible_ps,n-1)
	
	@classmethod
	def fisher_exact( cls, C, alternative="less" ):
		"""
		Creates an object representing the result of Fisher’s exact test for a single contingency table `C`. This is unrelated to Fisher’s method of combining *p* values. Note that, for most scientific applications, the restrictive conditions of this test are not met and Boschloo’s exact test is more appropriate.
		
		For automatic weighting, the sum over `C` is taken as the number of degrees of freedom.
		
		Parameters
		----------
		C: 2×2 array or nested iterable
			The contingency table.
		
		alternative: "less" or "greater"
		
		Examples
		--------
		.. code-block:: python3
		
			contingency_table = [[10,4],[5,4]]
			ctr = CTR.fisher_exact(contingency_table,alternative="greater")
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
		
		return cls( p, possible_ps, np.sum(C) )
	
	@classmethod
	def boschloo_exact( cls, C, alternative="less", n=32, atol=1e-10 ):
		"""
		Creates an object representing the result of Boschloo’s exact for a single contingency table `C` using SciPy’s implementation.
		
		For automatic weighting, the sum over `C` is taken as the number of degrees of freedom.
		
		Parameters
		----------
		C: 2×2 array or nested iterable
			The contingency table.
		
		alternative: "less" or "greater"
		
		n
			The same parameter of SciPy’s `boschloo_exact`.
		
		atol
			*p* values that are closer than this are treated as identical.
		
		Examples
		--------
		.. code-block:: python3
		
			contingency_table = [[10,4],[5,4]]
			ctr = CTR.boschloo_exact(contingency_table,alternative="greater")
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
		
		return cls( p, possible_ps, np.sum(C) )

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
	("edgington_sym"   ,"weighted"): lambda p,q,w:  w.dot(p+1-q)         ,
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

def get_statistic(method,weights):
	if method in (method for method,_ in combining_statistics):
		if weights is None:
			return combining_statistics[method,"normal"]
		else:
			try:
				return combining_statistics[method,"weighted"]
			except KeyError:
				raise ValueError(f'No weighted version of "{method}" method')
	else:
		if not callable(method):
			raise ValueError(f'Method "{method}" is neither known nor callable.')
		return method

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
		* If "two-sided", the compound research hypothesis is that the subtests exhibit either of the two above trends. Beware that this is not necessarily the same as just doubling the *p* value of the respective one-sided test, since for some combining methods, a compound dataset may exhibit **both** trends.
	
	weights: iterable of numbers or "dof"
		Weights for individual results. Does not work for minimum-based methods (Tippett and Simes).
		
		If `"dof"`, each test will be weighted with its degrees of freedom, i.e., the number of samples that can independently affect the test’s result. As there is some potential for ambiguity as to what the degrees of freedom of a given test are, please check that they are what you expect by looking at the tests’ documentations or the attribute `dof` of a CombinableTestResult. This particularly applies when combining different tests.
	
	n_samples
		Number of samples used for Monte Carlo simulation. High numbers increase the accuracy, but also the runtime and memory requirements.
	
	rtol: non-negative float
	atol: non-negative float
		Values of the statistics closer than specified by `atol` and `rtol` are regarded as identical (as in `numpy.isclose`). A small value (such as the default) may improve the results if numerical noise makes values different.
	
	RNG
		NumPy random-number generator used for the Monte Carlo simulation.
		If `None`, it will be automatically generated.
	
	sampling_method: "proportional" or "stochastic"
		How to sample from the *p* value distributions of the subtests.
		
		If `"proportional"`, the frequency of *p* values for each individual result will be exactly proportional to its probability – except for rounding. Only the rounding and the order of elements will be random.
		
		If `"stochastic"`, the values will be randomly sampled and thus their sampled frequencies are subject to stochastic fluctuations. This usually leads to slightly less accurate results, but the simulations are statistically independent.
		
		The author of these lines cannot think of any disadvantage to the first approach and has not found any in numerical experiments.
	
	Returns
	-------
	pvalue
		The estimated combined *p* value.
	
	std
		The estimated standard deviation of *p* values when repeating the sampling. This is accurate for stochastic sampling and overestimating for proportional sampling.
	
	Examples
	--------
	.. code-block:: python3
	
		ctrs = [
			CTR.sign_test( [24,58,10], [65,51,61], alternative="less" ),
			CTR.mann_whitney_u( [20,44,14,68], [73,22,80], alternative="less" ),
			CTR( ttest_ind( [28,36,11], [93,76,70,83], alternative="less" ).pvalue ),
		]
		compound_p = combine(ctrs,alternative="less").pvalue
	"""
	
	if len(ctrs)==1:
		p = ctrs[0].p
		q = ctrs[0].q
		if alternative=="less":
			return Combined_P_Value(p,0)
		elif alternative=="greater":
			return Combined_P_Value(q,0)
		elif alternative=="two-sided":
			return Combined_P_Value( min(2*p,2*q,1), 0 )
	
	statistic = get_statistic(method,weights)
	
	# required_args is a subset of {"p","q","w"} indicating the arguments of the combining statistic.
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
		for x in {"p","q"} & required_args:
			data_null[x] = np.empty((len(ctrs),n_samples))
			for ctr,target in zip(ctrs,data_null[x]):
				target[:] = ctr.nulldist.sample(which=x,**sampling_kwargs)
	
	data_orig = {
			x : np.array([getattr(ctr,x) for ctr in ctrs])
			for x in ["p","q"]
		}
	
	if isinstance(weights,str) and weights == "dof":
		weights = [ ctr.dof for ctr in ctrs ]
	if weights is not None:
		data_null["w"] = data_orig["w"] = np.asarray(weights)
	
	err_kwargs = {"divide":"ignore","invalid":"ignore"} if (method in statistics_with_inf) else {}
	with np.errstate(**err_kwargs):
		orig_stat  = apply_statistics(statistic,data_orig,alternative=alternative)
		null_stats = apply_statistics(statistic,data_null,alternative=alternative)
	
	return counted_p( orig_stat, null_stats, rtol=rtol, atol=atol )

def direction( ctrs, weights=None, method="mudholkar_george" ):
	"""
	A service function to indicate whether the `ctrs` are rather trending towards high or low *p* values.
	
	If you are combining two-sidedly, this tells you the direction of the strongest trend, whether it’s significant or not (that’s what `combine` is for). Beware that for some methods such as Fisher’s, a compound dataset may exhibit a significant trend in **both** directions and this function won’t tell you. This cannot happen for symmetric methods (Mudholkar–George, Stouffer, and symmetrised Edgington).
	
	Parameters
	----------
	`weights` and `method` as for `combine`.
	
	Returns
	-------
	direction: string
		One of "less", "greater" or "equal". The latter is only returned if the statistics of the combining method are exactly equal in either direction.
	
	Examples
	--------
	.. code-block:: python3
	
		ctrs = [
			CTR.sign_test( [24,58,10], [65,51,61], alternative="less" ),
			CTR.mann_whitney_u( [20,44,14,68], [73,22,80], alternative="less" ),
			CTR( ttest_ind( [28,36,11], [93,76,70,83], alternative="less" ).pvalue ),
		]
		trend = direction(ctrs)
	"""
	
	if len(ctrs)==1:
		stats = { "less": ctrs[0].p, "greater": ctrs[0].q }
	else:
		statistic = get_statistic(method,weights)
		
		data_orig = {
				x : np.array([getattr(ctr,x) for ctr in ctrs])
				for x in ["p","q"]
			}
		
		if isinstance(weights,str) and weights == "dof":
			weights = [ ctr.dof for ctr in ctrs ]
		if weights is not None:
			data_orig["w"] = np.asarray(weights)
		
		err_kwargs = {"divide":"ignore","invalid":"ignore"} if (method in statistics_with_inf) else {}
		with np.errstate(**err_kwargs):
			stats = {
				case: apply_statistics(statistic,data_orig,alternative=case)
				for case in ["less","greater"]
			}
	
	if stats["less"] < stats["greater"]:
		return "less"
	elif stats["greater"] < stats["less"]:
		return "greater"
	else:
		return "equal"

