from pytest import mark, raises
from itertools import chain, product, combinations_with_replacement
import numpy as np
from math import prod, sqrt

from scipy.stats import combine_pvalues
from scipy.special import erf, erfinv

from combine_pvalues_discrete.ctr import CTR, combine, combining_statistics, assert_one_sided
from combine_pvalues_discrete.tools import sign_test, assert_matching_p_values

n_samples = 10000

@mark.parametrize(
	  "  p  , alternative, result",
	[
	  ( 0.01, "less"     , 0.01   ),
	  ( 0.01, "greater"  , 0.99   ),
	  ( 0.01, "two-sided", 0.02   ),
	  ( 0.99, "less"     , 0.99   ),
	  ( 0.99, "greater"  , 0.01   ),
	  ( 0.99, "two-sided", 0.02   ),
	])
def test_single_p(p,alternative,result):
	assert np.isclose(
			combine([CTR(p)],alternative=alternative).pvalue,
			result
		)

def test_assert_one_sided_good():
	assert_one_sided("less")
	assert_one_sided("greater")

@mark.parametrize("bad_input",["two-sided","TWO-SIDED","wrzlprmft","LESS","Less"])
def test_assert_one_sided_bad(bad_input):
	with raises(Exception):
		assert_one_sided(bad_input)

def test_zero_p():
	with raises(ValueError):
		CTR( 0, [0,1,2,3] ),

examples = [
	CTR( 0.5, [0.5,      1] ),
	CTR( 1.0, [0.5,      1] ),
	CTR( 0.3, [0.3, 0.5, 1] ),
	CTR( 0.7, [0.2, 0.7, 1] ),
]

@mark.parametrize(
		"combo",
		chain(*(
			combinations_with_replacement(examples,r)
			for r in range(1,3)
		))
	)
@mark.parametrize("method,variant",combining_statistics)
@mark.parametrize("sampling_method",["proportional","stochastic"])
@mark.parametrize("alt",["less","greater","two-sided"])
def test_commutativity_and_associativity(combo,method,variant,sampling_method,alt,rng):
	get_p = lambda combo,weights: combine(
				combo,
				weights = weights,
				RNG = rng,
				n_samples = n_samples,
				method = method,
				sampling_method = sampling_method,
				alternative = alt,
			).pvalue
	
	n = len(combo)
	combo = np.array(combo)
	weights = rng.random(n) if variant=="weighted" else None
	result_1 = get_p(combo,weights)
	
	new_order = rng.choice(range(n),size=n,replace=False)
	combo = combo[new_order]
	weights = weights[new_order] if variant=="weighted" else None
	result_2 = get_p(combo,weights)
	
	assert_matching_p_values(
			result_1,
			result_2,
			n_samples,
			factor = 3.5 if sampling_method=="proportional" else 4,
		)

@mark.parametrize("example",examples)
def test_combine_single(example):
	assert combine([example]).pvalue == example.p

# Reproducing a sign test by combining single comparisons:

@mark.parametrize( "n,replicate", product( range(2,15), range(20) ) )
@mark.parametrize( "method", ["fisher","mudholkar_george"] )
def test_comparison_to_sign_test(n,replicate,method,rng):
	def my_sign_test_onesided(X,Y):
		ctrs = [
				CTR( 0.5 if x<y else 1, [0.5,1.0] )
				for x,y in zip(X,Y)
			]
		return combine(
				ctrs,
				n_samples = n_samples,
				method = method,
				RNG = rng,
			).pvalue
	
	X = rng.random(n)
	Y = rng.random(n)
	
	assert_matching_p_values(
		my_sign_test_onesided(X,Y),
		sign_test(X,Y)[0],
		n_samples,
	)

# Reproducing `combine_pvalues` for continuous tests and comparing:

def emulate_continuous_combine_ps(ps,**kwargs):
	ctrs = [ CTR(p) for p in ps ]
	return combine(ctrs,n_samples=n_samples,**kwargs).pvalue

# Cannot compare with Pearson’s and Tippett’s method due to SciPy Issue #15373
@mark.parametrize( "method", ["fisher","mudholkar_george","stouffer"] )
@mark.parametrize( "n", range(2,17,4) )
@mark.parametrize( "magnitude", ["small","normal"] )
@mark.parametrize("sampling_method",["proportional","stochastic"])
def test_compare_with_combine_pvalues(n,method,magnitude,sampling_method,rng):
	ps = 10**rng.uniform( -3 if magnitude=="small" else -1, 0, n )
	
	assert_matching_p_values(
		emulate_continuous_combine_ps(
			ps,
			method = method,
			RNG = rng,
			sampling_method = sampling_method,
		),
		combine_pvalues(ps,method=method)[1],
		n_samples,
		factor = 3 if sampling_method=="proportional" else 4,
	)

@mark.parametrize( "n", range(2,15) )
@mark.parametrize( "magnitude", ["small","normal"] )
@mark.parametrize("sampling_method",["proportional","stochastic"])
def test_compare_with_combine_pvalues_weighted(n,magnitude,sampling_method,rng):
	ps = 10**rng.uniform( -3 if magnitude=="small" else -1, 0, n )
	weights = rng.random(n)
	
	assert_matching_p_values(
		emulate_continuous_combine_ps(
			ps,
			weights=weights,
			method="stouffer",
			RNG=rng,
			sampling_method = sampling_method,
		),
		combine_pvalues(ps,method="stouffer",weights=weights)[1],
		n_samples,
		factor = 3 if sampling_method=="proportional" else 4,
	)

@mark.parametrize("method,variant",combining_statistics)
@mark.parametrize("variables", ["one", "all"])
@mark.parametrize("sampling_method",["proportional","stochastic"])
@mark.parametrize("alt",["less","greater"])
def test_monotony(method,variant,variables,sampling_method,alt,rng):
	# Test that result increases monotonously with respect to input.
	n,k = 5,7
	changing_values = {
			"less": 10**np.linspace(-3,-0.1,n),
			"greater": 1-10**np.linspace(-3,-0.1,n),
		}[alt]
	pvalues = rng.random(k)
	weights = rng.uniform(0.1,1,size=k)
	combined_ps = []
	errors = []
	for changing_value in changing_values:
		if variables == "one":
			pvalues[0] = changing_value
		else:
			pvalues = np.full(k,changing_value)
		ctrs = [ CTR(p) for p in pvalues ]
		combined_p,error = combine(
					ctrs,
					method = method,
					weights = weights if variant=="weighted" else None,
					alternative = alt,
					n_samples = n_samples,
					sampling_method = sampling_method,
					RNG = rng,
				)
		combined_ps.append(combined_p)
		errors.append(error)
	
	errors = np.array(errors)
	diff_errors = 1.5*(errors[:-1]+errors[1:])
	
	assert np.all( np.diff(combined_ps) >= -diff_errors )

# CDF of the standard normal distribution and its inverse for Stouffer’s method.
phi = lambda z: (1+erf(z/sqrt(2)))/2
phiinv = lambda x: sqrt(2)*erfinv(2*x-1)

@mark.parametrize(
			"  method  ,   alt      ,  solution   ",
		[
			("tippett" , "less"     , 1-(1-0.4)**3),
			("simes"   , "less"     , 0.9         ),
			("stouffer", "less"     , phi((phiinv(0.4)+phiinv(0.7)+phiinv(0.9))/sqrt(3)) ),
			("tippett" , "greater"  , 1-0.9**3    ),
			("simes"   , "greater"  , 0.3         ),
			("stouffer", "greater"  , phi((phiinv(0.6)+phiinv(0.3)+phiinv(0.1))/sqrt(3)) ),
			("tippett" , "two-sided", (1-0.8**3)  ),
			("stouffer", "two-sided", 2*phi((phiinv(0.6)+phiinv(0.3)+phiinv(0.1))/sqrt(3)) ),
			( lambda p:np.min(p,axis=0), "less" , 1-(1-0.4)**3 ),
		]
	)
@mark.parametrize("sampling_method",["proportional","stochastic"])
def test_simple_case(method,solution,sampling_method,alt,rng):
	assert_matching_p_values(
		emulate_continuous_combine_ps(
			[0.9,0.7,0.4],
			method = method,
			alternative = alt,
			RNG = rng,
			sampling_method = sampling_method,
		),
		solution,
		n_samples
	)

@mark.parametrize("sampling_method",["proportional","stochastic"])
def test_simple_weighted_case(sampling_method,rng):
	assert_matching_p_values(
		emulate_continuous_combine_ps(
			[0.9,0.7,0.4],
			weights = [1,2,3],
			method = "stouffer",
			RNG = rng,
			sampling_method=sampling_method,
		),
		phi( (phiinv(0.9)+2*phiinv(0.7)+3*phiinv(0.4)) / sqrt(1**2+2**2+3**2) ),
		n_samples,
	)

@mark.parametrize("method",(
		method
		for method,variant in combining_statistics
		if variant=="weighted"
	))
@mark.parametrize("sampling_method",["proportional","stochastic"])
@mark.parametrize("alt",["less","greater","two-sided"])
def test_identical_weights(method,sampling_method,alt,rng):
	n = 10
	ps = rng.random(n)
	weights = np.full(n,rng.exponential())
	
	results = [
		emulate_continuous_combine_ps(
			ps,
			RNG=rng,
			method=method,
			weights=w,
			alternative=alt,
			sampling_method=sampling_method,
		)
		for w in [weights,None]
	]
	assert_matching_p_values(*results,n=n_samples,factor=4,compare=True)

@mark.parametrize("method,variant",combining_statistics)
def test_less_greater_symmetry(method,variant,rng):
	m = 10
	
	ctrs_L,ctrs_G = [],[]
	for _ in range(m):
		n = rng.randint(5,10)
		data = rng.normal(size=n)
		ctrs_L.append( CTR.sign_test( data) )
		ctrs_G.append( CTR.sign_test(-data) )
	
	kwargs = dict(
			weights = rng.random(m) if variant=="weighted" else None,
			method = method,
			n_samples = n_samples,
			RNG = rng,
		)
	result_L = combine(ctrs_L,alternative="less"   ,**kwargs).pvalue
	result_G = combine(ctrs_G,alternative="greater",**kwargs).pvalue
	
	assert_matching_p_values( result_L, result_G, n=n_samples, compare=True )

@mark.parametrize(
		"method_A,method_B",
	[
		( "mudholkar_george", "mudholkar_george" ),
		( "fisher"          , "pearson"          ),
		( "edgington_sym"   , "edgington_sym"    ),
	])
@mark.parametrize("weighted",[False,True])
def test_complement_symmetry(method_A,method_B,weighted,rng):
	m = 10
	
	ctrs_A,ctrs_B = [],[]
	for _ in range(m):
		n = rng.randint(5,10)
		data = rng.normal(size=n)
		ctrs_A.append( CTR.sign_test( data) )
		ctrs_B.append( CTR.sign_test(-data) )
	
	kwargs = dict(
			weights = rng.random(m) if weighted else None,
			n_samples = n_samples,
			RNG = rng,
			alternative="less",
		)
	result_A =   combine(ctrs_A,method=method_A,**kwargs).pvalue
	result_B = 1-combine(ctrs_B,method=method_B,**kwargs).pvalue
	
	assert_matching_p_values( result_A, result_B, n=n_samples, compare=True )

