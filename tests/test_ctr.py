from pytest import mark, raises
from itertools import chain, product, combinations_with_replacement
import numpy as np
from math import prod, sqrt

from scipy.stats import combine_pvalues
from scipy.special import erf, erfinv

from combine_pvalues_discrete.ctr import CTR, combine, combining_statistics
from combine_pvalues_discrete.tools import sign_test, assert_matching_p_values

size = 100000

examples = [
	CTR( 0.5, [0.5,      1] ),
	CTR( 1.0, [0.5,      1] ),
	CTR( 0.3, [0.3, 0.5, 1] ),
	CTR( 0.7, [0.2, 0.7, 1] ),
]

@mark.parametrize(
		"seed, combo",
		enumerate(chain(*(
			combinations_with_replacement(examples,r)
			for r in range(1,3)
		)))
	)
@mark.parametrize("method,variant",combining_statistics)
def test_commutativity_and_associativity(seed,combo,method,variant):
	RNG = np.random.default_rng(seed)
	
	get_p = lambda combo,weights: combine(
				combo,
				weights = weights,
				RNG = RNG,
				size = size,
				method = method
			).pvalue
	
	n = len(combo)
	combo = np.array(combo)
	weights = RNG.random(n) if variant=="weighted" else None
	result_1 = get_p(combo,weights)
	
	new_order = RNG.choice(range(n),size=n,replace=False)
	combo = combo[new_order]
	weights = weights[new_order] if variant=="weighted" else None
	result_2 = get_p(combo,weights)
	
	assert_matching_p_values(result_1,result_2,size,factor=3)

@mark.parametrize("example",examples)
def test_combine_single(example):
	assert combine([example]).pvalue == example.p

# Reproducing a sign test by combining single comparisons:

@mark.parametrize( "n,replicate", product( range(2,15), range(20) ) )
def test_comparison_to_sign_test(n,replicate):
	RNG = np.random.default_rng((n+20)**3*replicate)
	
	def my_sign_test_onesided(X,Y):
		ctrs = [
				CTR( 0.5 if x<y else 1, [0.5,1.0] )
				for x,y in zip(X,Y)
			]
		return combine(ctrs,size=size,RNG=RNG,method="fisher").pvalue
	
	X = RNG.random(n)
	Y = RNG.random(n)
	
	assert_matching_p_values(
		my_sign_test_onesided(X,Y),
		sign_test(X,Y)[0],
		size,
	)

# Reproducing `combine_pvalues` for continuous tests and comparing:

def emulate_continuous_combine_ps(ps,RNG=np.random.default_rng(),**kwargs):
	ctrs = [ CTR(p) for p in ps ]
	return combine(ctrs,RNG=RNG,size=size,**kwargs).pvalue

# Cannot compare with Pearson’s and Tippett’s method due to SciPy Issue #15373
@mark.parametrize( "method", ["fisher","mudholkar_george","stouffer"] )
@mark.parametrize( "n", range(2,15) )
@mark.parametrize( "magnitude", ["small","normal"] )
def test_compare_with_combine_pvalues(n,method,magnitude):
	RNG = np.random.default_rng(n)
	ps = 10**RNG.uniform( -3 if magnitude=="small" else -1, 0, n )
	
	assert_matching_p_values(
		emulate_continuous_combine_ps(ps,RNG=RNG,method=method),
		combine_pvalues(ps,method=method)[1],
		size,
	)

@mark.parametrize( "n", range(2,15) )
@mark.parametrize( "magnitude", ["small","normal"] )
def test_compare_with_combine_pvalues_weighted(n,magnitude):
	RNG = np.random.default_rng(n)
	ps = 10**RNG.uniform( -3 if magnitude=="small" else -1, 0, n )
	weights = RNG.random(n)
	
	assert_matching_p_values(
		emulate_continuous_combine_ps(ps,RNG=RNG,method="stouffer",weights=weights),
		combine_pvalues(ps,method="stouffer",weights=weights)[1],
		size,
	)

@mark.parametrize( "method,variant", combining_statistics )
@mark.parametrize("variables", ["one", "all"])
def test_monotony(method,variant,variables):
	# Test that result increases monotonously with respect to input.
	n,k = 5,7
	changing_values = np.linspace(0.1,0.9,n)
	weights = np.random.random(k)
	pvalues = np.random.random(k)
	combined_ps = []
	for changing_value in changing_values:
		RNG = np.random.default_rng(42)
		if variables == "one":
			pvalues[0] = changing_value
		else:
			pvalues = np.full(k,changing_value)
		combined_p = emulate_continuous_combine_ps(
					pvalues,
					method = method,
					weights = weights if variant=="weighted" else None,
					RNG = RNG
				)
		combined_ps.append(combined_p)
	
	assert np.all( np.diff(combined_ps) >= 0 )

# CDF of the standard normal distribution and its inverse for Stouffer’s method.
phi = lambda z: (1+erf(z/sqrt(2)))/2
phiinv = lambda x: sqrt(2)*erfinv(2*x-1)

@mark.parametrize(
			"  method  ,   solution  ",
		[
			("tippett" , 1-(1-0.4)**3),
			("simes"   , 0.9         ),
			("stouffer", phi((phiinv(0.4)+phiinv(0.7)+phiinv(0.9))/sqrt(3)) ),
		]
	)
def test_simple_case(method,solution):
	assert_matching_p_values(
		emulate_continuous_combine_ps( [0.9,0.7,0.4], method=method ),
		solution,
		size
	)

def test_simple_weighted_case():
	assert_matching_p_values(
		emulate_continuous_combine_ps( [0.9,0.7,0.4], weights=[1,2,3], method="stouffer" ),
		phi( (phiinv(0.9)+2*phiinv(0.7)+3*phiinv(0.4)) / sqrt(1**2+2**2+3**2) ),
		size,
	)

@mark.parametrize("method",(
		method
		for method,variant in combining_statistics
		if variant=="weighted"
	))
def test_identical_weights(method):
	RNG = np.random.default_rng(abs(hash(method)))
	n = 10
	ps = RNG.random(n)
	weights = np.full(n,RNG.exponential())
	
	results = [
		emulate_continuous_combine_ps(ps,RNG=RNG,method=method,weights=w)
		for w in [weights,None]
	]
	assert_matching_p_values(*results,n=size,factor=4,compare=True)


