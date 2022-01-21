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
@mark.parametrize("method",combining_statistics)
def test_commutativity_and_associativity(seed,combo,method):
	RNG = np.random.default_rng(seed)
	combo_1 = list(combo)
	combo_2 = combo_1.copy()
	RNG.shuffle(combo_2)
	
	results = [
			combine(combo,RNG=RNG,size=size,method=method).pvalue
			for combo in [combo_1,combo_2]
		]
	assert_matching_p_values(*results,size,factor=3)


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

def emulate_continuous_combine_ps(ps,method,RNG=np.random.default_rng()):
	ctrs = [ CTR(p) for p in ps ]
	return combine(ctrs,RNG=RNG,size=size,method=method).pvalue

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

@mark.parametrize( "method", combining_statistics )
def test_monotony(method):
	ps = np.linspace(0.1,0.9,5)
	combined_ps = [
			emulate_continuous_combine_ps( np.full(3,p), method=method )
			for p in ps
		]
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



