from pytest import mark, raises
from itertools import chain, product, combinations_with_replacement
import numpy as np
from math import prod

from scipy.stats import combine_pvalues

from combine_pvalues_discrete.ctr import CTR
from combine_pvalues_discrete.pdist import PDist
from combine_pvalues_discrete.tools import sign_test, assert_matching_p_values

size = 100000

examples = [
	CTR.from_test( 0.5, [0.5,      1] ),
	CTR.from_test( 1.0, [0.5,      1] ),
	CTR.from_test( 0.3, [0.3, 0.5, 1] ),
	CTR.from_test( 0.7, [0.2, 0.7, 1] ),
]

@mark.parametrize(
		"seed, combo",
		enumerate(chain(*(
			combinations_with_replacement(examples,r)
			for r in range(1,3)
		)))
	)
def test_commutativity_and_associativity(seed,combo):
	RNG = np.random.default_rng(seed)
	combo = list(combo)
	x = prod(combo)
	RNG.shuffle(combo)
	y = prod(combo)
	assert x == y
	assert_matching_p_values(
			x.get_result(size=size,RNG=RNG).pvalue,
			y.get_result(size=size,RNG=RNG).pvalue,
			size, factor=3
		)


# Reproducing a sign test by combining single comparisons:

@mark.parametrize( "n,replicate", product( range(2,15), range(20) ) )
def test_comparison_to_sign_test(n,replicate):
	RNG = np.random.default_rng((n+20)**3*replicate)
	
	def my_sign_test_onesided(X,Y):
		return prod(
			CTR.from_test( 0.5 if x<y else 1, [0.5,1.0] )
			for x,y in zip(X,Y)
		).get_result(size=size,RNG=RNG).pvalue
	
	X = RNG.random(n)
	Y = RNG.random(n)
	
	assert_matching_p_values(
		my_sign_test_onesided(X,Y),
		sign_test(X,Y)[0],
		size,
	)

# Reproducing `combine_pvalues` for continuous tests and comparing:

def emulate_continuous_combine_ps(ps,RNG):
	ctr = prod( CTR.from_test(p,[]) for p in ps )
	return ctr.get_result(RNG=RNG,size=size).pvalue

@mark.parametrize( "n", range(2,15) )
def test_compare_with_combine_pvalues(n):
	RNG = np.random.default_rng(n)
	ps = 10**RNG.uniform(-3,0,n)
	
	assert_matching_p_values(
		emulate_continuous_combine_ps(ps,RNG),
		combine_pvalues(ps)[1],
		size,
	)

