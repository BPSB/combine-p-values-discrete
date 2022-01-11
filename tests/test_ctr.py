from pytest import mark, raises
from itertools import chain, product, combinations_with_replacement
import numpy as np

from scipy.stats import combine_pvalues

from combine_pvalues_discrete.logpdist import LogPDist
from combine_pvalues_discrete.ctr import CTR
from combine_pvalues_discrete.tools import tree_prod, sign_test

examples = [
	CTR( 0.5, LogPDist.uniform_from_ps([0.5,    1]) ),
	CTR( 1.0, LogPDist.uniform_from_ps([0.5,    1]) ),
	CTR( 0.3, LogPDist.uniform_from_ps([0.3,0.5,1]) ),
	CTR( 0.7, LogPDist.uniform_from_ps([0.2,0.7,1]) ),
]

@mark.parametrize(
		"combo",
		chain(*( combinations_with_replacement(examples,r) for r in range(1,3) ))
	)
def test_commutativity_and_associativity(combo):
	combo = list(combo)
	x = tree_prod(combo)
	np.random.shuffle(combo)
	y = tree_prod(combo)
	assert x == y
	assert x.combined_p == y.combined_p

@mark.parametrize("example",examples)
def test_from_discrete_test(example):
	ps = 10**(example.nulldist.logps[example.nulldist.probs>0])
	assert example == CTR.from_discrete_test(10**example.logp,ps)


# Reproducing a sign test by combining single comparisons:

def my_sign_test_onesided(X,Y):
	return tree_prod(
		CTR.from_discrete_test( 0.5 if x<y else 1, [0.5,1.0] )
		for x,y in zip(X,Y)
	).combined_p

sign_tests = [ lambda X,Y:sign_test(X,Y)[0], my_sign_test_onesided ]

@mark.parametrize( "n,replicate", product( range(2,15), range(20) ) )
def test_comparison_to_sign_test(n,replicate):
	n = 13
	X = np.random.random(n)
	Y = np.random.random(n)
	
	p_values = [ test(X,Y) for test in sign_tests ]
	
	np.testing.assert_almost_equal( *p_values )

# Reproducing `combine_pvalues` for continuous tests and comparing:

def emulate_continuous_combine_ps(ps):
    return tree_prod(
        CTR.from_continuous_test(p,density=10000)
        for p in ps
    ).combined_p

@mark.parametrize( "n", range(2,15) )
def test_compare_with_combine_pvalues(n):
	ps = 10**np.random.uniform(-3,0,n)
	
	np.testing.assert_allclose(
		combine_pvalues(ps)[1],
		emulate_continuous_combine_ps(ps),
		rtol=1e-3, atol=1e-4
	)

