from pytest import mark, raises
import numpy as np

from combine_pvalues_discrete.logpdist import LogPDist

def test_dimension_mismatch():
	with raises(ValueError):
		LogPDist([1],[0,1])

def test_no_end_at_1():
	with raises(ValueError):
		LogPDist([-2,-1],[0.5,0.5])

def test_no_normalised_probs():
	with raises(ValueError):
		LogPDist([-2, 0],[0.5,0.7])

def test_no_equidistance():
	with raises(ValueError):
		LogPDist( [-3,-1,0], [0.1,0.2,0.7] )
	
a = LogPDist( [-2,0], [0.1,0.9] )
b = LogPDist( [-1,0], [0.2,0.8] )
c = LogPDist( [-3,-2,-1,0], [0.02,0.08,0.18,0.72] )

@mark.parametrize("density",np.linspace(10,300,20))
def test_convolution(density):
	A = a._rescaled(np.random.randint(10,300))
	B = b._rescaled(np.random.randint(10,300))
	C = c._rescaled(np.random.randint(10,300))
	assert A*B == B*A == C

def test_uniform_from_ps():
	assert LogPDist( [-2,-1,0], [0.01,0.09,0.9] ) == LogPDist.uniform_from_ps([0.01,0.1,1])

def test_uniform_continuous():
	min_p = 1e-4
	density = 100
	uniform = LogPDist.uniform_continuous(min_p=min_p,density=density)
	
	np.testing.assert_almost_equal( 10**uniform[0][0], min_p )
	np.testing.assert_almost_equal( len(uniform), -np.log10(min_p)*density )
	assert uniform._index_of_logps(0) == len(uniform)-1
	
	for logp, cumprob in zip(uniform.logps,uniform.cumprobs):
		np.testing.assert_almost_equal( 10**logp, cumprob )
		np.testing.assert_almost_equal( uniform.cdf(logp), cumprob )
