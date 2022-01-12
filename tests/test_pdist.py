from pytest import mark, raises
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import uniform, ks_1samp
from itertools import combinations

from combine_pvalues_discrete.pdist import PDist

def test_core_stuff():
	dists = ( PDist([]), PDist([1]), PDist([0.5,1]) )
	for i,dist in enumerate(dists):
		assert dist.continuous or i
		assert dist == dist
	for dist1,dist2 in combinations(dists,2):
		assert dist1 != dist2

def test_no_end_at_1():
	with raises(ValueError):
		PDist([0.1,0.5])

def test_negative_p():
	with raises(ValueError):
		PDist([-0.1,0.5])

def test_zero_p():
	with raises(ValueError):
		PDist([0.1,0.5,0])

def test_p_more_than_one():
	with raises(ValueError):
		PDist([-0.1,0.5,2])

def test_correct_almost_1():
	dist = PDist([0.1,1-1e-14])
	assert dist.ps[-1] == 1


@mark.parametrize("size",range(1,100))
def test_normalisation(size):
	dist = PDist( list(np.random.random(size-1)) + [1] )
	assert np.sum(dist.probs) == 1

@mark.parametrize("size",range(1,100))
def test_cumprobs(size):
	dist = PDist( list(np.random.random(size-1)) + [1] )
	np.testing.assert_almost_equal( dist.ps, np.cumsum(dist.probs) )

@mark.parametrize("size",2**np.arange(0,10))
@mark.parametrize("n",10**np.arange(4,7))
def test_sampling(size,n):
	RNG = np.random.default_rng(size*n)
	if size:
		dist = PDist( list(RNG.random(size-1)) + [1] )
		sample = dist.sample(RNG=RNG,size=n)
		for p,prob in dist:
			assert_allclose( np.average(sample==p), prob, atol=3/np.sqrt(n) )
			assert_allclose( np.average(sample<=p),   p , atol=3/np.sqrt(n) )
	else:
		dist = PDist([])
		sample = dist.sample(RNG=RNG,size=n)
		assert ks_1samp(sample,uniform.cdf).pvalue > 0.05
