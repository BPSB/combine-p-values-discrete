from pytest import mark, raises
import numpy as np
from scipy.stats import uniform, ks_1samp
from itertools import combinations

from combine_pvalues_discrete.pdist import PDist
from combine_pvalues_discrete.tools import assert_matching_p_values

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

@mark.parametrize("n_ps",2**np.arange(0,10))
@mark.parametrize("n_samples",10**np.arange(4,7))
@mark.parametrize("method",("stochastic","proportional"))
def test_sampling(n_ps,n_samples,method,rng):
	if n_ps>1:
		dist = PDist( list(rng.random(n_ps-1)) + [1] )
		sample = dist.sample(RNG=rng,size=n_samples,method=method)
		for p,prob in zip(dist,dist.probs):
			assert np.isclose( np.average(sample==p), prob, atol=3/np.sqrt(n_samples) )
			assert_matching_p_values( np.average(sample<=p), p, n_samples )
	else:
		dist = PDist([])
		sample = dist.sample(RNG=rng,size=n_samples,method=method)
		assert ks_1samp(sample,uniform.cdf).pvalue > 0.05

@mark.parametrize("n_ps",2**np.arange(0,10))
@mark.parametrize("n_samples",10**np.arange(4,7))
@mark.parametrize("method",("stochastic","proportional"))
def test_sampling_complement(n_ps,n_samples,method,rng):
	if n_ps>1:
		dist = PDist( list(rng.random(n_ps-1)) + [1] )
		sample = dist.sample_complement(RNG=rng,size=n_samples,method=method)
		for p,prob in zip(dist,dist.probs):
			q = dist.complement(p)
			assert np.isclose( np.average(sample==q), prob, atol=3/np.sqrt(n_samples) )
			assert_matching_p_values( np.average(sample<=q), q, n_samples, factor=4 )
	else:
		dist = PDist([])
		sample = dist.sample_complement(RNG=rng,size=n_samples,method=method)
		assert ks_1samp(sample,uniform.cdf).pvalue > 0.01

@mark.parametrize("n_ps",2**np.arange(0,10))
@mark.parametrize("method",("stochastic","proportional"))
def test_sampling_both(n_ps,method,rng):
	if n_ps>1:
		dist = PDist( list(rng.random(n_ps-1)) + [1] )
	else:
		dist = PDist([])
	
	sampling_methods = {
			"p": dist.sample,
			"q": dist.sample_complement,
			"b": dist.sample_both,
		}
	samples = {}
	for key,sampling_method in sampling_methods.items():
		RNG = np.random.default_rng(42)
		samples[key] = sampling_method(RNG=RNG,size=10000,method=method)
	assert np.all( samples["p"] == samples["b"][0] )
	assert np.all( samples["q"] == samples["b"][1] )

def test_complement():
	dist = PDist([0.1,0.5,1])
	assert dist.complement(0.1)==1
	assert dist.complement(0.5)==0.9
	assert dist.complement(1)==0.5

