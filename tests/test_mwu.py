from pytest import mark, raises
from itertools import count
import numpy as np
from scipy.stats import mannwhitneyu, uniform, ks_1samp
from math import prod

from combine_pvalues_discrete.ctr import CTR
from combine_pvalues_discrete.tools import assert_matching_p_values

size = 100000

# All MWU tests occur with the alternative "less".

def test_simplest_case():
	assert (
		CTR.from_mann_whitney_u([0],[1],alternative="less")
		==
		CTR.from_test( 0.5, [0.5,1.0] )
	)

def combine_mwus( pairs, RNG, **kwargs ):
	return prod(
			CTR.from_mann_whitney_u(X,Y,**kwargs)
			for X,Y in pairs
		).combined_p(RNG=RNG,size=size)

def create_data(RNG,n,max_size=10,trend=0):
	"""
		Creates a dataset of unequal pairs (up to `max_size`) of normally distributed numbers with a trend towards the first half of a pair containing smaller values.
		If `trend` is zero, this conforms with the null hypothesis.
	"""
	return [(
			RNG.normal(size=RNG.integers(2,max_size,endpoint=True))-trend,
			RNG.normal(size=RNG.integers(2,max_size,endpoint=True))
		) for _ in range(n) ]

def test_null_distribution():
	RNG = np.random.default_rng(23)
	
	p_values = [
		combine_mwus( create_data(RNG,10), alternative="less", RNG=RNG )
		for _ in range(300)
	]
	
	assert ks_1samp(p_values,uniform.cdf).pvalue > 0.05

def create_surrogate(RNG,pairs):
	"""
	Creates a single artificial dataset complying with the null hypothesis (surrogate).
	This dataset has the same shape as `pairs`.
	"""
	return [
		[ RNG.normal(size=len(member)) for member in pair ]
		for pair in pairs
	]

def mwu_logp_sum(pairs):
	return sum(
		np.log10(mannwhitneyu(X,Y,alternative="less",method="exact").pvalue)
		for X,Y in pairs
	)

@mark.parametrize("trend",np.linspace(-0.7,0.7,10))
def test_compare_with_surrogates(trend):
	RNG = np.random.default_rng(hash(np.exp(trend)))
	dataset = create_data(RNG,10,max_size=5,trend=trend)
	
	p_from_combine = combine_mwus(dataset,alternative="less",RNG=RNG)
	
	n = 1000
	
	original_logp_sum = mwu_logp_sum(dataset)
	surrogate_logp_sums = [
		mwu_logp_sum( create_surrogate(RNG,dataset) )
		for _ in range(n)
	]
	p_from_surrogates = np.average( original_logp_sum >= surrogate_logp_sums )
	
	assert_matching_p_values(
			p_from_surrogates,
			p_from_combine,
			n = min(size,n),
			factor=3, compare=True
		)

