from pytest import mark
from math import prod
from itertools import count
import numpy as np
from scipy.stats import uniform, ks_1samp

from combine_pvalues_discrete.ctr import CTR
from combine_pvalues_discrete.tools import sign_test, assert_matching_p_values

size = 100000

# All sign tests occur with the alternative "less".

def test_simplest_case():
	assert (
		CTR.from_sign_test([0],[1],alternative="less")
		==
		CTR.from_test( 0.5, [0.5,1.0] )
	)

def combine_sign_tests( pairs, RNG, **kwargs ):
	return prod(
			CTR.from_sign_test(X,Y,**kwargs)
			for X,Y in pairs
		).combined_p(size=size,RNG=RNG)

def create_data(RNG,n,max_size=10,trend=0):
	"""
		Creates a dataset of pairs (up to `max_size`) of normally distributed numbers with a trend towards the first half of a pair containing smaller values.
		If `trend` is zero, this conforms with the null hypothesis.
	"""
	return [(
			RNG.normal(size=size)-trend,
			RNG.normal(size=size)
		) for size in RNG.integers(2,max_size,endpoint=True,size=n) ]

def test_null_distribution():
	RNG = np.random.default_rng(42)
	
	p_values = [
		combine_sign_tests(create_data(RNG,10),alternative="less",RNG=RNG)
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

def sign_test_logp_sum(pairs):
	return sum(
		np.log10(sign_test(X,Y,alternative="less")[0])
		for X,Y in pairs
	)

@mark.parametrize("trend",np.linspace(-0.7,0.7,11))
def test_compare_with_surrogates(trend):
	RNG = np.random.default_rng(hash(np.exp(trend)))
	dataset = create_data(RNG,10,max_size=5,trend=trend)
	
	p_from_combine = combine_sign_tests(dataset,alternative="less",RNG=RNG)
	
	n = 1000
	
	original_logp_sum = sign_test_logp_sum(dataset)
	surrogate_logp_sums = [
		sign_test_logp_sum( create_surrogate(RNG,dataset) )
		for _ in range(n)
	]
	p_from_surrogates = np.average( original_logp_sum > surrogate_logp_sums )
	
	assert_matching_p_values(
			p_from_surrogates,
			p_from_combine,
			n = min(size,n),
			factor=3, compare=True
		)


