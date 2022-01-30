from pytest import mark, raises
from itertools import count
import numpy as np
from scipy.stats import mannwhitneyu
from math import prod

from combine_pvalues_discrete.ctr import CTR, combine, combining_statistics
from combine_pvalues_discrete.tools import sign_test, assert_matching_p_values, assert_discrete_uniform

size = 100000

# All tests occur with the alternative "less".

def test_simple_mwu():
	assert (
		CTR.mann_whitney_u([0],[1],alternative="less")
		==
		CTR( 0.5, [0.5,1.0] )
	)

def test_simple_signtest():
	assert (
		CTR.sign_test([0],[1],alternative="less")
		==
		CTR( 0.5, [0.5,1.0] )
	)

def mwu_combine( data, **kwargs ):
	ctrs = [ CTR.mann_whitney_u(X,Y,alternative="less") for X,Y in data ]
	return combine(ctrs,size=size,**kwargs).pvalue

def mwu_data(RNG,n,trend=0):
	"""
		Creates a dataset of `n` unequal pairs of normally distributed numbers with a trend towards the first half of a pair containing smaller values.
		If `trend` is zero, this conforms with the null hypothesis.
	"""
	return [(
			RNG.normal(size=RNG.randint(2,6))-trend,
			RNG.normal(size=RNG.randint(2,6))
		) for _ in range(n) ]

def mwu_logp_sum(data):
	return sum(
		np.log10(mannwhitneyu(X,Y,alternative="less",method="exact").pvalue)
		for X,Y in data
	)

def signtest_combine( data, **kwargs ):
	ctrs = [ CTR.sign_test(X,Y,alternative="less") for X,Y in data ]
	return combine(ctrs,size=size,**kwargs).pvalue

def signtest_data(RNG,n,trend=0):
	"""
		Creates a dataset of `n` pairs of normally distributed numbers with a trend towards the first half of a pair containing smaller values.
		If `trend` is zero, this conforms with the null hypothesis.
	"""
	return [(
			RNG.normal(size=size)-trend,
			RNG.normal(size=size)
		) for size in RNG.randint(15,21,size=n) ]

def signtest_logp_sum(data):
	return sum(
		np.log10(sign_test(X,Y,alternative="less")[0])
		for X,Y in data
	)

tests = {
		"signtest": ( signtest_combine, signtest_data, signtest_logp_sum ),
		"mwu_test": (      mwu_combine,      mwu_data,      mwu_logp_sum ),
	}

@mark.slow
@mark.parametrize("method,variant",combining_statistics)
@mark.parametrize("test",tests)
def test_null_distribution(method,variant,test,rng):
	test_and_combine,create_data,_ = tests[test]
	n = 20
	p_values = [
		test_and_combine(
			create_data(rng,n),
			RNG = rng,
			method = method,
			weights = rng.random(n) if variant=="weighted" else None
		)
		for _ in range(30)
	]
	
	assert_discrete_uniform(p_values)

def create_surrogate(RNG,pairs):
	"""
	Creates a single artificial dataset complying with the null hypothesis (surrogate).
	This dataset has the same shape as `pairs`.
	"""
	return [
		[ RNG.normal(size=len(member)) for member in pair ]
		for pair in pairs
	]

@mark.parametrize("trend",np.linspace(-0.7,0.7,10))
@mark.parametrize("test",tests)
def test_compare_with_surrogates(trend,test,rng):
	test_and_combine,create_data,logp_sum = tests[test]
	dataset = create_data(rng,10,trend=trend)
	
	p_from_combine = test_and_combine(dataset,RNG=rng)
	
	n = 1000
	
	original_logp_sum = logp_sum(dataset)
	surrogate_logp_sums = [
		logp_sum( create_surrogate(rng,dataset) )
		for _ in range(n)
	]
	p_from_surrogates = np.average( original_logp_sum >= surrogate_logp_sums )
	
	assert_matching_p_values(
			p_from_surrogates,
			p_from_combine,
			n = min(size,n),
			factor=3, compare=True,
		)

