from pytest import mark, raises
from itertools import count, product
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr, kendalltau, fisher_exact, boschloo_exact
import math

from combine_pvalues_discrete.ctr import CTR, combine, combining_statistics
from combine_pvalues_discrete.tools import sign_test, assert_matching_p_values, assert_discrete_uniform

n_samples = 100000

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

@mark.parametrize("n",range(3,10))
def test_simple_wilcoxon(n,rng):
	data = np.arange(1,n+1)*rng.choice([-1,1],n)
	
	wilcoxon_kwargs = dict(alternative="less",mode="exact")
	control_p = wilcoxon(data,None,**wilcoxon_kwargs).pvalue
	control_all_ps = list({
			wilcoxon( np.arange(1,n+1)*signs, **wilcoxon_kwargs ).pvalue
			for signs in product([-1,1],repeat=n)
		})
	
	assert (
		CTR.wilcoxon_signed_rank(data,alternative="less")
		==
		CTR( control_p, control_all_ps )
	)

@mark.parametrize(
		"  x    ,    y   ,    alt   ,   p  ,      all_ps         ",
	[
		([1,3,2], [4,5,0], "less"   ,  5/6 , [ 1/6, 1/2, 5/6, 1 ]),
		([1,3,2], [4,5,0], "greater",  1/2 , [ 1/6, 1/2, 5/6, 1 ]),
		([1,2,2], [3,3,5], "less"   ,   1  , [ 1/3,           1 ]),
		([1,2,2], [3,3,5], "greater",  2/3 , [ 2/3,           1 ]),
	])
def test_simple_spearman(x,y,alt,p,all_ps):
	assert CTR.spearmanr( x, y, alternative=alt ).approx( CTR(p,all_ps) )

@mark.parametrize("alt",["less","greater"])
def test_spearman_large_dataset(rng,alt):
	x,y = rng.normal(size=(2,100))
	ctr = CTR.spearmanr(x,y,alternative=alt)
	assert ctr.p == spearmanr(x,y,alternative=alt).pvalue
	assert ctr.nulldist.continuous

def test_spearman_large_perfect_dataset():
	n = 100
	ctr = CTR.spearmanr(range(n),range(n),alternative="greater")
	assert ctr.p >= 1/math.factorial(n)

@mark.parametrize("n",range(2,9))
def test_spearman_nulldist_length(n):
	assert (
		len( CTR.spearmanr(range(n),range(n)).nulldist.ps )
		==
		math.comb(n+1,3) + (n!=3) # OEIS A126972
	)

def spearman_data(RNG,n,trend=0):
	x,y = RNG.normal(size=(2,n))
	y = (1-trend)*y + trend*x
	return x,y

@mark.parametrize("alt",["less","greater"])
@mark.parametrize("n",range(3,7))
def test_spearman_null(n,alt,rng):
	m = 1000 if n<5 else 100
	p_values = [
			CTR.spearmanr(
				*spearman_data(RNG=rng,n=n),
				alternative = alt
			).p
			for _ in range(m)
		]
	
	assert_discrete_uniform(p_values)

@mark.parametrize("n",range(3,9))
def test_spearman(n,rng):
	m = 1000
	
	x,y = spearman_data(RNG=rng,n=n,trend=0.8)
	orig_ρ = spearmanr(x,y).correlation

	null_ρs = np.array([
			spearmanr(*spearman_data(RNG=rng,n=n)).correlation
			for _ in range(m)
		])
	
	assert_matching_p_values(
			np.average( orig_ρ <= null_ρs ),
			CTR.spearmanr(x,y,alternative="greater").p,
			n = m,
		)

@mark.parametrize(
		"    x    ,     y    ,    alt   ,    p   ,                all_ps                 ",
	[
		([1,3,2,4], [4,5,0,6], "less"   ,  23/24 , [ 1/24, 1/6, 3/8, 5/8, 5/6, 23/24, 1 ]),
		([1,3,2,4], [4,5,0,6], "greater",   1/6  , [ 1/24, 1/6, 3/8, 5/8, 5/6, 23/24, 1 ]),
	])
def test_simple_kendall(x,y,alt,p,all_ps):
	result = CTR.kendalltau( x, y, alternative=alt )
	control_p = kendalltau(x,y,alternative=alt).pvalue
	assert result.approx( CTR(p,all_ps) )
	assert np.isclose( result.p, control_p )

@mark.parametrize(
		"      C      ,   alt    ,    p    ,            all_ps            ",
	[
		([[2,3],[4,0]], "less"   ,   5/42  , [  5/42 , 25/42 ,  20/21 , 1 ]),
		([[2,3],[4,0]], "greater",    1    , [  1/21 , 17/42 ,  37/42 , 1 ]),
		([[1,7],[2,7]], "less"   ,  93/170 , [ 21/170, 93/170,  78/85 , 1 ]),
		([[1,7],[2,7]], "greater", 149/170 , [  7/85 , 77/170, 149/170, 1 ]),
	])
def test_simple_fisher_exact(C,alt,p,all_ps):
	result = CTR.fisher_exact( C, alternative=alt )
	control_p = fisher_exact(C,alternative=alt)[1]
	assert result.approx( CTR(p,all_ps) )
	assert np.isclose( result.p, control_p )

def test_simple_boschloo():
	C = [[1,2],[3,0]]
	result = CTR.boschloo_exact( C, alternative="less" )
	control_p = boschloo_exact(C,alternative="less" ).pvalue
	assert np.isclose( result.p, control_p )
	all_ps = [ 1/64, 0.079305, 0.273032, 11/32, 0.57860, 49/64, 1 ]
	assert result.approx( CTR(0.079305,all_ps), tol=1e-5 )

# -----------------

def mwu_combine( data, **kwargs ):
	ctrs = [ CTR.mann_whitney_u(X,Y,alternative="less") for X,Y in data ]
	return combine(ctrs,**kwargs).pvalue

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

def mwu_invert(data):
	return [ (-X,-Y) for X,Y in data ]

def signtest_combine( data, **kwargs ):
	ctrs = [ CTR.sign_test(X,Y,alternative="less") for X,Y in data ]
	return combine(ctrs,**kwargs).pvalue

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

def signtest_invert(data):
	return [ (Y,X) for X,Y in data ]

tests = {
		"signtest": ( signtest_combine, signtest_data, signtest_logp_sum, signtest_invert ),
		"mwu_test": (      mwu_combine,      mwu_data,      mwu_logp_sum,      mwu_invert ),
	}

@mark.slow
@mark.parametrize("method,variant",combining_statistics)
@mark.parametrize("sampling_method",["proportional","stochastic"])
@mark.parametrize("test",tests)
@mark.parametrize("alt",["less","greater","two-sided"])
def test_null_distribution(method,variant,test,sampling_method,alt,rng):
	test_and_combine,create_data,*_ = tests[test]
	n = 10
	p_values = [
		test_and_combine(
			create_data(rng,n),
			alternative = alt,
			RNG = rng,
			method = method,
			n_samples = 10000,
			sampling_method = sampling_method,
			weights = rng.random(n) if variant=="weighted" else None
		)
		for _ in range(30)
	]
	
	assert_discrete_uniform( p_values, factor=4.1 )

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
@mark.parametrize("sampling_method",["proportional","stochastic"])
@mark.parametrize("test",tests)
@mark.parametrize("alt",["less","greater","two-sided"])
def test_compare_with_surrogates(trend,test,sampling_method,alt,rng):
	test_and_combine,create_data,logp_sum,invert = tests[test]
	dataset = create_data(rng,10,trend=trend)
	
	p_from_combine = test_and_combine(
			dataset,
			method = "fisher",
			n_samples = n_samples,
			alternative = alt,
			RNG = rng
		)
	
	n = 100
	
	evaluate = {
		"less":      lambda data: logp_sum(data),
		"greater":   lambda data: logp_sum(invert(data)),
		"two-sided": lambda data: min( logp_sum(data), logp_sum(invert(data)) ),
	}[alt]
	
	original_logp_sum = evaluate(dataset)
	surrogate_logp_sums = [
		evaluate( create_surrogate(rng,dataset) )
		for _ in range(n)
	]
	p_from_surrogates = np.average( original_logp_sum >= surrogate_logp_sums )
	
	assert_matching_p_values(
			p_from_surrogates,
			p_from_combine,
			n = min(n_samples,n),
			factor=3.3, compare=True,
		)

