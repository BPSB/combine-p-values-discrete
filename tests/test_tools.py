import math
from collections import namedtuple
from pytest import mark
import numpy as np
from statsmodels.stats.descriptivestats import sign_test as sm_sign_test

from combine_pvalues_discrete.tools import (
		is_unity,
		searchsorted_closest,
		sign_test,
		counted_p, std_from_true_p, assert_matching_p_values,
	)


@mark.parametrize(
		"       thing    , result",
		[
			( 1          , True  ),
			( 1.0        , True  ),
			( np.array(1), True  ),
			( True       , True  ),
			( 3          , False ),
			( 1.1        , False ),
			( np.array(4), False ),
			( False      , False ),
			( [1]        , False ),
			( "1"        , False ),
			( None       , False ),
			( is_unity   , False ),
		]
	)
def test_is_unity(thing,result):
	assert is_unity(thing)==result

def test_searchsorted_closest():
	np.testing.assert_array_equal(
		searchsorted_closest( [ 1, 2.5, 3, 7 ], [ 1, 2, 3.1, 9, 0 ] ),
		                                        [ 0, 1,  2 , 3, 0 ]
	)

def test_searchsorted_closest_single_input():
	assert searchsorted_closest([1,2,3],2.1) == 1

@mark.parametrize("n",range(1,20))
def test_signtest_with_statsmodels(n):
	X = np.random.normal(size=n)
	
	p = sign_test(X,alternative="two-sided")[0]
	p_sm = sm_sign_test(X)[1]
	
	Y = np.random.normal(n)
	p_2s = sign_test(X+Y,Y,alternative="two-sided")[0]
	
	assert np.isclose(p,p_sm)
	assert p == p_2s

@mark.parametrize("n",range(1,20))
def test_signtest_with_statsmodels_onesided(n):
	X = np.random.normal(size=n)
	
	# Because there is no readily available one-sided test:
	if np.mean(X>0)>0.5:
		X = -X
	elif np.mean(X>0)==0.5:
		return

	p = sign_test(X,alternative="less")[0]
	p_sm = sm_sign_test(X)[1]/2

	Y = np.random.normal(n)
	p_2s = sign_test(X+Y,Y,alternative="less")[0]

	assert np.isclose(p,p_sm)
	assert p == p_2s

def test_signtest_order():
	n = 20
	X = np.zeros(n)
	Y = np.ones(n)
	assert np.isclose( sign_test(X,Y,alternative="less")[0], 2**-n )

def test_counted_p():
	null_stats = [1,2,3,4,5,6,7,8,9]
	assert counted_p(0.5,null_stats).pvalue == 0.1
	assert counted_p(3.5,null_stats).pvalue == 0.4
	assert counted_p(10 ,null_stats).pvalue == 1.0

def test_std_counted_p():
	RNG = np.random.default_rng(42)
	
	n = 1000  # number of points per dataset
	m = 10000 # number of datasets
	k = 30    # number of different p values tested
	nulls = RNG.uniform(0,1,size=(n,m))
	true_ps = np.logspace(-2,0,k)
	estimated_ps,estimated_stds = counted_p( true_ps[None,None,:], nulls[:,:,None] )
	assert estimated_ps.shape == (m,k)
	stds = std_from_true_p(true_ps,n)
	assert stds.shape == (k,)
	control = np.std(estimated_ps,axis=0)
	
	deviations = true_ps - np.mean(estimated_ps,axis=0)
	# Corrections because the p value is estimated conservatively and, e.g., can never be below 1/(n+1):
	size_offset = (1-true_ps)/(n+1)
	assert np.all( np.abs(deviations+size_offset) <= 3*stds/np.sqrt(m) )
	
	assert_matching_p_values(
			np.mean(estimated_ps,axis=0),
			true_ps,
			n=n,
			factor = 3/np.sqrt(m)
		)
	np.testing.assert_allclose( stds, control, rtol=3/np.sqrt(m) )
	np.testing.assert_allclose( stds[:-1], np.mean(estimated_stds,axis=0)[:-1], rtol=3/np.sqrt(m) )
	# Last element is expected to be unequal, because the estimate cannot reasonably be zero.


