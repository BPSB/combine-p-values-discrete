import math
from collections import namedtuple
from pytest import mark, raises
import numpy as np
from statsmodels.stats.descriptivestats import sign_test as sm_sign_test

from combine_pvalues_discrete.tools import (
		is_empty,
		searchsorted_closest,
		has_ties,
		sign_test,
		counted_p, std_from_true_p,
		assert_matching_p_values, assert_discrete_uniform,
	)

@mark.parametrize(
		" argument     , result",[
		( None         , True  ),
		( []           , True  ),
		( ()           , True  ),
		( np.array([]) , True  ),
		( [0]          , False ),
		( [1]          , False ),
		(  0           , False ),
		(  1           , False ),
		( np.array([0]), False ),
		( np.array([1]), False ),
	])
def test_is_empty(argument,result):
	assert is_empty(argument) == result

def test_searchsorted_closest():
	np.testing.assert_array_equal(
		searchsorted_closest( [ 1, 2.5, 3, 7 ], [ 1, 2, 3.1, 9, 0 ] ),
		                                        [ 0, 1,  2 , 3, 0 ]
	)

def test_searchsorted_closest_single_input():
	assert searchsorted_closest([1,2,3],2.1) == 1

@mark.parametrize(
		" argument     , result",[
		( [0,4,1,3,2], False ),
		( [0,1,3,4,1], True  ),
	])
def test_has_ties(argument,result):
	assert has_ties(argument) == result

@mark.parametrize("n",range(1,20))
def test_signtest_with_statsmodels(n):
	# sm: from statsmodels
	X = np.random.normal(size=n)
	
	p = sign_test(X,alternative="two-sided")[0]
	p_sm = sm_sign_test(X)[1]
	
	Y = np.random.normal(n)
	p_2sample = sign_test(X+Y,Y,alternative="two-sided")[0]
	
	assert np.isclose(p,p_sm)
	assert p == p_2sample

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
	p_2sample = sign_test(X+Y,Y,alternative="less")[0]

	assert np.isclose(p,p_sm)
	assert p == p_2sample

@mark.parametrize("n",range(1,20))
def test_signtest_best_case(n):
	X = np.zeros(n)
	Y = np.ones(n)
	assert np.isclose( sign_test(X,Y,alternative="less")[0], 2**-n )

def test_counted_p():
	null_stats = [1,2,3,4,5,6,7,8,9]
	assert counted_p(0.5,null_stats).pvalue == 0.1
	assert counted_p(3.5,null_stats).pvalue == 0.4
	assert counted_p(10 ,null_stats).pvalue == 1.0
	assert counted_p(0.7,null_stats,atol=0.2).pvalue == 0.1
	assert counted_p(0.9,null_stats,atol=0.2).pvalue == 0.2
	assert counted_p(4.1,null_stats,atol=0.2).pvalue == 0.5
	assert counted_p(4.9,null_stats,atol=0.2).pvalue == 0.6
	assert counted_p(8.6,null_stats,atol=0.2).pvalue == 0.9
	assert counted_p(8.9,null_stats,atol=0.2).pvalue == 1.0

def test_std_counted_p(rng):
	n = 1000  # number of points per dataset
	m = 10000 # number of datasets
	k = 30    # number of different p values tested
	nulls = rng.uniform(0,1,size=(n,m))
	true_ps = np.logspace(-2,0,k)
	
	estimated_ps,estimated_stds = counted_p(
			true_ps[None,None,:],
			nulls[:,:,None],
		)
	
	assert_matching_p_values(
			np.mean(estimated_ps,axis=0),
			true_ps,
			n=n,
			threshold=1e-3,
		)
	assert estimated_ps.shape == (m,k)
	empirical_stds = np.std(estimated_ps,axis=0)
	
	true_stds = std_from_true_p(true_ps,n)
	assert true_stds.shape == (k,)
	
	deviations = true_ps - np.mean(estimated_ps,axis=0)
	# Corrections because the p value is estimated conservatively and, e.g., can never be below 1/(n+1):
	size_offset = (1-true_ps)/(n+1)
	assert np.all( np.abs(deviations+size_offset) <= 3*true_stds/np.sqrt(m) )
	# (√m to get standard error)
	
	np.testing.assert_allclose( true_stds, empirical_stds, rtol=3/np.sqrt(m) )
	np.testing.assert_allclose( true_stds[:-1], np.mean(estimated_stds,axis=0)[:-1], rtol=3/np.sqrt(m) )
	# Last element is expected to be unequal, because the estimate cannot reasonably be zero.

def test_assert_matching_p_values_one_sample():
	assert_matching_p_values(0.04,0.05,n=30)
	assert_matching_p_values([0.04,0.051,0.048],0.05,n=30)
	assert_matching_p_values([0.04,0.061,0.02],[0.05,0.06,0.02],n=30)
	with raises(AssertionError):
		assert_matching_p_values([0.047,0.061,0.02],[0.05,0.01,0.02],n=300)

def test_assert_matching_p_values_two_sample():
	assert_matching_p_values(0.04,0.05,n=30,compare=True)
	assert_matching_p_values([0.04,0.051,0.048],0.05,n=30,compare=True)
	assert_matching_p_values([0.04,0.061,0.02],[0.05,0.06,0.02],n=(30,40),compare=True)
	with raises(AssertionError):
		assert_matching_p_values([0.047,0.061,0.02],[0.05,0.01,0.02],n=3000,compare=True)
	with raises(AssertionError):
		assert_matching_p_values(0.1,0.9,n=10,compare=True)



@mark.parametrize("size",range(10,100,10))
@mark.parametrize("n_values",range(2,10))
def test_assert_discrete_uniform(size,n_values,rng):
	moderate_value = rng.uniform(0.3,0.7)
	values = sorted( np.hstack(( rng.random(n_values-2), [moderate_value,1] )) )
	probs = np.diff(values,prepend=0)
	data = rng.choice( values, p=probs, size=size, replace=True )
	assert_discrete_uniform(data,threshold=1e-4)

def test_assert_discrete_uniform_perfect():
	data = [ 0.1, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7, 1.0, 1.0, 1.0 ]
	assert_discrete_uniform(data,threshold=0.5)

def test_assert_discrete_uniform_fail():
	data = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.7, 0.7 ]
	with raises(AssertionError):
		assert_discrete_uniform(data)

