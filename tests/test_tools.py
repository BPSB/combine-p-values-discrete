from pytest import mark
import math
import numpy as np
from statsmodels.stats.descriptivestats import sign_test as sm_sign_test

from combine_pvalues_discrete.tools import is_unity, searchsorted_closest, tree_prod, sign_test, std_counted_p


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

@mark.parametrize("n_factors",range(1,20))
def test_tree_prod(n_factors):
	factors = np.random.uniform( size=n_factors )
	assert np.isclose( tree_prod(factors), math.prod(factors) )

class MultCounter(object):
	def __init__(self,mults=0):
		self.mults = mults
	
	def __mul__(self,other):
		return MultCounter(max(self.mults,other.mults)+1)

@mark.parametrize("n_factors",range(1,20))
def test_tree_prod_mult_count(n_factors):
	factors = [ MultCounter() for _ in range(n_factors) ]
	product = tree_prod(factors)
	assert n_factors <= 2**product.mults < 2*n_factors

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

def test_std_counted_p():
	n = 1000
	m = 10000
	k = 30
	nulls = np.random.uniform(0,1,size=(n,m))
	true_ps = np.logspace(-2,0,k)
	std_ps = std_counted_p(true_ps,n)
	
	estimated_ps = np.mean(nulls[:,:,None]<true_ps[None,None,:],axis=0)
	assert estimated_ps.shape == (m,k)
	control = np.std(estimated_ps,axis=0)
	assert np.all( np.abs(np.mean(estimated_ps,axis=0)-true_ps) <= 3*std_ps/np.sqrt(m) )
	np.testing.assert_allclose( std_ps, control, rtol=3/np.sqrt(m) )

