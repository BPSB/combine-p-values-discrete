from pytest import mark
import math
import numpy as np

from combine_pvalues_discrete.tools import is_unity, searchsorted_closest, tree_prod

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
	np.testing.assert_almost_equal( tree_prod(factors), math.prod(factors) )

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




