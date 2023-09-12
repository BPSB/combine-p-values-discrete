from pytest import mark
import numpy as np

from combine_pvalues_discrete.ctr import CTR, direction, combining_statistics

@mark.parametrize(
			"  ps  , method, result",
		[
			([0.1,0.1,0.9], "fisher" , "less"   ),
			([0.1,0.9,0.9], "fisher" , "greater"),
			([0.1,0.5,0.9], "fisher" , "equal"  ),
			([0.1,0.99],    "fisher" , "greater"),
			([0.1,0.99],    "pearson", "greater"),
		]
	)
def test_simple(ps,method,result):
	ctrs = [ CTR(p) for p in ps ]
	assert direction(ctrs,method=method) == result

@mark.parametrize("method,variant",combining_statistics)
def test_big(method,variant,rng):
	n = 100
	ctrs = [ CTR.sign_test(rng.normal(-1,1,10)) for _ in range(n) ]
	weights = rng.random(n) if variant=="weighted" else None
	assert direction(ctrs,weights,method) == "less"

