#!/usr/bin/python3

from setuptools import setup
from io import open

setup(
		name = "combine_pvalues_discrete",
		description = "A Python toolbox for combining p values from tests with a discrete statistics",
		long_description = open("README.rst", encoding="utf8").read(),
		python_requires = ">=3.7",
		packages = ["combine_pvalues_discrete"],
		install_requires = ["numpy","scipy>1.7.0"],
		setup_requires = ["setuptools_scm","pytest-runner","pytest-rng"],
		tests_require = ["pytest","statsmodels"],
		use_scm_version = {"write_to": "combine_pvalues_discrete/version.py"},
		classifiers = [
				"License :: OSI Approved :: BSD License",
				"Operating System :: POSIX",
				"Operating System :: MacOS :: MacOS X",
				"Operating System :: Microsoft :: Windows",
				"Programming Language :: Python",
				"Topic :: Scientific/Engineering",
			],
	)

