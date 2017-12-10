#!/usr/bin/env python

import os, sys

from setuptools import find_packages
import numpy

_packages = find_packages()

from os.path import join, dirname
# from setuptools import setup
from distutils.core import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

src_path = dirname(os.path.abspath(__file__))

ext_modules = cythonize([
    Extension(
        'DataBoxEngine',
        sources=['DataBoxEngine.pyx', ],
        libraries=['gdal', ],  #
        #         extra_link_args=[ 'libcrypto.a', 'libssl.a' ],
        extra_compile_args=['-Wno-cpp', '-g0', '-O3', "-I" + numpy.get_include()],
        # '-Wall', "-fPIC", "-std=c++11", '-g0', '-O3',
    )
],
    language='c++',
)

setup(name='DataBoxEngine',
      version='1.0',
      author="SWIG Docs",
      description="""Simple swig DataBoxEngine from docs""",
      ext_modules=ext_modules,
      #        py_modules=["hivetiles"],
      packages=_packages
      )
