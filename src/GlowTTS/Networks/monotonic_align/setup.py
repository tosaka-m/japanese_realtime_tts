'''
Copyright (c) 2020 Jaehyeon Kim
MIT License
https://opensource.org/licenses/mit-license.php
'''

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'monotonic_align',
  ext_modules = cythonize("core.pyx"),
  include_dirs=[numpy.get_include()]
)
