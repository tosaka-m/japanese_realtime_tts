from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize('src/jrtts/GlowTTS/Networks/monotonic_align/core.pyx'),
    include_dirs=[numpy.get_include()]
)

