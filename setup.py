from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extension = Extension(
    name='monotonic_align',
    sources=['src/jrtts/GlowTTS/Networks/monotonic_align/core.pyx'],
    include_dirs=[numpy.get_include()]
    )

setup(
    ext_modules = cythonize(extension)
)

