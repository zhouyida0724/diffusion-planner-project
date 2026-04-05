"""Setup for building Cython extensions."""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "fast_geometry",
        ["fast_geometry.pyx"],
        include_dirs=[np.get_include()],
        # Avoid -ffast-math to preserve numerical behavior as much as possible.
        extra_compile_args=["-O3", "-march=native"],
        extra_link_args=["-O3"],
        # Some vectorized libm symbols live in libmvec on glibc.
        libraries=["m", "mvec"],
    )
]

setup(
    name="fast_geometry",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
        }
    ),
)