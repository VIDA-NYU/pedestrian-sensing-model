from distutils.core import setup
from Cython.Distutils import build_ext
from setuptools.extension import Extension
import numpy as np

myextensions = [
    Extension(name='optimized', sources=['optimized.pyx'],
              include_dirs=[np.get_include()],
              language='c++',
              extra_compile_args=["-std=c++11", "-O3"]
              )

]

setup(
    name='optimized',
    ext_modules = myextensions,
    cmdclass = {'build_ext': build_ext}
)

