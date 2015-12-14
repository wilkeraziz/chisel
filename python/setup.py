from setuptools import setup
from Cython.Build import cythonize
import numpy as np

ext_modules = cythonize(['chisel/mteval/fast_bleu.pyx'],
        language='c++',
        exclude=[])

setup(
    name = 'chisel',
    license = 'Apache 2.0',
    author = 'Wilker Aziz',
    author_email = 'will.aziz@gmail.com',
    packages=['chisel', 'chisel.util', 'chisel.ff', 'chisel.decoder', 'chisel.mteval', 'chisel.learning'],
    ext_modules=ext_modules,
    include_dirs=[np.get_include()]
)
