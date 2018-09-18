import os
import sys
import imp
import numpy

from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext

try:
    __doc__ = open('README.md').read()
except IOError:
    pass

__file__ = './'
ROOT            = 'ensemble_hic'
LOCATION        = os.path.abspath(os.path.dirname(__file__))
JUNK            = ['CVS']

NAME            = "ensemble_hic"
VERSION         = "0.1"
AUTHOR          = "Simeon Carstens"
EMAIL           = "simeon.carstens@mpibpc.mpg.de"
URL             = "http://www.simeon-carstens.com"
SUMMARY         = "Inferential Structure Determination of Chromosome Ensembles from Hi-C Data"
DESCRIPTION     = __doc__
LICENSE         = 'MIT'
REQUIRES        = ['numpy', 'scipy', 'csb']

module = Extension('ensemble_hic._ensemble_hic',
                   define_macros = [('MAJOR_VERSION', '0'),
                                    ('MINOR_VERSION', '1'),
                                    ('PY_ARRAY_UNIQUE_SYMBOL','ENSEMBLEHIC')],
                   include_dirs = [numpy.get_include(), './ensemble_hic/c'],
                   extra_compile_args = ['-Wno-cpp'],
                   sources = ['./ensemble_hic/c/_ensemblehicmodule.c',
                              './ensemble_hic/c/mathutils.c',
                              './ensemble_hic/c/nblist.c',
                              './ensemble_hic/c/forcefield.c',
                              './ensemble_hic/c/prolsq.c',
                              ])

os.environ['CFLAGS'] = '-Wno-cpp'
setup(
    name=NAME,
    packages=find_packages(exclude=('tests',)),
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    description=SUMMARY,
    long_description=DESCRIPTION,
    license=LICENSE,
    requires=REQUIRES,
    ext_modules=[module] + cythonize("ensemble_hic/*.pyx","ensemble_hic/*.pyd"), 
    include_dirs = [numpy.get_include(), 'ensemble_hic/c'],
    cmdclass={'build_ext': build_ext},
    classifiers=(
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Software Development :: Libraries')
    )

