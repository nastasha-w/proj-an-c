#
# Script to build the read_eagle python extension module
#
# Run with
#
# python ./setup.py install --prefix=<dir>
#

# Edit this to set the location of HDF5. This directory should
# contain the HDF5 lib/ and include/ directories.
hdf5_location =  "/apps/skylake/software/mpi/gcc/6.4.0/openmpi/3.0.0/hdf5/1.10.1/" #(OzStar)
#hdf5_location = "/usr/" #(quasar/galaxy -- Leiden)
#hdf5_location = "/gpfs/COSMA/hdf5/gnu_4.1.2/1.8.5-patch1/" #(cosma5 -- Durham, pre-2018 update)
#hdf5_location = "/cosma/local/hdf5//intel_2018/1.8.20/" #(cosma5 -- Durham, post-2018 update)

import sys
from distutils.core import setup, Extension
import numpy.distutils.misc_util

numpy_include_dir = numpy.distutils.misc_util.get_numpy_include_dirs()
idirs             = numpy_include_dir + [hdf5_location+"/include"]
ldirs             = [hdf5_location+"/lib"]

if sys.platform.startswith("win"):
    # On Windows, must not specify run time search path
    rdirs = []
    # For static HDF5 library
    #extra_compile_args = ["-D_HDF5USEDLL_",]
    # For dynamic HDF5 library
    extra_compile_args = ["-DH5_BUILT_AS_DYNAMIC_LIB",]
else:
    # Set runtime library search path on non-Windows systems
    rdirs = ldirs
    # No need for extra args in this case
    extra_compile_args = []

read_eagle_module = Extension('_read_eagle',
                              sources = ['./src/_read_eagle.c','./src/read_eagle.c'],
                              libraries=["hdf5"],
                              include_dirs =idirs,
                              library_dirs =ldirs,
                              runtime_library_dirs=rdirs,
                              extra_compile_args=extra_compile_args,
                          )

setup (name         = 'ReadEagle',
       version      = '1.0',
       description  = 'Code for reading P-H key sorted Eagle snapshots',
       ext_modules  = [read_eagle_module],
       py_modules   = ['read_eagle'],
       package_dir  = {'' : 'src'})
