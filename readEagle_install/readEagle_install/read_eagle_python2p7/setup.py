#
# Script to build the read_eagle python extension module
#
# Run with
#
# python ./setup.py install --prefix=<dir>
#
# On Cosma need to load the gnu_comp/4.1.2 module and the
# corresponding HDF5 module first.
#

# Edit this to set the location of HDF5
#hdf5_location = "/gpfs/COSMA/hdf5/gnu_4.1.2/1.8.5-patch1/"
hdf5_location = "/opt/homebrew/" # new mac, /usr/ on galaxy
from distutils.core import setup, Extension
import numpy.distutils.misc_util

numpy_include_dir = numpy.distutils.misc_util.get_numpy_include_dirs()
idirs             = numpy_include_dir + [hdf5_location+"/include"]
ldirs             = [hdf5_location+"/lib"]
rdirs             = ldirs

read_eagle_module = Extension('_read_eagle',
                              sources = ['_read_eagle.c','read_eagle.c'],
                              libraries=["hdf5"],
                              define_macros=[("H5_USE_16_API","1")],
                              include_dirs =idirs,
                              library_dirs =ldirs,
                              runtime_library_dirs=rdirs)

setup (name         = 'ReadEagle',
       version      = '0.1',
       description  = 'Code for reading P-H key sorted Eagle snapshots',
       ext_modules  = [read_eagle_module],
       py_modules   = ['read_eagle'])
