Nastasha Wijers, 2018-06-04

Thanks to Jon Davies for doing this without instructions, so I knew what to
include.

This package contains what should be all the files you need to run my python
wrapper for HsmlAndProject. If you only want HsmlAndProject, or its OpenMP
version, you can ignore everything but the HsmlAndProject_OMP directory.

The python wrapper is contained in the proj-an repo. 
The script is reasonably tested for BAHAMAS.

mac branch: the compiler option CC is changed from gcc to gcc-mp-9 to deal with
the gcc -> clang issue. (Also, a header was replaced in one of the 
HsmlAndProject files.)

-------------------------------------------------------------------------------
Contents
-------------------------------------------------------------------------------
- HsmlAndProject_OMP: my HsmlAndProject files, modified from Volker Springel's
  version. This takes two SPH particle properties, and projects the first onto
  a 2d grid, and gives a weighted average of the second by the first on the 
  same grid. See the README file inside for documentation.
- readEagle_install: a package I got from Chris Barber for reading EAGLE output
  files. The only thing I added is a note on installing the read_eagle_files 
  package if you already have read_eagle. Please add/share installation notes
  for different systems as you get them. 
- interp2d: contains a 2d interpolation function I wrote at the start of my 
  PhD, and the makefile to compile it.
- python_wrapper: the main projection script make_maps_v3_master.py, the python
  scripts it depends on, a script to run 2d projections from the command line,
  and a README file with instructions and documentation.


-------------------------------------------------------------------------------
Instructions 
-------------------------------------------------------------------------------
To run all of this, you will need a C compiler, with an OpenMP installation for
the OMP versions, to compile the C functions used here. The makefiles included 
here use gcc. 
I have not included Serena Bertone's ion balance or emission tables, but you 
will need these (or similar tables in the same format) to calculate column 
densities or surface brightnesses respectively. (Ion balance tables are part of 
my specwizard package.) To use simulation outputs, you will, of course, also 
need access to those outputs. 

To get the python wrapper on proj-an working, compile the HsmlAndProject versions 
in the  HsmlAndProject_OMP folder (instructions in folder), make sure read_eagle 
and read_eagle_files are set up (see readEagle_install and the instructions 
inside) and the 2d interpolation scheme used for ion balance and emission tables 
in interp2d (Just run make in that folder. make clean also works there).

For instructions on the python wrapper itself, see the proj-an folder.



