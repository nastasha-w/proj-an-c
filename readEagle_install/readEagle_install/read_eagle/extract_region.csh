#!/bin/tcsh

# Name of one file from the snapshot
set infile  = /gpfs/data/Eagle/mfTestRuns/EAGLE_RUNS/L0100N1440/ANARCHY_CSP_048/COOL_OWLS_LIM_1/AGN_HALO_11p60_SNe_T0_5p5_x2Fe_x2SNIA/data/snapshot_002_z009p993/snap_002_z009p993.0.hdf5

# Name of the output file
set outfile = ./test.hdf5

# Coordinates of the region to read in -
# xmin,xmax,ymin,ymax,zmin,zmax
set region = "55,56,60,61,26,27"

./extract_region.py -r $region $infile $outfile
