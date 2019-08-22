#!/bin/tcsh
#
# Make a random sample of an Eagle snapshot
#
#BSUB -L /bin/tcsh
#BSUB -n 36
#BSUB -J sample_snapshot
#BSUB -oo sample.out
#BSUB -q cosma
#

module purge
module load python
module load platform_mpi/gnu_4.4.6/8.2.0

# Run the program
mpirun python ./sample_volume.py \
    test/snapshot_002_z009p993/snap_002_z009p993.0.hdf5 \
    ./test-sample/snap_002_z009p993 \
    0.001

