from read_eagle_files import *
data_type='sub'
dir='/disks/eagle/L0050N0752/Z0p10_W1p00_E_3p0_0p3_ALPHA1p0e6_rhogas1_reposlim3p0soft_100mgas'
snap=28
read = read_eagle_file(dir, filename=data_type, snapnum=snap)
boxsize=read.a*read.boxsize/read.h

#READING & CONVERTING
halo_mass=read.read_data_array('Subhalo/Mass')
halo_mass=(read.a**(-read.a_scaling))*halo_mass/(1.9891*10**33) #to Msun
