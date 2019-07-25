import read_eagle_files as eagle

# specify what to read in from where
dir = '/net/galaxy/data1/crain/eagle_startup/L0012N0188/Z0p10_W1p00_E_3p0_0p3_ALPHA1p0e6_rhogas1_reposlim3p0soft_100mgas'
file_type = 'snap'
snapnum = 20      # this is z = 0.87, each z has its own number

# units are cgs!
file = eagle.read_eagle_file(dir, file_type, snapnum, gadgetunits=False, suppress=False)

# gas particle density [g/cm^3]
density = file.read_data_array('PartType0/Density', suppress=False)
# gas particle temperature [K]
temperature = file.read_data_array('PartType0/Temperature', suppress=False)
# smoothed gas particle metallicity
metallicity = file.read_data_array('PartType0/SmoothedMetallicity', suppress=False)

# smoothed element abundances (mass fractions)
ab_H  = file.read_data_array('PartType0/SmoothedElementAbundance/Hydrogen', suppress=False)
ab_He = file.read_data_array('PartType0/SmoothedElementAbundance/Helium', suppress=False)

# file.a, file.h, file.z, file.boxsize give you the exp. factor, hubble const., redshift and comoving boxsize of the snapshot
# so boxsize in proper Mpc at the z of the snapshot
boxsize = file.a*file.boxsize/file.h
