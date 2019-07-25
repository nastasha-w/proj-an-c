"""
NEW VERSION THAT USES JOHN HELLY'S ROUTINES
...this allows you to just read a segment of a file. Much faster for making
images etc.

These require :
setenv PYTHONPATH /gpfs/data/jch/Eagle/Extract_Region/read_eagle_python/lib/python2.7/site-packages/:$PYTHONPATH



This function reads in arrays from Eagle snapshots, FOF output and 
subfind output in hdf5 format.  
The default converts output to physics quantities in GCS units 
using the conversion factors stored in the hdf5 files.

To run, make sure this file is in your PYTHONPATH and imported.
This function is set up as a class, so first you need to 
initialise it, then read in the attributes or data arrays as
required.

USE:
file = read_eagle_file(directory, file_type, snapnum,  
    gadgetunits=False, suppress=False)

    directory: the directory path which contains the data 
     folder you are reading the array from
    file_type: from the list of 'snap', 'snip', 'group', 
    'group_snip', 'sub' and 'particles'
    snapnum: snapshot number
    gadgetunits can be changed to True to produce data in
     Gadget units without carrying out conversion CGS
    suppress can be changed to True so as not to print
     progress to screen

     file now has the following properties
      self.fname 
      self.NumFiles
      self.NumPartTot
      self.NumPartFile

    read_file_attibutes() is also called, providing the following
    variables

    file now has the following extra properties
     self.a              - the expansion factor
     self.h              - the hubble constant
     self.z              - the redshift

array = file.read_data_array(var, suppress=False)
    this will read the data array 'var', e.g. PartType0/Mass
    or FOF/GroupMassType, and retrun the data to array

    file now has the following extra properties
     self.data           - the specified array
     self.a_scaling      - the a exponent conversion from gadgetunits
     self.h_scaling      - the h exponent conversion from gadgetunits
     self.CGSconversion  - the CGS conversion factor from gadgetunits

This was started by Michelle Furlong, Durham, 17 April 2012. 
Major update occured on 18 July 2012, adding functionality to read
all file types using the on function and accounting for changes
to the eagle output, in particlular the particle files.

Known issues:
Functionality to read group_snips needs to be added

Updates:
Both Eagle directory file names and standard Gadget directory file
names are checked
Files with .0.hdf5 and .hdf5 are checked
There is a check to make sure the correct number of elements are read
in from an array
JCH's read script is added for reading snaps, snips and particle data,
this is much faster and can also read specified regions regions
"""

import numpy as N
import h5py
import os
import fnmatch
import time
import read_eagle

metals = ['Carbon'    ,
           'Helium'   ,
           'Hydrogen' ,
           'Iron'     ,
           'Magnesium',
           'Neon'     ,
           'Nitrogen' ,
           'Oxygen'   ,
           'Silicon'  ,
          ] 

#list of Varible with dimenstions greater than 1 required
n3 = ['Coordinates', 'Velocity', 'Velocities', 'CentreOfMass', 'Spin',   
    'GroupCentreOfPotential', 'CentreOfPotential', 'GasSpin']

n6 = ['MassType', 'GroupMassType', 'GroupLengthType', 'GroupOffsetType', 'HalfMassProj', 'HalfMassRad', 'Mass_10kpc',  'Mass_20kpc', 'Mass_30kpc', 'Mass_40kpc', 'Mass_50kpc', 'Mass_70kpc', 'Mass_001kpc', 'Mass_003kpc', 'Mass_005kpc', 'Mass_010kpc',  'Mass_020kpc', 'Mass_030kpc', 'Mass_040kpc', 'Mass_050kpc', 'Mass_070kpc', 'Mass_100kpc', 'SubLengthType', 'MassTwiceHalfMassRad']

n9 = ['InertiaTensor']

filename_options = ['snap', 'group', 'group_snip', 'sub', 'particles', 'snip']

class read_eagle_file():
    def __init__( self, model_dir, filename, snapnum, suppress=True, gadgetunits=False, subsample_factor=1):
   
        if filename not in filename_options:
            print "[read_eagle_file] ERROR: This filename is not supported, please choose from:"
            print filename_options
            print "exiting...."
            exit(-1)

        self.filename = filename
        self.fname = self.get_full_dir(model_dir, filename, snapnum)
        print "trying to read data from ", self.fname
        self.subsample_factor = int(subsample_factor)   # deliberately an integer!
        self.snap = snapnum

        return


    def get_full_dir(self, model_dir, filename, snapnum):
        """Get filename, including full path"""
        dirpath = model_dir+"/data/"
        if filename == 'snap':
            #check to see if using new file format
            try:
                dirname = "snapshot_%.3i_z***p***" % snapnum
                dir = fnmatch.filter( os.listdir(dirpath), dirname)[0]
                string = dir.rsplit('_')
                fname = dirpath + dir + "/snap_%.3i_%s." % (snapnum, string[-1])
            #otherwise use old format
            except:
                dirname = "snapshot_%.3i/" % snapnum
                fname = "snap_%.3i." % snapnum
        elif filename == 'snip':
            #check to see if using new file format
            try:
                dirname = "snipshot_%.3i_z***p***" % snapnum
                dir = fnmatch.filter( os.listdir(dirpath), dirname)[0]
                string = dir.rsplit('_')
                fname = dirpath + dir + "/snip_%.3i_%s." % (snapnum, string[-1])
            #otherwise use old format
            except:
                dirname = "snipshot_%.3i/" % snapnum
                fname = "snip_%.3i." % snapnum
        elif filename == 'particles':
            try:
                dirname = "particledata_%.3i_z***p***" % snapnum
                dir = fnmatch.filter( os.listdir(dirpath), dirname)[0]
                string = dir.rsplit('_')
                fname = dirpath + dir + "/eagle_subfind_particles_%.3i_%s." % (snapnum, string[-1])
            except:
                dir = "particledata_%.3i/" % snapnum
                fname = dirpath + dir +"eagle_subfind_particles_%.3i." % snapnum
        else:
            try:
                if filename == 'group_snip':
                    dirname = "groups_snip_%.3i_z???p???" % snapnum
                else:
                    dirname = "groups_%.3i_z***p***" % snapnum
                dir = fnmatch.filter( os.listdir(dirpath), dirname)[0]
                string = dir.rsplit('_')
                if 'group' in filename:
                    fname = dirpath + dir + "/group_tab_%.3i_%s." % (snapnum, string[-1])
                elif 'sub' in filename:
                    fname = dirpath + dir + "/eagle_subfind_tab_%.3i_%s." % (snapnum, string[-1])
            except:
                dir = "groups_%.3i/" % snapnum
                if filename == 'group':
                    fname = dirpath + dir + "group_tab_%.3i." % snapnum
                elif filename == 'sub':
                    try:
                        fname = dirpath + dir + "eagle_subfind_tab_%.3i." % snapnum
                    except:
                        fname = dirpath + dir + "/sub_%.3i." % (snapnum)

        try:
            try:
                #open first file
                f = h5py.File( fname+"0.hdf5", 'r' )
            except:
                #this will become redundant with the new eagle file format
                f = h5py.File( fname+"hdf5", 'r' )
        except:
            print "[read_eagle_file] ERROR: File", fname, "does not exist/can not be opened"
            print "exiting...."
            exit(-2)

        self.NumFiles = f['Header'].attrs['NumFilesPerSnapshot']
        if filename == 'snap' or filename=='snip' or filename == 'particles':
           self.NumPartTot = f['Header'].attrs['NumPart_Total']
           self.NumPartFile = f['Header'].attrs['NumPart_ThisFile']
        else:
            NumPartTot = N.zeros(3)
            NumPartFile = N.zeros(3)
            NumPartTot[2] = f['Header'].attrs['NumPart_Total'][4]
            NumPartFile[2] = f['Header'].attrs['NumPart_ThisFile'][4]
            try:
                NumPartTot[0] = f['Header'].attrs['TotNgroups']
                NumPartFile[0] = f['Header'].attrs['Ngroups']
                if filename == 'sub':
                    NumPartTot[1] = f['Header'].attrs['TotNsubgroups']
                    NumPartFile[1] = f['Header'].attrs['Nsubgroups']
            except:
                NumPartTot[0] = f['Header'].attrs['NumPart_Total'][0]
                NumPartFile[0] = f['Header'].attrs['NumPart_ThisFile'][0]
                if filename == 'sub':
                    NumPartTot[1] = f['Header'].attrs['NumPart_Total'][1]
                    NumPartFile[1] = f['Header'].attrs['NumPart_ThisFile'][1]
            self.NumPartTot = NumPartTot
            self.NumPartFile = NumPartFile

        self.read_file_attributes(f)

        f.close()

        return fname

    def read_file_attributes(self, f):
        #read header data
        #try:
        #    f = h5py.File( self.fname+"0.hdf5", 'r' )
        #except:
        #    f = h5py.File( self.fname+"hdf5", 'r' )
        z = f['Header'].attrs['Redshift']
        self.z = round(z, 2)
        try:
            self.a = f['Header'].attrs['ExpansionFactor']
        except:
            self.a = f['Header'].attrs['Time']
        self.h = f['Header']. attrs['HubbleParam']
        self.boxsize = f['Header']. attrs['BoxSize']  # [h^-1 Mpc]
        
        #Read conversion units
        self.gamma        = f['Constants'].attrs['GAMMA']
        self.solar_mass   = f['Constants'].attrs['SOLAR_MASS']   # [g]
        self.boltzmann    = f['Constants'].attrs['BOLTZMANN']    # [erg/K]
        self.protonmass   = f['Constants'].attrs['PROTONMASS']   # [g]
        self.sec_per_year = f['Constants'].attrs['SEC_PER_YEAR']

        self.solarabundance        = f['Parameters/ChemicalElements'].attrs['SolarAbundance']
        self.solarabundance_oxygen = f['Parameters/ChemicalElements'].attrs['SolarAbundance_Oxygen']
        self.solarabundance_iron   = f['Parameters/ChemicalElements'].attrs['SolarAbundance_Iron']

        try:
        	self.rho_unit = f['Units'].attrs['UnitDensity_in_cgs']
        	self.omega_b = f['Header'].attrs['OmegaBaryon']
        except: #if no baryons in file
            self.rho_unit = 0 
            self.omega_b = 0

        try:
            self.bh_seed = f['RuntimePars'].attrs['SeedBlackHoleMass_Msun']
        except:
            self.bh_seed = False

        return

    def read_data_array(self, var, gadgetunits=False, suppress=False, interval=1, region=None, requiredlength=1e9):
        #open the file
        try:
            f = h5py.File( self.fname+"0.hdf5", 'r' )
        except:
            f = h5py.File( self.fname+"hdf5", 'r' )
        string = var.rsplit('/')
        try:
            parttype = N.int(string[0][-1])  #...select the integer bit...
        except:
            parttype = 6 #no parttype can have this value

        #read data, use JCH's script for snaps, snips and particledata
        if self.filename == 'snap' or self.filename == 'snip' or self.filename == 'particles':
            #read convresion factors (even if we don't use them)
            try :
                self.a = f['Header'].attrs['ExpansionFactor']
            except:
                self.a = f['Header'].attrs['Time']
            self.h = f['Header']. attrs['HubbleParam']
            try:
                if not 'ElementAbundance' in string[-1]:
                    self.a_scaling = f[var].attrs['aexp-scale-exponent']
                    self.h_scaling = f[var].attrs['h-scale-exponent']
                    self.CGSconversion = f[var].attrs['CGSConversionFactor']
                else:
                    self.a_scaling = f[var+'/'+metals[0]].attrs['aexp-scale-exponent']
                    self.h_scaling = f[var+'/'+metals[0]].attrs['h-scale-exponent']
                    self.CGSconversion = f[var+'/'+metals[0]].attrs['CGSConversionFactor']
            except:
                print "Warning, no conversion factors found in file 0!"
                self.a_scaling = 0
                self.h_scaling = 0
                self.CGSconversion = 1

           #close file opened for reading attributes
            f.close()
            try:
                f = read_eagle.EagleSnapshot( self.fname+"0.hdf5" )
                # select a portion for reading. Only relevant files will be openned.    

                if region is None:
                    region = (0, f.boxsize, 0, f.boxsize, 0, f.boxsize)
                if self.subsample_factor > 1:
                    f.set_sampling_rate(1/self.subsample_factor)
                f.select_region(*region)
                #read data from first file
                if not(suppress): print '[read_eagle_file] reading variable ', var
                self.data = f.read_dataset(parttype, var[9:])
                f.close()
            except:
                print 'Warning: No hash tables!  Any region selection is ignored.'
                self.data = self.read_nohashtable_array( var, gadgetunits=gadgetunits, suppress=suppress, interval=interval )

            if not(gadgetunits):
                conversion = self.a ** self.a_scaling * (self.h ** self.h_scaling) * self.CGSconversion
                if not(suppress):
                    print '[read_eagle_file] converting to physical quantities in CGS units: '
                    print '                     aexp-scale-exponent = ', self.a_scaling
                    print '                     h-scale-exponent    = ', self.h_scaling
                    print '                     CGSConversionFactor = ', self.CGSconversion
                    

                self.data = self.data*conversion
                if not(suppress): print '[read_eagle_file] returning data in CGS units'
            if not(suppress): print '[read_eagle_file] finished reading snapshot'
            if not(suppress): print 

            return self.data
        
        else:
            #test if there are any particles or this type in this
            #snapshot, otherwise return empty array
            if 'ParticleIDs' in var:
                parttype = 2
            elif 'FOF' in var:
                parttype = 0 #we set NumPartTot[0] to the total number of FOF in get_full_dir()
            elif 'Subhalo' in var:
                parttype = 1 #we set NumPartTot[1] to the total number of subhalos in get_full_dir()
            if self.NumPartTot[parttype] == 0:
                print "No particles of type ", parttype, " in snapshot ", self.snap
                return []

            #set up multi-dimensional arrays
            if string[-1] in n3:
                self.data = N.empty([0,3])
            elif string[-1] in n6:
                self.data = N.empty([0,6])
            elif string[-2] == 'Mass':
                self.data = N.empty([0,6])
            elif string[-1] in n9:
                self.data = N.empty([0,9])
            elif ('ElementAbundance') in string[-1]:
                self.dummy = N.empty([len(metals), self.NumPartTot[parttype]])
                self.data = N.empty([len(metals),0])
            else:
                self.data = N.empty([0])


            #read data from first file
            if not(suppress): print '[read_eagle_file] reading variable ', var

            # loop over files to find the first one with data in it.
            read=True
            withdata = -1
            j = 0
            Ndata = 0
            while read :
                f = h5py.File( self.fname+str(j)+".hdf5", 'r' )
                if 'ParticleIDs' in var:
                    Ndata = f['Header'].attrs['Nids']
                elif 'FOF' in var:
                    Ndata = f['Header'].attrs['Ngroups']
                elif 'Subhalo' in var:
                    Ndata = f['Header'].attrs['Nsubgroups']
                if Ndata == 0:
                    if not suppress: print "Num part in file", j, " is ", Ndata, "continuing..."
                    f.close()
                    j = j + 1
                    if j >= self.NumFiles :
                        print "No particles found in any file!!!"
                        return self.data
                else:
                    if withdata < 0: withdata = j
                    #read data
                    self.read_var(f, var, j, suppress)
                    f.close()
                    j = j+1
                    if j >= self.NumFiles or len(self.data) > requiredlength:
                        read = False

#        self.test_array(var)
 
        #read conversion factors (even if we don't use them)
        f = h5py.File( self.fname+str(withdata)+".hdf5", 'r' )
        try:
            self.a = f['Header'].attrs['ExpansionFactor']
        except:
            self.a = f['Header'].attrs['Time']
        self.h = f['Header']. attrs['HubbleParam']
        if not 'ElementAbundance' in string[-1]:
            self.a_scaling = f[var].attrs['aexp-scale-exponent']
            self.h_scaling = f[var].attrs['h-scale-exponent']
            self.CGSconversion = f[var].attrs['CGSConversionFactor']
        else:
            self.a_scaling = f[var+'/'+metals[0]].attrs['aexp-scale-exponent']
            self.h_scaling = f[var+'/'+metals[0]].attrs['h-scale-exponent']
            self.CGSconversion = f[var+'/'+metals[0]].attrs['CGSConversionFactor']

        if not(gadgetunits):
            conversion = self.a ** self.a_scaling * (self.h ** self.h_scaling) * self.CGSconversion
            if not(suppress):
                print '[read_eagle_file] converting to physical quantities in CGS units: '
                print '                     aexp-scale-exponent = ', self.a_scaling
                print '                     h-scale-exponent    = ', self.h_scaling
                print '                     CGSConversionFactor = ', self.CGSconversion
                
            self.data = self.data*conversion
            if not(suppress): print '[read_eagle_file] returning data in CGS units'
        if not(suppress): print '[read_eagle_file] finished reading snapshot'
        if not(suppress): print 

        return self.data


    def read_var(self, f, var, j, suppress, interval=1):
        string = var.rsplit('/')
        try:
            if not('ElementAbundance' in string[-1]):
                self.data = N.append( self.data, f[var][::self.subsample_factor], axis=0)
                #if not(suppress): print '         min/max value ', N.min(f[var]), N.max(f[var]) 
            else:
                for i in range(0, len(metals)):  
                    self.dummy[i] = N.copy(f[var+'/'+metals[i]])[::self.subsample_factor]
                self.data = N.append(self.data, self.dummy, axis=1)
            return True
        except:
                print '[read_eagle_file] ERROR: Variable '+var+' not found, file ', j
                return False

    def read_group_array(self, var, gadgetunits=False, suppress=False, interval=1):
        """to read a group file, you need to read it without assuming the hash table exists"""
        self.data = self.read_nohashtable_array(var, gadgetunits=gadgetunits, suppress=suppress, interval=interval)
        return self.data

    def read_nohashtable_array(self, var, gadgetunits=False, suppress=False, interval=1):
        """old style reading routine that doesn not require hash table"""
        #open the file
        try:
            f = h5py.File( self.fname+"0.hdf5", 'r' )
        except:
            f = h5py.File( self.fname+"hdf5", 'r' )

        #Set up array depending on what we're reading in
        string = var.rsplit('/')

        if string[-1] in n3:
            self.data = N.empty([0,3])
        elif string[-1] in n6:
            self.data = N.empty([0,6])
        elif string[-1] in n9:
            self.data = N.empty([0,9])
        elif ('ElementAbundance') in string[-1]:
            self.dummy = N.empty([len(metals), self.NumPartFile[4]])
            self.data = N.empty([len(metals),0])
        else:
            self.data = N.empty([0])

        #read data from first file
        if not(suppress): print '[read_eagle_file] reading variable ', var

        # loop over files to find the first one with data in it.
        j = 0
        len = 0
        while len == 0 :
            f = h5py.File( self.fname+str(j)+".hdf5", 'r' )
            string = var.rsplit('/')
            if self.filename == 'snap' or self.filename=='snip' or self.filename == 'particles':
                NumPartFile = f['Header'].attrs['NumPart_ThisFile']
                if 'PartType' in string[-2]: 
                    parttype = N.int(string[-2][-1])
                len = NumPartFile[parttype]
            else:
                if 'ParticleIDs' in var:
                    len = f['Header'].attrs['Nids']
                elif 'FOF' in var:
                    len = f['Header'].attrs['Ngroups']
                elif 'Subhalo' in var:
                    len = f['Header'].attrs['Nsubgroups']
            if len == 0:
                if not suppress: print "Num part in file", j, " is ", len, "continuing..."
                f.close()
                j = j + 1
                if j >= self.NumFiles :
                    print "No particles found in any file!!!"
                    # exit()  # don't want to exit , this might be high z
        # now we know the first file "j" has particles in it.
        self.read_nohashtable_var(f, var, j,  suppress)

        #read convresion factors
        try :
            self.a = f['Header'].attrs['ExpansionFactor']
        except :
            self.a = f['Header'].attrs['Time']
        self.h = f['Header']. attrs['HubbleParam']
        if not 'ElementAbundance' in string[-1]:
            self.a_scaling = f[var].attrs['aexp-scale-exponent']
            self.h_scaling = f[var].attrs['h-scale-exponent']
            self.CGSconversion = f[var].attrs['CGSConversionFactor']
        else:
            self.a_scaling = f[var+'/'+metals[0]].attrs['aexp-scale-exponent']
            self.h_scaling = f[var+'/'+metals[0]].attrs['h-scale-exponent']
            self.CGSconversion = f[var+'/'+metals[0]].attrs['CGSConversionFactor']
        f.close()

        for j in range(j+1, self.NumFiles):
            f = h5py.File( self.fname+str(j)+".hdf5", 'r' )
            string = var.rsplit('/')
            if self.filename == 'snap' or self.filename == 'snip' or self.filename == 'particles':
                NumPartFile = f['Header'].attrs['NumPart_ThisFile']
                if 'PartType' in string[-2]: 
                    parttype = N.int(string[-2][-1])
                len = NumPartFile[parttype]
            else:
                if 'ParticleIDs' in var:
                    len = f['Header'].attrs['Nids']
                elif 'FOF' in var:
                    len = f['Header'].attrs['Ngroups']
                elif 'Subhalo' in var:
                    len = f['Header'].attrs['Nsubgroups']
            if len == 0 :
                if not suppress: print "Num part in file", j, " is ", len, "continuing..."
                continue

            #if not(suppress): print '     reading file - part ', j 
            self.read_nohashtable_var(f, var, j, suppress)

            f.close()

#        self.test_array(var)

     #   if not(gadgetunits):
      #      conversion = self.a ** self.a_scaling * (self.h ** self.h_scaling) * self.CGSconversion
       #     if not(suppress):
        #        print '[read_eagle_file] converting to physical quantities in CGS units: '
        #        print '                     aexp-scale-exponent = ', self.a_scaling
        #        print '                     h-scale-exponent    = ', self.h_scaling
        #        print '                     CGSConversionFactor = ', self.CGSconversion
        #        
        #    self.data = self.data*conversion
        #    if not(suppress): print '[read_eagle_file] returning data in CGS units'
        #if not(suppress): print '[read_eagle_file] finished reading snapshot'
        #if not(suppress): print 

        return self.data


    def read_nohashtable_var(self, f, var, j, suppress, interval=1):
        """old reading routine that does not use hash table"""
        string = var.rsplit('/')
        try:
            if not('ElementAbundance' in string[-1]):
                self.data = N.append( self.data, f[var][::self.subsample_factor], axis=0)
                #if not(suppress): print '         min/max value ', N.min(f[var]), N.max(f[var]) 
            else:
                for i in range(0, len(metals)):  
                    self.dummy[i] = N.copy(f[var+'/'+metals[i]])[::self.subsample_factor]
                self.data = N.append(self.data, self.dummy, axis=1)
            return True
        except:
                print '[read_eagle_file] ERROR: Variable '+var+' not found, file ', j
                print "[read_eagle_file] returning value of False"
                #self.data = N.append( self.data, f[var][::self.subsample_factor], axis=0)
                return False



    def test_array(self, var):
        string = var.rsplit('/')
        """This function checks that all elements of the array were read in"""
        data_len = len(self.data) 
        if self.filename == 'snap' or  self.filename == 'snip' or self.filename == 'particles':
            if 'PartType' in string[-2]: 
                parttype = N.int(string[-2][-1])
                if 'ElementAbundance' in string[-1]:
                    data_len = len(self.data.T)
            elif 'ElementAbundance' in string[-2]:
                parttype = N.int(string[-3][-1])
            else:
                print "WARNING: Test functionality doesn't exist for ", var
                return 

        else:
            if 'ParticleIDs' in var:
                parttype = 2
            elif 'FOF' in var:
                parttype = 0 #we set NumPartTot[0] to the total number of FOF in get_full_dir()
            elif 'Subhalo' in var:
                parttype = 1 #we set NumPartTot[1] to the total number of subhalos in get_full_dir()
            else:
                print "WARNING: Test functionality doesn't exist for ", var
                return 


        if 'ParticleIDs' not in var:
            if data_len != self.NumPartTot[parttype] and self.subsample_factor == 1:
                print '[read_eagle_file] ERROR: Length of', var, ' should be', self.NumPartTot[parttype], ', array of length ', data_len, 'read in!'
                print "[read_eagle_file] Exiting..."
                exit(-1)
        
        return


#--------------------------------------------------------------------

if __name__ == "__main__":
    dir = '/gpfs/data/Eagle/mfTestRuns/CodeDev/SubFind/AddVariables'
    snap = 28

    print 'Testing groupfile'
    f = read_eagle_file(dir, 'group', snap)
    data = f.read_data_array('/FOF/Mass', requiredlength=10)

    print 'Testing snapshot'
    f = read_eagle.EagleSnapshot("/gpfs/data/Eagle/mfTestRuns/CodeDev/SubFind/AddVariables/data/snapshot_028_z000p000/snap_028_z000p000.0.hdf5" )
    f.close()
    read = read_eagle_file(dir, 'snap', snap)
    start=time.time()
    mass = read.read_data_array('PartType0/Mass')
    print time.time() - start
 
    exit()

