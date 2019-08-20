#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 18:34:41 2018

@author: wijers

SimfileClone class: more advanced version of readfileclone, and example objects
to work with
"""

import numpy as np
import make_maps_v3_master as m3
import eagle_constants_and_units as c

class Simfileclone_data_entry:
    '''
    Single array with the scalings included as attriutes in the EAGLE output
    '''
    def __init__(self, data, asc, hsc, cgs):
        self.a_scaling = asc 
        self.h_scaling = hsc
        self.CGSconversion = cgs
        self.data = data

# use EAGLE cosmological parameters    
class Simfileclone_data:
    '''
    Cosmological and box parameters, and a dictionary containing data entries 
    with names that should match those in the EAGLE output files, e.g.
    'PartType0/Temperature'
    'PartType0/SmoothedElementAbundance/Helium'
    'PartType1/Coordinates'
    '''
    def __init__(self, z=0., boxsize = 10.*c.hubbleparam, data = None):
        self.z = z        
        self.hubble = m3.Hubble(self.z)
        self.omegam = c.omega0
        self.omegab = c.omegabaryon
        self.omegalambda = c.omegalambda
        self.h = c.hubbleparam
        self.a = 1./(1.+self.z)
        self.boxsize = boxsize
        self.data = {}
        if data is not None:
            self.add_data(data)

    def add_data(self,data): 
        '''
        data: {name: Simfileclone_data_entry instance} dict
        '''
        self.data.update(data)
        # make sure any coordinates fit in the box
        keys = np.array(data.keys())
        coordkeys = keys[np.array(['Coordinates' in key for key in keys])]
        for key in coordkeys:
            self.data[key].data %= self.boxsize

class Simfileclone:
    '''
    for testing purposes: contains properties and grids mimicking 
    read_eagle_files Simfile objects
    cosmology fixed to eagle values
    
    data:          Simfileclone_data instance
    regionsupport: allow roughly mimicked read-in region selection
                   (does not come with the speed advantage of the EAGLE 
                   version)
    '''
    def __init__(self, data, regionsupport=False):
        self.data = data
        self.z = data.z        
        self.h = data.h
        self.a = data.a
        self.boxsize = data.boxsize
        self.omegam = data.omegam
        self.omegab = data.omegab
        self.omegalambda = data.omegalambda
        self.regionsupported = regionsupport
        

    def readarray(self,name,region=None,rawunits=False):
        print("Reading in %s"%name) # to keep track of double read-ins
        arr = self.data.data[name]
        self.a_scaling = arr.a_scaling 
        self.h_scaling = arr.h_scaling
        self.CGSconversion = arr.CGSconversion
        self.CGSconvtot = self.a**self.a_scaling * self.h**self.h_scaling * self.CGSconversion
        if self.regionsupported and region is not None:
            sel = self.getregionsel(region, name)
        else:
            sel = m3.Sel()
        if rawunits:
            return arr.data[sel.val]
        else:
            return arr.data[sel.val]*self.CGSconvtot 
        
    def getregionsel(self, region, name): # mimicks the region selection in read_eagle, at least roughly 
        pt = (name.split('/'))[0] # PartType#
        xkey = '%s/%s'(pt,'Coordinates')
        lkey = '%s/%s'(pt,'SmoothingLength')

        if xkey not in self.data.keys() or lkey not in self.data.keys():
            print('Region selection is not possible for %s since smoothing length and coordinate data are needed for this')
            self.regionsupported = False
            return m3.Sel()
        mask = m3.Sel()
        region = np.array(region)
        region %= self.boxsize
        if region[0] < region[1]:
            mask.comb(region[0] <= self.data.data[xkey].data[:,0] + self.data.data[lkey].data)
            mask.comb(region[1] >  self.data.data[xkey].data[:,0] - self.data.data[lkey].data)
        else:
            mask.comb(np.logical_or(region[0] <= self.data.data[xkey].data[:,0] + self.data.data[lkey].data,\
                                    region[1] >  self.data.data[xkey].data[:,0] - self.data.data[lkey].data))
        if region[2] < region[3]:
            mask.comb(region[2] <= self.data.data[xkey].data[:,1] + self.data.data[lkey].data)
            mask.comb(region[3] >  self.data.data[xkey].data[:,1] - self.data.data[lkey].data)
        else:
            mask.comb(np.logical_or(region[2] <= self.data.data[xkey].data[:,1] + self.data.data[lkey].data,\
                                    region[3] >  self.data.data[xkey].data[:,1] - self.data.data[lkey].data))
        if region[4] < region[5]:
            mask.comb(region[4] <= self.data.data[xkey].data[:,2] + self.data.data[lkey].data)
            mask.comb(region[5] >  self.data.data[xkey].data[:,2] - self.data.data[lkey].data)
        else:
            mask.comb(np.logical_or(region[4] <= self.data.data[xkey].data[:,2] + self.data.data[lkey].data,\
                                    region[5] >  self.data.data[xkey].data[:,2] - self.data.data[lkey].data))    
        return mask




# a few entry classes with pre-specified asc, hsc, cgs
class Pos_entry(Simfileclone_data_entry):
    '''
    units: cMpc/h
    position, smoothing length
    '''
    def __init__(self, data):
        Simfileclone_data_entry.__init__(self, data, 1, -1, c.unitlength_in_cm)

class Vel_entry(Simfileclone_data_entry):
    '''
    units: km/s * a^0.5
    velocity
    '''
    def __init__(self, data, NumPart=None):
        if data.shape == (3,) and NumPart is not None:
            data = np.array((data,)*NumPart)
        Simfileclone_data_entry.__init__(self, data, 0.5, 0., c.unitvelocity_in_cm_per_s)


def getgrid(partperside, scale):
    coordsbase = np.indices((partperside,)*3)[0] 
    coordsbase = (np.asarray(coordsbase,dtype=np.float)+0.5)/float(partperside)
    coords = np.empty((np.prod(coordsbase.shape),3))
    coords[:,0] = np.ndarray.flatten(coordsbase)*scale
    coords[:,1] = np.ndarray.flatten(np.swapaxes(coordsbase,0,1))*scale
    coords[:,2] = np.ndarray.flatten(np.swapaxes(coordsbase,0,2))*scale
    return Pos_entry(coords)



### rho, T grid, solar metallicities
z = 3.
numpart = 10**6
pos = getgrid(100, 100.*c.hubbleparam)
lsm = Pos_entry(np.ones(numpart))
vel = Vel_entry(np.array([10.,100.,-250.]),numpart)

Tvals = 10**(2. +  np.arange(1000)/1000.*(9.-2.))
rhovals = 10**(-33. +  np.arange(1000)/1000.*(-23. + 33.)) / c.unitdensity_in_cgs
T = Simfileclone_data_entry((np.array((Tvals,)*len(rhovals))).flatten(), 0, 0 , 1.)
rho3 = Simfileclone_data_entry((np.array([(rho,)*len(Tvals) for rho in rhovals]) / ((1./(1.+z))**-3*c.hubbleparam**2) ).flatten(), -3, 2, c.unitdensity_in_cgs)
noSFR = Simfileclone_data_entry(np.zeros(numpart).astype(bool), 0, 0, 1.)

# solar values from Rob Wiersma's tables
dct_sol = {}
dct_sol['Hydrogen']  = np.ones(numpart)*0.70649785
dct_sol['Helium']    = np.ones(numpart)*0.28055534
dct_sol['Calcium']   = np.ones(numpart)*6.4355E-5
dct_sol['Carbon']    = np.ones(numpart)*0.0020665436
dct_sol['Iron']      = np.ones(numpart)*0.0011032152
dct_sol['Magnesium'] = np.ones(numpart)*5.907064E-4
dct_sol['Neon']      = np.ones(numpart)*0.0014144605
dct_sol['Nitrogen']  = np.ones(numpart)*8.3562563E-4
dct_sol['Oxygen']    = np.ones(numpart)*0.0054926244
dct_sol['Silicon']   = np.ones(numpart)*6.825874E-4
dct_sol['Sulfur']    = np.ones(numpart)*4.0898522E-4
met_sol              = np.ones(numpart)*0.0129

# primordial values
dct_pri = {}
dct_pri['Hydrogen']  = np.ones(numpart)*0.752
dct_pri['Helium']    = np.ones(numpart)*0.248
dct_pri['Calcium']   = np.zeros(numpart)
dct_pri['Carbon']    = np.zeros(numpart)
dct_pri['Iron']      = np.zeros(numpart)
dct_pri['Magnesium'] = np.zeros(numpart)
dct_pri['Neon']      = np.zeros(numpart)
dct_pri['Nitrogen']  = np.zeros(numpart)
dct_pri['Oxygen']    = np.zeros(numpart)
dct_pri['Silicon']   = np.zeros(numpart)
dct_pri['Sulfur']    = np.zeros(numpart)
met_pri              = np.zeros(numpart)

# 0.1 solar values (H, He interpolated; sum with Ztot works)
dct_0p1 = {}
dct_0p1['Hydrogen']  = 0.752 + 0.1*(0.70649785-0.752)
dct_0p1['Helium']    = 0.248 + 0.1*(0.28055534-0.248)
dct_0p1['Calcium']   = 0.1*6.4355E-5
dct_0p1['Carbon']    = 0.1*0.0020665436
dct_0p1['Iron']      = 0.1*0.0011032152
dct_0p1['Magnesium'] = 0.1*5.907064E-4
dct_0p1['Neon']      = 0.1*0.0014144605
dct_0p1['Nitrogen']  = 0.1*8.3562563E-4
dct_0p1['Oxygen']    = 0.1*0.0054926244
dct_0p1['Silicon']   = 0.1*6.825874E-4
dct_0p1['Sulfur']    = 0.1*4.0898522E-4
dct_0p1['Metallicity'] = 0.1*0.0129

solar_rhoT_z3 = Simfileclone_data(z=z, boxsize=100.*c.hubbleparam)
solar_rhoT_z3.add_data({'PartType0/Coordinates': pos,\
                     'PartType0/SmoothingLength': lsm,\
                     'PartType0/Velocity': vel,\
                     'PartType0/Density': rho3,\
                     'PartType0/Temperature': T,\
                     'PartType0/Metallicity': Simfileclone_data_entry(met_sol, 0, 0, 1.),\
                     'PartType0/SmoothedMetallicity': Simfileclone_data_entry(met_sol, 0, 0, 1.),\
                     'PartType0/OnEquationOfState': noSFR })

solar_rhoT_z3.add_data({'PartType0/ElementAbundance/%s'%elt: Simfileclone_data_entry(dct_sol[elt], 0, 0, 1.) for elt in dct_sol.keys()})
solar_rhoT_z3.add_data({'PartType0/SmoothedElementAbundance/%s'%elt: Simfileclone_data_entry(dct_sol[elt], 0, 0, 1.) for elt in dct_sol.keys()})

# Pt metallicities are solar, Sm are primordial 
Zvar_rhoT_z3 = Simfileclone_data(z=z, boxsize=100.*c.hubbleparam)
Zvar_rhoT_z3.add_data({'PartType0/Coordinates': pos,\
                     'PartType0/SmoothingLength': lsm,\
                     'PartType0/Velocity': vel,\
                     'PartType0/Density': rho3,\
                     'PartType0/Temperature': T,\
                     'PartType0/Metallicity': Simfileclone_data_entry(met_sol, 0, 0, 1.),\
                     'PartType0/SmoothedMetallicity': Simfileclone_data_entry(met_pri, 0, 0, 1.),\
                     'PartType0/OnEquationOfState': noSFR })

Zvar_rhoT_z3.add_data({'PartType0/ElementAbundance/%s'%elt: Simfileclone_data_entry(dct_sol[elt], 0, 0, 1.) for elt in dct_sol.keys()})
Zvar_rhoT_z3.add_data({'PartType0/SmoothedElementAbundance/%s'%elt: Simfileclone_data_entry(dct_pri[elt], 0, 0, 1.) for elt in dct_pri.keys()})

# redshift zero
z = 0.
numpart = 10**6
pos = getgrid(100, 100.*c.hubbleparam)
lsm = Pos_entry(np.ones(numpart))
vel = Vel_entry(np.array([10.,100.,-250.]),numpart)

Tvals = 10**(2. +  np.arange(1000)/1000.*(9.-2.))
rhovals = 10**(-33. +  np.arange(1000)/1000.*(-23. + 33.)) / c.unitdensity_in_cgs
T = Simfileclone_data_entry((np.array((Tvals,)*len(rhovals))).flatten(), 0, 0 , 1.)
rho0 = Simfileclone_data_entry((np.array([(rho,)*len(Tvals) for rho in rhovals]) / ((1./(1.+z))**-3*c.hubbleparam**2) ).flatten(), -3, 2, c.unitdensity_in_cgs)
noSFR = Simfileclone_data_entry(np.zeros(numpart).astype(bool), 0, 0, 1.)

# solar values from Rob Wiersma's tables
dct2_sol = {}
dct2_sol['Hydrogen']  = np.ones(numpart)*0.70649785
dct2_sol['Helium']    = np.ones(numpart)*0.28055534
dct2_sol['Calcium']   = np.ones(numpart)*6.4355E-5
dct2_sol['Carbon']    = np.ones(numpart)*0.0020665436
dct2_sol['Iron']      = np.ones(numpart)*0.0011032152
dct2_sol['Magnesium'] = np.ones(numpart)*5.907064E-4
dct2_sol['Neon']      = np.ones(numpart)*0.0014144605
dct2_sol['Nitrogen']  = np.ones(numpart)*8.3562563E-4
dct2_sol['Oxygen']    = np.ones(numpart)*0.0054926244
dct2_sol['Silicon']   = np.ones(numpart)*6.825874E-4
dct2_sol['Sulfur']    = np.ones(numpart)*4.0898522E-4
met2_sol              = np.ones(numpart)*0.0129

# primordial values
dct2_pri = {}
dct2_pri['Hydrogen']  = np.ones(numpart)*0.752
dct2_pri['Helium']    = np.ones(numpart)*0.248
dct2_pri['Calcium']   = np.zeros(numpart)
dct2_pri['Carbon']    = np.zeros(numpart)
dct2_pri['Iron']      = np.zeros(numpart)
dct2_pri['Magnesium'] = np.zeros(numpart)
dct2_pri['Neon']      = np.zeros(numpart)
dct2_pri['Nitrogen']  = np.zeros(numpart)
dct2_pri['Oxygen']    = np.zeros(numpart)
dct2_pri['Silicon']   = np.zeros(numpart)
dct2_pri['Sulfur']    = np.zeros(numpart)
met2_pri              = np.zeros(numpart)


# Pt metallicities are solar, Sm are primordial 
Zvar_rhoT_z0 = Simfileclone_data(z=z, boxsize=100.*c.hubbleparam)
Zvar_rhoT_z0.add_data({'PartType0/Coordinates': pos,\
                     'PartType0/SmoothingLength': lsm,\
                     'PartType0/Velocity': vel,\
                     'PartType0/Density': rho0,\
                     'PartType0/Temperature': T,\
                     'PartType0/Metallicity': Simfileclone_data_entry(met_sol, 0, 0, 1.),\
                     'PartType0/SmoothedMetallicity': Simfileclone_data_entry(met_pri, 0, 0, 1.),\
                     'PartType0/OnEquationOfState': noSFR })

Zvar_rhoT_z0.add_data({'PartType0/ElementAbundance/%s'%elt: Simfileclone_data_entry(dct2_sol[elt], 0, 0, 1.) for elt in dct_sol.keys()})
Zvar_rhoT_z0.add_data({'PartType0/SmoothedElementAbundance/%s'%elt: Simfileclone_data_entry(dct2_pri[elt], 0, 0, 1.) for elt in dct_sol.keys()})