# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:16:50 2017

@author: wijers

Calculate line emission or ion column densities from EAGLE simulation outputs,
project these quantities or EAGLE output quantities onto a 2D grid

supports line emission and column density from ions/lines in 
make_maps_opts_locs, 
and can calculate both total quantities in a column along one of the coordinate 
axes, and average quatities weighted by one of the supported quantities

projections along more general directions are not supported.

Uses the v3 version of HsmlAndProject, which allows the projection of larger
numbers of particles than before, and fixes an issue with the minimum 
smoothing length in the case of non-square pixels


TODO: 
    - test wishlisting
    - implement generic selections using wishlisting (can be basic only, initially)
"""

version = 3.21 # matches corresponding make_maps version
# naming of outputs updated to be more sensible, e.g. cares less about 
# projection axis; 
# sacrifices naming equal to previous versions

###############################################################################
###############################################################################
#############################                    ##############################
#############################       README       ##############################    
#############################                    ##############################
###############################################################################
###############################################################################

# This is a make_maps version that tries to incorporate all the other
# make_maps_v3 versions. It still uses make_maps_opts_locs for small tables and 
# lists, and file locations.
#
# The main function is make_map, but a lot of the functionality written out in
# previous versions in split into functions here: emission and column density
# calculations, for example, are done per particle (lumninosity and number of
# ions, respectively), and have a column area / conversion to surface 
# brightness done afterwards. These functions can be used separately to 
# calculate e.g. phase diagrams weighted by ion number etc.
#
# Note for these purposes that read-in and deletion of variables is usually
# done via the Vardict (`variable dictionay') class. 
# (And at a minimum through Simfile, which at 
# present is a thin wrapper for read_eagle_files.readfile, but is meant to
# be expandible to similar simulations, e.g. Hydrangea/C-EAGLE or OWLS, without
# changing the rest of the code too much.)
#
# a Vardict object keeps track of the particle properties read in or calculated
# (e.g. temperature or lumninosity). Of course, for single use, this is just
# a somewhat overcomplicted way to store variables. However, it is useful for
# combining more than one function, without having to change the function
# depending on what else you want to calculate (or fill it with if statements).
#
# Vardict simply stores a wishlist, and uses delif to remove a variable only if 
# it is not on the wishlist. The variable 'last' can be set to True to force 
# deletion, which is easy for the last function in a calculation. Note that
# the generation and updating of the wishlist is not trivial; this just allows
# this part to be independent of e.g. emission calculation, reducing the chance
# of errors cropping up due to unnecessary fiddling with functions.
#
# In order for this to work properly, a consistent naming of variables is 
# important. Variables read in from EAGLE directly are named by their hdf5 
# path, without the PartType#/ .  Derived quantities have the following names:
#
# logT        for log10 Temperature [K]
# lognH       for log10 hydrogen number density [cm^-3], proper volume density
# propvol     for volume [cm^-3], proper
# luminosity  for emission line luminosity; conversion to cgs stored in Vardict
# coldens     for column density; conversion to cgs stored in Vardict
# eos         for particles on the equation of state (boolean)
# Lambda_over_nH2 cooling rate/nH^2 [erg cm^3/s]
# tcool       cooling time (internal energy/cooling rate) [s]
# note that these variables do not always mean exactly the same thing: 
# for example lognH  will e.g. depend on the hydrogen number density used. 
# Take this into account in wishlist generation and calculation order. 
###############################################################################
###############################################################################


################################
#       imports (not all)      #
################################
import numpy as np
import eagle_constants_and_units as c
reload(c)

import ctypes as ct
#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import mpl_toolkits.axes_grid1 as axgrid
import ion_header as ionh
import string
#import time

import make_maps_opts_locs as ol
reload(ol)
import h5py
import numbers as num # for instance checking
# !! imports for simulation read-in are done in Readfile class for the 
#    simulation under consideration (prevents unececessary errors if an unused
#    read-in script is not installed)


##########################
#      functions 1       #
##########################

#### cosmological basics

def comoving_distance_cm(z,simfile=None): # assumes Omega_k = 0
    if z < 1e-8:
        print('Using 0 comoving distance from z. \n')
        return 0.
    if simfile is None:
        # use cosmological parameters for EAGLE from its constants and units file
        hpar = c.hubbleparam 
        omega0 = c.omega0
        omegalambda = c.omegalambda
    else:
        # use cosmological parameters for the simulation file (read in in Simfile.__init__ from hdf5 output files)
        hpar = simfile.h # use hubble parameter from the simulation; c.hubble is 100 km/s/Mpc 
        z = simfile.z    # override input z by the value for the used snapshot
        omega0 = simfile.omegam
        omegalambda = simfile.omegalambda
    
    def integrand(zi):
        return (omega0*(1.+zi)**3 + omegalambda)**0.5
    zi_arr = np.arange(0,z+z/512.,z/512.)
    com = np.trapz(1./integrand(zi_arr),x=zi_arr)
    return com * c.c/(c.hubble*hpar)

def ang_diam_distance_cm(z,simfile=None):
    if simfile is None:
        return comoving_distance_cm(z,simfile)/(1.+z)
    else:
        return comoving_distance_cm(simfile.z,simfile)/(1.+simfile.z)

def lum_distance_cm(z,simfile=None):
    if simfile is None:
        return comoving_distance_cm(z,simfile)*(1.+z)
    else:
        return comoving_distance_cm(simfile.z,simfile)*(1.+simfile.z)
        
def Hubble(z,simfile=None):
    if simfile is None:
        # use cosmological parameters for EAGLE from its constants and units file
        hpar = c.hubbleparam 
        omega0 = c.omega0
        omegalambda = c.omegalambda
    else:
        # use cosmological parameters for the simulation file (read in in Simfile.__init__ from hdf5 output files)
        hpar = simfile.h # use hubble parameter from the simulation; c.hubble is 100 km/s/Mpc 
        z = simfile.z    # override input z by the value for the used snapshot
        omega0 = simfile.omegam
        omegalambda = simfile.omegalambda
        
    return (c.hubble*hpar)*(omega0*(1.+z)**3 + omegalambda)**0.5
      
def solidangle(alpha,beta): # alpha = 0.5 * pix_length_1/D_A, beta = 0.5 * pix_length_2/D_A
    #from www.mpia.de/~mathar/public/mathar20051002.pdf
    # citing  A. Khadjavi, J. Opt. Soc. Am. 58, 1417 (1968).
    # stored in home/papers
    # using the exact formula, with alpha = beta, 
    # the python exact formula gives zero for alpha = beta < 10^-3--10^-4
    # assuming the pixel sizes are not so odd that the exact formula is needed in one direction and gives zero for the other,
    # use the Taylor expansion to order 4 
    # testing the difference between the Taylor and python exact calculations shows that 
    # for square pixels, 10**-2.5 is a reasonable cut-off
    # for rectangular pixels, the cut-off seems to be needed in both values
    if alpha < 10**-2.5 or beta <10**-2.5:
        return 4*alpha*beta - 2*alpha*beta*(alpha**2+beta**2)
    else: 
        return 4*np.arccos(((1+alpha**2 +beta**2)/((1+alpha**2)*(1+beta**2)))**0.5)



#### emission/asorption table finding and interpolation

def readstrdata(filen,separator = None,headerlength=1):
    # input: file name, charachter separating columns, number of lines at the top to ignore
    # separator None means any length of whitespace is considered a separator
    # only for string data
    
    data = open(filen,'r')
    array = []
    # skip header:
    for i in range(headerlength):
        data.readline()
    for line in data:
        line = line.strip() # remove '\n'
        columns = line.split(separator)
        columns = [str(col) for col in columns]
        array.append(columns)
    return np.array(array)       
    
# finds emission tables for element and interpolates them to zcalc if needed and possible
def findemtables(element,zcalc):
    
    #### checks and setup
    
    if not element in ol.elements:
        print("There will be an error somewhere: %s is not included or misspelled. \n" % element)
    
    if zcalc < 0. and zcalc > 1e-4:
        zcalc = 0.0
        zname = ol.zopts[0]
        interp = False

    elif zcalc in ol.zpoints:
        # only need one table
        zname = ol.zopts[ol.zpoints.index(zcalc)]
        interp = False
    
    elif zcalc <= ol.zpoints[-1]:
        # linear interpolation between two tables
        zarray = np.asarray(ol.zpoints)
        zname1 = ol.zopts[len(zarray[zarray<zcalc])-1]
        zname2 = ol.zopts[-len(zarray[zarray>zcalc])]
        interp = True
    else:
        print("Chosen z value requires extrapolation. This has not been implemented. \n") 
        
    
    #### read in the tables; interpolate tables in z if needed and possible
    
    if not interp:
        tablefilename = ol.dir_emtab%zname + element + '.hdf5'
        tablefile = h5py.File(tablefilename, "r")
        #energies = np.array(tablefile.get('header/spectrum/logenergy_ryd'))
        #fluxes = np.array(tablefile.get('header/spectrum/logflux'))
        logTK =   np.array(tablefile.get('logt'),dtype=np.float32)  
        logrhocm3 =   np.array(tablefile.get('logd'),dtype=np.float32)
        lines =   np.array(tablefile.get('lines'),dtype=np.float32)  
        
        
        tablefile.close()
    
    if interp: #linear interpolation: 1./(a1-a0) * ( (a1-a)*f0 + (a-a0)*f1 )
        tablefilename1 = ol.dir_emtab%zname1 + element + '.hdf5'
        tablefile1 = h5py.File(tablefilename1, "r")
        #energies = np.array(tablefile.get('header/spectrum/logenergy_ryd'))
        #fluxes = np.array(tablefile.get('header/spectrum/logflux'))
        logTK1 =   np.array(tablefile1.get('logt'),dtype=np.float32)  
        logrhocm31 =   np.array(tablefile1.get('logd'),dtype=np.float32)
        lines1 =   np.array(tablefile1.get('lines'),dtype=np.float32) 
        
        tablefile1.close()
        
        tablefilename2 = ol.dir_emtab%zname2 + element + '.hdf5'
        tablefile2 = h5py.File(tablefilename2, "r")
        #energies = np.array(tablefile.get('header/spectrum/logenergy_ryd'))
        #fluxes = np.array(tablefile.get('header/spectrum/logflux'))
        logTK2 =   np.array(tablefile2.get('logt'),dtype=np.float32)  
        logrhocm32 =   np.array(tablefile2.get('logd'),dtype=np.float32)
        lines2 =   np.array(tablefile2.get('lines'),dtype=np.float32) 
        
        tablefile2.close()
        
        if (np.all(logTK1 == logTK2) and np.all(logrhocm31 == logrhocm32)):
            print("interpolating 2 emission tables")
            lines = 1./(float(zname2)-float(zname1)) * ( (float(zname2)-zcalc)*lines1 + (zcalc-float(zname1))*lines2 )
            logTK = logTK1
            logrhocm3 = logrhocm31
        else: 
            print("Temperature and density ranges of the two interpolation z tables don't match. \n")
            print("Using nearest z table in stead.")
            if abs(zcalc - float(zname1)) < abs(zcalc - float(zname2)):
                logTK = logTK1
                logrhocm3 = logrhocm31
                lines = lines1
            else:
                logTK = logTK2
                logrhocm3 = logrhocm32
                lines = lines2
    
    return lines, logTK, logrhocm3

           
# calculate emission using C function (interpolator)
def find_emdenssq(z,elt,lognH,logT,lineind):

    p_emtable, logTK, lognHcm3 = findemtables(elt,z)
    emtable = p_emtable[:,:,lineind]
    NumPart = len(lognH)
    inlogemission = np.zeros(NumPart,dtype=np.float32)
    
    if len(logT) != NumPart:
        print('logrho and logT should have the same length')
        return None

    # need to compile with some extra options to get this to work: make -f make_emission_only
    print("------------------- C interpolation function output --------------------------\n")
    cfile = ol.c_interpfile

    acfile = ct.CDLL(cfile)
    interpfunction = acfile.interpolate_2d
    # ion balance tables are temperature x density x line no.
    interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           ct.c_longlong , \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK)*len(lognHcm3),)), \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognHcm3),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]

    # argument conversion

    res = interpfunction(logT.astype(np.float32),\
               lognH.astype(np.float32),\
               ct.c_longlong(NumPart),\
               np.ndarray.flatten(emtable.astype(np.float32)),\
               logTK.astype(np.float32),\
               ct.c_int(len(logTK)), \
               lognHcm3.astype(np.float32),\
               ct.c_int(len(lognHcm3)),\
               inlogemission \
              )
    
    print("-------------- C interpolation function output finished ----------------------\n")
    
    if res != 0:
        print('Something has gone wrong in the C function: output %s. \n',str(res))
        return None
        
    return inlogemission



def findiontables(ion,z):
    # README in dir_iontab:
    # files are hdf5, contain ionisation fraction of a species for rho, T, z
    
    
    #### checks and setup
    
    if not ion in ol.ions:
        print("There will be an error somewhere: %s is not included or misspelled. \n" % ion)
    
    tablefilename = ol.dir_iontab %ion + '.hdf5'      
    tablefile = h5py.File(tablefilename, "r")
    logTK =   np.array(tablefile.get('logt'),dtype=np.float32)  
    lognHcm3 =   np.array(tablefile.get('logd'),dtype=np.float32)
    ziopts = np.array(tablefile.get('redshift'),dtype=np.float32) # monotonically incresing, first entry is zero
    balance_d_t_z = np.array(tablefile.get('ionbal'),dtype=np.float32)
    tablefile.close()

    if z < 0. and z > 1e-4:
        z = 0.0
        zind = 0 
        interp = False
        
    elif z in ziopts:
        # only need one table
        zind = np.argwhere(z == ziopts)
        interp = False
    
    elif z <= ziopts[-1]:
        # linear interpolation between two tables
        zind1 = np.sum(ziopts<z)-1
        zind2 = -sum(ziopts>z)
        interp = True
    else:
        print("Chosen z value requires extrapolation. This has not been implemented. \n") 
        
    
    #### read in the tables; interpolate tables in z if needed and possible
    
    if not interp:
       balance = np.squeeze(balance_d_t_z[:,:,zind]) # for some reason, extra dimensions are tacked on 
    
    if interp: #linear interpolation: 1./(a1-a0) * ( (a1-a)*f0 + (a-a0)*f1 )
        balance1 = balance_d_t_z[:,:,zind1]
        balance2 = balance_d_t_z[:,:,zind2]
        
        print("interpolating 2 emission tables")
        balance = 1./( ziopts[zind2] - ziopts[zind1]) * ( (ziopts[zind2]-z)*balance1 + (z-ziopts[zind1])*balance2 )

    
    return balance, logTK, lognHcm3

def find_ionbal(z,ion,lognH,logT):
    
    # compared to the line emission files, the order of the nH, T indices in the balance tables is switched
    balance, logTK, lognHcm3 = findiontables(ion,z) #(np.array([[0.,0.],[0.,1.],[0.,2.]]), np.array([0.,1.,2.]), np.array([0.,1.]) ) 
    NumPart = len(lognH)
    inbalance = np.zeros(NumPart,dtype=np.float32)
    
    if len(logT) != NumPart:
        print('logrho and logT should have the same length')
        return None

    # need to compile with some extra options to get this to work: make -f make_emission_only
    print("------------------- C interpolation function output --------------------------\n")
    cfile = ol.c_interpfile

    acfile = ct.CDLL(cfile)
    interpfunction = acfile.interpolate_2d # just a linear interpolator; works for non-emission stuff too
    # ion balance tables are density x temperature x redshift 

    interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           ct.c_longlong , \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK)*len(lognHcm3),)), \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognHcm3),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]

 
    res = interpfunction(lognH.astype(np.float32),\
               logT.astype(np.float32),\
               ct.c_longlong(NumPart),\
               np.ndarray.flatten(balance.astype(np.float32)),\
               lognHcm3.astype(np.float32),\
               ct.c_int(len(lognHcm3)),\
               logTK.astype(np.float32),\
               ct.c_int(len(logTK)), \
               inbalance \
              )

    print("-------------- C interpolation function output finished ----------------------\n")
    
    if res != 0:
        print('Something has gone wrong in the C function: output %s. \n',str(res))
        return None
        
    return inbalance


### cooling tables -> cooling rates. 

def getcoolingtable(tablefile, per_elt = True):
    '''
    retrieves the per element or total metal tables from Wiersma, Schaye, 
    & Smith 2009 given an hdf5 file and whether to get the per element data or 
    the total metals data
    a full file is ~1.2 MB, so reading in everything before processing 
    shouldn't be too hard on memory 
    '''
    
    cooldct = {} # to hold the read-in data
    
    # Metal-free cooling tables are on a T, nH, Hefrac grid
    lognHcm3   =   np.log10(np.array(tablefile.get('Metal_free/Hydrogen_density_bins'),dtype=np.float32))  # stored as nH [cm^-3], but log space even intervals
    logTK      =   np.log10(np.array(tablefile.get('Metal_free/Temperature_bins'),dtype=np.float32)) # stored as T [K], but log space even intervals
    Hemassfrac =   np.array(tablefile.get('Metal_free/Helium_mass_fraction_bins'),dtype=np.float32)
   
    lambda_over_nH2_mf = np.array(tablefile.get('Metal_free/Net_Cooling'),dtype=np.float32)
    
    cooldct['Metal_free'] = {'lognHcm3': lognHcm3, 'logTK': logTK, 'Hemassfrac': Hemassfrac, 'Lambda_over_nH2': lambda_over_nH2_mf}
    cooldct['Metal_free']['mu'] = np.array(tablefile.get('Metal_free/Mean_particle_mass'),dtype=np.float32)
    
    # electron density table and the solar table to scale it by
    cooldct['Electron_density_over_n_h'] = {}
    cooldct['Electron_density_over_n_h']['solar'] = np.array(tablefile.get('Solar/Electron_density_over_n_h'),dtype=np.float32)
    cooldct['Electron_density_over_n_h']['solar_logTK'] = np.log10(np.array(tablefile.get('Solar/Temperature_bins'),dtype=np.float32))
    cooldct['Electron_density_over_n_h']['solar_lognHcm3'] = np.log10(np.array(tablefile.get('Solar/Hydrogen_density_bins'),dtype=np.float32))
    cooldct['Electron_density_over_n_h']['table'] = np.array(tablefile.get('Metal_free/Electron_density_over_n_h'),dtype=np.float32) # same bins as the metal-free cooling tables
    
    # solar abundance data
    elts = np.array(tablefile.get('/Header/Abundances/Abund_names')) # list of element names (capital letters)
    abunds = np.array(tablefile.get('/Header/Abundances/Solar_number_ratios'),dtype=np.float32) 
    cooldct['solar_nfrac'] = {elts[ind]: abunds[ind] for ind in range(len(elts))}
    cooldct['solar_mfrac'] = {'total_metals': 0.0129} # from Rob Wiersma's IDL routine compute_cooling_Z.pro, documentation. Solar mass fraction
    
    eltsl = list(elts)
    eltsl.remove('Hydrogen')
    eltsl.remove('Helium')
    # per-element or total cooling rates
    if per_elt:
        for elt in eltsl: # will get NaN values for Helium, Hydrogen if these are not removed from the list
            cooldct[elt] = {}
            cooldct[elt]['logTK']    = np.log10(np.array(tablefile.get('%s/Temperature_bins'%(elt)),dtype=np.float32))  # stored as nH [cm^-3], but log space even intervals
            cooldct[elt]['lognHcm3'] = np.log10(np.array(tablefile.get('%s/Hydrogen_density_bins'%(elt)),dtype=np.float32))  # stored as nH [cm^-3], but log space even intervals
            cooldct[elt]['Lambda_over_nH2'] = np.array(tablefile.get('%s/Net_Cooling'%(elt)),dtype=np.float32)  # stored as nH [cm^-3], but log space even intervals
            
    else:
        elt = 'Total_Metals'
        cooldct[elt] = {}
        cooldct[elt]['logTK']    = np.log10(np.array(tablefile.get('%s/Temperature_bins'%(elt)),dtype=np.float32))  # stored as nH [cm^-3], but log space even intervals
        cooldct[elt]['lognHcm3'] = np.log10(np.array(tablefile.get('%s/Hydrogen_density_bins'%(elt)),dtype=np.float32))  # stored as nH [cm^-3], but log space even intervals
        cooldct[elt]['Lambda_over_nH2'] = np.array(tablefile.get('%s/Net_Cooling'%(elt)),dtype=np.float32)  # stored as nH [cm^-3], but log space even intervals
        
    return cooldct
    
def findcoolingtables(z, method = 'per_element'):
    '''
    gets the per element cooling tables from Wiersema, Schaye, & Smith 2009,
    does linear redshift interpolation to the selected z value if required
    
    methods: 'per_element' or 'total_metals'. See readme file with the cooling 
    tables, or Wiersema, Schaye & Smith 2009, eq 4
    (basically, the difference is that total_metals uses all metals and assumes
    solar element abundance ratios)
    '''
    wdir = ol.dir_coolingtab
    szopts = readstrdata(wdir + 'redshifts.dat', headerlength = 1) # file has list of redshifts for which there are tables
    szopts = szopts.T[0] # 2d->1d array
    zopts = np.array([float(sz) for sz in szopts])
    #print(zopts)
    #print(szopts)
    if method == 'per_element':
        perelt = True
    elif method == 'total_metals':
        perelt = False
    
    if z < 0. and z > -1.e-4:
        z = 0.0
        zind = 0 
        interp = False
        
    elif z in zopts:
        # only need one table
        zind = np.argwhere(z == zopts)[0,0]
        interp = False
    
    elif z <= zopts[-1]:
        # linear interpolation between two tables
        zind1 = np.sum(zopts<z)-1
        zind2 = -1*np.sum(zopts>z)
        interp = True
    else:
        print("Chosen z value requires extrapolation. This has not been implemented. \n") 
 
    
    if not interp:
        tablefilename = wdir + 'z_%s.hdf5'%(szopts[zind])      
        tablefile = h5py.File(tablefilename, "r")
        tabdct_out = getcoolingtable(tablefile, per_elt = perelt)
        tablefile.close()
    
    else: # get both cooling tables, interpolate in z
        #print()
        tablefilename1 = wdir + 'z_%s.hdf5'%(szopts[zind1])      
        tablefile1 = h5py.File(tablefilename1, "r")
        tabdct1 = getcoolingtable(tablefile1, per_elt = perelt)
        tablefile1.close()

        tablefilename2 = wdir + 'z_%s.hdf5'%(szopts[zind2])      
        tablefile2 = h5py.File(tablefilename2, "r")
        tabdct2 = getcoolingtable(tablefile2, per_elt = perelt)
        tablefile2.close()

        tabdct_out = {}
        
        keys = tabdct1.keys()
        
        # metal-free: if interpolation grid match (they should), interpolate 
        # the tables in z. Electron density tables have the same grid points
        # separate from the other because it contains helium
        if (np.all(tabdct1['Metal_free']['lognHcm3']   == tabdct2['Metal_free']['lognHcm3']) and\
            np.all(tabdct1['Metal_free']['logTK']      == tabdct2['Metal_free']['logTK']) ) and\
            np.all(tabdct1['Metal_free']['Hemassfrac'] == tabdct2['Metal_free']['Hemassfrac' ]):
            tabdct_out['Metal_free'] = {}
            tabdct_out['Electron_density_over_n_h'] = {}
            tabdct_out['Metal_free']['lognHcm3']   = tabdct2['Metal_free']['lognHcm3']
            tabdct_out['Metal_free']['logTK']      = tabdct2['Metal_free']['logTK']
            tabdct_out['Metal_free']['Hemassfrac'] = tabdct2['Metal_free']['Hemassfrac']
            tabdct_out['Metal_free']['Lambda_over_nH2'] =\
                1./(zopts[zind2]-zopts[zind1]) *\
                ( (zopts[zind2]-z)*tabdct1['Metal_free']['Lambda_over_nH2'] +\
                  (z-zopts[zind1])*tabdct2['Metal_free']['Lambda_over_nH2']   )
            tabdct_out['Electron_density_over_n_h']['table'] =\
                1./(zopts[zind2]-zopts[zind1]) *\
                ( (zopts[zind2]-z)*tabdct1['Electron_density_over_n_h']['table'] +\
                  (z-zopts[zind1])*tabdct2['Electron_density_over_n_h']['table']   )
            tabdct_out['Metal_free']['mu'] =\
                1./(zopts[zind2]-zopts[zind1]) *\
                ( (zopts[zind2]-z)*tabdct1['Metal_free']['mu'] +\
                  (z-zopts[zind1])*tabdct2['Metal_free']['mu']   )
        else:
            print('Failed to interpolate Metal_free tables due to mismatch in interpolation grids')
            
        #interpolate solar electron density grids
        if (np.all(tabdct1['Electron_density_over_n_h']['solar_logTK']    == tabdct2['Electron_density_over_n_h']['solar_logTK'] ) and\
            np.all(tabdct1['Electron_density_over_n_h']['solar_lognHcm3'] == tabdct2['Electron_density_over_n_h']['solar_lognHcm3']) ):
            if 'Electron_density_over_n_h' not in tabdct_out.keys():
                tabdct_out['Electron_density_over_n_h'] = {}

            tabdct_out['Electron_density_over_n_h']['solar_logTK']    = tabdct2['Electron_density_over_n_h']['solar_logTK']
            tabdct_out['Electron_density_over_n_h']['solar_lognHcm3'] = tabdct2['Electron_density_over_n_h']['solar_lognHcm3']

            tabdct_out['Electron_density_over_n_h']['solar'] =\
                1./(zopts[zind2]-zopts[zind1]) *\
                ( (zopts[zind2]-z)*tabdct1['Electron_density_over_n_h']['solar'] +\
                  (z-zopts[zind1])*tabdct2['Electron_density_over_n_h']['solar']   )
        else:
            print('Failed to interpolate Solar tables due to mismatch in interpolation grids')
        
        if tabdct1['solar_nfrac'] == tabdct2['solar_nfrac']:
            tabdct_out['solar_nfrac'] = tabdct2['solar_nfrac']
        else:
            print('Failed to assign solar number fraction list due to mismatch between tables')
        if tabdct1['solar_mfrac'] == tabdct2['solar_mfrac']:
            tabdct_out['solar_mfrac'] = tabdct2['solar_mfrac']
        else:
            print('Failed to assign mass number fraction list due to mismatch between tables')
            
         # we've just done these:
        keys.remove('Metal_free')
        keys.remove('Electron_density_over_n_h')
        keys.remove('solar_nfrac')
        keys.remove('solar_mfrac')
        # total_metals and the elements all work the same way
        for key in keys: 
            if (np.all(tabdct1[key]['lognHcm3']   == tabdct2[key]['lognHcm3']) and\
                np.all(tabdct1[key]['logTK']      == tabdct2[key]['logTK']) ):
                tabdct_out[key] = {}
                tabdct_out[key]['lognHcm3']   = tabdct2[key]['lognHcm3']
                tabdct_out[key]['logTK']      = tabdct2[key]['logTK']
                tabdct_out[key]['Lambda_over_nH2'] =\
                    1./(zopts[zind2]-zopts[zind1]) *\
                    ( (zopts[zind2]-z)*tabdct1[key]['Lambda_over_nH2'] +\
                      (z-zopts[zind1])*tabdct2[key]['Lambda_over_nH2']   )
            else:
                print('Failed to interpolate %s tables due to mismatch in interpolation grids'%key)
           
    return tabdct_out
    
    
def find_coolingrates(z, dct, method = 'per_element', **kwargs):
    '''
    !! Comparison of cooling times to Wiersma, Schaye, & Smith (2009), where 
    the used tables come from and which the calculations should match, shows 
    that cooling contours might be off by ~0.1-0.2 dex (esp. their fig. 4, 
    where the intersecting lines mean differences are more visible than they 
    would be inother plots)
    
    
    arguments:        
    z:      redshift (used to find the radiation field)
            if Vardict is used, the redshift it contains is used for unit 
            conversions
    dct:    dictionary: should contain 
                lognH [cm^-3] hydrogen number density, log10
                logT [K] temperature, log10
                mass fraction per element: dictionary
                    element name (lowercase, sulfur): mass fraction
                Density [cm^-3] density 
                    (needed to get number density from mass fraction)
            or Vardict instance: kwargs must include
                T4EOS (True/False; excluding SFR should be done beforehand via 
                selections)
                hab    if lognH is not read in already:
                    'SmoothedElementAbundance/Hydrogen', 
                    'ElementAbundance/Hydrogen',
                    or hydrogen mass fraction (float) 
                abunds ('Sm', 'Pt', or dct of float values [mass fraction, 
                       NOT solar])
                       in case of a dictionary, all lowercase or all uppercase 
                       element names should both work; tested on uppercase
                       spelling: Sulfur, not Sulpher
                last   (boolean, default True): delete all vardict entries 
                       except the final answer
    method: 'per_element' or 'total_metals'
            if dct is a dictionary, element abundances should have helium in 
            both cases, only 'metallicity' if the 'total_metals' method is used
            
    returns: 
    cooling rate Lambda/n_H^2 [erg/s cm^3] for the particles for which 
    the data was supplied
    '''
    
    if method == 'per_element':
        elts_geq_he = ol.eltdct_to_ct.keys()
        elts_geq_he.remove('hydrogen')
        elts_geq_h = list(np.copy(elts_geq_he))
        elts_geq_he.remove('helium')
    elif method == 'total_metals':
        elts_geq_h = ['helium', 'metallicity']
        elts_geq_he = ['metallicity']
    delafter_abunds = False # if abundances are overwritten with custom values, delete after we're done to avoid confusion with the EAGLE output values (same names)

    if isinstance(dct, Vardict):
        vard = True
        partdct = dct.particle
        if 'last' in kwargs.keys():
            last = kwargs['last']
        else:
            last = True

        eltab_base = kwargs['abunds']
        if not (isinstance(eltab_base, str) or isinstance(eltab_base, dict)): # tuple like in make_maps?
            eltab_base = eltab_base[0]        
        if isinstance(eltab_base,str):
            if eltab_base == 'Sm' or 'SmoothedElementAbundance' in eltab_base: # example abundance is accepted
                eltab_base = 'SmoothedElementAbundance/%s'
            elif eltab_base == 'Pt' or 'ElementAbundance' == eltab_base[:16]: # example abundance is accepted
                eltab_base = 'ElementAbundance/%s'
            else:
                print('eltab value %s is not a valid option'%eltab_base)
                return -1            
        elif isinstance(eltab_base, dict):
            delafter_abunds = True
            eltab_base =  'ElementAbundance/%s'
            try: # allow uppercase or lowercase element names
                dct.particle.update({eltab_base%string.capwords(elt): kwargs['abunds'][elt] for elt in elts_geq_he})
            except KeyError:
                dct.particle.update({eltab_base%string.capwords(elt): kwargs['abunds'][string.capwords(elt)] for elt in elts_geq_he})
                
        if not dct.isstored_part('logT'):     
            dct.getlogT(last=last,logT = kwargs['T4EOS'])
        # modify wishlist for our purposes
        wishlist_old = list(np.copy(dct.wishlist))
        dct.wishlist = list(set(dct.wishlist + ['Density']))
        if not dct.isstored_part('lognH'):
            if 'last' in kwargs.keys():
                skwargs = kwargs.copy()
                del skwargs['last']
            else:
                skwargs = kwargs
            dct.getlognH(last=False,**skwargs)
        if not dct.isstored_part('Density'): # if we already had lognH, might need to read in Density explicitly
            dct.readif('Density')
        NumPart = len(dct.particle['lognH'])            

    else:
        vard = False
        NumPart = len(dct['lognH'])  
        partdct = dct
        # some value checking: array lengths
        if not np.all(np.array([ len(dct['logT'])==NumPart, len(dct['Density'])==NumPart])):
            print("Lengths of lognH (%i), logT (%i) and Density (%i) should match, but don't"%(NumPart, len(dct['logT']), len(dct['Density'])))
        if not np.all(np.array( [ True if not hasattr(dct[elt], '__len__') else\
                                  len(dct[elt])==NumPart or len(dct[elt])==1\
                                  for elt in elts_geq_h ] )):
            print("Element mass fractions must be numbers or arrays of length 1 or matching the logT, lognH and Density")
            return -1
        eltab_base =  'ElementAbundance/%s'
        if method == 'per_element': # for total_metals, keys are sorted out in place
            dct.update({eltab_base%(string.capwords(elt)): dct[elt] for elt in elts_geq_h}) # allows retrieval of values to be independent of dictionary/Vardict use
        
    cooldct = findcoolingtables(z, method = method)    
    lambda_over_nH2 = np.zeros(NumPart,dtype=np.float32)
    
    
    # do the per-element cooling interpolation and number (per_element) or mass (total_metals) fraction rescaling
    if method == 'per_element':
        for elt in elts_geq_he:
            incool = np.zeros(NumPart,dtype=np.float32)
            logTK = cooldct[ol.eltdct_to_ct[elt]]['logTK']
            lognHcm3 = cooldct[ol.eltdct_to_ct[elt]]['lognHcm3']
            table = cooldct[ol.eltdct_to_ct[elt]]['Lambda_over_nH2'] # temperature x density
            
            # need to compile with some extra options to get this to work: make -f make_emission_only
            print("------------------- C interpolation function output --------------------------\n")
            cfile = ol.c_interpfile
        
            acfile = ct.CDLL(cfile)
            interpfunction = acfile.interpolate_2d # just a linear interpolator; works for non-emission stuff too
        
            interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                                   np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                                   ct.c_longlong , \
                                   np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK)*len(lognHcm3),)), \
                                   np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK),)), \
                                   ct.c_int,\
                                   np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognHcm3),)), \
                                   ct.c_int,\
                                   np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]
        
         
            res = interpfunction(partdct['logT'].astype(np.float32),\
                       partdct['lognH'].astype(np.float32),\
                       ct.c_longlong(NumPart),\
                       np.ndarray.flatten(table.astype(np.float32)),\
                       logTK.astype(np.float32),\
                       ct.c_int(len(logTK)), \
                       lognHcm3.astype(np.float32),\
                       ct.c_int(len(lognHcm3)),\
                       incool \
                      )
        
            print("-------------- C interpolation function output finished ----------------------\n")
            
            if res != 0:
                print('Something has gone wrong in the C function: output %s. \n',str(res))
                return -2
            
            # rescale by ni/nh / (ni/nh)_solar; ni = rho*massfraction_i/mass_i
            if vard:
                dct.readif(eltab_base%string.capwords(elt))
            incool *= partdct[eltab_base%string.capwords(elt)]
            if vard:
                dct.delif(eltab_base%string.capwords(elt), last=last)
            incool /= (ionh.atomw[string.capwords(elt)] * c.u)
            incool /= cooldct['solar_nfrac'][ol.eltdct_to_ct[elt]] #partdct[eltab_base%('Helium')].astype(np.float32)scale by ni/nH / (ni/nH)_solar
            
            lambda_over_nH2 += incool
        

        lambda_over_nH2  *= partdct['Density'] # ne/nh / (ne/nh)_solar * element-indepent part of ni/nh / (ni/nh)_solar ( = density / nH)
        if vard:
            cgsfd = dct.CGSconv['Density']
            if cgsfd != 1.:
                lambda_over_nH2  *= cgsfd
        lambda_over_nH2 /= 10**partdct['lognH']     
    # end of per element

    elif method == 'total_metals':
        logTK = cooldct['Total_Metals']['logTK']
        lognHcm3 = cooldct['Total_Metals']['lognHcm3']
        table = cooldct['Total_Metals']['Lambda_over_nH2'] # temperature x density
        
        # need to compile with some extra options to get this to work: make -f make_emission_only
        print("------------------- C interpolation function output --------------------------\n")
        cfile = ol.c_interpfile
    
        acfile = ct.CDLL(cfile)
        interpfunction = acfile.interpolate_2d # just a linear interpolator; works for non-emission stuff too
    
        interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                               np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                               ct.c_longlong , \
                               np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK)*len(lognHcm3),)), \
                               np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK),)), \
                               ct.c_int,\
                               np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognHcm3),)), \
                               ct.c_int,\
                               np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]
    
     
        res = interpfunction(partdct['logT'].astype(np.float32),\
                   partdct['lognH'].astype(np.float32),\
                   ct.c_longlong(NumPart),\
                   np.ndarray.flatten(table.astype(np.float32)),\
                   logTK.astype(np.float32),\
                   ct.c_int(len(logTK)), \
                   lognHcm3.astype(np.float32),\
                   ct.c_int(len(lognHcm3)),\
                   lambda_over_nH2 \
                  )
    
        print("-------------- C interpolation function output finished ----------------------\n")
        
        if res != 0:
            print('Something has gone wrong in the C function: output %s. \n',str(res))
            return -2
        
        # rescale by ni/nh / (ni/nh)_solar; ni = rho*massfraction_i/mass_i
        if vard:
            eltab = kwargs['abunds']
            if eltab == 'Pt':
                metkey = 'Metallicity'
                dct.readif(metkey)
            elif eltab == 'Sm':
                metkey = 'SmoothedMetallicity'
                dct.readif(metkey)
            else: #dictionary
                metkey = 'metallicity'
                if 'metallicity' in eltab.keys():
                    partdct[metkey] = eltab[metkey]
                else:
                    partdct[metkey] = eltab['Metallicity']             
        else: #dictionary
            metkey = 'metallicity'
        if vard:
            dct.readif(metkey)
        lambda_over_nH2 *= partdct[metkey]
        if vard:
            dct.delif(metkey, last=last)
        lambda_over_nH2 /= cooldct['solar_mfrac']['total_metals'] #scale by mi/mH / (mi/mH)_solar (average particle mass is unknown -> cannot get particle density)
    # end of total metals
    
    # restore wishlist,vardict to pre-call version; clean-up
    if vard:
        eltab = kwargs['abunds']
        if (isinstance(eltab, num.Number) or isinstance(eltab, dict)): # not the actual EAGLE values were used
            if method == 'total_metals' and metkey in partdct.keys(): # may have already been caught by delif
                del partdct[metkey]
        if 'Density' not in wishlist_old and 'Density' in dct.wishlist:
            dct.wishlist.remove('Density')
        dct.delif('Density',last=last)
        if delafter_abunds: # abundances were not the EAGLE output values
            for elt in elts_geq_he:
                if dct.isstored_part(eltab_base%string.capwords(elt)):
                    dct.delif(eltab_base%string.capwords(elt), last=True)

    
    ## finish rescaling the cooling rates: ne/nh / (ne/nh)_solar
    incool = np.zeros(NumPart,dtype=np.float32)
    fHe      = cooldct['Metal_free']['Hemassfrac']   
    lognHcm3 = cooldct['Metal_free']['lognHcm3']
    logTK    = cooldct['Metal_free']['logTK']
    table    = cooldct['Electron_density_over_n_h']['table'] #  fHe x temperature x density
    
    # get helium number fractions
    if vard:
        dct.readif(eltab_base%('Helium'))
    if not hasattr(partdct[eltab_base%('Helium')], '__len__'): # single-number abundance, while a full array is needed for the C function
        partdct[eltab_base%('Helium')] = partdct[eltab_base%('Helium')]*np.ones(NumPart, dtype=np.float32)
    
    print("------------------- C interpolation function output --------------------------\n")
    cfile = ol.c_interpfile

    acfile = ct.CDLL(cfile)
    interpfunction = acfile.interpolate_3d # just a linear interpolator

    interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           ct.c_longlong , \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK)*len(lognHcm3)*len(fHe),)), \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(fHe),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognHcm3),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]

 
    res = interpfunction(partdct[eltab_base%('Helium')].astype(np.float32),\
               partdct['logT'].astype(np.float32),\
               partdct['lognH'].astype(np.float32),\
               ct.c_longlong(NumPart),\
               np.ndarray.flatten(table.astype(np.float32)),\
               fHe.astype(np.float32),\
               ct.c_int(len(fHe)),\
               logTK.astype(np.float32),\
               ct.c_int(len(logTK)), \
               lognHcm3.astype(np.float32),\
               ct.c_int(len(lognHcm3)),\
               incool \
              )
    
    print("-------------- C interpolation function output finished ----------------------\n")
    if res != 0:
        print('Something has gone wrong in the C function: output %s. \n',str(res))
        return -2
    
    lambda_over_nH2 *= incool
    
    # 2d interpolation for solar values
    incool = np.zeros(NumPart,dtype=np.float32)
    
    logTK    = cooldct['Electron_density_over_n_h']['solar_logTK'] 
    lognHcm3 = cooldct['Electron_density_over_n_h']['solar_lognHcm3']
    table    = cooldct['Electron_density_over_n_h']['solar'] # temperature x density

    cfile = ol.c_interpfile
    
    acfile = ct.CDLL(cfile)
    interpfunction = acfile.interpolate_2d # just a linear interpolator; works for non-emission stuff too

    interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           ct.c_longlong , \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK)*len(lognHcm3),)), \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognHcm3),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]

 
    res = interpfunction(partdct['logT'].astype(np.float32),\
               partdct['lognH'].astype(np.float32),\
               ct.c_longlong(NumPart),\
               np.ndarray.flatten(table.astype(np.float32)),\
               logTK.astype(np.float32),\
               ct.c_int(len(logTK)), \
               lognHcm3.astype(np.float32),\
               ct.c_int(len(lognHcm3)),\
               incool \
              )
    if res != 0:
        print('Something has gone wrong in the C function: output %s. \n',str(res))
        return -2
    lambda_over_nH2 /= incool
    
    
    ## add the metal-free cooling
    incool = np.zeros(NumPart,dtype=np.float32)
    fHe      = cooldct['Metal_free']['Hemassfrac']   
    lognHcm3 = cooldct['Metal_free']['lognHcm3']
    logTK    = cooldct['Metal_free']['logTK']
    table    = cooldct['Metal_free']['Lambda_over_nH2'] # fHe x temperature x density
    
    
    print("------------------- C interpolation function output --------------------------\n")
    cfile = ol.c_interpfile

    acfile = ct.CDLL(cfile)
    interpfunction = acfile.interpolate_3d # just a linear interpolator

    interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           ct.c_longlong , \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK)*len(lognHcm3)*len(fHe),)), \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(fHe),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognHcm3),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]

 
    res = interpfunction(partdct[eltab_base%('Helium')].astype(np.float32),\
               partdct['logT'].astype(np.float32),\
               partdct['lognH'].astype(np.float32),\
               ct.c_longlong(NumPart),\
               np.ndarray.flatten(table.astype(np.float32)),\
               fHe.astype(np.float32),\
               ct.c_int(len(fHe)),\
               logTK.astype(np.float32),\
               ct.c_int(len(logTK)), \
               lognHcm3.astype(np.float32),\
               ct.c_int(len(lognHcm3)),\
               incool \
              )
    
    print("-------------- C interpolation function output finished ----------------------\n")

    lambda_over_nH2 += incool
    del incool

    if vard:
        dct.delif('lognH', last=last)
        dct.delif('logT',  last=last)
        dct.delif(eltab_base%('Helium'), last=last)
        dct.add_part('Lambda_over_nH2', lambda_over_nH2)
        
    return lambda_over_nH2

        

def find_coolingtimes(z,dct, method = 'per_element', **kwargs):
    '''
    !! Comparison to Wiersma, Schaye, & Smith (2009), where the used tables
    come from and which the calculations should match, shows that cooling 
    contours might be off by ~0.1-0.2 dex (esp. their fig. 4, where the 
    intersecting lines mean differences are more visible than they would be in
    other plots)
    
    arguments: see find_coolingrates
    
    returns internal energy / cooling rate
    negative values -> cooling times
    postive values -> heating times
    '''
    # cooling tables have mean particle mass as a function of temperature, density, and helium fraction (mu)
    
    
    if isinstance(dct, Vardict):
        vard = True
        partdct = dct.particle
        if 'last' in kwargs.keys():
            last = kwargs['last']
        else:
            last = True
        if not dct.isstored_part('logT'):     
            dct.getlogT(last=last,logT = kwargs['T4EOS'])
        # modify wishlist for our purposes
        wishlist_old = list(np.copy(dct.wishlist))
        dct.wishlist = list(set(dct.wishlist +  ['Density', 'lognH', 'logT']))
        if not dct.isstored_part('lognH'):
            if 'last' in kwargs.keys():
                skwargs = kwargs.copy()
                del skwargs['last']
            else:
                skwargs = kwargs
            dct.getlognH(last=False,**skwargs)
        NumPart = len(dct.particle['lognH'])
        
        eltab_base = kwargs['abunds']
        delafter_abunds = False
        if not (isinstance(eltab_base, str) or isinstance(eltab_base, dict)): # make_maps-style tuple: none of that nonsense here
            eltab_base = eltab_base[0]        
        if isinstance(eltab_base,str):
            if eltab_base == 'Sm' or 'SmoothedElementAbundance' in eltab_base: # example abundance is accepted
                eltab_base = 'SmoothedElementAbundance/%s'
            elif eltab_base == 'Pt' or 'ElementAbundance' == eltab_base[:16]: # example abundance is accepted
                eltab_base = 'ElementAbundance/%s'
            else:
                print('eltab value %s is not a valid option'%eltab_base)
                return -1            
        elif isinstance(eltab_base, dict):
            delafter_abunds = True
            eltab_base =  'ElementAbundance/%s'
            if 'Helium' in kwargs['abunds']:
                hekey = 'Helium'
            else:
                hekey = 'helium'
            dct.particle.update({eltab_base%('Helium'): kwargs['abunds'][hekey] * np.ones(NumPart, dtype=np.float32)})      
        dct.wishlist += [eltab_base%('Helium')]
        
    else:
        vard = False
        NumPart = len(dct['lognH'])  
        partdct = dct
        # some value checking: array lengths
        if not np.all(np.array([ len(dct['logT'])==NumPart, len(dct['Density'])==NumPart])):
            print("Lengths of lognH (%i), logT (%i) and Density (%i) should match, but don't"%(NumPart, len(dct['logT']), len(dct['Density'])))
        dct['ElementAbundance/Helium'] = dct['helium']
        eltab_base =  'ElementAbundance/%s'
    
    Lambda = find_coolingrates(z,dct, method=method, **kwargs)
    Lambda *= 10**(2*partdct['lognH']) 

        
    
    # get mu
    cooldct = findcoolingtables(z, method = method)
    mu = np.zeros(NumPart,dtype=np.float32)
    fHe      = cooldct['Metal_free']['Hemassfrac']   
    lognHcm3 = cooldct['Metal_free']['lognHcm3']
    logTK    = cooldct['Metal_free']['logTK']
    table    = cooldct['Metal_free']['mu'] # fHe x temperature x density

    print("------------------- C interpolation function output --------------------------\n")
    cfile = ol.c_interpfile

    acfile = ct.CDLL(cfile)
    interpfunction = acfile.interpolate_3d # just a linear interpolator

    interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           ct.c_longlong , \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(fHe)*len(logTK)*len(lognHcm3),)), \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(fHe),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognHcm3),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]

 
    res = interpfunction(partdct[eltab_base%('Helium')].astype(np.float32),\
               partdct['logT'].astype(np.float32),\
               partdct['lognH'].astype(np.float32),\
               ct.c_longlong(NumPart),\
               np.ndarray.flatten(table.astype(np.float32)),\
               fHe.astype(np.float32),\
               ct.c_int(len(fHe)),\
               logTK.astype(np.float32),\
               ct.c_int(len(logTK)), \
               lognHcm3.astype(np.float32),\
               ct.c_int(len(lognHcm3)),\
               mu \
              )
    
    print("-------------- C interpolation function output finished ----------------------\n")
    if res != 0:
        print('Something has gone wrong in the C function: output %s. \n',str(res))
        return -2
    if vard:
        dct.delif('Lambda_over_nH2', last=last)
    
    tcool =  partdct['Density']/mu*(10**partdct['logT']) #internal energy; mu is metal-free, so there will be some small errors in n when there are metals present 
    cgc = (1.5*c.boltzmann/c.u) 
    if vard:
        cgc *= dct.CGSconv['Density'] # to CGS units
    tcool *= cgc # to CGS units
    del mu
    
    tcool /= Lambda # tcool = Uint/Lambda_over_V
    del Lambda
    
    # clean up wishlist, add end result to vardict
    if vard:
        if 'Density' not in wishlist_old and 'Density' in dct.wishlist:
            dct.wishlist.remove('Density')
            dct.delif('Density', last=last)
        if eltab_base%('Helium') not in wishlist_old and eltab_base%('Helium') in dct.wishlist:
            dct.wishlist.remove(eltab_base%('Helium'))
            dct.delif(eltab_base%('Helium'), last=last)
        if 'lognH' not in wishlist_old and 'lognH' in dct.wishlist:
            dct.wishlist.remove('lognH')
            dct.delif('lognH', last=last)
        if 'logT' not in wishlist_old and 'logT' in dct.wishlist:
            dct.wishlist.remove('logT')
            dct.delif('logT', last=last)
        if delafter_abunds: # abundances were not the EAGLE output values
            if dct.isstored_part(eltab_base%('Helium')):
                dct.delif(eltab_base%('Helium'), last=True)
        dct.add_part('tcool', tcool)
    return tcool
    


def getBenOpp1chemabundtables(vardict,excludeSFR,eltab,hab,ion,last=True,updatesel=True,misc=None):
    # ion names used here and in table naming -> ions in ChemicalAbundances table
    print('Getting ion balance from simulation directly (BenOpp1)')
    iontranslation = {'c2':  'CarbonII',\
                      'c3':  'CarbonIII',\
                      'c4':  'CarbonIV',\
                      'h1':  'HydrogenI',\
                      'mg2': 'MagnesiumII',\
                      'ne8': 'NeonVIII',\
                      'n5':  'NitrogenV',\
                      'o6':  'OxygenVI',\
                      'o7':  'OxygenVII',\
                      'o8':  'OxygenVIII',\
                      'si2': 'SiliconII',\
                      'si3': 'SiliconIII',\
                      'si4': 'SiliconIV'}
    mass_over_h = {'hydrogen':  1.,\
                   'carbon':    c.atomw_C/c.atomw_H,\
                   'magnesium': c.atomw_Mg/c.atomw_H,\
                   'neon':      c.atomw_Ne/c.atomw_H,\
                   'nitrogen':  c.atomw_N/c.atomw_H,\
                   'oxygen':    c.atomw_O/c.atomw_H,\
                   'silicon':   c.atomw_Si/c.atomw_H}
    if ion not in iontranslation.keys():
        print('ChemicalAbundances tables are not available ')
    # store chemical abundances as ionfrac, to match what is used otherwise
    # chemical abundance arrays may contain NaN values. Set those to zero. 
    vardict.readif('ChemicalAbundances/%s'%(iontranslation[ion]),region = 'auto',rawunits = True,out =False, setsel = None,setval = None)
    if ion != 'h1':
        vardict.readif(eltab, region='auto', rawunits=True, out=False, setsel=None, setval=None)
        vardict.readif(hab, region='auto', rawunits=True, out=False, setsel=None, setval=None)
        vardict.particle['ChemicalAbundances/%s'%(iontranslation[ion])] /=\
            vardict.particle[eltab]/(mass_over_h[ol.elements_ion[ion]]*vardict.particle[hab]) # convert num. dens. rel to hydrogen to num dens. rel. to total element (eltab and hab are mass fractions)

        vardict.particle['ChemicalAbundances/%s'%(iontranslation[ion])][vardict.particle[eltab]==0] = 0. #handle /0 errors properly: no elements -> no ions of that element 
        vardict.add_part('ionfrac',vardict.particle['ChemicalAbundances/%s'%(iontranslation[ion])])
        # correct zero values: NaN values -> zero, infinite values (element abundance is zero) -> zero
        vardict.particle['ionfrac'][np.isnan(vardict.particle['ionfrac'])] = 0.
        vardict.particle['ionfrac'][np.isinf(vardict.particle['ionfrac'])] = 0.
        
    else:
        vardict.add_part('ionfrac',vardict.particle['ChemicalAbundances/%s'%(iontranslation[ion])])
    vardict.delif('ChemicalAbundances/%s'%(iontranslation[ion]),last=True) # got mangled -> remove to avoid confusion
    if hab != eltab: # we still need eltab later
        vardict.delif(hab, last=last)



#### i/o processing for projections and particle selection 

def translate(old_dct, old_nm, centre, boxsize, periodic):

    if type(boxsize) == float: # to handle velocity space slicing with the correct periodicity
        boxsize = (boxsize,)*3
        
    if not periodic:
        print 'Translating particle positions: (%.2f, %.2f, %.2f) -> (0, 0, 0) Mpc' \
          % (centre[0], centre[1], centre[2])
    else: 
        print 'Translating particle positions: (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f) Mpc' \
          % (centre[0], centre[1], centre[2], 0.5*boxsize[0], 0.5*boxsize[1], 0.5*boxsize[2])
    
    # translates old coordinates into coordinates wrt centre
    # taking into account the periodicity of the box
    # for periodic boundary conditions, the coordinates must be translated into the [0, BoxSize] range for compatibility with the C function

    # change arrays in place and use dict to avoid copying and save memory
    # docs and test: broadcasting checks trailing dimensions first, so boxsize/center operation should work even if coordinate array has 3 elements 
    
    centre = np.array(centre)
    boxsize = np.array(boxsize)
    if not periodic:     
#        old_dct[old_nm][:,0] = (old_dct[old_nm][:,0] - centre[0] + 0.5*boxsize[0])%boxsize[0] - 0.5*boxsize[0]
#        old_dct[old_nm][:,1] = (old_dct[old_nm][:,1] - centre[1] + 0.5*boxsize[1])%boxsize[1] - 0.5*boxsize[1]
#        old_dct[old_nm][:,2] = (old_dct[old_nm][:,2] - centre[2] + 0.5*boxsize[2])%boxsize[2] - 0.5*boxsize[2]
         old_dct[old_nm] += 0.5*boxsize - centre
         old_dct[old_nm] %= boxsize
         old_dct[old_nm] -= 0.5*boxsize
        
    if periodic: 
#        old_dct[old_nm][:,0] = (old_dct[old_nm][:,0] - centre[0] + 0.5*boxsize[0])%boxsize[0]
#        old_dct[old_nm][:,1] = (old_dct[old_nm][:,1] - centre[1] + 0.5*boxsize[1])%boxsize[1]
#        old_dct[old_nm][:,2] = (old_dct[old_nm][:,2] - centre[2] + 0.5*boxsize[2])%boxsize[2]
         old_dct[old_nm] += 0.5*boxsize - centre
         old_dct[old_nm] %= boxsize
         
    old_dct[old_nm] = old_dct[old_nm].astype(np.float32)



def nameoutput(ptypeW,simnum,snapnum,version,kernel,npix_x,L_x,L_y,L_z,centre,BoxSize,hconst,excludeSFRW,excludeSFRQ,velcut,axis,var,abundsW,ionW,parttype,ptypeQ,abundsQ,ionQ,quantityW,quantityQ,simulation,LsinMpc,misc): 
    # some messiness is hard to avoid, but it's contained
    # Ls and centre have not been converted to Mpc when this function is called

    # box and axis
    zcen = ''
    xypos = ''
    if LsinMpc:
        Lunit = ''
        hfac = 1.
    else:
        Lunit = '-hm1'
        hfac = hconst
    if axis == 'z':
        if L_z*hfac < BoxSize * hconst**-1:
            zcen = '_zcen%s%s' %(str(centre[2]),Lunit)
        if L_x*hfac < BoxSize * hconst**-1 or L_y*hfac < BoxSize * hconst**-1:
            xypos = '_x%s-pm%s%s_y%s-pm%s%s' %(str(centre[0]),str(L_x),Lunit,str(centre[1]),str(L_y),Lunit)
        sLp = str(L_z)

    elif axis == 'y':
        if L_y*hfac < BoxSize * hconst**-1:
            zcen = '_ycen%s%s' % (str(centre[1]),Lunit)
        if L_x*hfac < BoxSize * hconst**-1 or L_z*hfac < BoxSize * hconst**-1:
            xypos = '_z%s-pm%s%s_x%s-pm%s%s' %(str(centre[2]),str(L_z),Lunit,str(centre[0]),str(L_x),Lunit)
        sLp = str(L_y)

    elif axis == 'x':
        if L_x*hfac < BoxSize * hconst**-1:
            zcen = '_xcen%s%s' % (str(centre[0]),Lunit)
        if L_y*hfac < BoxSize * hconst**-1 or L_z*hfac < BoxSize * hconst**-1:
            xypos = '_y%s-pm%s%s_z%s-pm%s%s' %(str(centre[1]),str(L_y),Lunit,str(centre[2]),str(L_z),Lunit)
        sLp = str(L_x)
        
 
    axind = '_%s-projection' %axis

    # EOS particle handling
    if excludeSFRW == True:
        SFRindW = '_noEOS'
    elif excludeSFRW == False:
        SFRindW = '_wiEOS'
    elif excludeSFRW == 'T4':
        SFRindW = '_T4EOS'
    elif excludeSFRW == 'from':
        SFRindW = '_fromSFR'
    elif excludeSFRW == 'only':
        SFRindW = '_onlyEOS'
    
    if excludeSFRQ == True:
        SFRindQ = '_noEOS'
    elif excludeSFRQ == False:
        SFRindQ = '_wiEOS'
    elif excludeSFRQ == 'T4':
        SFRindQ = '_T4EOS'
    elif excludeSFRQ == 'from':
        SFRindQ = '_fromSFR'
    elif excludeSFRQ == 'only':
        SFRindQ = '_onlyEOS'
        
        
    # abundances 
    if ptypeW in ['coldens', 'emission']:
        if abundsW[0] not in ['Sm','Pt']:
            sabundsW = '%smassfracAb'%str(abundsW[0])  
        else: 
            sabundsW = abundsW[0] + 'Ab'
        if type(abundsW[1]) == float:
            sabundsW = sabundsW + '-%smassfracHAb'%str(abundsW[1])
        elif abundsW[1] != abundsW[0]:
            sabundsW = sabundsW + '-%smassfracHAb'%abundsW[1]

    if ptypeQ in ['coldens', 'emission']:
        if abundsQ[0] not in ['Sm','Pt']:
            sabundsQ = str(abundsQ[0]) + 'massfracAb'
        else: 
            sabundsQ = abundsQ[0] + 'Ab'
        if type(abundsQ[1]) == float:
            sabundsQ = sabundsQ + '-%smassfracHAb'%str(abundsQ[1])
        elif abundsQ[1] != abundsQ[0]:
            sabundsQ = sabundsQ + '-%sHAb'%abundsQ[1]
                        

    # miscellaneous: ppv/ppp box, simulation, particle type        
    if velcut:
        vind = '_velocity-sliced'
    else:
        vind = ''

    if var != 'REFERENCE':
        ssimnum = simnum +var
    else:
        ssimnum = simnum
    if simulation == 'bahamas':
        ssimnum = 'BA-%s'%ssimnum    
    if simulation == 'eagle-ioneq':
        ssimnum = 'EA-ioneq-%s'%ssimnum      

    if parttype != '0':
        sparttype = '_PartType%s'%parttype
    else:
        sparttype = ''

    #avoid / in file names
    if ptypeW == 'basic':
        squantityW = quantityW
        squantityW = squantityW.replace('/','-')
    if ptypeQ == 'basic':
        squantityQ = quantityQ
        squantityQ = squantityQ.replace('/','-')
    
    # putting it together: ptypeQ = None is set to get resfile for W
    if ptypeQ is None: #output outputW name
        if ptypeW == 'coldens' or ptypeW == 'emission':
            resfile = ol.ndir + '%s_%s_%s_%s_test%s_%s_%sSm_%spix_%sslice' %(ptypeW,ionW,ssimnum,snapnum,str(version),sabundsW,kernel,str(npix_x),sLp) + zcen + xypos + axind + SFRindW + vind

        elif ptypeW == 'basic':
            resfile = ol.ndir + '%s%s_%s_%s_test%s_%sSm_%spix_%sslice' %(squantityW,sparttype,ssimnum,snapnum,str(version),kernel,str(npix_x),sLp) + zcen + xypos + axind + SFRindW + vind

    if ptypeQ is not None: # naming for quantityQ output
        if ptypeQ == 'basic':
            squantityQ = squantityQ + SFRindQ
        else:
            squantityQ = '%s_%s_%s'%(ptypeQ,ionQ,sabundsQ) + SFRindQ
        if ptypeW == 'basic':
            squantityW = squantityW + SFRindW
        else:
            squantityW = '%s_%s_%s'%(ptypeW,ionW,sabundsW) + SFRindW        
               
        resfile = ol.ndir + '%s_%s%s_%s_%s_test%s_%sSm_%spix_%sslice' %(squantityQ,squantityW,sparttype,ssimnum,snapnum,str(version),kernel,str(npix_x),sLp) + zcen + xypos + axind + vind

    
    #if misc is not None: 
    #    # key-value expansion means names are not strictly deterministic for misc options, but the naming should be unambiguous, so that's ok
    #    namelist = [(key,misc[key]) for key in misc.keys()]
    #    namelist = ['%s-%s'%(str(pair[0]),str(pair[1])) for pair in namelist]
    #    misctail = '_%s'%('_'.join(namelist))
    #else: 
    #    misctail = ''

    if misc is not None: # if if if : if we want to use chemical abundances from Ben' Oppenheimer's recal variations
        if 'usechemabundtables' in misc:
            if misc['usechemabundtables'] == 'BenOpp1':
                resfile = resfile + '_BenOpp1-chemtables'
        
    #resfile = resfile + misctail
    if ptypeQ == None:
        print('saving W result to: '+resfile+'\n')
    else: 
        print('saving Q result to: '+resfile+'\n')
    return resfile



def inputcheck(simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, \
         ptypeW,\
         ionW, abundsW, quantityW,\
         ionQ, abundsQ, quantityQ, ptypeQ,\
         excludeSFRW, excludeSFRQ, parttype,\
         theta, phi, psi, \
         var, axis, log, velcut,\
         periodic, kernel, saveres,\
         simulation, LsinMpc,\
         select, misc, ompproj):
             
    '''
    Checks the input to make_map();
    This is not an exhaustive check; it does handle the default/auto options
    return numbers are not ordered; just search <return ##>    
    '''
    # max used number: 34
    
    # basic type and valid option checks
    if type(var) != str:
        print (' should be a string.\n')
        return 1
    if kernel not in ol.kernel_list:
        print('%s is not a kernel option. Options are: \n' % kernel)
        print ol.kernel_list
        return 2    
    if axis not in ['x','y','z']:
        print('Axis must be "x", "y", or "z".')
        return 11    
    if (theta,psi,phi) != (0.,0.,0.):
        print('Warning: rotation is not implemented in this code!\n  Using zero rotation version') 
    if type(periodic) != bool:
        print('periodic should be True or False.\n')
        return 12
    if type(log) != bool:
        print('log should be True or False.\n')
        return 13
    if type(saveres) != bool:
        print('saveres should be True or False.\n')
        return 14
    if type(velcut) != bool:
        print('velcut should be True or False.\n')
        return 15
    if type(snapnum) != int:
        print('snapnum should be an integer.\n')
        return 21
    if type(simnum) != str:
        print('simnum should be a string')
        return 22
    if (type(centre[0]) != float and type(centre[0]) != int) or (type(centre[1]) != float and type(centre[1]) != int) or (type(centre[2]) != float and type(centre[2]) != int):
        print('centre should contain 3 floats')
        return 29
    if type(ompproj) != bool:
        if ompproj == 1:
            ompproj = True
        elif ompproj == 0:
            ompproj = False
        else:
            print('ompproj should be a boolean or 0 or 1')
            return 32
      

    if simulation not in ['eagle', 'bahamas', 'Eagle', 'Bahamas', 'EAGLE', 'BAHAMAS', 'eagle-ioneq']:
        print('Simulation %s is not a valid choice; should be "eagle", "eagle-ioneq" or "bahamas"'%str(simulation))
        return 30
    elif simulation == 'Eagle' or simulation == 'EAGLE':
        simulation = 'eagle'
        print('Preferred form of simulation names is all lowercase (%s)'%simulation)
    elif simulation == 'Bahamas' or simulation == 'BAHAMAS':
        simulation = 'bahamas'
        print('Preferred form of simulation names is all lowercase (%s)'%simulation)

    if (simulation == 'eagle' or simulation == 'eagle-ioneq') and LsinMpc is None:
        LsinMpc = True
    elif simulation == 'bahamas' and LsinMpc is None:
        LsinMpc = False

    if simulation == 'eagle' and (len(simnum) != 10 or simnum[0] != 'L' or simnum[5] != 'N'):
        print('incorrect simnum format %s; should be L####N#### for eagle\n'%simnum)
        return 23
    elif simulation == 'bahamas' and (simnum[0] != 'L' or simnum[4] != 'N'):
        print('incorrect simnum format %s; should be L*N* for bahamas\n'%simnum)
        return 31
    elif simulation == 'eagle-ioneq' and simnum != 'L0025N0752':
        print('For eagle-ioneq, only L0025N0752 is avilable')
        return 33

    centre = [float(centre[0]),float(centre[1]),float(centre[2])]
    if (type(L_x) != float and type(L_x) != int) or (type(L_y) != float and type(L_y) != int) or (type(L_z) != float and type(L_z) != int):
        print('L_x, L_y, and L_z should be floats')
        return 24
    L_x, L_y, L_z = (float(L_x),float(L_y),float(L_z))
    if type(npix_x) != int or type(npix_y) != int or npix_x < 1 or npix_y <1:
        print('npix_x, npix_y should be positive integers')
        return 25    

    # combination-dependent checks
    if var == 'auto':
        if simnum == 'L0025N0752' and simulation != 'eagle-ioneq':
            var = 'RECALIBRATED'
        else: 
            var = 'REFERENCE'

    if misc is not None: # if if if : if we want to use chemical abundances from Ben' Oppenheimer's recal variations
        if 'usechemabundtables' in misc:
            if misc['usechemabundtables'] == 'BenOpp1':
                if simulation != 'eagle-ioneq':
                    print('chemical abundance tables are only avaiable for the eagle-ioneq simulation')
                    return 34
                if ptypeW == 'coldens':
                    if 'Sm' in abundsW:
                        print('chemical abundance tables are only for particle abundances')
                        return 34
                    elif abundsW in ['auto', None]:
                        abundsW = 'Pt'
                if ptypeQ == 'coldens':
                    if 'Sm' in abundsQ:
                        print('chemical abundance tables are only for particle abundances')
                        return 34
                    elif abundsQ in ['auto', None]:
                        abundsQ = 'Pt'
                
     
    iseltQ, iseltW = (False, False)

    if ptypeW not in ['emission', 'coldens', 'basic']:
        print('ptypeW should be one of emission, coldens, or basic (str).\n')
        return 3
    elif ptypeW in ['emission','coldens']:
        parttype = '0'
        if ionW in ol.elements_ion.keys():            
            iseltW = False
        elif ionW in ol.elements and ptypeW == 'coldens':
            iseltW = True
        else:
            print('%s is an invalid ion option for ptypeW %s\n'%(ionW,ptypeW))
            return 26
        if type(abundsW) not in [list,tuple,np.ndarray]:
            abundsW = [abundsW,'auto']
        else:
            abundsW = list(abundsW) # tuple element assigment is not allowed, sometimes needed 
        if abundsW[0] not in ['Sm','Pt','auto']:
            if type(abundsW[0]) not in [float, int]:
                print('Abundances must be either smoothed ("Sm") or particle ("Pt") abundances, automatic ("auto"), or a solar units abundance (float)')
                return 4
            elif iseltW:
                abundsW[0] = abundsW[0] * ol.solar_abunds_ea[ionW]
            else:
                abundsW[0] = abundsW[0] * ol.solar_abunds_ea[ol.elements_ion[ionW]]
        elif abundsW[0] == 'auto':
            if ptypeW == 'emission':
                abundsW[0] = 'Sm'
            else:
                abundsW[0] = 'Pt'
        if abundsW[1] not in ['Sm','Pt','auto']:
            if type(abundsW[1]) not in [float, int]:
                print('Abundances must be either smoothed ("Sm") or particle ("Pt") abundances, automatic ("auto"), or a solar units abundance (float)')
                return 30
        elif abundsW[1] == 'auto':
            if type(abundsW[0]) == float:
                abundsW[1] = 0.752 # if element abundance is fixed, use primordial hydrogen abundance                
            else: 
                abundsW[1] = abundsW[0]
    else:
        if quantityW is None:
            print('For pytpeW basic, quantityW must be specified.\n')
            return 5
        elif type(quantityW) != str:
            print('quantityW must be a string.\n')
            return 6
    

    
            
    if ptypeQ not in ['emission', 'coldens', 'basic',None]:
        print('ptypeQ should be one of emission, coldens, basic (str), or None.\n')
        return 7

    elif ptypeQ in ['emission','coldens']:
        parttype = '0'
        if ionQ in ol.elements_ion.keys():            
            iseltQ = False
        elif ionQ in ol.elements and ptypeQ == 'coldens':
            iseltQ = True
        else:
            print('%s is an invalid ion option for ptypeQ %s\n'%(ionQ,ptypeQ))
            return 8

        if type(abundsQ) not in [list,tuple,np.ndarray]:
            abundsQ = [abundsQ,'auto'] 
        else:
            abundsQ = list(abundsQ)
        if abundsQ[0] not in ['Sm','Pt','auto']:
            if type(abundsQ[0]) not in [float, int]:
                print('Abundances must be either smoothed ("Sm") or particle ("Pt") abundances, automatic ("auto"), or a solar units abundance (float)')
                return 9
            elif iseltQ:
                abundsQ[0] = abundsQ[0] * ol.solar_abunds_ea[ionQ]
            else:
                abundsQ[0] = abundsQ[0] * ol.solar_abunds_ea[ol.elements_ion[ionQ]]
        elif abundsQ[0] == 'auto':
            if ptypeQ == 'emission':
                 abundsQ[0] = 'Sm'
            else:
                abundsQ[0] = 'Pt'
        if abundsQ[1] not in ['Sm','Pt','auto']:
            if type(abundsQ[1]) not in [float, int]:
                print('Abundances must be either smoothed ("Sm") or particle ("Pt") abundances, automatic ("auto"), or a solar units abundance (float)')
                return 28
        elif abundsQ[1] == 'auto':
            if type(abundsQ[0]) == float:
                abundsQ[1] = 0.752 # if element abundance is fixed, use primordial hydrogen abundance                
            else: 
                abundsQ[1] = abundsQ[0]

    elif type(quantityQ) != str and quantityQ is not None:
            print('quantityQ must be a string or None.\n')
            return 27
    
        
    if ptypeW == 'basic' or ptypeQ == 'basic':
        if parttype not in ['0','1','4','5']: # parttype only matters if it is used
            if parttype in [0,1,4,5]:
                parttype = str(parttype)
            else:
                print('parttype should be "0", "1", "4", or "5" (str).\n')
                return 16
                
    if excludeSFRW not in [True,False,'T4','only']:
        if excludeSFRW != 'from':
            print('Invalid option for excludeSFRW: %s'%excludeSFRW)
            return 17
        elif not (ptypeW == 'emission' and ionW == 'halpha'):
            excludeSFRW = 'only'
            print('Unless calculation is for halpha emission, fromSFR will default to onlySFR.\n')
        
    if ptypeQ is not None:
        if (excludeSFRW in [False,'T4']) and (excludeSFRQ not in [False,'T4']):
            print('ExcludeSFR options %s and %s are not compatible'%(excludeSFRW,excludeSFRQ))
            return 18
        elif excludeSFRW in ['from','only']:
            if excludeSFRQ not in ['from', 'only']:
                print('ExcludeSFR options %s and %s are not compatible'%(excludeSFRW,excludeSFRQ))
                return 19
            elif excludeSFRQ == 'from' and not (ptypeQ == 'emission' and ionQ == 'halpha'):
                excludeSFRQ = 'only'
                print('Unless calculation is for halpha emission, fromSFR will default to onlySFR.\n')
                
        elif excludeSFRW != excludeSFRQ and excludeSFRW == True:    
            print('ExcludeSFR options %s and %s are not compatible'%(excludeSFRW,excludeSFRQ))
            return 20
            
    if parttype != '0': #EOS is only relevant for parttype 0 (gas)
        excludeSFRW = False
        excludeSFRQ = False        
        
    # if nothing has gone wrong, return all input, since setting quantities in functions doesn't work on global variables    
    return 0, iseltW, iseltQ, simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, \
         ptypeW,\
         ionW, abundsW, quantityW,\
         ionQ, abundsQ, quantityQ, ptypeQ,\
         excludeSFRW, excludeSFRQ, parttype,\
         theta, phi, psi, \
         var, axis, log, velcut,\
         periodic, kernel, saveres,\
         simulation, LsinMpc, misc, ompproj



def partselect_pos(simfile, centre, Ls, Axis1, Axis2, Axis3, parttype='0'): # this is handled by read_eagle; the hash tables are sufficient to specify the region

    '''
    Uses the read_eagle hash tables to select a region to be used on read-ins
    '''
    hconst = simfile.h
    BoxSize = simfile.boxsize
    if Ls[Axis1] >= BoxSize * hconst**-1 and Ls[Axis2] >= BoxSize * hconst**-1:
        if Ls[Axis3] >= BoxSize * hconst**-1:
            region = None
        else:
            region = np.array([0., BoxSize, 0., BoxSize, 0., BoxSize])
            region[[2*Axis3, 2*Axis3+1]] = [(centre[Axis3]-Ls[Axis3]/2.)*hconst, (centre[Axis3]+Ls[Axis3]/2.)*hconst]
    
    else : 
        region = np.array([0., BoxSize, 0., BoxSize, 0., BoxSize])
        region[[2*Axis3, 2*Axis3+1]] = [(centre[Axis3]-Ls[Axis3]/2.)*hconst, (centre[Axis3]+Ls[Axis3]/2.)*hconst]
        lsmooth = simfile.readarray('PartType%s/SmoothingLength'%parttype, rawunits=True, region = region)
        margin = np.max(lsmooth)
        del lsmooth # read it in again later, if it's needed again
        region[[2*Axis2, 2*Axis2+1] ] = [(centre[Axis2]-Ls[Axis2]/2.)*hconst - margin ,(centre[Axis2]+Ls[Axis2]/2.)*hconst + margin]
        region[[2*Axis1, 2*Axis1+1]]  = [(centre[Axis1]-Ls[Axis1]/2.)*hconst - margin ,(centre[Axis1]+Ls[Axis1]/2.)*hconst + margin]
    print region
    return region


def partselect_vel_region(simfile, centre, Ls, Axis1, Axis2, Axis3, parttype = '0'):  # obsolete
    '''
    returns a region for particle read-in (as does partselect_pos),
    and the velocity coordinates along the projection direction
    '''              
    hconst = simfile.h
    BoxSize = simfile.boxsize
    z = simfile.z
    hf = Hubble(z,simfile=simfile)
    
    # full selection in position directions (Axis0, Axis1)
    Ls_pos = np.copy(Ls)
    Ls_pos[Axis3] = BoxSize
    region = partselect_pos(simfile, centre, Ls_pos, Axis1, Axis2, Axis3, parttype=parttype)
    print 'partselect vel region :', region 

    # further selection: only need velocity in the projection direction
    velp = simfile.readarray('PartType%s/Velocity'%parttype, rawunits=True,region=region)[:,Axis3] 
    vconv = simfile.a **simfile.a_scaling * (simfile.h ** simfile.h_scaling) * simfile.CGSconversion
    maxdzvelp = np.max(np.abs(velp))*vconv/hf/(hconst**-1*simfile.a*c.unitlength_in_cm) #convert gadget velocity to cgs velocity to cgs position to gadget comoving coordinate units
    del velp # read in again with new region in ppv_selselect_coordsgen
    
    region[[2*Axis3, 2*Axis3+1]] = [ (centre[Axis3]-Ls[Axis3]/2.)*hconst - maxdzvelp ,(centre[Axis3]+Ls[Axis3]/2.)*hconst + maxdzvelp]
    print('Velocity space selection spatial region: %s'%str(region))
        
    return region


def ppv_selselect_coordsgen(centre, Ls, Axis1, Axis2, Axis3, periodic, vardict, parttype = '0'): # changed to only need region input
    '''
    separated into a function, but needs to be used carefully
    given position-equivalent velocity space selection, 
    partselect_vel_region output,
    adds coordinates, box3, etc. in position-position-velocity space to vardict
    '''
    hconst = vardict.simfile.h
    BoxSize = vardict.simfile.boxsize
    hf = Hubble(vardict.simfile.z,simfile=vardict.simfile)
    
    # coords are stored in float64, but I don't need that precision here. (As long as particles end up in about the same slice, it should be ok. It's converted to float32 for projection anyway)
    coords = vardict.simfile.readarray('PartType%s/Coordinates'%parttype,rawunits=True).astype(np.float32)[vardict.readsel.val,:] # hubble is also in cgs; ~1e24 cm length coordinates shouldn't overflow
    conv = vardict.simfile.a ** vardict.simfile.a_scaling * (vardict.simfile.h ** vardict.simfile.h_scaling) * vardict.simfile.CGSconversion
    vardict.add_part('coords_cMpc-vel',coords)
    del coords
    
    vardict.readif('Velocity', rawunits=True,region=vardict.region)
    vconv = vardict.simfile.a **vardict.simfile.a_scaling * (vardict.simfile.h ** vardict.simfile.h_scaling) * vardict.simfile.CGSconversion
    vardict.particle['Velocity'] = vardict.particle['Velocity'][:,Axis3] * vconv

    boxvel = BoxSize*conv*hf
    vardict.particle['coords_cMpc-vel'][:,Axis3] *= conv*hf
    vardict.particle['coords_cMpc-vel'][:,Axis3] += vardict.particle['Velocity']
    vardict.delif('Velocity',last=True)
    vardict.particle['coords_cMpc-vel'][:,Axis3] %= boxvel
    
    # avoid fancy indexing to avoid array copies
    vardict.particle['coords_cMpc-vel'][:,Axis1] *= hconst**-1
    vardict.particle['coords_cMpc-vel'][:,Axis2] *= hconst**-1
    BoxSize = BoxSize * hconst**-1        
        
    Ls[Axis3] = Ls[Axis3]/BoxSize * boxvel
    centre[Axis3] = centre[Axis3]/BoxSize * boxvel
    box3 = list((BoxSize,)*3)
    box3[Axis3] = boxvel
    
    translate(vardict.particle, 'coords_cMpc-vel', centre, box3, periodic)
    # This selection can miss particles at the edge due to numerical errors
    # this is shown by testing on a whole box projection with Readfileclone,
    # with a box designed to create these edges cases
    #
    # using >=  and <= runs the risk of double-counting particles when slicing 
    # the box, however, so this is not an ideal solution either. 
    #
    # tests on the 12.5 cMpc box showed no loss of particles on the whole box 
    # selection
    if periodic:
        sel = Sel({'arr': 0.5*boxvel - 0.5*Ls[Axis3] <= vardict.particle['coords_cMpc-vel'][:,Axis3]})
        sel.comb({'arr':  0.5*boxvel + 0.5*Ls[Axis3] >  vardict.particle['coords_cMpc-vel'][:,Axis3]})
    else:
        sel = Sel({'arr': -0.5*Ls[Axis3] <= vardict.particle['coords_cMpc-vel'][:,Axis3]})
        sel.comb({'arr':   0.5*Ls[Axis3] >  vardict.particle['coords_cMpc-vel'][:,Axis3]})

    
    vardict.add_box('box3',box3)
    vardict.overwrite_box('centre',centre)
    vardict.overwrite_box('Ls',Ls)
    vardict.update(sel)


def ppp_selselect_coordsadd(centre, Ls, periodic, vardict, parttype = '0',keepcoords=True): # changed to only need region input
    '''
    separated into a function, but needs to be used carefully
    adds coordinates, box3, etc. in position space to vardict 
    (depending on wishlist setting in vardict)
    periodic refers to the project and is only used in setting the target 
    range in coordinate translation. The coordinates are always taken to be 
    periodic
    '''

    BoxSize = vardict.simfile.boxsize/vardict.simfile.h
    
    # coords are stored in float64, but I don't need that precision here. (As long as particles end up in about the same slice, it should be ok. It's converted to float32 for projection anyway)
    coords = vardict.simfile.readarray('PartType%s/Coordinates'%parttype, rawunits=True,region=vardict.region).astype(np.float32) 
    vardict.readif('SmoothingLength',rawunits=True)
    lmax= np.max(vardict.particle['SmoothingLength'])
    vardict.delif('SmoothingLength')
    hconst = vardict.simfile.h
    lmax /= hconst
    coords /= hconst # convert to Mpc from Mpc/h
    box3 = list((BoxSize,)*3)

    vardict.add_part('coords_cMpc-vel',coords)
    
    translate(vardict.particle, 'coords_cMpc-vel', centre, box3, periodic)

    doselection = np.array([0,1,2])[np.array(Ls) < BoxSize]
    if len(doselection) > 0:
        if periodic:
            i = doselection[0]
            sel = Sel({'arr': 0.5*BoxSize - 0.5*Ls[i] - lmax <= vardict.particle['coords_cMpc-vel'][:,i]})
            sel.comb({'arr':  0.5*BoxSize + 0.5*Ls[i] + lmax >  vardict.particle['coords_cMpc-vel'][:,i]})
            if len(doselection) > 1:
                for i in doselection[1:]:
                    sel.comb({'arr':  0.5*BoxSize + 0.5*Ls[i] + lmax >  vardict.particle['coords_cMpc-vel'][:,i]})
                    sel.comb({'arr':  0.5*BoxSize - 0.5*Ls[i] - lmax <= vardict.particle['coords_cMpc-vel'][:,i]})
        else:
            i = doselection[0]
            sel = Sel({'arr':  -0.5*Ls[i] - lmax <= vardict.particle['coords_cMpc-vel'][:,i]})
            sel.comb({'arr':    0.5*Ls[i] + lmax >  vardict.particle['coords_cMpc-vel'][:,i]})
            if len(doselection) > 1:
                for i in doselection[1:]:
                    sel.comb({'arr':   0.5*Ls[i] + lmax >  vardict.particle['coords_cMpc-vel'][:,i]})
                    sel.comb({'arr':  -0.5*Ls[i] - lmax <= vardict.particle['coords_cMpc-vel'][:,i]})
    else:
        sel = Sel() # default empty selection if we're doing the whole box
        
    vardict.add_box('box3',box3)
    vardict.add_box('centre',centre)
    vardict.add_box('Ls',Ls)
    vardict.delif('coords_cMpc-vel',last= not keepcoords)
    vardict.update(sel)



##### small helper functions for the main projection routine

def get_eltab_names(abunds,iselt,ion): #assumes 
    if abunds[0] == 'Sm':
        if not iselt:            
            eltab = 'SmoothedElementAbundance/%s' %string.capwords(ol.elements_ion[ion])  
        else:
            eltab = 'SmoothedElementAbundance/%s' %string.capwords(ion)
    elif abunds[0] =='Pt': # auto already set in inputcheck
        if not iselt:            
            eltab = 'ElementAbundance/%s' %string.capwords(ol.elements_ion[ion])
        else:
            eltab = 'ElementAbundance/%s' %string.capwords(ion)
    else: 
        eltab = abunds[0] #float

    if abunds[1] == 'Sm':
        hab = 'SmoothedElementAbundance/Hydrogen'
    elif abunds[1] =='Pt': # auto already set in inputcheck
        hab = 'ElementAbundance/Hydrogen'
    else: 
       hab = abunds[1] #float            

    return eltab, hab
    
#################
#    classes    #
#################

class Simfile:
    '''
    Presently, a thin wrapper for read_eagle_files.readfile
    intended to possibly expand read-in to e.g. C-EAGLE/Hydrangea, OWLS, BAHAMAS:
    contain more or less the same data (so that no rewrite of the calculations is needed),
    but have different read-in options (regions or just selections) and libraries

    anything beyond raw data passing is handled by Vardict. 
    if necessary, e.g. tables or separate files can be used to set e.g., a, h, scalings
    for non-EAGLE outputs
    if region handling is not included, Vardict and the selections should be given a case
    or option to handle that, rather than using consecutive selections on each array  
    '''
    def readarray_eagle(self,name,region=None,rawunits=False):
        arr = self.readfile.read_data_array(name, gadgetunits=rawunits, suppress=False,region=region)
        self.a_scaling = self.readfile.a_scaling 
        self.h_scaling = self.readfile.h_scaling
        self.CGSconversion = self.readfile.CGSconversion
        self.CGSconvtot = self.a**self.a_scaling * self.h**self.h_scaling * self.CGSconversion
        return arr
    def readarray_bahamas(self,name,region=None,rawunits=False): #region is useless here
        if region is not None:
            print('Warning (readarray_bahamas): region selection will not have any effect')
        arr = self.readfile.read_var(name, gadgetunits=rawunits, verbose=True)
        # CGS conversion should be safe to just take from the first file
        self.CGSconvtot = self.readfile.convert_cgs(name, 0, verbose=True)
        self.a_scaling = self.readfile.a_scaling 
        self.h_scaling = self.readfile.h_scaling
        self.CGSconversion = self.readfile.CGSconversion
        return arr
        
    def __init__(self,simnum,snapnum,var,file_type = ol.file_type,simulation = 'eagle'):
        if simulation == 'eagle':
            import read_eagle_files as eag
            simdir  = ol.simdir_eagle%simnum + '/' + var 
            self.readfile = eag.read_eagle_file(simdir, file_type, snapnum, gadgetunits=True, suppress=False)
            # pass down readfile properties for reference 
            self.boxsize = self.readfile.boxsize   
            self.h = self.readfile.h
            self.a = self.readfile.a
            self.z = 1./self.a -1.
            # omegam and omegalambda are not retrieved by read_eagle, but are needed to get the hubble parameter H(z)
            # try cases are extracted from read_eagle files and read_bahamas_files
            self.filenamebase = self.readfile.fname
            try:
                self.hdf5file = h5py.File( self.filenamebase+"0.hdf5", 'r' )
            except:
                self.hdf5file = h5py.File( self.filenamebase+"hdf5", 'r' )
            self.omegam = self.hdf5file['Header'].attrs['Omega0']
            self.omegalambda = self.hdf5file['Header'].attrs['OmegaLambda']
            self.omegab = self.hdf5file['Header'].attrs['OmegaBaryon']
            self.hdf5file.close()
            
            self.region_supported = True
            self.readarray = self.readarray_eagle

        elif simulation == 'eagle-ioneq':
            import read_eagle_files_noneq_noregion as refn #modified read_bahamas_files
            simdir  = ol.simdir_eagle_noneq
            if var == 'REFERENCE':
                ioneq_str = 'ioneq_'
            elif var == 'ssh':
                ioneq_str = 'ioneq_SS_'
            else:
                print('var option %s is not valid for eagle-ioneq'%var)
                return None
            print simdir, file_type, snapnum, ioneq_str
            self.readfile = refn.Gadget(simdir, file_type, snapnum, gadgetunits=True, suppress=False, sim = 'EAGLE-IONEQ', ioneq_str=ioneq_str, add_data_dir = False)
            # pass down readfile properties for reference 
            self.boxsize = self.readfile.boxsize   
            self.h = self.readfile.h
            self.a = self.readfile.a
            self.z = 1./self.a -1.
            # omegam and omegalambda are not retrieved by read_eagle, but are needed to get the hubble parameter H(z)
            # try cases are extracted from read_eagle files and read_bahamas_files
            self.filenamebase = self.readfile.filename
            try:
                self.hdf5file = h5py.File( self.filenamebase+"0.hdf5", 'r' )
            except:
                self.hdf5file = h5py.File( self.filenamebase+"hdf5", 'r' )
            self.omegam = self.hdf5file['Header'].attrs['Omega0']
            self.omegalambda = self.hdf5file['Header'].attrs['OmegaLambda']
            self.omegab = self.hdf5file['Header'].attrs['OmegaBaryon']
            self.hdf5file.close()
            
            self.region_supported = False
            self.readarray = self.readarray_bahamas # modified read_bahamas_files has same methods

        elif simulation == 'bahamas':
            import read_bahamas_files as bhm #also has OWLS option
            simdir = ol.simdir_bahamas%simnum
            self.readfile = bhm.Gadget(simdir, file_type, snapnum, gadgetunits=True, verbose=True,sim = 'BAHAMAS')
            self.boxsize = self.readfile.boxsize   
            self.h = self.readfile.h
            self.a = self.readfile.a
            self.z = 1./self.a -1.
            # omegam and omegalambda are not retrieved by read_eagle, but are needed to get the hubble parameter H(z)
            # try cases are extracted from read_eagle files and read_bahamas_files
            self.filenamebase = self.readfile.filename
            try:
                self.hdf5file = h5py.File( self.filenamebase+"0.hdf5", 'r' )
            except:
                self.hdf5file = h5py.File( self.filenamebase+"hdf5", 'r' )
            self.omegam = self.hdf5file['Header'].attrs['Omega0']
            self.omegalambda = self.hdf5file['Header'].attrs['OmegaLambda']
            self.omegab = self.hdf5file['Header'].attrs['OmegaBaryon']
            self.hdf5file.close()
            
            self.region_supported = False
            self.readarray = self.readarray_bahamas
        else:
            print('Simulation %s is not supported; Simfile object will be useless.'%simulation)
        
class Sel:
    '''
    an array mask for 1D arrays (selections):
    default value is slice(None,None,None), can be set as a boolean array
    can be combined with other 1D masks through logical_and 
    
    just here as a conveniet way to initalise particle selection for unknown
    array sizes
    
    key: if input is a dict containing the array, key to use to retrieve it 
    note: if a sel instance is input, no deep copy is made - its arrays can be 
         modified in-function
    '''
    # asarray uses copy=False
    def __init__(self,arr=None,key='arr'):
        if isinstance(arr,np.ndarray):
            self.seldef = True
            self.val = np.asarray(arr,dtype=bool) 
        elif isinstance(arr,dict):
            if key in arr.keys():
                self.seldef = True
                self.val = np.asarray(arr[key],dtype=bool)
            elif len(arr.keys()) ==1: # must the the one element, right?
                key_old = key
                key = arr.keys()[0]
                if key_old is not None:
                    print('Warning: invalid key %s for dictionary input to Sel.__init__; using only key %s in stead.'%(str(key_old), str(key)))
                self.seldef = True
                self.val = np.asarray(arr[key],dtype=bool)
            else:
                print('Error: invalid key %s for dictionary input to Sel.__init__')    
        else: # default setting
            self.seldef = False
            self.val = slice(None,None,None)     
    def __str__(self):
        return 'Sel(%s, seldef=%s)'%(str(self.val),str(self.seldef))

    def comb(self,arr,key=None): # for combining two selections of the same length into the first

        if isinstance(arr,Sel):
            if arr.seldef and self.seldef: # if arr is a non-default Sel instance, just use the array value
                 self.val &= arr.val
            elif not self.seldef:
                self.val = arr.val
                self.seldef = arr.seldef
            else: #arr.seldef, self.seldef == False
                pass # Sel instance is unaltered

        elif isinstance(arr,np.ndarray): # if the instance in non-default, combine with any arr value, otherwise replace the instance 
            if self.seldef:                
                self.val &= np.asarray(arr,dtype=bool)   #self.val  = np.logical_and(self.val,arr) 
            else:
                self.val = np.asarray(arr,dtype=bool)
            self.seldef = True
                
        elif isinstance(arr,dict):
            if key in arr.keys():
                if isinstance(arr[key],np.ndarray):
                    if not self.seldef:
                        self.val = np.asarray(arr[key],dtype=bool)
                        self.seldef = True
                    else:
                        self.val &= np.asarray(arr[key],dtype=bool)
                elif isinstance(arr[key],Sel): # might as well allow a Sel from a dictionary; contained array is not copied on call, so recursion won't waste memory
                    self.comb(arr[key])
                else: 
                    print('Error (Sel.comb): can not combine Sel instance with object %s; Sel instance unmodified'%type(arr[key]))
            elif len(arr.keys()) ==1: # must the the one element, right?
                key_old = key
                key = arr.keys()[0]
                if key_old is not None:
                    print('Warning (Sel.comb): invalid key %s for dictionary input to Sel.comb; using only key %s in stead.'%(str(key_old), str(key)))
                if isinstance(arr[key],np.ndarray):
                    if not self.seldef:
                        self.val = np.asarray(arr[key],dtype=bool)
                        self.seldef = True
                    else:
                        self.val &= np.asarray(arr[key],dtype=bool)
                elif isinstance(arr[key],Sel):
                    self.comb(arr[key])
                else: 
                    print('Error (Sel.comb): can not combine Sel instance with object %s; Sel instance unmodified'%type(arr[key]))
            else:
                print('Error (Sel.comb): invalid key %s for dictionary input to Sel.comb; Sel instance unmodified'%str(key))    
                
        else:
            print('Error (Sel.comb): arr must be a Sel instance, compatible array, or dict containing such an array')


    def refine(self,arr,key=None): # comb for when the second array is a selection of the first's True indices
    #        if isinstance(arr,Sel) and arr.seldef: # if arr is a non-default Sel instance, just use the array value; non-initalised Sel does nothing
    #            self.refine(arr.val)                    
    #        elif not isinstance(arr,Sel): # if the instance is non-default, combine with any arr value, otherwise replace the instance 
    #            if self.seldef:                
    #                self.val[self.val] = arr
    #            else:
    #               self.val = arr
    #            self.seldef = True

        if isinstance(arr,Sel):
            if arr.seldef and self.seldef: # if arr is a non-default Sel instance, just use the array value
                 self.val[self.val] = arr.val
            elif not self.seldef:
                self.val = arr.val
                self.seldef = arr.seldef
            else: #arr.seldef, self.seldef == False
                pass # Sel instance is unaltered

        elif isinstance(arr,np.ndarray): # if the instance in non-default, combine with any arr value, otherwise replace the instance 
            if self.seldef:                
                self.val[self.val] = np.asarray(arr,dtype=bool)   #self.val  = np.logical_and(self.val,arr) 
            else:
                self.val = np.asarray(arr,dtype=bool)
            self.seldef = True
                
        elif isinstance(arr,dict):
            if key in arr.keys():
                if isinstance(arr[key],np.ndarray):
                    if not self.seldef:
                        self.val = np.asarray(arr[key],dtype=bool)
                        self.seldef = True
                    else:
                        self.val[self.val] = np.asarray(arr[key],dtype=bool)
                elif isinstance(arr[key],Sel): # might as well allow a Sel from a dictionary; contained array is not copied on call, so recursion won't waste memory
                    self.refine(arr[key])
                else: 
                    print('Error (Sel.refine): can not combine Sel instance with object %s; Sel instance unmodified'%type(arr[key]))
            elif len(arr.keys()) ==1: # must the the one element, right?
                key_old = key
                key = arr.keys()[0]
                if key_old is not None:
                    print('Warning (Sel.refine): invalid key %s for dictionary input to Sel.comb; using only key %s in stead.'%(str(key_old), str(key)))
                if isinstance(arr[key],np.ndarray):
                    if not self.seldef:
                        self.val = np.asarray(arr[key],dtype=bool)
                        self.seldef = True
                    else:
                        self.val[self.val] = np.asarray(arr[key],dtype=bool)
                elif isinstance(arr[key],Sel): # might as well allow a Sel from a dictionary; contained array is not copied on call, so recursion won't waste memory
                    self.refine(arr[key])
                else: 
                    print('Error (Sel.refine): can not combine Sel instance with object %s; Sel instance unmodified'%type(arr[key]))
            else:
                print('Error (Sel.refine): invalid key %s for dictionary input to Sel.refine; Sel instance unmodified'%str(key))    
                
        else:
            print('Error (Sel.comb): arr must be a Sel instance, compatible array, or dict containing such an array')

    def reset(self):
        self.seldef = False
        self.val = slice(None,None,None)

        

class Vardict:
    '''
    Stores read-in and derived variables in a dictionary, and keeps track of
        what will still be needed in the next calculation
    In the next calculation, then tracks what still needs to be read in, and
        what is still stored
    wishlist: variables desired after the calculation (depends on both W and Q, 
        so construction is done separately); list
    parttype: PartType as in the EAGLE output (see project docstring); string
    simfile: Simfile instance - tells the function where to find its 
        data
    sel:      selection relative to total stored array in region: used on read-ins
    '''

    def __init__(self,simfile,parttype,wishlist,region = None, readsel = None):
        self.simfile = simfile
        self.parttype = parttype
        self.wishlist = wishlist
        self.particle = {} # name:var dictionary of particle attributes
        self.box = {}      # name:var dictionary of box attributes
        self.CGSconv = {}  #name:units dictionary (stores factor by which to multiply corresponding array to get CGS)
        # convenient to store as default    
        self.region = region 
        if readsel is None:
            self.readsel = Sel()
        elif isinstance(readsel,Sel):
            self.readsel = readsel
        else:
            self.readsel = Sel(readsel)
            
    def delif(self,name, last = False): 
        '''
        deletes a variable if it is not on a list of variables to save, 
        and removes corresponding dictionary entry, always if it is the last use
        only works on particle quantities; 
        box quantities are not worth the trouble of deleting
        '''
        if name not in self.wishlist or last:
            del self.particle[name] # remove dictionary element           
    
    def isstored_box(self,name):
        return name in self.box.keys()
    
    def isstored_part(self,name):
        return name in self.particle.keys()
     
    def readif(self,name,region = 'auto',rawunits = False,out =False, setsel = None,setval = None):  # reads in name (of parttype) from EAGLE if it is not already read in (on lst), 
        '''
        reads in data that is not already read in, using a dictionary to trak read-in particles
        name:     name of EAGLE output to read in (not including 'PartType/') (str)
        parttype: parttype of array to read in (str)
        region, gadgetunits: as in read_data_array
        sel :     a Sel instance  
        cnvfact:  save cgs conversion factor to this variable unless set to None
        out:      saves result to the variable if not None
        setsel and setval: sets values in mask setsel to setval if setval is not None 
            !! cnvfact is not stored in Vardict
        '''
        if region == 'auto':
            region = self.region
        sel = self.readsel
        if not self.isstored_part(name):
            self.particle[name] = self.simfile.readarray('PartType%s/%s' %(self.parttype,name), rawunits=True,region=region)[sel.val] #coordinates are always needed, so these will not be read in this way; other arrays are 1D
            self.CGSconv[name] = self.simfile.a ** self.simfile.a_scaling * (self.simfile.h ** self.simfile.h_scaling) * self.simfile.CGSconversion                               
            if not rawunits: # do CGS conversion here since read_eagle_files does not seem to modify the array in place
                self.particle[name] *= self.CGSconv[name]
                self.CGSconv[name] = 1.
        if setval != None:
            self.particle[name][setsel] = setval
        if out:
            return self.particle[name]

    def add_part(self,name,var):
        if name in self.particle.keys():
            print('Warning: variable <%s> will be overwritten by Vardict.add_part' %name)
        self.particle[name] = var

    def add_box(self,name,var):
        if name in self.box.keys():
            print('Warning: variable <%s> will be overwritten by Vardict.add_box' %name)
        self.box[name] = var
    
    def update(self,selname,key=None):
        '''
        updates stored particle property arrays to only contain only the new 
        selection elements, and updates the read-in selection
        '''
        if not isinstance(selname,Sel) and not isinstance(selname,dict):
            selname = Sel({'arr': selname},'arr') # if input is a numpy array, don't do a second copy for comb/refine calls
        if selname.seldef and self.readsel.seldef:
            if selname.val.shape == self.readsel.val.shape:
                self.tempsel = selname.val[self.readsel.val]
                for name in self.particle.keys():
                    self.particle[name] = (self.particle[name])[self.tempsel,...] 
                del self.tempsel
                self.readsel.comb(selname)
            elif selname.val.shape[0] == self.particle[self.particle.keys()[0]].shape[0]: # if keys()[0] happens to be coordinates or something, we just want to match the zero index
                for name in self.particle.keys():  
                    self.particle[name] = (self.particle[name])[selname.val,...] 
                self.readsel.refine(selname)
        elif selname.seldef:
            for name in self.particle.keys():
                self.particle[name] = (self.particle[name])[selname.val,...] 
            self.readsel.comb(selname)
        else:
            pass # selname id undefines; no update necessary
    
    def overwrite_part(self,name,var):
        self.particle[name] = var
            
    def overwrite_box(self,name,var):
        self.box[name] = var    

    ## functions to get specific derived particle properties; wishlist generation counts on naming being get<name of property to store>
    # note: each function 'cleans up' all other quantites it uses! adjust wishlists in lumonisity etc. calculation to save quantities needed later on 
    def getlognH(self,last=True,**kwargs):
        if not 'hab' in kwargs.keys():
            print('hab must be specified to calculate lognH')
            return None
        self.readif('Density', rawunits=True)
        if type(kwargs['hab']) == str:
            self.readif(kwargs['hab'],rawunits = True)
            self.add_part('lognH', np.log10(self.particle[kwargs['hab']]) + np.log10(self.particle['Density']) + np.log10( self.CGSconv['Density']/(c.atomw_H*c.u) ) )
            self.delif(kwargs['hab'],last=last)
        else:
            self.add_part('lognH', np.log10(self.particle['Density']) + np.log10(self.CGSconv['Density'] * kwargs['hab']/(c.atomw_H*c.u) ) )
        self.delif('Density',last=last)    
     
    def getpropvol(self,last=True):
        self.readif('Density', rawunits=True)
        self.readif('Mass', rawunits=True)
        self.add_part('propvol', (self.particle['Mass']/self.particle['Density']) *(self.CGSconv['Mass']/self.CGSconv['Density']))
        self.delif('Mass',last=last)
        self.delif('Density',last=last)
    
    def getlogT(self,last=True,**kwargs):
        if kwargs['logT']:
            self.readif('OnEquationOfState',rawunits=True)
            self.add_part('eos',self.particle['OnEquationOfState'] > 0.)
            self.delif('OnEquationOfState',last=last)
            self.readif('Temperature',rawunits=True,setsel = self.particle['eos'],setval = 1e4)
            self.delif('eos',last=last)
        else:
            self.readif('Temperature',rawunits=True)
        self.add_part('logT',np.log10(self.particle['Temperature']))
        self.delif('Temperature',last=last)
        

class Readfileclone: # for testing purposes: contains properties and grids mimicking read_eagle_files readfiles objects
    def __init__(self,z=0.,coords='auto',vel = 'auto',boxsize=10./c.hubbleparam**-1,lsmooth= 1.1/c.hubbleparam**-1):
        self.z = z        
        self.hub = Hubble(self.z)
        self.h = c.hubbleparam
        self.a = 1./(1.+self.z)
        self.boxsize = boxsize

        if coords == 'auto': # 10x10x10 grid of halfway cell centres, scaled to box size
            numcens = 10
            coordsbase = np.indices((numcens,)*3)[0] 
            coordsbase = (np.asarray(coordsbase,dtype=np.float)+0.5)/float(numcens)
            self.Coordinates = np.empty((np.prod(coordsbase.shape),3))
            self.Coordinates[:,0] = np.ndarray.flatten(coordsbase)*self.boxsize
            self.Coordinates[:,1] = np.ndarray.flatten(np.swapaxes(coordsbase,0,1))*self.boxsize
            self.Coordinates[:,2] = np.ndarray.flatten(np.swapaxes(coordsbase,0,2))*self.boxsize
            del coordsbase
        else:
            self.Coordinates = coords

        if vel == 'auto':
            self.Velocity = np.empty(self.Coordinates.shape)
            self.Velocity[:,0] = 100. #km/s
            self.Velocity[:,1] = -50. #km/s
            self.Velocity[:,2] = 150. #km/s
        elif vel.shape == (3,):
            self.Velocity = np.empty(self.Coordinates.shape)
            self.Velocity[:,0] = vel[0] #km/s
            self.Velocity[:,1] = vel[1] #km/s
            self.Velocity[:,2] = vel[2] #km/s            
        else:
            self.Velocity = vel

        self.SmoothingLength = lsmooth*np.ones(self.Coordinates.shape[0])
        self.readarray = self.read_data_array
            
    def read_data_array(self, name, gadgetunits=False, region=None, suppress=False):
        # select correct entry, and set conversion factors
        if 'Coordinates' in name:
            out = np.copy(self.Coordinates)
            if not gadgetunits:
                out *= self.h**-1*self.a*c.unitlength_in_cm
            self.a_scaling = 1
            self.h_scaling = -1 
            self.CGSconversion = c.unitlength_in_cm

        elif 'Velocity' in name:
            out = np.copy(self.Velocity)
            if not gadgetunits:
                out *= self.a**0.5*c.unitvelocity_in_cm_per_s
            self.a_scaling = 0.5
            self.h_scaling = 0
            self.CGSconversion = c.unitvelocity_in_cm_per_s

        elif 'SmoothingLength' in name:
            out = np.copy(self.SmoothingLength)
            if not gadgetunits:
                out *= self.h**-1*self.a*c.unitlength_in_cm
            self.a_scaling = 1
            self.h_scaling = -1 
            self.CGSconversion = c.unitlength_in_cm

        else:
            print('No handling for %s has been implemented'%name)

        #region handling
        self.mask = Sel()
        if region is not None:
            self.mask.comb(region[0] <= self.Coordinates[:,0])
            self.mask.comb(region[1] >  self.Coordinates[:,0])
            self.mask.comb(region[2] <= self.Coordinates[:,1])
            self.mask.comb(region[3] >  self.Coordinates[:,1])
            self.mask.comb(region[4] <= self.Coordinates[:,2]) 
            self.mask.comb(region[5] >  self.Coordinates[:,2])
        if len(out.shape) == 2:
            return out[self.mask.val,:]
        else:
            return out[self.mask.val]    

    def readarray(self,name,region=None,rawunits=False):
        return self.read_data_array(self, name, gadgetunits=rawunits, region=region, suppress=False)
    


#################################
# main functions, using classes #
#################################


def luminosity_calc(vardict,excludeSFR,eltab,hab,ion,last=True,updatesel=True):
    '''
    Calculate the per particle luminosity of an emission line (ion)
    vardict should already contain the particle selection for which to 
    calculate the luminosities 
    last and updatesel defaults set for single use

    At this stage, it only matters if excludeSFR is 'T4' or something else; 
    EOS pre-selection has already been done.
    eltab and hab can either be a (PartType#/ excluded) hdf5 path or an element
    mass fraction (float).
    
    outputs SPH particle line luminosities in erg/s *1e10, 
    and the 1e10 conversion factor back to CGS (prevents risk of float32 
    overflow)
    '''
    print('Calculating particle luminosities...')    
    
    if type(eltab) == str:
        vardict.readif(eltab,rawunits = True)
        if updatesel:
            vardict.update(vardict.particle[eltab] > 0.)

    
    if not vardict.isstored_part('propvol'):
        vardict.readif('Density', rawunits=True)
        vardict.readif('Mass', rawunits=True)
        vardict.add_part('propvol', (vardict.particle['Mass']/vardict.particle['Density']) *(vardict.CGSconv['Mass']/vardict.CGSconv['Density']))
        vardict.delif('Mass',last=last)
    
    print 'Min, max, median of particle volume [cgs]: %.5e %.5e %.5e' \
        % (np.min(vardict.particle['propvol']), np.max(vardict.particle['propvol']), np.median(vardict.particle['propvol']))

    if not vardict.isstored_part('lognH'):
        vardict.readif('Density', rawunits=True)
        if type(hab) == str:
            vardict.readif(hab,rawunits = True)
            vardict.add_part('lognH', np.log10(vardict.particle[hab]) + np.log10(vardict.particle['Density']) + np.log10( vardict.CGSconv['Density']/(c.atomw_H*c.u) ) )
            if eltab != hab:
                vardict.delif(hab,last=last)
        else:
            vardict.add_part('lognH', np.log10(vardict.particle['Density']) + np.log10( vardict.CGSconv['Density'] * hab/(c.atomw_H*c.u) ) )
        vardict.delif('Density',last=last)

    print 'Min, max, median of particle log10 nH [cgs]: %.5e %.5e %.5e' \
        % (np.min(vardict.particle['lognH']), np.max(vardict.particle['lognH']), np.median(vardict.particle['lognH']))    
    
    if not vardict.isstored_part('logT'):
        if excludeSFR == 'T4':
            vardict.readif('OnEquationOfState',rawunits=True)
            vardict.add_part('eos',vardict.particle['OnEquationOfState'] > 0.)
            vardict.delif('OnEquationOfState',last=last)
            vardict.readif('Temperature',rawunits=True,setsel = vardict.particle['eos'],setval = 1e4)
            vardict.delif('eos',last=last)
        else:
            vardict.readif('Temperature',rawunits=True)
        vardict.add_part('logT',np.log10(vardict.particle['Temperature']))
        vardict.delif('Temperature',last=last)
    
    print 'Min, max, median of particle log temperature [K]: %.5e %.5e %.5e' \
        % (np.min(vardict.particle['logT']), np.max(vardict.particle['logT']), np.median(vardict.particle['logT']))
    

    lineind = ol.line_nos_ion[ion]
    vardict.add_part('emdenssq', find_emdenssq(1./vardict.simfile.a-1.,ol.elements_ion[ion],vardict.particle['lognH'],vardict.particle['logT'],lineind))
    print 'Min, max, median of particle emdenssq: %.5e %.5e %.5e' \
        % (np.min(vardict.particle['emdenssq']), np.max(vardict.particle['emdenssq']), np.median(vardict.particle['emdenssq']))
    vardict.delif('logT',last = last)

    # for agreement with Cosmoplotter
    # also: using SPH_KERNEL_GADGET; check what EAGLE uses!!
    #lowZ = eltabund < 10**-15
    #eltabund[lowZ] = 0.

    # using units of 10**-10 * CGS, to make sure overflow of float32 does not occur in C 
    # (max is within 2-3 factors of 10 of float32 overflow in one simulation)    
    if type(eltab) == str:
        luminosity = vardict.particle[eltab]/ol.solar_abunds[ol.elements_ion[ion]]*10**(vardict.particle['emdenssq'] + 2*vardict.particle['lognH'] + np.log10(vardict.particle['propvol']) -10.)   
        vardict.delif(eltab,last=last)
    else:
        luminosity = eltab/ol.solar_abunds[ol.elements_ion[ion]]*10**(vardict.particle['emdenssq'] + 2*vardict.particle['lognH'] + np.log10(vardict.particle['propvol']) -10.)
    vardict.delif('lognH',last=last)
    vardict.delif('emdenssq',last=last)
    vardict.delif('propvol',last=last)
    vardict.delif(eltab,last=last)
    
    print 'Min, max, median of particle lumninosity [1e10 cgs]: %.5e %.5e %.5e' \
        % (np.min(luminosity), np.max(luminosity), np.median(luminosity)) 
    
    CGSconv = 1e10
    print('  done.\n')    

    return luminosity, CGSconv # array, cgsconversion




def lumninosty_to_Sb(vardict,Ls,Axis1,Axis2,Axis3,npix_x,npix_y,ion):
    '''
    converts cgs lumninosity (erg/s) to cgs surface brightness 
    (photons/s/cm2/steradian)
    ion needed because conversion depends on the line energy
    '''
    zcalc = 1./vardict.simfile.a-1.
    comdist = comoving_distance_cm(zcalc,simfile=vardict.simfile)
    longlen = max(Ls)/2. * c.cm_per_mpc  
    if comdist > longlen: # even at larger values, the projection along z-axis = projection along sightline approximation will break down
        ldist = comdist*(1.+zcalc)
        adist = comdist/(1.+zcalc)
    else: 
        ldist = longlen*(1.+zcalc)
        adist = longlen/(1.+zcalc) 

    # conversion (x, y are axis placeholders and may actually repreesnt different axes in the simulation, as with numpix_x, and numpix_y)
    halfangle_x = 0.5*Ls[Axis1]/(1.+zcalc)/npix_x * c.cm_per_mpc/adist
    halfangle_y = 0.5*Ls[Axis2]/(1.+zcalc)/npix_y * c.cm_per_mpc/adist

    #solidangle = 2*np.pi*(1-np.cos(2.*halfangle_x))
    #print ("solid angle per pixel: %f" %solidangle)
    return 1./(4*np.pi*ldist**2)*(1.+zcalc)/ol.line_eng_ion[ion]*1./solidangle(halfangle_x,halfangle_y)



def Nion_calc(vardict,excludeSFR,eltab,hab,ion,last=True,updatesel=True,misc=None):
    ionbal_from_outputs = False
    if misc is not None:
        if 'usechemabundtables' in misc:
            if misc['usechemabundtables'] == 'BenOpp1':
                ionbal_from_outputs = True
      
    if type(eltab) == str:
        vardict.readif(eltab,rawunits=True)
        if updatesel:
            vardict.update(vardict.particle[eltab]>0.)   
    
    if not ionbal_from_outputs: # if not misc option for getting ionfrac from Ben Oppenheimer's modified RECAL-L0025N0752 runs with non-equilibrium ion fractions
        if not vardict.isstored_part('lognH'):
            vardict.readif('Density', rawunits=True)
            if type(hab) == str:
                vardict.readif(hab,rawunits = True)
                vardict.add_part('lognH', np.log10(vardict.particle[hab]) + np.log10(vardict.particle['Density']) + np.log10( vardict.CGSconv['Density']/(c.atomw_H*c.u) ) )
                if eltab != hab:
                    vardict.delif(hab,last=last)
            else:
                vardict.add_part('lognH', np.log10(vardict.particle['Density']) + np.log10( vardict.CGSconv['Density'] * hab/(c.atomw_H*c.u) ) )
            vardict.delif('Density',last=last)   
    
        print 'Min, max, median of particle log10 nH [cgs]: %.5e %.5e %.5e' \
            % (np.min(vardict.particle['lognH']), np.max(vardict.particle['lognH']), np.median(vardict.particle['lognH'])) 
    
        if not vardict.isstored_part('logT'):
            if excludeSFR == 'T4':
                vardict.readif('OnEquationOfState',rawunits=True)
                vardict.add_part('eos',vardict.particle['OnEquationOfState'] > 0.)
                vardict.delif('OnEquationOfState',last=last)
                vardict.readif('Temperature',rawunits=True,setsel = vardict.particle['eos'],setval = 1e4)
                vardict.delif('eos',last=last)
            else:
                vardict.readif('Temperature',rawunits=True)
            vardict.add_part('logT',np.log10(vardict.particle['Temperature']))
            vardict.delif('Temperature',last=last)
    
        print 'Min, max, median of particle log temperature [K]: %.5e %.5e %.5e' \
            % (np.min(vardict.particle['logT']), np.max(vardict.particle['logT']), np.median(vardict.particle['logT']))
    
        vardict.add_part('ionfrac',find_ionbal(1./vardict.simfile.a-1.,ion,vardict.particle['lognH'],vardict.particle['logT']))
    # get ion balance; misc option for en Oppenheimer's ion balamce tables in L0025N0752 modified recal
    else:
        # gets ionfrac entry
        getBenOpp1chemabundtables(vardict,excludeSFR,eltab,hab,ion,last=last,updatesel=updatesel,misc=misc)
    
         

    vardict.readif('Mass',rawunits=True)
    
    if type(eltab) == str:
        Nion = vardict.particle[eltab]*vardict.particle['ionfrac']*vardict.particle['Mass'] 
        vardict.delif('ionfrac',last=last)
        vardict.delif('Mass',last=last)
        vardict.delif(eltab,last=last)
    else: 
        Nion = eltab*vardict.particle['ionfrac']*vardict.particle['Mass'] 
        vardict.delif('ionfrac',last=last)
        vardict.delif('Mass',last=last)

        
    
    ionmass = ionh.atomw[string.capwords(ol.elements_ion[ion])]
    to_cgs_numdens = vardict.CGSconv['Mass']/(ionmass*c.u)
    
    return Nion, to_cgs_numdens # array, cgsconversion
    
    
def Nelt_calc(vardict,excludeSFR,eltab,ion,last=True,updatesel=True):

    if type(eltab) == str:
        vardict.readif(eltab,rawunits=True)
        if updatesel:
            vardict.update(vardict.particle[eltab]>0.)   

    vardict.readif('Mass',rawunits=True)    

    if type(eltab) == str:
        Nelt = vardict.particle[eltab]*vardict.particle['Mass'] 
        vardict.delif('Mass',last=last)
        vardict.delif(eltab,last=last)
    else: 
        Nelt = eltab*vardict.particle['Mass'] 
        vardict.delif('Mass',last=last)


    ionmass = ionh.atomw[string.capwords(ion)]
    to_cgs_numdens = vardict.CGSconv['Mass']/(ionmass*c.u)
    return Nelt, to_cgs_numdens

    
def Nion_to_coldens(vardict,Ls,Axis1,Axis2,Axis3,npix_x,npix_y):
    afact = vardict.simfile.a
    area = (Ls[Axis1]/ np.float32(npix_x)) * (Ls[Axis2] / np.float32(npix_y)) * c.cm_per_mpc ** 2 *afact**2
    return 1./area


def luminosity_calc_halpha_fromSFR(vardict,excludeSFR,last=True,updatesel=True):
    
    if not vardict.isstored_part('eos'):
        vardict.readif('OnEquationOfState', rawunits=True) 
        vardict.add_part('eos',vardict.particle['OnEquationOfState']> 0.)
        vardict.delif('OnEquationOfState',last=last)
    if updatesel:
        vardict.update(vardict.particle['eos'])
        vardict.readif('StarFormationRate', rawunits=True)
    else:
        vardict.readif('StarFormationRate', rawunits=True,setsel= not vardict.particle['eos'],setval=0.)
    vardict.delif('eos',last=last)
    convtolum = 1./(5.37e-42) 
    return vardict.particle['StarFormationRate'], convtolum # array, cgsconversion


def readbasic(vardict,quantity,excludeSFR,last = True,**kwargs):
    '''
    for some derived quantities, certain keywords are required
    '''
    # Temperature: requires setting T=1e4 K for EOS particles depending on excludeSFR setting
    if quantity == 'Temperature':
        if excludeSFR == 'T4':
            if not vardict.isstored_part('eos'):
                vardict.readif('OnEquationOfState',rawunits=True)
                vardict.add_part('eos',vardict.particle['OnEquationOfState'] > 0.)
                vardict.delif('OnEquationOfState',last=last)
            vardict.readif('Temperature',rawunits=True,setsel = vardict.particle['eos'],setval = 1e4)
            vardict.delif('eos',last=last)
        else:
            vardict.readif('Temperature',rawunits=True)

    # Mass is not actually stored for DM: just use ones
    elif vardict.parttype == '1' and quantity == 'Mass': 
        vardict.readif('Coordinates',rawunits=True)
        vardict.add_part('Mass',np.ones((vardict.particle['Coordinates'].shape[0],)))   

    # derived properties with vardict read-in method
    elif quantity == 'lognH': 
        vardict.getlognH(last=last,**kwargs)
    elif quantity == 'logT': 
        vardict.getlogT(last=last,logT = excludeSFR == 'T4') # excludeSFR setting determines wheather to use T4 or not  
    elif quantity == 'propvol': 
        vardict.getpropvol(last=last)
    # default case: standard simulation quantity read-in         
    else:
        vardict.readif(quantity,rawunits=True)




def project(NumPart,Ls,Axis1,Axis2,Axis3,box3,periodic,npix_x,npix_y,kernel,dct,tree,ompproj=True):
    '''
    dct must be a dictionary containing arrays 'coords', 'lsmooth', 'qW', 'qQ' (prevents copying of large arrays)
    '''

    # positions [Mpc / cm/s], kernel sizes [Mpc] and input quantities
    # a quirk of HsmlAndProject is that it only works for >= 100 particles. Pad with zeros if less.
    if NumPart >=100:
        pos = dct['coords'].astype(np.float32)
        Hsml = dct['lsmooth'].astype(np.float32)
        qW = dct['qW'].astype(np.float32)
        qQ = dct['qQ'].astype(np.float32)
    
    else:
        qQ = np.zeros((100,),dtype = np.float32)
        qQ[:NumPart] = dct['qQ'].astype(np.float32)  
        qW = np.zeros((100,),dtype = np.float32)
        qW[:NumPart] = dct['qW'].astype(np.float32)
        Hsml = np.zeros((100,),dtype = np.float32)
        Hsml[:NumPart] = dct['lsmooth'].astype(np.float32)
        pos = np.ones((100,3),dtype = np.float32)*1e8 #should put the particles outside any EAGLE projection region
        pos[:NumPart,:] = dct['coords'].astype(np.float32)
        NumPart = 100
    
    # ==============================================
    # Putting everything in right format for C routine
    # ==============================================
    
    print '\n--- Calling findHsmlAndProject ---\n'
    
    # define edges of the map wrt centre coordinates [Mpc]
    # in the periodic case, the C function expects all coordinates to be in the [0, BoxSize] range (though I don't think it actually reads Xmin etc. in for this)    
    # these need to be defined wrt the 'rotated' axes, e.g. Zmin, Zmax are always the min/max along the projection direction    
    if not periodic: # 0-centered
        Xmin = -1.0 * Ls[Axis1]/ 2.0
        Xmax = Ls[Axis1] / 2.0
        Ymin = -1.0 * Ls[Axis2] / 2.0
        Ymax = Ls[Axis2] / 2.0
        Zmin = -1.0 * Ls[Axis3] / 2.0
        Zmax = Ls[Axis3] / 2.0
  
    else: # half box centered (BoxSize used for x-y periodic boundary conditions)
        Xmin, Ymin = (0.,)*2
        Xmax,Ymax = (box3[Axis1],box3[Axis2])
        Zmin, Zmax = (0.5*(box3[Axis3] - Ls[Axis3]), 0.5*(box3[Axis3] + Ls[Axis3]))
    
    BoxSize = box3[Axis1]
        
    # maximum kernel size [Mpc] (modified from Marijke's version)
    Hmax = 0.5*min(Ls[Axis1],Ls[Axis2]) # Axis3 might be velocity; whole different units, so just ignore
        
    # arrays to be filled with resulting maps
    ResultW = np.zeros((npix_x, npix_y)).astype(np.float32)
    ResultQ = np.zeros((npix_x, npix_y)).astype(np.float32)
    
    # input arrays for C routine (change in c_pos <-> change in pos)
    c_pos = pos[:,:]
    c_Hsml = Hsml[:]
    c_QuantityW = qW[:]
    c_QuantityQ = qQ[:]
    c_ResultW = ResultW[:,:]
    c_ResultQ = ResultQ[:,:]
    
    # check if HsmlAndProject changes 
    print 'Total quantity W in: %.5e' % (np.sum(c_QuantityW))
    print 'Total quantity Q in: %.5e' % (np.sum(c_QuantityQ))
    
    # path to shared library
    if ompproj:
        sompproj = '_omp'
    else:
        sompproj = ''
    if tree:
        # in v3, projection can use more particles than c_int max,
        # but the tree building cannot
        if not ct.c_int(NumPart).value == NumPart:
            print(' ***         Warning         ***\n\nNumber of particles %i overflows C int type.\n This will likely cause the tree building routine in HsmlAndProjcet_v3 to fail.\nSee notes on v3 version.\n\n*****************************\n')
        if periodic:
            lib_path = ol.hsml_dir + 'HsmlAndProject_v3_%s_perbc%s.so' %(kernel, sompproj)       
        else:    
            lib_path = ol.hsml_dir + 'HsmlAndProject_v3_%s%s.so' %(kernel, sompproj)       
    else:
        if periodic:
            lib_path = ol.hsml_dir + 'HsmlAndProject_v3_notree_%s_perbc%s.so' %(kernel, sompproj)       
        else:    
            lib_path = ol.hsml_dir + 'HsmlAndProject_v3_notree_%s%s.so' %(kernel, sompproj)
    
    print('Using projection file: %s \n' % lib_path)
    # load the library
    my_library = ct.CDLL(lib_path)
    
    # set the parameter types (numbers with ctypes, arrays with ndpointers)
    my_library.findHsmlAndProject.argtypes = [ct.c_long,
                                  np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,3)), 
                                  np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)), 
                                  np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)), 
                                  np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)), 
                                  ct.c_float,
                                  ct.c_float,
                                  ct.c_float,
                                  ct.c_float,
                                  ct.c_float,
                                  ct.c_float,
                                  ct.c_int,
                                  ct.c_int,
                                  ct.c_int,
                                  ct.c_int,
                                  ct.c_int,
                                  ct.c_int,
                                  ct.c_float,
                                  ct.c_double,
                                  np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(npix_x,npix_y)), 
                                  np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(npix_x,npix_y))]
    
    # set the return type
    my_library.findHsmlAndProject.restype = None
    
    print '----------'
    
    # call the findHsmlAndProject C routine
    my_library.findHsmlAndProject(ct.c_long(NumPart),   # number of particles in map
                                  c_pos,                # positions wrt to centre (NumPart, 3)
                                  c_Hsml,               # SPH kernel
                                  c_QuantityW,          # quantity to be mapped by projection (or weighting for average)
                                  c_QuantityQ,          # quantity to be mapped by averaging
                                  ct.c_float(Xmin),     # left edge of map
                                  ct.c_float(Xmax),     # right edge of map
                                  ct.c_float(Ymin),     # bottom edge of map
                                  ct.c_float(Ymax),     # top edge of map
                                  ct.c_float(Zmin),     # near edge of map
                                  ct.c_float(Zmax),     # far edge of map
                                  ct.c_int(npix_x),     # number of pixels in x direction
                                  ct.c_int(npix_y),     # number of pixels in y direction
                                  ct.c_int(ol.desngb),  # number of neightbours for SPH interpolation
                                  ct.c_int(Axis1),      # horizontal axis (x direction)
                                  ct.c_int(Axis2),      # vertical axis (y direction)
                                  ct.c_int(Axis3),      # projection axis (z direction)
                                  ct.c_float(Hmax),     # maximum allowed smoothing kernel
                                  ct.c_double(BoxSize), # size of simulation box
                                  c_ResultW,            # RESULT: map of projected QuantityW (npix_x, npix_y)
                                  c_ResultQ)            # RESULT: map of projected QuantityQ weighted by QuantityW (npix_x, npix_y)
    
    print '----------'
    
    # check if mapped quantities conserved (but first one also counts some particles outside the map actually)
    print 'Total quantity W in:  %.5e' % (np.sum(c_QuantityW))
    print 'Total quantity W out: %.5e' % (np.sum(ResultW))
    print 'Total quantity Q in:  %.5e' % (np.sum(c_QuantityQ))
    print 'Total quantity Q out: %.5e' % (np.sum(ResultQ))

    return ResultW, ResultQ





########################################################################################################################





#### tables of particle properties needed for different functions, subfunctions requiring further quantites, and setting for wishlist quantities (T4EOS (bool) for Temperature and logT, hab for lognH)

# thin wrapper for functions with some extra information for wishlisting
class DocumentedFunction:
    '''
    function: the function to call
    subfunctions: list of other DocumentedFunction instances
    neededquantities: Vardict.particle entries the function uses (excl. only
    needed by subfunctions)
    settings: e.g. T4EOS, hab: quantities that determine the value of the 
              needed quantities
    '''
    def __init__(self,function, subfunctions, neededquantities, neededsettings):
        self.function = function
        self.subfunctions = subfunctions
        self.needed = neededquantities
        self.reqset = neededsettings
        
    def __call__(self,*args,**kwargs):
        return self.function(*args,**kwargs)

def getwishlist(funcname,**kwargs): 
    # key words: eltab, hab (str, float, or None (unspecified) values), 
    # matches output of settings, so these can be re-input as kwargs for subfunctions 
    '''
    Outputs:
    - the raw and derived quantities a function uses, specifying 
      subfunctions' outputs, but not their internal requiremtnt,
    - the names of subfunctions used to get derived quanitites 
      (required format 'get<output quantity label in vardict.particle>')
      list is in order of calling, so that settings reflect the last saved 
      version
    - settings: options for quantities used in the subfunctions in case of 
      ambiguities. 
        Temperature, logT: bool - T4EOS used (True) or not (False)
        eltab, hab: str or float, as produced by get_eltab_names
      this is used when the vardict.particle key for a quantity does not 
      reflect this information, but the values it stores depend on it: e.g. 
      eltab and hab labels contain this information, but the derived lognH 
      does not. In normal calculations, this is useful, since the rest of e.g.
      a luminosity calculation is independent of these settings, but in saving
      quantities, it could cause the wrong quantity to be used in a second 
      calculation

    !!! If a function uses different settings internally, the wishlist 
    combination routine can easily fail; see that function for notes on how to 
    handle functions like that !!!

    '''
    
    # set kwargs defaults:
    if not 'eltab' in kwargs.keys():
        kwargs.update({'eltab': None})
    if not 'hab' in kwargs.keys():
        kwargs.update({'hab': None})
    if not 'Temperature' in kwargs.keys():
        kwargs.update({'Temperature': None})
    if not 'logT' in kwargs.keys():
        kwargs.update({'logT': None})
        
    if funcname == 'luminosity_calc':
        if kwargs['eltab'] is None or kwargs['logT']is None or kwargs['hab'] is None:
            print('eltab, hab, and logT must be specified to generate the wishlist for luminosity_calc.')
            return None
        else:
            subs = ['getpropvol', 'getlognH', 'getlogT']
            needed = ['lognH','logT','propvol']
            settings = {'logT': kwargs['logT'], 'hab': kwargs['hab'], 'eltab': kwargs['eltab']}
            if isinstance(kwargs['eltab'],str):
                needed += [kwargs['eltab']]
            return (needed, subs, settings)

    elif funcname == 'Nion_calc':
        if kwargs['eltab'] is None or kwargs['logT']is None or kwargs['hab'] is None:
            print('eltab, hab, and logT must be specified to generate the wishlist for Nion_calc.')
        else:
            subs = ['getlognH', 'getlogT']
            needed = ['lognH','logT','Mass']
            settings = {'logT': kwargs['logT'], 'hab': kwargs['hab'], 'eltab': kwargs['eltab']}
            if isinstance(kwargs['eltab'],str):
                needed += [kwargs['eltab']]
            return (needed, subs, settings)

    elif funcname == 'Nelt_calc':
        if kwargs['eltab'] is None:
            print('eltab must be specified to generate the wishlist for Nelt_calc.')
        else:
            subs = []
            needed = ['Mass']
            settings = {'eltab': kwargs['eltab']}
            if isinstance(kwargs['eltab'],str):
                needed += [kwargs['eltab']]
            return (needed, subs, settings)

    elif funcname == 'luminosity_calc_halpha_fromSFR':
        subs = []
        needed = ['eos','StarFormationRate'] # OnEquationOfState is (so far) only ever used to get eos
        settings = {}
        return (needed,subs,settings)

    elif funcname == 'getlognH':
        if kwargs['hab'] is None:
            print('hab must be specified to generate the wishlist for getlognH.')
        else:
            subs = []
            needed = ['Density']
            settings = {'hab': kwargs['hab']}
            if isinstance(kwargs['hab'],str):
                needed += [kwargs['hab']]
            return (needed,subs,settings)

    elif funcname == 'getlogT':
        if kwargs['logT'] is None:
            print('logT must be specified to generate the wishlist for getlogT.')
        subs = []
        needed = ['Temperature']
        if kwargs['logT']:
            needed += ['OnEquationOfState']
        settings = {'Temperature': kwargs['logT']}
        return (needed, subs, settings)
            
    elif funcname == 'getpropvol':
        subs = []
        needed = ['Mass','Density']
        settings = {}
        return (needed,subs,settings)

    else:
        print('No entry in getwishlist for function %s'%funcname)
        return None

# checks a set of functions and returns needed, subs, settings for the whole lot, assuming the settings agree between all the subs 
def getsubswishlist(subs,settings):
    neededsubs = set()
    subsubs = set()
    settingssubs = {}
    for sub in subs:
        neededsub, subsub, settingssub = getwishlist(sub,**settings)
        neededsubs |= set(neededsub)
        subsubs |= set(subsub )
        settingssubs.update(settingssub)
    return neededsubs, list(subsubs), settingssubs

# settings dependencies for subfunction outputs:
settingsdep = {\
'hab': {'lognH'},\
}

def removesettingsdiffs(overlap,settings1,settings2):
    '''
    removes anything from overlap with a settings discrepancy
    returns new overlap
    '''
    settingskeys12 = set(settings1.keys())&set(settings2.keys()) # settings relevant for both functions, with updated settings2
    settingseq = {key: settings1[key] == settings2[key] for key in settingskeys12} # check where settings agree
    # if a quantity is needed in both functions, but with different settings, saving it will only cause trouble
    for key in settingseq.keys():
        if settingseq[key] == False:
           overlap.remove(key) 
           if key in settingsdep:
               overlap -= settingsdep[key]
    return overlap
    
# wishlist given getwishlist output for the first and second functions to run         
def combwishlist(neededsubssettingsfirst,neededsubssettingssecond):
    '''
    !!! May fail if a function or (nested) subfunction uses different settings
    internally !!!
    (Probably best not to try so save anything from something like that anyway;
    do not include that quantity in the getwishlist needed/subs list, and set 
    the relevant setting to None. Then any wishlisting will not include these 
    quantities.)
    '''
    used1, subs1, settings1 = neededsubssettingsfirst 
    needed2, subs2, settings2 = neededsubssettingssecond

    ## want to examine the whole tree of quantities used by function1 -> loop over nested subfunctions
    
    nextsubs = subs1
    used1 = set(used1)
    ## again, assumes first function uses the same settings throughout
    while len(nextsubs) > 0: # while there is a next layer of subfunctions
        usedsub, nextsubs, settingssub = getsubswishlist(nextsubs,settings1)
        settings1.update(settingssub) # add next layer of settings to everything function 1 uses
        used1 |= usedsub # add next layer of used quantities to everything function 1 uses
    


    needed12 = set(needed2)&used1 # stuff for second function that the first already gets
    needed12 = removesettingsdiffs(needed12,settings1,settings2)

    # loop over nested subfunctions: if a subfunction result is not saved, check if a quantity it uses should be
    nextsubs = subs2    
    while len(nextsubs) > 0: # loops over all nested subfunctions
        # checks which of the subfunctions don't have their outcomes on the wishlist already
        nextsubproducts = set([nextsub[3:] for nextsub in nextsubs]) # remove 'get' from the function name
        nextsubproducts -= needed12 # get the subfunction products that are not already stored
        nextsubs = ['get%s'%nextproduct for nextproduct in nextsubproducts] # gets the subfunctions whose products are not already stored, so those whose ingredients we want to keep in function 1 gets them

        # gets the requirements for the selected subfunctions and any additiona settings
        neededsub, nextsubs, settingssub = getsubswishlist(nextsubs,settings2)
        settings2.update(settingssub) # add next layer of settings to everything function 2 needs
        needed12 |= neededsub&used1 # add next layer of used quantities to everything function 2 needs
        needed12 = removesettingsdiffs(needed12,settings1,settings2)
        
    return list(needed12)

##########################################################################################



               
def make_map(simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, \
         ptypeW,\
         ionW=None, abundsW='auto', quantityW=None,\
         ionQ=None, abundsQ='auto', quantityQ=None, ptypeQ=None,\
         excludeSFRW=False, excludeSFRQ=False, parttype='0',\
         theta=0.0, phi=0.0, psi=0.0, \
         var='auto', axis='z',log=True, velcut=False,\
         periodic=True, kernel='C2', saveres=False,\
         simulation='eagle', LsinMpc=None,\
         select=None, misc=None, ompproj=False):
    
    """
    ------------------
     input simulation
    ------------------
    simnum:    only L####N####; string
    var:       'REFERENCE', 'RECALIBRATED', etc.; string
               default ('auto') means 'REFERENCE' for all but L0025N0752
    snapnum:   number of the snapshot; integer
    simulation:which simulation to use; 'eagle' or 'bahamas' (or 'eagle-ioneq')
               default 'eagle'
    -----------------
     what to project (particle selection)
    -----------------
    centre:    centre of the box to select particles in; list of 3 floats
    L_x (y,z): total length of the region to project in the x, (y,z) direction; 
               float
    LsinMpc:   are L_x,y,z and centre given in Mpc (True) or Mpc/h (False); 
               boolean or None
               if None: True for EAGLE, False for BAHAMAS
               default None
    log:       return log of projected values; boolean
               default True is strongly recommended to prevent float32 overflow
               in the output
    theta, psi, phi: angles to rotate image before plotting; float
               UNIMPLEMENTED after modifications to Marijke's make_maps
    axis:      axis to project along; string
               options: 'x', 'y', 'z'
    velcut:    slice by velocity in stead of position; boolean
               uses hubble flow equivalent of region given in position space
               default: False
               (box position along projection axis defines velocity cut through 
               Hubble flow)
    npix_x,    number of pixels to use in the prjection in the x and y 
    npix_y:    directions (int, > 0). Naming only uses the number of x pixels, and 
               the minimum smoothing length uses the pixel diagonal, so using 
               non-square pixels will not improve the resolution along one 
               direction by much
    
    The chosen region is assumed to be a continuous block that does not overlap 
    the boundaries of the box. Use emission_calc(_perfile) in make_maps for 
    regions overlapping boundaries, or split such regions up before mapping.
               
     -----------------
     quantities to project
    -----------------
    two quantities can be projected: W and Q
    W is projected directly: the output map is the sum of the particle 
     contributions in each grid cell
    for Q, a W-weighted average is calculated in each cell
    parameters describing what quantities to calculate have W/Q versions, that
     do the same thing, but for the different quantities. For the Q options,
     None can be used for all if no weighted average is desired
    
    ptypeW/    the category of quantity to project (str)
    ptypeQ:    options are 'basic', 'emission', 'coldens', and for ptypeQ, None
               'basic' means a quantity stored in the EAGLE output
               default: None for ptypeQ
    ionW/      required for ptype options 'emission' and 'coldens'
    ionQ:      for ptype option 'basic', option is ignored
               ion/element for which to calculated the column density 
               (ptype 'coldens')
               or ion/line of which to calculated the emission 
               (ptype 'emission')
               see make_maps_opts_locs.py for options
    quantityW/ required for ptype option 'basic'
    quantityQ: for ptype options 'emission' and 'coldens', option is ignored
               the quantity from the EAGLE output to project (string)
               should be the path in the hdf5 file starting after PartType#/
    parttype:  required for ptype option 'basic'
               for ptype options 'emission' and 'coldens', option is ignored
               the particle type for which to project (string!)
               0 (default): gas
               1: DM
               4: Stars
               5: BHs                     
    
    
    -------------------
     technical choices
    -------------------
    abundsW/   type of SPH abundances; string, float, or tuple(option,option)
    abundsQ:   if one option is given,
               smoothed/particle abundances are used for both nH and element                
               abundances 
               float is fixed element abundance in eagle solar units 
               (see make_maps_opts_locs); for emission and coldens,
               the primordial hydrogen abundance is then used to calculate
               lognH
               if tuple option is given,
               then the first element (index 0) is for the element abundance, 
               and the second for hydrogen (lognH calculation for emission and 
               absorption); float option here is in (absolute)  mass fraction
               'auto' is smoothed for ptype 'emission', particle for 'coldens',
               and same as the one-option setting for hydrogen
               options are 'Sm', 'Pt', 'auto', float, or a tuple of these
    kernel:    smoothing kernel to use in projections; string
               options: 'C2', 'gadget'
               default: 'C2'
               see HsmlAndProject for other options
    periodic:  use periodic boundary conditions (not along projection axis); 
               boolean
    excludeSFRW/ how to handle particle on the equation of state; string/bool  
    excludeSFRQ: options are 
               True   -> exclude EOS particles
               False  -> include EOS particles at face temperature
               'T4'   -> include EOS particles at T = 1e4 K
               'only' -> include only EOS particles
               'from' -> use only EOS particles or calculate halpha
                          emission from the star formation rate (ptype 
                          'emission', currently only for ion 'halpha')
               since Q and W must use the same particles, only False and T4
               or from and only can be combined with each other
    misc:      intended for one-off uses without messing with the rest
               dict or None (default)
               used in nameoutput with simple key-value naming
               no checks in inputcheck: typically single-funciton modifications
               checks are done there
    ompproj:   use the OpenMP implementation of the C projection routine (bool)
    --------
     output
    --------
    always:    2D array of projected emission
    optional:  .npz file containing the 2D array (naming is automatic)

               
    modify make_maps_opts_locs for locations of interpolation files (c), 
    projection routine (c), ion balance and emission tables (hdf5) and write 
    locations
    
    --------------
     misc options
    --------------
    'usechemabundtables': 'BenOpp1'
                           use Ben Oppenheimer's ChemicalAbundances tables
                           ion inclusion is checked, whether the simulation 
                           contains these tables is not
                           only for simulation='eagle-ioneq', var='REFERENCE' 
                           or 'ssh', simunum='L0025N0752', limited snapshots
                           only useful for column densities
                           NOT YET IMPLEMENTED CORRECTLY
    """    
    ########################
    #   setup and checks   #
    ########################
    
    # Must come first! (including 'auto' option handling)
    res = inputcheck(simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, \
         ptypeW,\
         ionW, abundsW, quantityW,\
         ionQ, abundsQ, quantityQ, ptypeQ,\
         excludeSFRW, excludeSFRQ, parttype,\
         theta, phi, psi, \
         var, axis, log, velcut,\
         periodic, kernel, saveres,\
         simulation, LsinMpc,\
         select, misc, ompproj)
    if type(res) == int:
        print('Input error %i'%res)
        return None
 
    iseltW, iseltQ, simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, \
         ptypeW,\
         ionW, abundsW, quantityW,\
         ionQ, abundsQ, quantityQ, ptypeQ,\
         excludeSFRW, excludeSFRQ, parttype,\
         theta, phi, psi, \
         var, axis, log, velcut,\
         periodic, kernel, saveres,\
         simulation, LsinMpc, misc, ompproj = res[1:]
    
    print('Processed input:')
    print 'iseltW, iseltQ, simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y'   
    print iseltW, iseltQ, simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, 
    print 'ptypeW, ionW, abundsW, quantityW, ionQ, abundsQ, quantityQ, ptypeQ, excludeSFRW, excludeSFRQ, parttype'
    print ptypeW,\
         ionW, abundsW, quantityW,\
         ionQ, abundsQ, quantityQ, ptypeQ,\
         excludeSFRW, excludeSFRQ, parttype
    print 'theta, phi, psi, var, axis, log, velcut, periodic, kernel, saveres, simulation, LsinMpc, ompproj'
    print theta, phi, psi, \
         var, axis, log, velcut,\
         periodic, kernel, saveres,\
         simulation, LsinMpc, ompproj
    
    print 'misc'
    print misc
    print '\n'
 

        
        
    ##### Wishlist generation: preventing doing calculations twice
    
    wishlist = ['coords_cMpc-vel']
    if ptypeQ != None:
        if ptypeQ == 'basic' and quantityQ != 'Temperature': #temperature checks come later
            wishlist += [quantityQ]       
        elif (ptypeQ=='basic' and quantityQ =='Temperature') and ((excludeSFRQ == 'T4' and excludeSFRW == 'T4') or (excludeSFRQ != 'T4' and excludeSFRW != 'T4')):
            wishlist += ['Temperature']
        elif (ptypeQ == 'emission' and excludeSFRQ != 'from') or ptypeQ == 'coldens': # element abundances to be added when the right option is determined below 
            wishlist += ['Density','Mass']
        elif ptypeQ == 'emission' and excludeSFRQ == 'from':
            wishlist += ['StarFormationRate']
        else:
            wishlist = []
            
    # since different function require smoothed or particle abundances: 
    # does some imput checking
    # set eltabW/Q, habW/Q: input for read_eagle element retrieval
    # set iseltW/Q (for the coldens case): calculate an ion or element coldens
    # updates the wishlist to include all EAGLE output quantities desired.
    
    if ptypeW in ['emission', 'coldens']:
        eltabW, habW = get_eltab_names(abundsW,iseltW,ionW)
    else:
        eltabW, habW = (None,None)
    
    if ptypeQ in ['emission', 'coldens'] and excludeSFRQ != 'from':
        eltabQ, habQ = get_eltab_names(abundsQ,iseltQ,ionQ)
        wishlist += [habQ,eltabQ]
    else:
        eltabQ, habQ = (None,None)
        
    wishlistW = ['coords_cMpc-vel']
    if ptypeW == 'basic': #temperature checks come later
        wishlistW += [quantityW]       
    elif ptypeW == 'emission' and excludeSFRW != 'from': # element abundances to be added when the right option is determined below 
        wishlistW += ['Density','Mass']
    elif ptypeQ == 'emission' and excludeSFRQ == 'from':
        wishlist = ['StarFormationRate']
    else:
        wishlist = []
    


    #### more advanced wishlist options: 
    if (ptypeW in ['emission','coldens'] and excludeSFRW != 'from') and (ptypeQ in ['emission','coldens'] and excludeSFRQ != 'from'):
        if habQ == habW: #same abundance choice
            wishlist.append('lognH')
            # only needed to get lognH
            wishlist.remove('Density')
            wishlist.remove(habQ)
        if ptypeW == 'emission' and ptypeQ == 'emission':
            wishlist.append('propvol')
            if habW == habQ:   
                wishlist.remove('Mass')
        
    if (ptypeQ in ['emission','coldens'] and excludeSFRQ != 'from')and ((excludeSFRQ == 'T4' and excludeSFRW == 'T4') or (excludeSFRQ != 'T4' and excludeSFRW != 'T4')):
        # if Q needs temperature and W is not using a different version (T4 vs. not T4)
        if ptypeW in ['coldens', 'emission'] and excludeSFRW !='from':
            wishlist.append('logT')
        else:
            wishlist.append('Temperature') # don't want to save the temperature when logT is all we need

    #### set data file, setup axis handling            
    
    simfile = Simfile(simnum,snapnum,var,simulation=simulation)

    if axis == 'x':
        Axis1 = 1
        Axis2 = 2
        Axis3 = 0
    elif axis == 'y':
        Axis1 = 2
        Axis2 = 0
        Axis3 = 1
    else:
        Axis1 = 0
        Axis2 = 1
        Axis3 = 2

    #name output files (W file name is independent of Q options)
    # done before conversion to Mpc to preserve nice filenames in Mpc/h units
    resfile = nameoutput(ptypeW,simnum,snapnum,version,kernel,npix_x,L_x,L_y,L_z,centre,simfile.boxsize,simfile.h,excludeSFRW,excludeSFRQ,velcut,axis,var,abundsW,ionW,parttype,None,abundsQ,ionQ,quantityW,quantityQ,simulation,LsinMpc,misc)
    if ptypeQ !=None:
        resfile2 = nameoutput(ptypeW,simnum,snapnum,version,kernel,npix_x,L_x,L_y,L_z,centre,simfile.boxsize,simfile.h,excludeSFRW,excludeSFRQ,velcut,axis,var,abundsW,ionW,parttype,ptypeQ,abundsQ,ionQ,quantityW,quantityQ,simulation,LsinMpc,misc)

    Ls = np.array([L_x,L_y,L_z])
    # make sure centre, Ls are in Mpc (other option in Mpc/h)
    if not LsinMpc:
        Ls /= simfile.h
        centre = np.array(centre)/simfile.h
        

    ####################################
    # read-in and quantity calculation #
    ####################################
    
    
    if simfile.region_supported:
        if velcut: # in practice, velocity region selection return the whole box in the projection direction anyway
            # region, hubbleflow_cgs = partselect_vel_region(simfile, centre, Ls, Axis1, Axis2, Axis3, parttype = parttype)
            region = partselect_pos(simfile, centre, Ls, Axis1, Axis2, Axis3, parttype=parttype)
            if region is not None: # None means use the whole box
                region[[2*Axis3, 2*Axis3+1]] = [ 0.,simfile.boxsize]
                if np.all(region == np.array([0.,simfile.boxsize,0.,simfile.boxsize,0.,simfile.boxsize])):
                    region = None # Don't other imposing a region if everything is selected; will be the case for whole box slices.
        else:
            region = partselect_pos(simfile, centre, Ls, Axis1, Axis2, Axis3, parttype=parttype)
        # initiate vardict: set centre, Ls, box3, and coords if we're working in ppv space
    else:
        region = None

    vardict_WQ = Vardict(simfile,parttype,wishlist,region = region,readsel=Sel())
    vardict_WQ.add_box('centre',centre)
    vardict_WQ.add_box('Ls',Ls)

    # cases where Sels need to be used for read-in coordinate selection
    # else sets box3 for region selection only case
    if velcut and simfile.region_supported:
        ppv_selselect_coordsgen(centre, Ls, Axis1, Axis2, Axis3, periodic, vardict_WQ, parttype = parttype)
    elif not vardict_WQ.simfile.region_supported and not velcut: #set region by setting readsel 
        ppp_selselect_coordsadd(centre, Ls, periodic, vardict_WQ, parttype = parttype)        
    elif not vardict_WQ.simfile.region_supported and velcut: # ppv_coordselect uses previously set readsels
        Ls_temp = np.copy(Ls)
        Ls_temp[Axis3] = simfile.boxsize/simfile.h
        ppp_selselect_coordsadd(centre, Ls_temp, periodic, vardict_WQ, parttype = parttype) # does the Axis1, Axis2 selection
        ppv_selselect_coordsgen(centre, Ls, Axis1, Axis2, Axis3, periodic, vardict_WQ, parttype = parttype) # does the Axis3 selection
    else:
        box3 = (simfile.boxsize*simfile.h**-1,)*3
        vardict_WQ.add_box('box3',box3)


    # excludeSFR handling: use np.logical_not on selection array
    # this is needed for all calculations, so might as well do it here
    # only remaining checks on excludeSFR are for 'T4' and 'from'

    if excludeSFRW in ['from','only']: # only select EOS particles; difference in only in the emission calculation
        vardict_WQ.readif('OnEquationOfState',rawunits =True)
        eossel = Sel({'arr': vardict_WQ.particle['OnEquationOfState'] > 0.})
        vardict_WQ.delif('OnEquationOfState')
        vardict_WQ.update(eossel) #should significantly reduce memory impact of coordinate storage
        del eossel

    elif excludeSFRW == True: # only select non-EOS particles
        vardict_WQ.readif('OnEquationOfState',rawunits =True)
        eossel = Sel({'arr': vardict_WQ.particle['OnEquationOfState'] <= 0.})
        vardict_WQ.delif('OnEquationOfState')
        vardict_WQ.update(eossel) #will have less impact on coordinate storage
        del eossel
    # False and T4 require no up-front or general particle selection, just one instance in the temperature read-in


    # calculate the quantities to project: save outside vardict (and no links in it) to prevent modification by the next calculation 
    if ptypeQ is None:
        last = True
    else:
        last = False
    if ptypeW == 'basic':
        readbasic(vardict_WQ,quantityW,excludeSFRW,last = last)           
        qW  = vardict_WQ.particle[quantityW]
        multipafterW = Nion_to_coldens(vardict_WQ,Ls,Axis1,Axis2,Axis3,npix_x,npix_y)*vardict_WQ.CGSconv[quantityW]

    elif ptypeW == 'coldens' and not iseltW:
        qW, multipafterW = Nion_calc(vardict_WQ,excludeSFRW,eltabW,habW,ionW,last=last,updatesel=True,misc=misc)
        multipafterW *= Nion_to_coldens(vardict_WQ,Ls,Axis1,Axis2,Axis3,npix_x,npix_y)
    elif ptypeW == 'coldens' and iseltW:
        qW, multipafterW = Nelt_calc(vardict_WQ,excludeSFRW,eltabW,ionW,last=last,updatesel=True)
        multipafterW *= Nion_to_coldens(vardict_WQ,Ls,Axis1,Axis2,Axis3,npix_x,npix_y)
        
    elif ptypeW == 'emission' and excludeSFRW != 'from':
        qW, multipafterW = luminosity_calc(vardict_WQ,excludeSFRW,eltabW,habW,ionW,last=last,updatesel=True)
        multipafterW *= lumninosty_to_Sb(vardict_WQ,Ls,Axis1,Axis2,Axis3,npix_x,npix_y,ionW)
    elif ptypeW == 'emission' and excludeSFRW == 'from':
        if ionW == 'halpha':
            qW, multipafterW = luminosity_calc_halpha_fromSFR(vardict_WQ,excludeSFRW,last=last,updatesel=True)
        multipafterW *= lumninosty_to_Sb(vardict_WQ,Ls,Axis1,Axis2,Axis3,npix_x,npix_y,ionW)
    
    
    if ptypeQ == 'basic':
        readbasic(vardict_WQ,quantityQ,excludeSFRQ,last = True)              
        qQ  = vardict_WQ.particle[quantityQ]
        multipafterQ = vardict_WQ.CGSconv[quantityQ]
    
    elif ptypeQ == 'coldens' and not iseltQ:
        qQ, multipafterQ = Nion_calc(vardict_WQ,excludeSFRQ,eltabQ,habQ,ionQ,last=True,updatesel=False,misc=misc)
        multipafterQ *= Nion_to_coldens(vardict_WQ,Ls,Axis1,Axis2,Axis3,npix_x,npix_y)
    elif ptypeQ == 'coldens' and iseltQ:
        qQ, multipafterQ = Nelt_calc(vardict_WQ,excludeSFRQ,eltabQ,ionQ,last=True,updatesel=False)
        multipafterQ *= Nion_to_coldens(vardict_WQ,Ls,Axis1,Axis2,Axis3,npix_x,npix_y)
        
    elif ptypeQ == 'emission' and excludeSFRQ != 'from':
        qQ, multipafterQ = luminosity_calc(vardict_WQ,excludeSFRQ,eltabQ,habQ,ionQ,last=True,updatesel=False)
        multipafterQ *= lumninosty_to_Sb(vardict_WQ,Ls,Axis1,Axis2,Axis3,npix_x,npix_y,ionW)
    elif ptypeQ == 'emission' and excludeSFRQ == 'from':
        if ionQ == 'halpha':
            qQ, multipafterQ = luminosity_calc_halpha_fromSFR(vardict_WQ,excludeSFRQ,last=True,updatesel=False)
        multipafterQ *= lumninosty_to_Sb(vardict_WQ,Ls,Axis1,Axis2,Axis3,npix_x,npix_y,ionW)



    if not velcut:
        vardict_WQ.readif('Coordinates',rawunits=True)
        vardict_WQ.add_part('coords_cMpc-vel',vardict_WQ.particle['Coordinates']*simfile.h**-1)
        vardict_WQ.delif('Coordinates',last=True) # essentially, force delete
        translate(vardict_WQ.particle,'coords_cMpc-vel', vardict_WQ.box['centre'], vardict_WQ.box['box3'], periodic)

    NumPart = vardict_WQ.particle['coords_cMpc-vel'].shape[0]
    if parttype == '0':
        lsmooth = simfile.readarray('PartType%s/SmoothingLength'%parttype, rawunits=True, region=vardict_WQ.region)[vardict_WQ.readsel.val] * simfile.h**-1
        tree = False
    elif parttype == '1': # DM: has a physically reasonable smoothing length, but it is not in the output files 
        lsmooth = np.zeros(NumPart)
        tree = True
    elif parttype == '4' or parttype=='5':
        lsmooth = 0.5*np.ones(NumPart)*Ls[Axis1]/npix_x 
        tree = False
    
    
    # prevents largest particle values from overflowing float32 (extra factor 1e4 is a rough attempt to prevent overflow in the projection)
    maxlogW = np.log10(np.max(qW))
    overfW = (int(np.ceil(maxlogW))+4)/38
    qW = qW*10**(-overfW*38)
    multipafterW *= 10**(overfW*38) 
    if ptypeQ is not None:
        overfQ = int(np.ceil(np.log10(np.max(qQ)) + maxlogW)+4)/38 - overfW
        qQ = qQ*10**(-overfQ*38)
        multipafterQ *= 10**(overfQ*38) 
    else:
        qQ = np.zeros(qW.shape,dtype=np.float32)
        
    projdict = {'lsmooth':lsmooth, 'coords':vardict_WQ.particle['coords_cMpc-vel'],'qW': qW, 'qQ':qQ}
    resultW,resultQ = project(NumPart,vardict_WQ.box['Ls'],Axis1,Axis2,Axis3,vardict_WQ.box['box3'],periodic,npix_x,npix_y,kernel,projdict,tree,ompproj=ompproj)


    if log: # strongly recommended: log values should fit into float32 just fine, e.g. non-log cgs Mass overflows float32 
        resultW = np.log10(resultW) + np.log10(multipafterW)
        if ptypeQ is not None:
            resultQ = np.log10(resultQ) + np.log10(multipafterQ)
    else:
        resultW *= multipafterW
        if ptypeQ is not None:
            resultQ *= multipafterQ
        
    np.savez(resfile, resultW.astype(np.float32))
    if ptypeQ is not None:
        np.savez(resfile2, resultQ.astype(np.float32))

    return resultW, resultQ









#########################################
##### halo selection from FOF files #####
#########################################
    
def get_EA_FOF_MRCOP(simnum,snapnum,var = None,mdef='200c', outdct = None):
    import read_eagle_files as eag
    
    if var is None:
        var = 'REFERENCE'
    fo = eag.read_eagle_file(ol.simdir_eagle%simnum + var + '/', 'sub', snapnum,  gadgetunits=True, suppress=False) # read_eagle_files object
    if mdef == '200c':
        massstring = 'Group_M_Crit200'
        sizestring = 'Group_R_Crit200'
        masslabel = 'M200c_Msun'
        sizelabel = 'R200c_cMpc'
    copstring = 'GroupCentreOfPotential'
    coplabel = 'COP_cMpc'
    
    # output in cMpc, Msun
    mass = fo.read_data_array('FOF/%s'%massstring, gadgetunits=True)
    mass *= 1e10 # gadget units -> Msun
    size = fo.read_data_array('FOF/%s'%sizestring, gadgetunits=True)
    size /= fo.h # gadget units -> cMpc
    cop = fo.read_data_array('FOF/%s'%copstring, gadgetunits=True)
    cop /= fo.h # gadget units -> cMpc
    
    if outdct is None:
        outdct = {}
    outdct[masslabel] = mass
    outdct[sizelabel] = size
    outdct[coplabel]  = cop
    
    return outdct
    
    
        