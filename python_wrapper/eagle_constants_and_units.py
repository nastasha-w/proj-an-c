import numpy as np


# --------------------------------------------------------------------------------
#
#   List of constants, unit conversion factors and cosmological parameters (where
#   the latter are as used by EAGLE).
#
# --------------------------------------------------------------------------------


# units
unitlength_in_cm         = 3.085678e+24
unitmass_in_g            = 1.989e+43
unitvelocity_in_cm_per_s = 100000.0
unitdensity_in_cgs       = 6.76991117829e-31
unitenergy_in_cgs        = 1.989e+53
unitpressure_in_cgs      = 6.76991117829e-21
unittime_in_s            = 3.085678e+19

# header
omega0           = 0.307
omegabaryon      = 0.0482519
omegalambda      = 0.693
hubbleparam      = 0.6777

# constants
pi               = 3.1415927 #pie
gamma            = 1.66666666667 # monatomic gas
gravity          = 6.672e-08 #cgs
solar_mass       = 1.989e+33 #cgs
solar_lum        = 3.826e+33 #cgs
rad_const        = 7.565e-15 #?
avogadro         = 6.0222e+23 #dimensionless
boltzmann        = 1.38066e-16 #cgs
gas_const        = 83142500.0  #cgs
c                = 29979000000.0 #cgs
planck           = 6.6262e-27 #cgs
cm_per_mpc       = 3.085678e+24 #correct
protonmass       = 1.6726e-24 #cgs
electronmass     = 9.10953e-28 #cgs
electroncharge   = 4.8032e-10 #cgs (statCoulomb)
hubble           = 3.2407789e-18 # NOT THE FFING HUBBLE CONSTANT!! THIS IS H0/h !!!!!!!
t_cmb0           = 2.728 #cgs
sec_per_megayear = 3.155e+13 #correct
sec_per_year     = 31550000.0 #correct
thompson         = 6.65245e-25 #cgs
z_solar          = 0.012663729 # seems about right
stefan           = 7.5657e-15 # !!!! 1e-10 * cgs value
ev_to_erg        = 1.60217646e-12 #correct

# parameters
S_over_Si        = 0.605416
Ca_over_Si       = 0.0941736

# CHECK EVERYTHING WITH hubble IN IT !!!!!!!!!!!!!!!!
# derive useful extras
rho_crit_cgs     = 3.0e0*hubble*hubble
rho_crit_cgs    /= 8.0e0*pi*gravity
rho_b_cgs        = rho_crit_cgs * omegabaryon
rho_m_cgs        = rho_crit_cgs * omega0
rho_crit         = rho_crit_cgs / unitdensity_in_cgs
rho_b            = rho_crit     * omegabaryon
rho_m            = rho_crit     * omega0

# standard atomic weights
atomw_H          = 1.00794005e0
atomw_He         = 4.00260162e0
atomw_C          = 12.01069927e0
atomw_N          = 14.00669765e0
atomw_O          = 15.99940014e0
atomw_Ne         = 20.17969704e0
atomw_Mg         = 24.3049984e0
atomw_Si         = 28.08549881e0
atomw_S          = 32.06499863e0
atomw_Ca         = 40.0779953e0
atomw_Fe         = 55.84499741e0
u                = 1.66053892e-24
