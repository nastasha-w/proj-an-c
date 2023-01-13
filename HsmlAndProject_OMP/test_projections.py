#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import ctypes as ct
import sys
import matplotlib.pyplot as plt

def plotmaps(map_C, map_py, title):
    vmin = np.min(map_C[np.isfinite(map_C)])
    vmin = min(vmin, np.min(map_py[np.isfinite(map_py)]))
    vmax = np.max(map_C[np.isfinite(map_C)])
    vmax = max(vmax, np.max(map_C[np.isfinite(map_py)]))

    diff = map_C - map_py
    infmask_C = np.logical_not(np.isnan(map_C))
    infmask_py = np.logical_not(np.isnan(map_py))
    print('Nan values agree: ', np.all(infmask_C == infmask_py))
    compmap = np.logical_and(infmask_C, infmask_py)
    dmax = np.max(np.abs(diff[compmap]))
    dmin = -1. * dmax

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    img1 = ax1.imshow(map_C.T, origin='lower', interpolation='nearest',
                      cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title('C code map')
    plt.colorbar(img1, ax=ax1) 
    ax1.tick_params(labelbottom=False, labelleft=False)

    img2 = ax2.imshow(map_py.T, origin='lower', interpolation='nearest',
                      cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title('Python code map')
    plt.colorbar(img2, ax=ax2) 
    ax2.tick_params(labelbottom=False, labelleft=False)

    img3 = ax3.imshow(diff.T, origin='lower', interpolation='nearest',
                      cmap='RdBu', vmin=dmin, vmax=dmax)
    ax3.set_title('C - Python code map')
    plt.colorbar(img3, ax=ax3)
    ax3.tick_params(labelbottom=False, labelleft=False)

    ax4.hist(diff.flatten(), 
             bins=np.linspace(1.01 * dmin, 1.01 * dmax, 200))
    ax4.set_title('difference histogram')

    fig.suptitle(title)



### copied from proj-an/make_maps_v3_master.py
### modified search path for .so files
### desngb copied from proj-an/make_maps_opts_locs.py
desngb = 58
def project(NumPart, Ls, Axis1, Axis2, Axis3, box3, periodic, npix_x, npix_y,\
            kernel, dct, tree, ompproj=True, projmin=None, projmax=None):
    '''
    input:
    --------------------------------------------------------------------------
    - NumPart: number of SPH particles to project (int)
    - Ls:      dimensions (diameter) of the box to project (same units as 
               coordinates)
               length 3, indexable
    - Axis<i>: for Ls and coordinates, these variables control which axis is 
               the the projection axis (Axis3), and the orientation of the 
               other two. For a z-projection with a resulting array 
               (X index, Y index), use Axis1=0, Axis2=1, Axis3=2
    - box3:    the dimensions of the parent box (same units as Ls)
               length 3, indexable
    - periodic: is the projected region (perpendicular to the line of sight) a
               full slice of a periodic simulation (gas distributions at the 
               edges should be wrapped around the box), or a smaller part of 
               the simulation (gas contributions outside the projected region 
               edges should be ignored)
    - npix_x,y: how many pixels to use in the Axis1 and Axis2 directions, 
               respectively. Note that the minimum smoothing length is set to 
               the pixel diagonal, so very non-square pixels won't actually add
               much resolution in the higher-resolution direction.
               integers
    - kernel:  what shape to assume for the gas distribution respresented by a 
               single SPH particle 
               'C2' or 'gadget'
    - dct must be a dictionary containing arrays 
      'coords', 'lsmooth', 'qW', 'qQ' (prevents copying of large arrays)
      o 'coords': coordinates, (Numpart, 3) array. Coordinates should be 
                  transformed so that the projected region is a 
                  [-Ls[0] / 2., Ls[0] / 2.,\
                   -Ls[1] / 2., Ls[1] / 2.,\
                   -Ls[2] / 2., Ls[2] / 2. ]
                  box (if not periodic) or
                  [0., Ls[0], 0., Ls[1], 0., Ls[2]] if it is.
                  (The reason for this assumption in the periodic case is that 
                  it makes it easy to determine when something needs to be 
                  wrapped around the edge, and for the non-periodic case, it 
                  allows the code to ignore periodic conditions even though the
                  simulations are periodic and the selected region could 
                  therefore in principle require wrapping.)
      o 'lsmooth': gas smoothing lengths (same units as coords)
      o 'qW':     the array containing the particle property to directly,
                  project, and to weight qQ by
      o 'qQ':     the array to get a qW-weighted average for in each pixel
    - projmin, projmax: maximum coordinate values in projection direction
                  (override default values in Ls; I put this in for a specific
                  application)
              
    returns:
    --------------------------------------------------------------------------
    (ResultW, ResultQ) : tuple of npix_x, npix_y arrays (float32)
    - ResultW: qW projected onto the grid. The array contains the sum of qW 
               contributions to each pixel, not a qW surface density.
               the sums of ResultW and qW shoudl be the same to floating-point 
               errors when projecting a whole simulation, but apparent mass 
               loss in the projection may occur when projecting smaller 
               regions, where some particles in the qW array are (partially) 
               outside the projected region
    - ResultQ: qW-weighted average of qQ in each pixel
    '''

    # positions [Mpc / cm/s], kernel sizes [Mpc] and input quantities
    # a quirk of HsmlAndProject is that it only works for >= 100 particles. Pad with zeros if less.
    if NumPart >=100:
        pos = dct['coords'].astype(np.float32)
        Hsml = dct['lsmooth'].astype(np.float32)
        qW = dct['qW'].astype(np.float32)
        qQ = dct['qQ'].astype(np.float32)

    else:
        qQ = np.zeros((100,), dtype=np.float32)
        qQ[:NumPart] = dct['qQ'].astype(np.float32)
        qW = np.zeros((100,), dtype=np.float32)
        qW[:NumPart] = dct['qW'].astype(np.float32)
        Hsml = np.zeros((100,), dtype=np.float32)
        Hsml[:NumPart] = dct['lsmooth'].astype(np.float32)
        pos = np.ones((100,3), dtype=np.float32) * 1e8  #should put the particles outside any EAGLE projection region
        pos[:NumPart,:] = dct['coords'].astype(np.float32)
        NumPart = 100

    # ==============================================
    # Putting everything in right format for C routine
    # ==============================================

    print('\n--- Calling findHsmlAndProject ---\n')

    # define edges of the map wrt centre coordinates [Mpc]
    # in the periodic case, the C function expects all coordinates to be in the [0, BoxSize] range (though I don't think it actually reads Xmin etc. in for this)
    # these need to be defined wrt the 'rotated' axes, e.g. Zmin, Zmax are always the min/max along the projection direction
    if not periodic: # 0-centered
        Xmin = -0.5 * Ls[Axis1]
        Xmax =  0.5 * Ls[Axis1]
        Ymin = -0.5 * Ls[Axis2]
        Ymax =  0.5 * Ls[Axis2]
        if projmin is None:
            Zmin = -0.5 * Ls[Axis3]
        else:
            Zmin = projmin
        if projmax is None:
            Zmax = 0.5 * Ls[Axis3]
        else:
            Zmax = projmax

    else: # half box centered (BoxSize used for x-y periodic boundary conditions)
        Xmin, Ymin = (0.,) * 2
        Xmax, Ymax = (box3[Axis1], box3[Axis2])
        if projmin is None:
            Zmin = 0.5 * (box3[Axis3] - Ls[Axis3])
        else:
            Zmin = projmin
        if projmax is None:
            Zmax = 0.5 * (box3[Axis3] + Ls[Axis3])
        else:
            Zmax = projmax

    BoxSize = box3[Axis1]

    # maximum kernel size [Mpc] (modified from Marijke's version)
    Hmax = 0.5 * min(Ls[Axis1],Ls[Axis2]) # Axis3 might be velocity; whole different units, so just ignore

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
    print('Total quantity W in: %.5e' % (np.sum(c_QuantityW)))
    print('Total quantity Q in: %.5e' % (np.sum(c_QuantityQ)))

    # path to shared library
    sdir = './'
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
            lib_path = sdir + 'HsmlAndProject_v3_%s_perbc%s.so' %(kernel, sompproj)
        else:
            lib_path = sdir + 'HsmlAndProject_v3_%s%s.so' %(kernel, sompproj)
    else:
        if periodic:
            lib_path = sdir + 'HsmlAndProject_v3_notree_%s_perbc%s.so' %(kernel, sompproj)
        else:
            lib_path = sdir + 'HsmlAndProject_v3_notree_%s%s.so' %(kernel, sompproj)

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

    print('----------')

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
                                  ct.c_int(desngb),  # number of neightbours for SPH interpolation
                                  ct.c_int(Axis1),      # horizontal axis (x direction)
                                  ct.c_int(Axis2),      # vertical axis (y direction)
                                  ct.c_int(Axis3),      # projection axis (z direction)
                                  ct.c_float(Hmax),     # maximum allowed smoothing kernel
                                  ct.c_double(BoxSize), # size of simulation box
                                  c_ResultW,            # RESULT: map of projected QuantityW (npix_x, npix_y)
                                  c_ResultQ)            # RESULT: map of projected QuantityQ weighted by QuantityW (npix_x, npix_y)

    print('----------')

    # check if mapped quantities conserved (but first one also counts some particles outside the map actually)
    print('Total quantity W in:  %.5e' % (np.sum(c_QuantityW)))
    print('Total quantity W out: %.5e' % (np.sum(ResultW)))
    print('Total quantity Q in:  %.5e' % (np.sum(c_QuantityQ)))
    print('Total quantity Q out: %.5e' % (np.sum(ResultQ)))

    return ResultW, ResultQ


### kernel function taken from kernel_v3_omp.c and kernel_v3.h
def kernelfunc_c2(dnorm, lsmooth):
    #hsml_factor = 1.936492
    #corr_norm = 0.0294
    #corr_exp =  0.977
    coeff1 = 3.342253804929802

    out = coeff1 * 1. / lsmooth**3  * (1.0 - dnorm)**4 * (1. + 4. * dnorm)
    out[dnorm > 1.] = 0.
    return out

def kernelfunc_gd(dnorm, lsmooth):
    #hsml_factor = 2.0
    coeff1 = 2.546479089470   
    coeff2 = 15.278874536822
    #coeff3 = 45.836623610466
    #coeff4 = 30.557749073644
    coeff5 = 5.092958178941
    #coeff6 = -15.278874536822
    
    out = np.zeros(dnorm.shape, dtype=np.float32)
    mask1 = dnorm < 0.5
    out[mask1] = 1. / lsmooth**3 * \
                 (coeff1 + coeff2 * (dnorm[mask1] - 1.) * dnorm[mask1]**2)
    mask2 = np.logical_and(dnorm >= 0.5, dnorm <= 1.)
    out[mask2] = 1. / lsmooth**3 * coeff5 * (1. - dnorm[mask2])**3
    return out

def kernelfunc(xcens, ycens, pos2d, lsmooth, box2, 
               periodic=False, kernel='C2'):
    '''
    returns *sum-normalized* kernel values
    '''
    xdiff = xcens - pos2d[0]
    if periodic:
        xr = box2[1] - box2[0]
        xdiff = (xdiff + 0.5 * xr) % xr - 0.5 * xr
    ydiff = ycens - pos2d[1]
    if periodic:
        yr = box2[3] - box2[2]
        xdiff = (ydiff + 0.5 * yr) % yr - 0.5 * yr
    dnorm = np.sqrt((xdiff**2)[:, np.newaxis] + (ydiff**2)[np.newaxis, :]) 
    dnorm /= lsmooth

    if kernel == 'C2':
        out = kernelfunc_c2(dnorm, lsmooth)
    elif kernel == 'gadget':
        out = kernelfunc_gd(dnorm, lsmooth)
    else:
        raise ValueError('Invalid kernel option {}'.format(kernel))
    return out

def project_slow_test(Ls, Axis1, Axis2, Axis3, box3, periodic, 
                      npix_x, npix_y, kernel, dct):

    coords = dct['coords'].astype(np.float32)
    lsmooth = dct['lsmooth'].astype(np.float32)
    qW = dct['qW'].astype(np.float32)
    qQ = dct['qQ'].astype(np.float32)

    if periodic:
        xrange = [0., 2 * box3[Axis1] + 1]
        yrange = [0., 2 * box3[Axis2] + 1]
        zrange = [box3[Axis3] - 0.5 * Ls[Axis3], 
                  box3[Axis3] + 0.5 * Ls[Axis3]]
    else:
        xrange = [-0.5 * Ls[Axis1], 0.5 * Ls[Axis1]]
        yrange = [-0.5 * Ls[Axis2], 0.5 * Ls[Axis2]]
        zrange = [-0.5 * Ls[Axis3], 0.5 * Ls[Axis3]]
    #print(Ls)
    #print(Axis1, Axis2, Axis3)
    #print(xrange)
    #print(yrange)
    #print(zrange)
    pixsize_x = (xrange[1] - xrange[0]) / float(npix_x)
    pixsize_y = (yrange[1] - yrange[0]) / float(npix_y)
    xcens = np.arange(xrange[0] + 0.5 * pixsize_x, xrange[1], pixsize_x)
    ycens = np.arange(yrange[0] + 0.5 * pixsize_y, yrange[1], pixsize_y)

    box2 = [2 * box3[Axis1], 2 * box3[Axis1] +1, 
            2 * box3[Axis2], 2 * box3[Axis2] +1]
    outW = np.zeros((npix_x, npix_y), dtype=np.float32)
    outQ = np.zeros((npix_x, npix_y), dtype=np.float32)
    print(xcens)
    print(ycens)
    print(coords)
    for i in range(len(lsmooth)):
        pos2d = [coords[i, Axis1], coords[i, Axis2]]
        _lsmooth = lsmooth[i]
        if not periodic:
            if pos2d[0] + _lsmooth < xcens[0]:
                print('Skipping ', i)
                continue
            if pos2d[0] - _lsmooth > xcens[-1]:
                print('Skipping ', i)
                continue
            if pos2d[1] + _lsmooth < ycens[0]:
                print('Skipping ', i)
                continue
            if pos2d[1] - _lsmooth > ycens[-1]:
                print('Skipping ', i)
                continue
        if coords[i, Axis3] <= zrange[0] or coords[i, Axis3] > zrange[-1]:
            print('Skipping ', i)
            continue 

        _kf = kernelfunc(xcens, ycens, pos2d, _lsmooth, box2, 
                         periodic=periodic, kernel=kernel)
        ipixminx = np.floor((pos2d[0] - _lsmooth - 1.5 * pixsize_x) \
                            / pixsize_x)
        ipixmaxx = np.ceil((pos2d[0] + _lsmooth + 0.5 * pixsize_x) / pixsize_x)
        ipixminy = np.floor((pos2d[1] - _lsmooth  - 1.5 * pixsize_y) \
                            / pixsize_y)
        ipixmaxy = np.ceil((pos2d[1] + _lsmooth + 0.5 * pixsize_y) / pixsize_y)
        _xcnorm = np.arange((ipixminx + 0.5) * pixsize_x, 
                            (ipixmaxx + 1.) * pixsize_x, pixsize_x)
        _ycnorm = np.arange((ipixminy + 0.5) * pixsize_y, 
                            (ipixmaxy + 1.) * pixsize_y, pixsize_y)
        #print(_xcnorm)
        #print(_ycnorm)
        __kf = kernelfunc(_xcnorm, _ycnorm, pos2d, _lsmooth, box2,
                          periodic=periodic, kernel=kernel)
        _norm = np.sum(__kf)
        #print('norm: {}'.format(_norm))
        _kf *= 1. / _norm
        outW += qW[i] * _kf
        outQ += qQ[i] * qW[i] * _kf
    outQ /= outW
    # mimic C code:
    outQ[outW == 0] = 0.
    return outW, outQ

def printmap_text(_map):
    maxv = np.max(_map)
    halfv = 0.5 * maxv
    print('Max value: {maxv}')
    print(' >0, <= {}: +'.format(halfv))
    print(' >{}, <= {}: *'.format(halfv, maxv))
    print('\n\n\n')

    for i in range(map.shape(0), -1, -1):
        line = _map[i, :]
        pline = ''.join([' ' if lv <= 0. else \
                         '+' if lv <= halfv else \
                         '*' for lv in line])
        print(pline)


def test_projection(periodic=False, kernel='C2', omp=True):
    print('-'*40)
    msg = 'Starting test: periodic: {per}, kernel: {ker}, OpenMP: {omp}'
    print(msg.format(per=periodic, ker=kernel, omp=omp))
    print('-'*40)

    box3 = [0., 20., 0., 20., 0., 20.]
    Axis1 = 0
    Axis2 = 1
    Axis3 = 2
    npix_x = 50
    npix_y = 50
    Ls = [20., 20., 10.]

    lsmooth = np.array([1.5, 1.5, 1.5, 2., 2., 3., 1., 0.5, 1., 1.5])
    qW = np.array([1., 1., 2., 1., 1., 2., 0., 1., 1., 2.])
    qQ = np.array([1., 1., 1., 1., 2., 1., 1., 0., 1., 2.])
    xycoords = np.arange(1., 20., 2.)
    dx = np.array([0.01, -0.1, 0.03, 0.04, -0.05, -0.11, 0.07, 0.09,
                   0.12, -0.0001])
    dy = np.append(dx[4:], dx[:4])
    zcoords = np.array([6., 4., 7., 8., 9., 10., 11., 12., 16., 13.])
    coords = np.array([xycoords + dx, xycoords + dy, zcoords]).T
    if not periodic:
        coords[0, :] -= 0.5 * box3[1]
        coords[1, :] -= 0.5 * box3[3]
        coords[2, :] -= 0.5 * box3[5]
    dct = {'lsmooth': lsmooth, 'coords': coords, 'qW': qW, 'qQ': qQ}

    Numpart = len(lsmooth)
    
    mapW_C, mapQ_C = project(Numpart, Ls, Axis1, Axis2, Axis3, box3, 
                             periodic, npix_x, npix_y,
                             kernel, dct, False, ompproj=omp)

    mapW_py, mapQ_py = project_slow_test(Ls, Axis1, Axis2, Axis3, box3, 
                                         periodic, npix_x, npix_y, kernel, 
                                         dct)
    
    msg = 'Test {wq} {res} for periodic: {per}, kernel: {ker}, OpenMP: {omp}'
    msg_kw = {'per': periodic, 'ker': kernel, 'omp': omp}
    resW = np.allclose(mapW_C, mapW_py)
    _mW = msg.format(wq='mapW', res='succes' if resW else 'failed', **msg_kw)
    print(_mW)
    if not resW:
        plotmaps(mapW_C, mapW_py, _mW)
    # np.isclose(np.NaN, np.NaN) returns False.
    infmask_C = np.logical_not(np.isnan(mapQ_C))
    infmask_py = np.logical_not(np.isnan(mapQ_py))
    nansame = np.all(infmask_C == infmask_py)
    resQ = np.allclose(mapQ_C[infmask_C], mapQ_py[infmask_C]) \
           and nansame
    _mQ = msg.format(wq='mapQ', res='succes' if resQ else 'failed', **msg_kw)
    print(_mQ)
    if not resQ:
        plotmaps(mapQ_C, mapQ_py, _mQ)
    
    if omp: # test for race conditions
        coords_rctest = np.ones((200, 3), dtype=np.float32) 
        coords_rctest[0, :] *= box3[1]
        coords_rctest[1, :] *= box3[3]
        coords_rctest[2, :] *= 0.5 * box3[5]
        lsmooth_rctest = 0.51 * np.sqrt(2.) * 20. / 50. 
        lsmooth_rctest = lsmooth_rctest * np.ones((200,), dtype=np.float32)
        qW = np.ones((200,), dtype=np.float32)
        qQ = np.ones((200,), dtype=np.float32)
        if not periodic:
            coords[0, :] -= 0.5 * box3[1]
            coords[1, :] -= 0.5 * box3[3]
            coords[2, :] -= 0.5 * box3[5]
        
        dct = {'lsmooth': lsmooth_rctest, 'coords': coords_rctest,
               'qW': qW, 'qQ': qQ}
        Numpart = len(lsmooth_rctest)

        mapW_C, mapQ_C = project(Numpart, Ls, Axis1, Axis2, Axis3, box3, 
                             periodic, npix_x, npix_y,
                             kernel, dct, False, ompproj=omp)

        mapW_py, mapQ_py = project_slow_test(Ls, Axis1, Axis2, Axis3, box3, 
                                         periodic, npix_x, npix_y, kernel, 
                                         dct)
    
        msg = 'OpenMP rc test {wq} {res} for periodic: {per}, kernel: {ker}'
        msg_kw = {'per': periodic, 'ker': kernel}
        resW = np.allclose(mapW_C, mapW_py)
        sfw = 'succes' if resW else 'failed'
        _mW = msg.format(wq='mapW', res=sfw, **msg_kw)
        print(_mW)
        if not resW:
            plotmaps(mapW_C, mapW_py, _mW)
        resQ = np.allclose(mapQ_C, mapQ_py)
        infmask_C = np.logical_not(np.isnan(mapQ_C))
        infmask_py = np.logical_not(np.isnan(mapQ_py))
        nansame = np.all(infmask_C == infmask_py)
        resQ = np.allclose(mapQ_C[infmask_C], mapQ_py[infmask_C]) \
               and nansame
        sfq = 'succes' if resQ else 'failed'
        _mQ = msg.format(wq='mapQ', res=sfq, **msg_kw)
        print(_mQ)
        if not resQ:
            plotmaps(mapQ_C, mapQ_py, _mQ)
    print('\n'*3)
    plt.show()

def run_tests(index=None):
    if index is None:
        for index in range(8):
            run_tests(index)
    elif index in range(8):
        print('Running test {}'.format(index))
        periodic = bool(index // 4)
        kernel = ['C2', 'gadget'][(index % 4) // 2]
        omp = bool(index % 2)
        test_projection(periodic=periodic, kernel=kernel, omp=omp)
    else: 
        raise ValueError('index {} is not valid'.format(index))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        index = int(sys.argv[1])
    else:
        index = None
    run_tests(index=index)
