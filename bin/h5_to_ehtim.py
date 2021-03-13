#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh
import h5py

import string,math,sys,fileinput,glob,os,time,errno
import numpy.ma as ma

from astropy.time import Time as aTime
from astropy.coordinates import SkyCoord
from astropy.io import fits as pf




def load_im_hdf5_frankfurt(filename,header='old'):
    
    """
       orginal file by George Wong
       modified by Christian M. Fromm

       Read in an image from an hdf5 file.
       Args:
            filename (str)  : path to input hdf5 file
            sourename (str) : name of source
       optional Args:
            plot (integer) : create a 1 or not 0
            savefits (integer) : store image as fits 1 or not 0
            showplot (integer) : if show plot 1 or not 0
       Returns:
            Dictionary including:
            image: loaded image object
            + GRRT and GRMHD parameters
    """

    ###Load information from hdf5 file
    hfp = h5py.File(filename)

    if header=='new':
        ###get main image parameters
        dsource = hfp['header']['dsource'][()]                  # distance to source in cm
        jyscale = hfp['header']['scale'][()]                    # convert cgs intensity -> Jy flux density
        rf = hfp['header']['freqcgs'][()]                       # in cgs
        tunit = hfp['header']['units']['T_unit'][()]            # in seconds
        lunit = hfp['header']['units']['L_unit'][()]            # in cm
        DX = hfp['header']['camera']['dx'][()]                  # in GM/c^2
        nx = hfp['header']['camera']['nx'][()]                  # width in pixels
        time = hfp['header']['t'][()]*tunit/3600.               # time in hours


        ###emission parameters 
        rhigh=hfp['header']['emission']['Rhigh'][()]            # rhigh value for R-beta model
        rlow=hfp['header']['emission']['Rlow'][()]              # rlow value for R-beta model
        sigmacut=hfp['header']['emission']['sigmacut'][()]      # sigma cut used during GRRT (exclude regions with sigma>sigmacut
        betacrit=hfp['header']['emission']['betacrit'][()]      # critical beta value used during GRRT
        normflux=hfp['header']['emission']['normflux'][()]      # normalisation flux at 230GHz
        fov=hfp['header']['emission']['fov'][()]                # used field of view (-fov, fov)

        mbh=hfp['header']['bh']['mbh'][()]                      # mass of black hole
        spin=hfp['header']['bh']['spin'][()]                    # spin of black hole
        time=hfp['header']['bh']['time'][()]                    # time of the snapshot in M
        inc=hfp['header']['bh']['inc'][()]                      # viewing angle used during GRRT
        acc=hfp['header']['bh']['acc'][()]                      # accretion model of the used GRMHD

        ###get data
        unpoldat = np.copy(hfp['unpol'])                        # NX,NY
        hfp.close()

        ###Correct image orientation
        unpoldat = np.flip(unpoldat.transpose((1,0)),axis=0)


        ###--> use astropy to get source location
        #get source postion
        loc=SkyCoord.from_name(sourcename)

        #get RA and DEC in degree
        ra=loc.ra.deg
        dec=loc.dec.deg

        print('Source: %s located a RA: %s DEC: %s' %(sourcename,str(loc.ra), str(loc.dec)))

    if header=='old':


        ###get main image parameters
        dsource = hfp['header']['dsource'][()]                  # distance to source in cm
        jyscale = hfp['header']['scale'][()]                    # convert cgs intensity -> Jy flux density
        rf = hfp['header']['freqcgs'][()]                       # in cgs
        tunit = hfp['header']['units']['T_unit'][()]            # in seconds
        lunit = hfp['header']['units']['L_unit'][()]            # in cm
        DX = hfp['header']['camera']['dx'][()]                  # in GM/c^2
        nx = hfp['header']['camera']['nx'][()]                  # width in pixels
        time = hfp['header']['t'][()]*tunit/3600.               # time in hours


        ###get data
        unpoldat = np.copy(hfp['unpol'])                        # NX,NY
        hfp.close()

        ###Correct image orientation
        unpoldat = np.flip(unpoldat.transpose((1,0)),axis=0)

        ### Make a guess at the source based on distance and optionally fall back on mass
        #src = SOURCE_DEFAULT
        #src = "M87"
        if dsource > 4.e25 and dsource < 6.2e25: src = "M87"
        elif dsource > 2.45e22 and dsource < 2.6e22: src = "SgrA"
        
        # Fill in information according to the source
        if src == "SgrA":
            ra = 266.416816625
            dec = -28.992189444
        elif src == "M87": 
            ra = 187.70593075
            dec = 12.391123306

        rhigh = 0.
        rlow  = 0.
        sigmacut = 0.
        betacrit = 0.
        normflux = 0.
        fov      = 0.

    ###Process image to set proper dimensions
    fovmuas = DX / dsource * lunit * 2.06265e11
    psize_x = eh.RADPERUAS * fovmuas / nx
    Iim = unpoldat*jyscale


    #print('total flux', ma.sum(Iim))
    ##important ra is in hours!!!
    outim = eh.image.Image(Iim, psize_x,  ra * 12./180., dec, rf=rf, polrep='stokes', pol_prim='I', time=time)
    outim = outim.regrid_image(320*eh.RADPERUAS, 80)

    return outim

    ##return dictionary
    #return {'image':outim, 'rhigh':rhigh, 'rlow': rlow, 'sigmacut': sigmacut, 'betacrit': betacrit, 'normflux': normflux, 'fov': fov, 'mbh': mbh, 'spin': spin, 'time': time, 'inc':inc, 'acc': acc}




#================
#ref=load_im_hdf5(sys.argv[1],sys.argv[2],plot=1,showplot=0)
