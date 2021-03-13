#!/usr/bin/env python3
#
# Calculate image sizes in two ways: directly from the image and from the eht baselines
#
# NOTE: these scripts coherent averages BEFORE network calibration to save time
# this is not consistent with what is done on the real data, but should be ok for now

from sys import argv
from glob import glob
from time import time
from itertools import product
from h5_to_ehtim import load_im_hdf5_frankfurt 

import numpy as np
import joblib as jl
import scipy.interpolate as interp
from ruamel import yaml
#import yaml

import ehtim as eh
import ehtim.scattering.stochastic_optics as so

#==============================================================================
# Default parameters

params = {
    'imagedir'    : '/path/to/imagedir',                                           # directory of the hdf5 movie images
    'arrayfile'   : 'EHT2017.txt',                                                 # eht array file
    'obsfile'     : 'hops_3599_SGRA_lo_V0_both_scan_netcal_normalized_10s.uvfits', # sgr a* reference data set
    'outfile'     : 'sz', # base path for output text files, OUTFILE+'_obs_sizes.txt' and OUTFILE+'_frame_sizes.txt'

    'simtype'     : 'Illinois',         # Illinois or Frankfurt
   #'framedur'    : 98.5,               # frame duration in seconds, 5m for sgr a* -- now hardcoded with simtype
    'fov'         : 160 * eh.RADPERUAS, # field-of-view of regridded frames.
    'npix'        : 80,                 # number of pixels in regridded frames

    'cachefile'   : None,
    'cacheonly'   : False,

    'nshifts'     : 10, #5,  # number of time shifts
    'nrots'       : 10, #5,  # number of rotation angles
    'nseeds'      : 10, #10, # number of random seeds

    'tavg'        : 60,       # coherent average time
    'ttype'       : 'direct', # fourier transform type
    'nproc'       : 4,        # processes for network calibration
    'job'         : 0,
    'njobs'       : 1,

    # these synthetic data parameters come from Kotaro
    'gain_offset' : {'AA': 0.15, 'AP': 0.15, 'AZ': 0.15, 'LM': 0.6, 'PV': 0.15, 'SM': 0.15, 'JC': 0.15, 'SP': 0.15, 'SR': 0.0},
    'gainp'       : {'AA': 0.05, 'AP': 0.05, 'AZ': 0.05, 'LM': 0.5, 'PV': 0.05, 'SM': 0.05, 'JC': 0.05, 'SP': 0.15, 'SR': 0.0},
    'dterm_offset': 0.0,  # 0.05 # turn off dterms for now
    'gain_sigmat' : None, # 0.25
    'lc_smooth'   : 0.5,
}

#==============================================================================
# Load Movie from Data or from Cache

def loadmov(params):

    if params['cachefile'] is None:
        cache = params['imagedir'].rstrip('/') + '.cache'
    else:
        cache = params['cachefile']

    try:
        mov, obs, max_shift = jl.load(cache) # if cache
        print('Use cache file "{}".'.format(cache))
        return mov, obs, max_shift
    except FileNotFoundError:
        print('Cache file "{}" not found, load from image files in "{}"'.format(cache, params['imagedir']))
        pass

    # the sgra data file
    # want this to be normalized, network calibrated, but NOT calibrated to a Gaussian
    print('1. load Sgr A* data')
    obs_sgra = eh.obsdata.load_uvfits(params['obsfile'])

    # copy the correct mount types
    arr = eh.array.load_txt(params['arrayfile'])
    t_obs = list(obs_sgra.tarr['site'])
    t_eht = list(arr.tarr['site'])
    t_conv = {'AA':'ALMA','AP':'APEX','SM':'SMA','JC':'JCMT','AZ':'SMT','LM':'LMT','PV':'PV','SP':'SPT'}
    for t in t_conv.keys():
        if t in obs_sgra.tarr['site']:
            for key in ['fr_par','fr_elev','fr_off']:
                obs_sgra.tarr[key][t_obs.index(t)] = arr.tarr[key][t_eht.index(t_conv[t])]

    # Calculate Sgr A* size from the observation
    #size_eht = calc_size_michael(obs_sgra)

    # coherent average.
    # TODO -- generating data from scan average isn't consistent with real data.
    obs_start_hr = np.min(obs_sgra.unpack(['time'])['time'])
    obs_stop_hr  = np.max(obs_sgra.unpack(['time'])['time'])
    obs_duration = obs_stop_hr - obs_start_hr

    # load the sample images
    print('2. load GRMHD images')
    image_list = np.sort(glob(params['imagedir']+'/*.h5')) # TODO -- we assume this sort puts frames in time order

    if params['simtype']=='Illinois':
        im_list = [eh.io.load.load_im_hdf5(image).regrid_image(params['fov'], params['npix']) for image in image_list]
        framedur = 98.5 # seconds / frame, 5 M
    elif params['simtype']=='Frankfurt':
        im_list = [load_im_hdf5_frankfurt(image) for image in image_list] 
        framedur = 197.0 # seconds / frame, 10 M
    else:
        raise Exception("simtype must be 'Illinois' or 'Frankfurt'!")

    # scatter the sample images (just diffractive blur)
    print('3. diffractive scatter GRMHD images')
    sm = so.ScatteringModel()
    im_list_scat = [sm.Ensemble_Average_Blur(im) for im in im_list]

    # merge frames into a movie
    print('4. make GRMHD movie object')
    mov = eh.movie.merge_im_list(im_list_scat, framedur=framedur)

    # shift the movie start time to coincide with the observation start
    tshift = obs_start_hr - mov.start_hr
    mov = mov.offset_time(tshift)
    mov_start_hr = mov.start_hr
    mov_stop_hr  = mov.stop_hr
    mov_duration = mov_stop_hr - mov_start_hr

    # What is the maximum shift we can make and still fit the observation inside the movie
    max_shift = 0.95*(mov_duration - obs_duration)  # 95% is a fudge factor to account for edge effects
    if max_shift < 0:
        raise Exception('movie duration is less than the observation duration!')

    # standardize movie and observation metadata
    # TODO having issues on uiuc cluster with astropy iers table! Might need to revert to previous astropy
    obs     = obs_sgra.copy()
    obs.mjd = obs.mjd # 51545 # TODO this is a hack to get observe_same to work, but we need to fix to observe on the correct day

    mov.mjd = obs.mjd
    mov.ra  = obs.ra
    mov.dec = obs.dec
    mov.rf  = obs.rf

    print('5. save cache to "{}"'.format(cache))
    jl.dump((mov, obs, max_shift), cache)

    return mov, obs, max_shift

#==============================================================================
# Actually compute the sizes

def observe_and_norm(mov, obs_org, timeshift, seed, params):
    """ make a normalized observation, shifted and roated and with random noise/gains

        mov is the input movie object
        obs_org is the reference sgra* observation
        timeshift is how much to shift the movie start time, in hours
        seed is the random number generation seed for gains and thermal noise
    """
    mov = mov.copy()

    # shift the start time
    if timeshift != 0:
        print('offset times: ', timeshift)
        mov = mov.offset_time(timeshift)

    # simulate observation
    print('observe')
    obs = mov.observe_same(obs_org, ttype=params['ttype'], add_th_noise=True, ampcal=False, phasecal=False,
                           stabilize_scan_phase=True, stabilize_scan_amp=True,
                           gain_offset=params['gain_offset'], gainp=params['gainp'],
                           jones=True, inv_jones=False,
                           dcal=True, frcal=True, rlgaincal=True, neggains=True,
                           dterm_offset=params['dterm_offset'], sigmat=params['gain_sigmat'],
                           seed=seed,verbose=False)

    # switch polrep
    obs = obs.switch_polrep('circ')

    # coherently average
    obs = obs.avg_coherent(params['tavg'])

    # get lightcurve from movie
    movie_times = mov.times
    movie_fluxes = mov.lightcurve
    movlc = interp.UnivariateSpline(movie_times, movie_fluxes, ext=3)
    movlc.set_smoothing_factor(params['lc_smooth'])

    # network calibrate
    print("Network calibrate to the time-dependent total flux")
    obs_nc = eh.netcal(obs, movlc, processes=params['nproc'], gain_tol=1.0, pol='RRLL')
    #for repeat in range(2): # TODO -- in kotaro's script, do we need to netcal multiple times?
    #    obs_nc = eh.netcal(obs_nc, spl, processes=params['nproc'], gain_tol=1.0, pol='RRLL')

    # Normalize *all* baselines to a total flux density of unity
    obs_nc_norm = obs_nc.copy()
    for field in ['rrvis','llvis','rlvis','lrvis','rrsigma','llsigma','rlsigma','lrsigma']:
        obs_nc_norm.data[field] /= movlc(obs_nc_norm.data['time'])

    return (obs, obs_nc_norm)

def rotate(mov, rotang, params):
    if rotang == 0:
        return mov.copy()
    else:
        print('rotate frames')
        fov,  npix  = params['fov'],  int(params['npix'])
        fovL, npixL = np.sqrt(2)*fov, int(np.sqrt(2)*npix)
        # TODO: make faster? yes we may parallelize this
        return eh.movie.merge_im_list([
            mov.get_frame(i).regrid_image(fovL, npixL).rotate(rotang).regrid_image(fov, npix)
            for i in range(mov.nframes)])

def calc_size_image(im):
    """directly compute the covariance matrix size from an image im"""
    gparams = im.fit_gauss()
    return(gparams[0]/eh.RADPERUAS, gparams[1]/eh.RADPERUAS)

def calc_size_michael(obs):
    """ calculate size as we do for Sgr A* data
        The observation object obs must be ALREADY NORMALIZED SO ZERO BASELINE IS UNITY!!
    """
    debias = True

    # Flag data with very low S/N
    #obs = obs.flag_low_snr(10.0)

    # Average data in 1-minute intervals
    #obs = obs.avg_coherent(60.0)

    # Compute the minimum source size from ALMA-LMT
    #print("Minimum source size from ALMA-LMT/SMT-LMT (uas):")
    AALM = obs.unpack_bl('AA','LM',['amp','uvdist'],debias=debias)
    if len(AALM) == 0:
        AALM = obs.unpack_bl('AP','LM',['amp','uvdist'],debias=debias)
    AZLM = obs.unpack_bl('AZ','LM',['amp','uvdist'],debias=debias)

    # Mask for simultaneous measurements
    AZLM_mask = [x for x in AZLM if x['time'] in AALM['time']]
    AALM_mask = [x for x in AALM if x['time'] in AZLM['time']]

    s_list = []
    for k in range(len(AALM_mask)):
        r = AALM_mask[k]['amp']/(AZLM_mask[k]['amp'][0]+1e-12)
        u1 = AZLM_mask[k]['uvdist'][0]
        u2 = AALM_mask[k]['uvdist'][0]
        s_list.append(np.sqrt( 4.0*np.log(2)*np.log(r)/(-np.pi**2 * (u2**2 - u1**2)))/eh.RADPERUAS)
    s_list = np.array(s_list)
    s_list = s_list[~np.isnan(s_list)]
    s_list = s_list[~np.isinf(s_list)]
    s_min = np.max(s_list)
    s_min2 = np.median(s_list)
    #print(np.max(s_min))

    #print("Maximum Size:")
    s_list = []
    AZLM = obs.unpack_bl('AZ','LM',['amp','uvdist'],debias=debias)
    s_list = np.sqrt( np.log(1.0/AZLM['amp']) * 4.0*np.log(2)/(np.pi**2 * AZLM['uvdist']**2))/eh.RADPERUAS
    s_list = s_list[~np.isnan(s_list)]
    s_list = s_list[~np.isinf(s_list)]
    s_max = np.min(s_list)
    s_max2 = np.median(s_list)
    #print(s_max)

    # catch some issues
    if s_max < s_min:
        print("maximum size estimate less than minimum size!")

    return [[s_min, s_max], [s_min2, s_max2]]

#==============================================================================
# Main program

def main(params):

    mov_org, obs_org, max_shift = loadmov(params)
    if params['cacheonly'] is not False:
        return # skip all the rest

    # observe the movie with random samples of gains, rotation angles, and time shifts, and calculate the size
    print('generate synthetic data samples')

    shifts = -1 * np.linspace(0,max_shift,params['nshifts']) # shifts need to be negative!
    rots   = np.linspace(0,2*np.pi,params['nrots'])

    sizearr     = []
    sizearr_med = []

    # TODO parallelize this loop

    ntotal  = int(params['nshifts']) * int(params['nrots']) * int(params['nseeds'])
    job     = int(params['job'])
    njobs   = int(params['njobs'])
    each    = ntotal // njobs
    myrange = list(range(job*each+1, (job+1)*each+1))

    iter = 0
    for rotang in rots:
        mov = rotate(mov_org, rotang, params)
        for shift, seed in product(shifts, range(1,params['nseeds']+1)):
            iter += 1
            if iter not in myrange:
                continue

            try:
                tstart = time()
                print('=============\n',  iter, '\n=============')
                (obs, obs_norm) = observe_and_norm(mov, obs_org, shift, seed, params)
                size_obs = calc_size_michael(obs_norm)
                sizearr.append(size_obs[0])
                sizearr_med.append(size_obs[1])
                print('iteration time: ', time() - tstart)
            except:
                print('error in observation!')
                continue

    sizearr     = np.array(sizearr)
    sizearr_med = np.array(sizearr_med)

    # save the observation sampled sizes
    outarr = np.hstack((sizearr, sizearr_med))
    np.savetxt(params['outfile'] + '_obs_sizes.txt',outarr, fmt='%0.2f')

    # save the sizes from the frames
    sizearr_im = np.array([calc_size_image(mov_org.get_frame(i)) for i in range(mov_org.nframes)])
    np.savetxt(params['outfile'] + '_frame_sizes.txt', sizearr_im, fmt='%0.2f')

def make_size_hists(image_sizes_file, obs_sizes_file):
    """make some plots"""
    # get sizes of individual frames from the image domain
    sizearr_im = np.loadtxt(images_sizes_file)
    obsdat = np.loadtxt(obs_sizes_file)
    sizearr_obs1 = obsdat[:,0:2]
    sizearr_obs2 = obsdat[:,2:4]

#    plt.figure(1)
#    plt.hist(0.5*np.sqrt(sizearr_im[:,0]**2 + sizearr_im[:,1]**2),color='blue',alpha=0.75,label='Image Domain Mean Size',density=True)
#    plt.hist(sizearr_obs1[:,0],color='green',alpha=0.75,density=True,label='EHT data min size (max)')
#    plt.hist(sizearr_obs1[:,1],color='red',alpha=0.75,density=True,label='EHT data max size (min)')
#    plt.xlabel(r'(uas)') 
#    plt.legend()

#    plt.figure(2)
#    plt.hist(0.5*np.sqrt(sizearr_im[:,0]**2 + sizearr_im[:,1]**2),color='blue',alpha=0.75,label='Image Domain Mean Size',density=True)
#    plt.hist(sizearr_obs2[:,0],color='green',alpha=0.75,density=True,label='EHT data min size (med)')
#    plt.hist(sizearr_obs2[:,1],color='red',alpha=0.75,density=True,label='EHT data max size (med)')
#    plt.xlabel(r'(uas)') 
#    plt.legend()

    plt.figure(3)
    plt.hist(sizearr_im[:,0],color='blue',alpha=0.75,label='Image Domain Minor Axis',density=True)
    plt.hist(sizearr_obs1[:,0],color='green',alpha=0.75,density=True,label='EHT data min size (Max)')
    plt.hist(sizearr_obs2[:,0],color='red',alpha=0.75,density=True,label='EHT data min size (Med)')
    plt.xlabel(r'(uas)') 
    plt.legend()

    plt.figure(4)
    plt.hist(sizearr_im[:,1],color='blue',alpha=0.75,label='Image Domain Major Axis',density=True)
    plt.hist(sizearr_obs1[:,1],color='green',alpha=0.75,density=True,label='EHT data max size (Min)')
    plt.hist(sizearr_obs2[:,1],color='red',alpha=0.75,density=True,label='EHT data max size (Med)')
    plt.xlabel(r'(uas)') 
    plt.legend()


if __name__=='__main__':

    # Override parameters with command line arguments
    for arg in argv[1:]:
        if ':' in arg:
            k, v = arg.split(':')
            params[k] = v
        else:
            params[arg] = True

    # Load YAML input if needed
    if 'yaml' in params:
        with open(params['yaml'], 'r') as f:
            data = yaml.safe_load(f)
        params.update(data)

    # Really run the script
    main(params)
