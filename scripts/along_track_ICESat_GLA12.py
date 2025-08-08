#!/usr/bin/env python
u"""
along_track_ICESat_GLA12.py
Written by Tyler Sutterley (08/2025)

Fits a time-variable surface to ICESat data to create an
along-track GLAH12 data product

INPUTS:
    infile(s): track file(s) to run

COMMAND LINE OPTIONS:
    --help: list the command line options
    -H X, --hemisphere X: Region of interest to run
    --rgt-phase X: Repeat ground-track phase to run
    -A X, --along_track X: Along-track distance for the segments
    -S X, --search_radius X: Search radius for the surface fit
    -I X, --iteration X: Number of iterations for surface fit
    --order-spline X: Smoothing spline order for along-track coordinates
    --order-time X: Temporal fit polynomial order
    --order-space X: Spatial fit polynomial order
    --point-threshold X: Threshold for minimum number of points
    -T X, --tide X: Model for ocean tide correction
    -R X, --reanalysis X: Model for inverse-barometer correction
    -G X, --geoid X: Geoid height model for correction
    --dem-filter: Filter elevations against internal DEM
    --knots X: Number of knots for the spline fit
    --runs X: Number of Monte Carlo runs for the along-track coordinates
    -V, --verbose: Verbose output of run
    -M X, --mode X: Permissions mode of the directories and files

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        http://www.scipy.org/
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

REFERENCES:
    T. Schenk and B. M. Csatho, "A New Methodology for Detecting
        Ice Sheet Surface Elevation Changes From Laser Altimetry Data",
        IEEE Transactions on Geoscience and Remote Sensing, 50(9),
        3302--3316, (2012). https://doi.org/10.1109/TGRS.2011.2182357
    B. E. Smith, C. R. Bentley, and C. F. Raymond, "Recent elevation
        changes on the ice streams and ridges of the Ross Embayment
        from ICESat crossovers", Geophysical Research Letters,
        32(25), L21S09, (2005). https://doi.org/10.1029/2005GL024365
    B. E. Smith, N. Gourmelen, A. Huth and I. Joughin, "Connected
        subglacial lake drainage beneath Thwaites Glacier, West
        Antarctica", The Cryosphere, 11, 451--467 (2017).
        https://doi.org/10.5194/tc-11-451-2017
     T. C. Sutterley, I. Velicogna, E. J. Rignot, J. Mouginot,
        T. Flament, M. R. van den Broeke, J. M. van Wessem, C. H. Reijmer, 
        "Mass loss of the Amundsen Sea Embayment of West Antarctica from
        four independent techniques", Geophysical Research Letters,
        41(23), 8421--8428, (2014). https://doi.org/10.1002/2014GL061940

UPDATE HISTORY:
    Updated 08/2025: save robust spread (RDE) of height residuals
        can adjust the order of the spline fit for along-track coordinates
        drop empty campaigns if setting repeat ground track phase
    Updated 07/2025: save saturated waveform correction to output files
        save elevation change rate and uncertainty if computed
        use a Monte Carlo approach to calculate the along-track coordinates
    Forked 06/2025 from fit_surface_tiles.py
"""
import sys
import re
import os
import copy
import time
import logging
import argparse
import pathlib
import logging
import traceback
import collections
import numpy as np
import grounding_zones as gz

h5py = gz.utilities.import_dependency('h5py')
scipy = gz.utilities.import_dependency('scipy')
scipy.interpolate = gz.utilities.import_dependency('scipy.interpolate')
pyTMD = gz.utilities.import_dependency('pyTMD')
timescale = gz.utilities.import_dependency('timescale')

# compile regular expression operator for extracting information from file
rx = re.compile((r'GLAH(\d{2})_(\d{3})_(\d{1})(\d{1})(\d{2})_(\d{3})_'
    r'(\d{4})_(\d{1})_(\d{2})_(\d{4})\.H5$'), re.VERBOSE)

# ICESat campaigns (all repeat ground-track phases)
campaigns = ['1A', '1B', '2A', '2B', '2C', '3A',
    '3B', '3C', '3D', '3E', '3F', '3G', '3H',
    '3I', '3J', '3K', '2D', '2E', '2F']

# PURPOSE: keep track of threads
def info(args):
    logging.debug(pathlib.Path(sys.argv[0]).name)
    logging.debug(args)
    logging.debug(f'module name: {__name__}')
    if hasattr(os, 'getppid'):
        logging.debug(f'parent process: {os.getppid():d}')
    logging.debug(f'process id: {os.getpid():d}')

# PURPOSE: calculate the campaign bias correction
def campaign_bias_correction(campaign: str):
    """
    Additive bias correction based on the laser number
        for a ICESat campaign

    Parameters
    ----------
    campaign: str
        ICESat campaign
    """
    # corrections for each laser number
    correction = dict(laser1=0.0, laser2=-0.017, laser3=0.011)
    # find the laser number from the campaign
    laser, = re.findall(r'\d', campaign)
    try:
        return correction[f'laser{laser}']
    except KeyError as exc:
        return 0.0

# PURPOSE: read GLAH12 file and associated corrections
# convert heights from TOPEX/Poseidon to WGS84 and ITRF2020
def read_GLAH12_file(GRANULE,
        indices=Ellipsis,
        TIDE_MODEL=None,
        REANALYSIS=None,
        GEOID=None,
        DEM_FILTER=False
    ):
    # try to extract the file format from the granule name
    try:
        PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE = \
            rx.findall(GRANULE.name).pop()
    except (ValueError, IndexError):
        # output grounding zone HDF5 file (generic)
        FILE_FORMAT = 'generic'
    else:
        # output grounding zone HDF5 file for NSIDC granules
        file_format = 'GLAH{0}_{1}_{2}_{3}{4}{5}_{6}_{7}_{8}_{9}_{10}.h5'
        FILE_FORMAT = 'standard'

    # open the HDF5 file for reading
    fid = gz.io.multiprocess_h5py(GRANULE, mode='r')
    logging.info(GRANULE)
    # copy ICESat campaign name from ancillary data
    ancillary_data = fid['ANCILLARY_DATA']
    campaign = ancillary_data.attrs['Campaign'].decode('utf-8')
    # get variables and attributes
    # time of ICESat data
    J2000 = fid['Data_40HZ']['DS_UTCTime_40'][indices].copy()
    ts = timescale.time.Timescale().from_deltatime(
        J2000, epoch=timescale.time._j2000_epoch,
        standard='UTC')
    # number of 40Hz data points
    n_40HZ = len(ts)
    # campaign bias correction
    bias_corr = campaign_bias_correction(campaign)
    # saturation correction
    sat_corr = fid['Data_40HZ']['Elevation_Corrections']['d_satElevCorr'][indices]
    # Longitude (degrees East)
    d_lon = fid['Data_40HZ']['Geolocation']['d_lon'][indices]
    # Latitude (TOPEX/Poseidon ellipsoid degrees North)
    d_lat = fid['Data_40HZ']['Geolocation']['d_lat'][indices]
    # Elevation (height above TOPEX/Poseidon ellipsoid in meters)
    d_elev = fid['Data_40HZ']['Elevation_Surfaces']['d_elev'][indices]
    fv = fid['Data_40HZ']['Elevation_Surfaces']['d_elev'].fillvalue
    # distance to the reference ground track
    d_d2refTrk = fid['Data_40HZ']['Geophysical']['d_d2refTrk'][indices]
    # retide the elevation data
    d_ocElv = fid['Data_40HZ']['Geophysical']['d_ocElv'][indices]
    d_ocElv[d_ocElv == fv] = 0.0
    # replace the solid earth tide with IERS 2010 recommendations
    d_erElv = fid['Data_40HZ']['Geophysical']['d_erElv'][indices]
    d_erElv[d_erElv == fv] = 0.0
    # internal digital elevation model (DEM)
    d_DEM_elv = fid['Data_40HZ']['Geophysical']['d_DEM_elv'][indices]
    d_DEM_elv = np.ma.masked_equal(d_DEM_elv, fv)
    # high resolution DEM elevation array
    d_DEMhiresArElv = fid['Data_40HZ']['Geophysical']['d_DEMhiresArElv'][indices,:]
    d_DEMhiresArElv = np.ma.masked_equal(d_DEMhiresArElv, fv)
    # calculate the standard deviation of the high resolution DEM array
    d_DEMhiresRMS = np.ma.std(d_DEMhiresArElv, axis=1)

    # get the transform for converting to the latest ITRF
    transform = gz.crs.tp_itrf2008_to_wgs84_itrf2020()
    # transform the data to WGS84 ellipsoid in ITRF2020
    d_corr = d_elev[:] + sat_corr + bias_corr + d_erElv[:] + d_ocElv[:]
    lon, lat, data, tdec = transform.transform(d_lon, d_lat, d_corr, ts.year)
    # calculate the solid earth tides (tide free)
    tide_earth = pyTMD.compute.SET_displacements(lon, lat, J2000,
        EPSG=4326, EPOCH=timescale.time._j2000_epoch, TYPE='drift',
        TIME='UTC', ELLIPSOID='WGS84', TIDE_SYSTEM='tide_free',
        EPHEMERIDES='JPL')
    # remove the solid earth tides from the elevation data
    data -= tide_earth

    # quality summary HDF5 file
    VAR = 'QA'    
    if (FILE_FORMAT == 'standard'):
        a1 = (PRD,RL,VAR,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        f1 = GRANULE.with_name(file_format.format(*a1))
    elif (FILE_FORMAT == 'generic'):
        file1 = f'{GRANULE.stem}_{VAR}{GRANULE.suffix}'
        f1 = GRANULE.with_name(file1)

    # mask for reducing to valid values
    if f1.exists():
        # quality summary mask file
        fid1 = gz.io.multiprocess_h5py(f1, mode='r')
        quality_summary = fid1['Data_40HZ']['Quality']['qa_sum_flg'][indices]
        fid1.close()
    else:
        quality_summary = np.ones((n_40HZ), dtype=int)

    # mask invalid values
    elev = np.ma.array(data, fill_value=fv)
    elev.mask = (d_elev == fv) | np.isnan(elev.data) | (quality_summary > 0)
    # filter against internal digital elevation models (DEM)
    # but don't filter simply because the DEM is not available
    if DEM_FILTER:
        # check absolute heights against internal DEM
        max_DEM_diff = 250.0
        DEM_diff_mask = (np.abs(d_elev - d_DEM_elv) > max_DEM_diff) & \
            np.logical_not(d_DEM_elv.mask)
        elev.mask |= DEM_diff_mask   
        # check roughness from high resolution DEM standard deviations
        max_DEMhires_RMS = 50.0
        DEMhires_mask = (d_DEMhiresRMS > max_DEMhires_RMS) & \
            np.logical_not(d_DEMhiresRMS.mask)
        elev.mask |= DEMhires_mask

    # ocean tide model
    otide = np.ma.zeros((n_40HZ), fill_value=fv)
    if TIDE_MODEL and (FILE_FORMAT == 'standard'):
        VAR = f'{TIDE_MODEL}_TIDES'
        a2 = (PRD,RL,VAR,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        f2 = GRANULE.with_name(file_format.format(*a2))
    elif TIDE_MODEL and (FILE_FORMAT == 'generic'):
        VAR = f'{TIDE_MODEL}_TIDES'
        file2 = f'{GRANULE.stem}_{VAR}{GRANULE.suffix}'
        f2 = GRANULE.with_name(file2)
    if TIDE_MODEL:
        fid2 = gz.io.multiprocess_h5py(f2, mode='r')
        var = fid2['Data_40HZ']['Geophysical']['d_ocElv']
        otide.data[:] = var[indices]
        otide.mask = (otide.data[:] == var.fillvalue)
        fid2.close()
    else:
        # use default tide model
        var = fid['Data_40HZ']['Geophysical']['d_ocElv']
        otide.data[:] = var[indices]
        otide.mask = (otide.data[:] == var.fillvalue)
    # replace fill values with 0
    otide = otide.filled(fill_value=0.0)

    # inverse barometer or dynamic atmosphere correction
    IB = np.ma.zeros((n_40HZ), fill_value=fv)
    IB.mask = np.zeros((n_40HZ), dtype=bool)
    if REANALYSIS and (FILE_FORMAT == 'standard'):
        VAR = 'DAC' if (REANALYSIS == 'DAC') else f'{REANALYSIS}_IB'
        a3 = (PRD,RL,VAR,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        f3 = GRANULE.with_name(file_format.format(*a3))
    elif REANALYSIS and (FILE_FORMAT == 'generic'):
        VAR = 'DAC' if (REANALYSIS == 'DAC') else f'{REANALYSIS}_IB'
        file3 = f'{GRANULE.stem}_{VAR}{GRANULE.suffix}'
        f3 = GRANULE.with_name(file3)
    if REANALYSIS:
        fid3 = gz.io.multiprocess_h5py(f3, mode='r')
        key = 'd_dacElv' if (REANALYSIS == 'DAC') else 'd_ibElv'
        var = fid3['Data_40HZ']['Geophysical'][key]
        IB.data[:] = var[indices]
        IB.mask = (IB.data[:] == var.fillvalue)
        fid3.close()
    # replace fill values with 0
    IB = IB.filled(fill_value=0.0)

    # geoid height
    gdHt = np.ma.zeros((n_40HZ), fill_value=fv)
    if GEOID and (FILE_FORMAT == 'standard'):
        VAR = f'{GEOID}_GEOID'
        a5 = (PRD,RL,VAR,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        f5 = GRANULE.with_name(file_format.format(*a5))
    elif GEOID and (FILE_FORMAT == 'generic'):
        VAR = f'{GEOID}_GEOID'
        file5 = f'{GRANULE.stem}_{VAR}{GRANULE.suffix}'
        f5 = GRANULE.with_name(file5)
    if GEOID:
        fid5 = gz.io.multiprocess_h5py(f5, mode='r')
        var = fid5['Data_40HZ']['Geophysical']['d_gdHt']
        gdHt.data[:] = var[indices]
        gdHt.mask = (gdHt.data[:] == var.fillvalue)
        fid5.close()
    else:
        # use default geoid height
        var = fid['Data_40HZ']['Geophysical']['d_gdHt']
        gdHt.data[:] = var[indices]
        gdHt.mask = (gdHt.data[:] == var.fillvalue)

    # close the HDF5 file
    fid.close()

    # return variables
    return dict(lon=d_lon, lat=d_lat, h=elev,
        t=J2000, campaign=campaign, sat_corr=sat_corr,
        tide_ocean=otide, tide_earth=tide_earth, dac=IB,
        geoid=gdHt, d2rgt=d_d2refTrk)

def along_track_splines(d, x, y, z, max_order=3, **kwargs):
    """
    Use univariate splines to interpolate along-track coordinates
    """
    # set default keyword arguments
    kwargs.setdefault('s', 0)
    kwargs.setdefault('ext', 0)
    # try to create a spline with the given data
    for k in range(max_order, 0, -1):
        try:
            sx = scipy.interpolate.UnivariateSpline(d, x, k=k, **kwargs)
            sy = scipy.interpolate.UnivariateSpline(d, y, k=k, **kwargs)
            sz = scipy.interpolate.UnivariateSpline(d, z, k=k, **kwargs)
        except Exception as exc:
            pass
        else:
            return (sx, sy, sz)
    # if we get here, then we failed to create a spline
    raise ValueError(f'Failed to create spline for {d.size} points')

# PURPOSE: read ICESat ICESat/GLAS L2 GLA12 Ice Sheet elevation data
# and create an along-track GLA12 HDF5 file using a surface fit technique
def along_track_GLA12(track_file,
        HEM=None,
        ALONG_TRACK=200,
        SEARCH_RADIUS=600,
        ORDER_SPLINE=3,
        ORDER_TIME=1,
        ORDER_SPACE=1,
        RELATIVE=None,
        ITERATIONS=25,
        THRESHOLD=8,
        TIDE_MODEL=None,
        REANALYSIS=None,
        GEOID=None,
        DEM_FILTER=False,
        KNOTS=0,
        RUNS=2000,
        RGT_PHASE=None,
        MODE=0o775
    ):

    # input track file
    track_file = pathlib.Path(track_file).expanduser().absolute()
    RGT, = re.findall(r'(\d{4})', track_file.stem)
    base_dir = track_file.parents[1]
    # region of interest
    REGION = dict(N='GL', S='AA')
    # main variables to extract
    variables = ['lon', 'lat', 'h', 't', 'sat_corr',
        'tide_ocean', 'tide_earth', 'dac', 'geoid', 'd2rgt']

    # open the track file
    with h5py.File(track_file, 'r') as f1:
        logging.info(track_file)
        groups = sorted([g for g in f1.keys() if rx.match(g)])
        n = 0
        for granule in groups:
            indices = f1[granule]['index'][:].copy()
            n += len(indices)
        # try to extract the file format from the granule name
        try:
            PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE = \
                rx.findall(granule).pop()
        except (ValueError, IndexError):
            PRD, RL = ('GLAH12', '634')

        # create empty dictionary for the GLAH12 data
        GLAH12 = dict()
        lineage = sorted(groups)
        # create empty arrays for the variables
        for var in variables:
            GLAH12[var] = np.ma.zeros((n))
        # create empty arrays for the campaign
        GLAH12['campaign'] = np.zeros((n), dtype='i')
        #  create empty arrays for the repeat ground-track phase
        GLAH12['i_rgtp'] = np.zeros((n), dtype='i')
        # counter for filling arrays
        c1 = 0
        # loop over the granules
        for granule in groups:
            # indices for RGT
            indices = f1[granule]['index'][:].copy()
            c2 = c1 + len(indices)
            dinput = read_GLAH12_file(base_dir.joinpath(granule),
                indices=indices,
                TIDE_MODEL=TIDE_MODEL,
                REANALYSIS=REANALYSIS,
                GEOID=GEOID,
                DEM_FILTER=DEM_FILTER
            )
            # save the variables
            for var in variables:
                GLAH12[var][c1:c2] = dinput[var].copy()
            GLAH12['campaign'][c1:c2] = campaigns.index(dinput['campaign'])
            GLAH12['i_rgtp'][c1:c2] = f1[granule]['i_rgtp'][:].copy()
            # increment counter
            c1 = np.copy(c2)

    # reduce to valid
    mask = np.zeros((n), dtype=bool)
    for var in variables:
        mask |= GLAH12[var].mask
    # reduce to hemisphere
    if (HEM == 'N'):
        mask |= (GLAH12['lat'] <= 0.0)
    elif (HEM == 'S'):
        mask |= (GLAH12['lat'] > 0.0)
    # reduce to repeat ground-track phase
    if (RGT_PHASE is not None):
        mask |= (GLAH12['i_rgtp'] != RGT_PHASE)
    # check that there are some valid points
    if np.all(mask):
        raise ValueError(f'No valid points in {track_file.name}')
    # apply mask to all variables
    for var in variables:
        GLAH12[var].mask = mask
        GLAH12[var] = GLAH12[var].compressed()
    GLAH12['campaign'] = GLAH12['campaign'][~mask]
    GLAH12['i_rgtp'] = GLAH12['i_rgtp'][~mask]

    # convert longitude and latitude to Cartesian coordinates
    x, y, z = pyTMD.spatial.to_cartesian(GLAH12['lon'], GLAH12['lat'])
    # calculate distance coordinates in ECEF
    i = np.argmin(x + y + z)
    d = np.sqrt((x - x[i])**2 + (y - y[i])**2 + (z - z[i])**2)
    # sort by distance from the first point and keep only unique values
    d, s = np.unique(d, return_index=True)
    x = x[s]
    y = y[s]
    z = z[s]
    # sort the data
    for var in variables:
        GLAH12[var] = GLAH12[var][s]
    # sort the campaigns and repeat ground-track phase
    GLAH12['campaign'] = GLAH12['campaign'][s]
    GLAH12['i_rgtp'] = GLAH12['i_rgtp'][s]
    # adjust number of campaigns if RGT_PHASE is set
    if (RGT_PHASE == 1):
        campaign = copy.copy(campaigns[:3])
    elif (RGT_PHASE == 2):
        campaign = copy.copy(campaigns[2:])
    else:
        campaign = copy.copy(campaigns)
    # number of campaigns
    n_camp = len(campaign)

    # create a new set of distances
    dist = np.arange(0, np.max(d), ALONG_TRACK)
    # number of segments
    n_seg = len(dist)
    
    # weight by the distance from the reference ground track
    w = np.abs(GLAH12['d2rgt'])**(-1.0)
    
    # randomly sample the along-track coordinates
    # and calculate the median of the random samples
    xtemp = np.zeros((n_seg, RUNS))
    ytemp = np.zeros((n_seg, RUNS))
    ztemp = np.zeros((n_seg, RUNS))
    # indices for the data
    indices = np.arange(0, len(d))
    # number of samples for each run
    # if KNOTS is set, then use that number of samples
    # otherwise use the number of points divided by the number of runs
    n_samp = np.int64(KNOTS) if (KNOTS > 0) else (len(d)//RUNS)
    # create a random number generator
    rng = np.random.default_rng()
    # for each monte carlo run
    for N in range(RUNS):
        # randomly sample indices
        s = rng.choice(indices, size=n_samp, replace=False, shuffle=False)
        # verify that the samples are monotonicly increasing
        i = np.sort(s)
        # try using scipy interpolating splines
        sx, sy, sz = along_track_splines(d[i], x[i], y[i], z[i],
            max_order=ORDER_SPLINE, w=w[i], s=None)
        # interpolate the data for iteration
        xtemp[:,N] = sx(dist)
        ytemp[:,N] = sy(dist)
        ztemp[:,N] = sz(dist)
    # calculate the median of the random samples
    xi = np.median(xtemp, axis=1)
    yi = np.median(ytemp, axis=1)
    zi = np.median(ztemp, axis=1)

    # convert to geodetic and geocentric coordinates
    longitude, latitude, _ = pyTMD.spatial.to_geodetic(xi, yi, zi)
    latitude_geocentric = np.arctan(zi / np.sqrt(xi**2.0 + yi**2.0))
    # legendre polynomial of degree 2 (unnormalized)
    theta = (np.pi/2.0 - latitude_geocentric)
    P2 = 0.5*(3.0*np.cos(theta)**2 - 1.0)
    # body tide love numbers for degree 2
    k2 = 0.300
    h2 = 0.609

    # output variables
    segment = collections.OrderedDict()
    attributes = collections.OrderedDict()
    fill_value = -9999.0
    # segment identifiers
    segment['ref_pt'] = np.arange((n_seg)) + 1
    attributes['ref_pt'] = collections.OrderedDict()
    attributes['ref_pt']['contentType'] = "referenceInformation"
    attributes['ref_pt']['long_name'] = "Index of reference point"
    attributes['ref_pt']['valid_min'] = 1
    attributes['ref_pt']['valid_max'] = n_seg
    attributes['ref_pt']['coordinates'] = "longitude latitude"
    # ICESat campaign
    segment['campaign'] = np.arange((n_camp)) + 1
    attributes['campaign'] = collections.OrderedDict()
    attributes['campaign']['contentType'] = "referenceInformation"
    attributes['campaign']['long_name'] = "Index of ICESat campaign"
    attributes['campaign']['flag_meanings'] = copy.copy(campaign)
    attributes['campaign']['valid_min'] = 1
    attributes['campaign']['valid_max'] = n_camp
    # along-track distance
    segment['x_atc'] = np.copy(dist)
    attributes['x_atc'] = collections.OrderedDict()
    attributes['x_atc']['units'] = "meters"
    attributes['x_atc']['contentType'] = "derived"
    attributes['x_atc']['long_name'] = "Along-track distance"
    attributes['x_atc']['coordinates'] = "longitude latitude"
    # longitude
    segment['longitude'] = np.copy(longitude)
    attributes['longitude'] = collections.OrderedDict()
    attributes['longitude']['units'] = "degrees_east"
    attributes['longitude']['contentType'] = "physicalMeasurement"
    attributes['longitude']['long_name'] = "Longitude"
    attributes['longitude']['standard_name'] = "longitude"
    attributes['longitude']['description'] = "Longitude of point location"
    attributes['longitude']['valid_min'] = -180.0
    attributes['longitude']['valid_max'] = 180.0
    # latitude
    segment['latitude'] = np.copy(latitude)
    attributes['latitude'] = collections.OrderedDict()
    attributes['latitude']['units'] = "degrees_north"
    attributes['latitude']['contentType'] = "physicalMeasurement"
    attributes['latitude']['long_name'] = "Latitude"
    attributes['latitude']['standard_name'] = "latitude"
    attributes['latitude']['description'] = "Latitude of point location"
    attributes['latitude']['valid_min'] = -90.0
    attributes['latitude']['valid_max'] = 90.0
    # corrected height
    segment['h_corr'] = np.ma.zeros((n_seg,n_camp), fill_value=fill_value)
    segment['h_corr'].mask = np.ones((n_seg,n_camp), dtype=bool)
    attributes['h_corr'] = collections.OrderedDict()
    attributes['h_corr']['units'] = "meters"
    attributes['h_corr']['contentType'] = "physicalMeasurement"
    attributes['h_corr']['long_name'] = "Mean corrected height"
    attributes['h_corr']['description'] = "height above WGS84 ellipsoid"
    attributes['h_corr']['reference_system'] = "ITRF2020"
    attributes['h_corr']['coordinates'] = "delta_time longitude latitude"
    # uncertainty in corrected height
    segment['h_corr_sigma'] = np.ma.zeros((n_seg,n_camp), fill_value=fill_value)
    segment['h_corr_sigma'].mask = np.ones((n_seg,n_camp), dtype=bool)
    attributes['h_corr_sigma'] = collections.OrderedDict()
    attributes['h_corr_sigma']['units'] = "meters"
    attributes['h_corr_sigma']['contentType'] = "physicalMeasurement"
    attributes['h_corr_sigma']['long_name'] = "Height uncertainty"
    attributes['h_corr_sigma']['description'] = "uncertainty in corrected height"
    attributes['h_corr_sigma']['reference_system'] = "ITRF2020"
    attributes['h_corr_sigma']['coordinates'] = "delta_time longitude latitude"
    # delta time (J2000 seconds)
    segment['delta_time'] = np.ma.zeros((n_seg,n_camp), fill_value=fill_value)
    segment['delta_time'].mask = np.ones((n_seg,n_camp), dtype=bool)
    attributes['delta_time'] = collections.OrderedDict()
    attributes['delta_time']['units'] = "seconds since 2000-01-01T12:00:00"
    attributes['delta_time']['contentType'] = "physicalMeasurement"
    attributes['delta_time']['long_name'] = "Time in J2000 seconds"
    attributes['delta_time']['standard_name'] = "time"
    attributes['delta_time']['calendar'] = "standard"
    attributes['delta_time']['description'] = ("Number of UTC seconds since "
        "the J2000 epoch (2000-01-01T12:00:00.000000Z)")
    attributes['delta_time']['coordinates'] = "delta_time longitude latitude"
    # ocean tide
    segment['tide_ocean'] = np.ma.zeros((n_seg,n_camp), fill_value=fill_value)
    segment['tide_ocean'].mask = np.ones((n_seg,n_camp), dtype=bool)
    attributes['tide_ocean'] = collections.OrderedDict()
    attributes['tide_ocean']['units'] = "meters"
    attributes['tide_ocean']['contentType'] = "derived"
    attributes['tide_ocean']['long_name'] = "Average ocean tide"
    attributes['tide_ocean']['coordinates'] = "delta_time longitude latitude"
    # solid earth tide
    segment['tide_earth'] = np.ma.zeros((n_seg,n_camp), fill_value=fill_value)
    segment['tide_earth'].mask = np.ones((n_seg,n_camp), dtype=bool)
    attributes['tide_earth'] = collections.OrderedDict()
    attributes['tide_earth']['units'] = "meters"
    attributes['tide_earth']['contentType'] = "derived"
    attributes['tide_earth']['long_name'] = "Average solid earth tide"
    attributes['tide_earth']['description'] = ("Solid earth tides "
        "in the tide-free system")
    attributes['tide_earth']['coordinates'] = "delta_time longitude latitude"
    # solid earth tide conversion to mean-tide system
    # using conversion values from Mathews et al. (1997)
    segment['tide_earth_free2mean'] = np.zeros((n_seg))
    segment['tide_earth_free2mean'][:] = 0.3146*np.sqrt(5.0/(4.0*np.pi))*h2*P2
    attributes['tide_earth_free2mean'] = collections.OrderedDict()
    attributes['tide_earth_free2mean']['units'] = "meters"
    attributes['tide_earth_free2mean']['contentType'] = "derived"
    attributes['tide_earth_free2mean']['long_name'] = \
        "Earth Tide Free-to-Mean conversion"
    attributes['tide_earth_free2mean']['description'] = \
        ("Additive value to convert solid earth tide from the "
        "tide-free system to the mean tide system")
    attributes['tide_earth_free2mean']['coordinates'] = "longitude latitude"
    # inverse barometer or dynamic atmosphere correction
    segment['dac'] = np.ma.zeros((n_seg,n_camp), fill_value=fill_value)
    segment['dac'].mask = np.ones((n_seg,n_camp), dtype=bool)
    attributes['dac'] = collections.OrderedDict()
    attributes['dac']['units'] = "meters"
    attributes['dac']['contentType'] = "derived"
    attributes['dac']['long_name'] = "Average dynamic atmospheric correction"
    attributes['dac']['coordinates'] = "delta_time longitude latitude"
    # ICESat saturation correction
    segment['satwfc'] = np.ma.zeros((n_seg,n_camp), fill_value=fill_value)
    segment['satwfc'].mask = np.ones((n_seg,n_camp), dtype=bool)
    attributes['satwfc'] = collections.OrderedDict()
    attributes['satwfc']['units'] = "meters"
    attributes['satwfc']['contentType'] = "derived"
    attributes['satwfc']['long_name'] = "Average saturated waveform correction"
    attributes['satwfc']['coordinates'] = "delta_time longitude latitude"
    attributes['satwfc']['description'] = ("Saturated waveform correction "
        "applied to the corrected elevation estimates")
    # campaign bias correction
    segment['icbc'] = np.zeros((n_camp))
    segment['icbc'][:] = [campaign_bias_correction(c) for c in campaign]
    attributes['icbc'] = collections.OrderedDict()
    attributes['icbc']['units'] = "meters"
    attributes['icbc']['contentType'] = "derived"
    attributes['icbc']['long_name'] = "ICESat inter-campaign bias correction"
    attributes['icbc']['description'] = ("ICESat inter-campaign bias "
        "correction applied to the corrected elevation estimates")
    # geoid height
    segment['geoid_h'] = np.ma.zeros((n_seg), fill_value=fill_value)
    segment['geoid_h'].mask = np.ones((n_seg), dtype=bool)
    attributes['geoid_h'] = collections.OrderedDict()
    attributes['geoid_h']['units'] = "meters"
    attributes['geoid_h']['contentType'] = "derived"
    attributes['geoid_h']['long_name'] = "Average geoid height"
    attributes['geoid_h']['description'] = ("Geoidal undulation with "
        "respect to the WGS84 ellipsoid")
    attributes['geoid_h']['coordinates'] = "longitude latitude"
    # geoid conversion to mean-tide system
    # from Rapp 1991 (Consideration of Permanent Tidal Deformation)
    segment['geoid_free2mean'] = np.zeros((n_seg))
    segment['geoid_free2mean'][:] = -0.198*(1.0 + k2)*P2
    attributes['geoid_free2mean'] = collections.OrderedDict()
    attributes['geoid_free2mean']['units'] = "meters"
    attributes['geoid_free2mean']['contentType'] = "derived"
    attributes['geoid_free2mean']['long_name'] = "Geoid Free-to-Mean conversion"
    attributes['geoid_free2mean']['description'] = \
        ("Additive value to convert geoid heights from the "
        "tide-free system to the mean tide system")
    attributes['geoid_free2mean']['coordinates'] = "longitude latitude"
    # mean height
    segment['h_mean'] = np.ma.zeros((n_seg), fill_value=fill_value)
    segment['h_mean'].mask = np.ones((n_seg), dtype=bool)
    attributes['h_mean'] = collections.OrderedDict()
    attributes['h_mean']['units'] = "meters"
    attributes['h_mean']['contentType'] = "derived"
    attributes['h_mean']['long_name'] = "Average height from fit"
    attributes['h_mean']['reference_system'] = "ITRF2020"
    attributes['h_mean']['coordinates'] = "longitude latitude"
    # uncertainty in mean height
    segment['h_sigma'] = np.ma.zeros((n_seg), fill_value=fill_value)
    segment['h_sigma'].mask = np.ones((n_seg), dtype=bool)
    attributes['h_sigma'] = collections.OrderedDict()
    attributes['h_sigma']['units'] = "meters"
    attributes['h_sigma']['contentType'] = "derived"
    attributes['h_sigma']['long_name'] = "Uncertainty in average height from fit"
    attributes['h_sigma']['coordinates'] = "longitude latitude"
    # RDE of height residuals
    segment['h_robust_sprd'] = np.ma.zeros((n_seg), fill_value=fill_value)
    segment['h_robust_sprd'].mask = np.ones((n_seg), dtype=bool)
    attributes['h_robust_sprd'] = collections.OrderedDict()
    attributes['h_robust_sprd']['units'] = "meters"
    attributes['h_robust_sprd']['contentType'] = "derived"
    attributes['h_robust_sprd']['long_name'] = "Robust Spread"
    attributes['h_robust_sprd']['description'] = \
        "RDE of height residuals from surface-polynomial fit"
    attributes['h_robust_sprd']['coordinates'] = "longitude latitude"
    # misfit from fit
    segment['misfit_RMS'] = np.ma.zeros((n_seg), fill_value=fill_value)
    segment['misfit_RMS'].mask = np.ones((n_seg), dtype=bool)
    attributes['misfit_RMS'] = collections.OrderedDict()
    attributes['misfit_RMS']['units'] = "meters"
    attributes['misfit_RMS']['contentType'] = "derived"
    attributes['misfit_RMS']['long_name'] = "RMS misfit for the surface-polynomial fit"
    attributes['misfit_RMS']['coordinates'] = "longitude latitude"
    # east slope
    segment['e_slope'] = np.ma.zeros((n_seg), fill_value=fill_value)
    segment['e_slope'].mask = np.ones((n_seg), dtype=bool)
    attributes['e_slope'] = collections.OrderedDict()
    attributes['e_slope']['units'] = "1"
    attributes['e_slope']['contentType'] = "derived"
    attributes['e_slope']['long_name'] = "East-component slope"
    attributes['e_slope']['coordinates'] = "longitude latitude"
    # north slope
    segment['n_slope'] = np.ma.zeros((n_seg), fill_value=fill_value)
    segment['n_slope'].mask = np.ones((n_seg), dtype=bool)
    attributes['n_slope'] = collections.OrderedDict()
    attributes['n_slope']['units'] = "1"
    attributes['n_slope']['contentType'] = "derived"
    attributes['n_slope']['long_name'] = "North-component slope"
    attributes['n_slope']['coordinates'] = "longitude latitude"
    # elevation change rate and uncertainty
    attributes['dhdt'] = collections.OrderedDict()
    attributes['dhdt']['units'] = "meters/year"
    attributes['dhdt']['contentType'] = "derived"
    attributes['dhdt']['long_name'] = "Elevation change rate"
    attributes['dhdt']['coordinates'] = "longitude latitude"  
    attributes['dhdt_sigma'] = collections.OrderedDict()
    attributes['dhdt_sigma']['units'] = "meters/year"
    attributes['dhdt_sigma']['contentType'] = "derived"
    attributes['dhdt_sigma']['long_name'] = "Uncertainty in elevation change rate"
    attributes['dhdt_sigma']['coordinates'] = "longitude latitude"
    if (ORDER_TIME >= 1):
        segment['dhdt'] = np.ma.zeros((n_seg), fill_value=fill_value)
        segment['dhdt'].mask = np.ones((n_seg), dtype=bool)
        segment['dhdt_sigma'] = np.ma.zeros((n_seg), fill_value=fill_value)
        segment['dhdt_sigma'].mask = np.ones((n_seg), dtype=bool)
    # iterations
    segment['iterations'] = np.zeros((n_seg), dtype='i')
    attributes['iterations'] = collections.OrderedDict()
    attributes['iterations']['long_name'] = 'Number of fit iterations'
    attributes['iterations']['contentType'] = "derived"
    attributes['iterations']['units'] = '1'
    attributes['iterations']['coordinates'] = "longitude latitude"
    # degrees of freedom
    segment['DOF'] = np.zeros((n_seg), dtype='i')
    attributes['DOF'] = collections.OrderedDict()
    attributes['DOF']['long_name'] = 'Degrees of freedom'
    attributes['DOF']['contentType'] = "derived"
    attributes['DOF']['units'] = '1'
    attributes['DOF']['coordinates'] = "longitude latitude"
    # data count
    segment['count'] = np.zeros((n_seg, n_camp), dtype='i')
    attributes['count'] = collections.OrderedDict()
    attributes['count']['long_name'] = 'Number of data points'
    attributes['count']['contentType'] = "derived"
    attributes['count']['units'] = '1'
    attributes['count']['coordinates'] = "delta_time longitude latitude"
    # surface fit keywords
    kwargs = dict(CX=0, CY=0, FIT_TYPE='polynomial', BOUNDED=True,
        ORDER_TIME=ORDER_TIME, ORDER_SPACE=ORDER_SPACE,
        ITERATIONS=ITERATIONS, THRESHOLD=THRESHOLD,
        RELATIVE=RELATIVE, STDEV=1)
    # number of temporal terms
    n_terms = gz.fit._temporal_terms(**kwargs)

    # for each segment, fit a surface to the data
    for iseg in range(n_seg):
        # convert coordinates to local east-north-up (ENU)
        E, N, U = pyTMD.spatial.to_ENU(x, y, z,
            lon0=longitude[iseg], lat0=latitude[iseg])
        # distance from the segment center
        radius = np.sqrt(E**2 + N**2 + U**2)
        # find all points within the search radius
        ii, = np.nonzero(radius <= SEARCH_RADIUS)
        # reduce to the search radius
        t_in = GLAH12['t'][ii]
        e_in = E[ii]
        n_in = N[ii]
        h_in = GLAH12['h'][ii]
        c_in = GLAH12['campaign'][ii]
        otide = GLAH12['tide_ocean'][ii]
        setide = GLAH12['tide_earth'][ii]
        ib = GLAH12['dac'][ii]
        sat_corr = GLAH12['sat_corr'][ii]
        geoid = GLAH12['geoid'][ii]
        m_in = np.zeros_like(t_in, dtype=bool)
        # convert times from J2000 seconds
        ts = timescale.time.Timescale().from_deltatime(
            t_in, epoch=timescale.time._j2000_epoch,
            standard='UTC')
        # fit the surface
        try:
            fit = gz.fit.iterative_surface(
                ts.year, e_in, n_in, h_in, **kwargs)
        except Exception as e:
            continue
        # fit indices
        ifit = fit['indices']
        m_in[ifit] = True        
        # save the mean height and uncertainty
        segment['h_mean'][iseg] = fit['beta'][0].copy()
        segment['h_mean'].mask[iseg] = np.isnan(fit['beta'][0])
        segment['h_sigma'][iseg] = fit['error'][0].copy()
        segment['h_sigma'].mask[iseg] = np.isnan(fit['error'][0])
        segment['h_robust_sprd'][iseg] = fit['RDE'].copy()
        segment['h_robust_sprd'].mask[iseg] = np.isnan(fit['RDE'])
        # save the misfit RMS
        misfit_RMS = np.sqrt(fit['MSE'])
        segment['misfit_RMS'][iseg] = misfit_RMS.copy()
        segment['misfit_RMS'].mask[iseg] = np.isnan(misfit_RMS)
        # save the elevation change rate and uncertainty
        if (ORDER_TIME >= 1):
            segment['dhdt'][iseg] = fit['beta'][1].copy()
            segment['dhdt'].mask[iseg] = np.isnan(fit['beta'][1])
            segment['dhdt_sigma'][iseg] = fit['error'][1].copy()
            segment['dhdt_sigma'].mask[iseg] = np.isnan(fit['error'][1])
        # save the east and north slopes
        e_slope = fit['beta'][ORDER_TIME+2]
        e_slope_sigma = fit['error'][ORDER_TIME+2]
        segment['e_slope'][iseg] = e_slope.copy()
        segment['e_slope'].mask[iseg] = np.isnan(e_slope)
        n_slope = fit['beta'][ORDER_TIME+1]
        n_slope_sigma = fit['error'][ORDER_TIME+1]
        segment['n_slope'][iseg] = n_slope.copy()
        segment['n_slope'].mask[iseg] = np.isnan(n_slope)
        # save the geoid height
        geoid_h = np.nanmean(geoid[ifit])
        segment['geoid_h'][iseg] = geoid_h.copy()
        segment['geoid_h'].mask[iseg] = np.isnan(geoid_h)
        # save the number of iterations and degrees of freedom
        segment['iterations'][iseg] = np.copy(fit['iterations'])
        segment['DOF'][iseg] = np.copy(fit['DOF'])
        # calculate the mean for each campaign
        for icamp in range(n_camp):
            v, = np.nonzero((c_in == icamp) & (m_in) & np.isfinite(h_in))
            if len(v) == 0:
                continue
            # build design matrix for reducing with only spatial terms
            DMAT, _ = gz.fit._build_design_matrix(
                t_in[v], e_in[v], n_in[v], **kwargs)
            # spatial model
            spatial_model = np.dot(DMAT[:,n_terms:], fit['beta'][n_terms:])
            # weight all points equally
            weights = np.ones_like(v, dtype=np.float64)
            # calculate sum of the weights for normalizing
            w_sum = np.nansum(weights)
            # reduce the height
            reduced = h_in[v] - spatial_model
            h_corr = np.nansum(reduced * weights)/w_sum
            segment['h_corr'].data[iseg,icamp] = h_corr.copy()
            segment['h_corr'].mask[iseg,icamp] = np.isnan(h_corr)
            # estimate total uncertainties
            sigma_h = np.nansum((weights*(reduced - h_corr))**2)
            sigma_e = np.nansum((weights*(e_slope_sigma * e_in[v]))**2)
            sigma_n = np.nansum((weights*(n_slope_sigma * n_in[v]))**2)
            h_corr_sigma = np.sqrt((sigma_h + sigma_e + sigma_n)/w_sum)
            segment['h_corr_sigma'].data[iseg,icamp] = h_corr_sigma.copy()
            segment['h_corr_sigma'].mask[iseg,icamp] = np.isnan(h_corr_sigma)
            # save the time and geophysical corrections
            delta_time = np.nansum(t_in[v] * weights)/w_sum
            segment['delta_time'][iseg,icamp] = delta_time.copy()
            segment['delta_time'].mask[iseg,icamp] = np.isnan(delta_time)
            tide_ocean = np.nansum(otide[v] * weights)/w_sum
            segment['tide_ocean'][iseg,icamp] = tide_ocean.copy()
            segment['tide_ocean'].mask[iseg,icamp] = np.isnan(tide_ocean)
            tide_earth = np.nansum(setide[v] * weights)/w_sum
            segment['tide_earth'][iseg,icamp] = tide_earth.copy()
            segment['tide_earth'].mask[iseg,icamp] = np.isnan(tide_earth)
            dac = np.nansum(ib[v] * weights)/w_sum
            segment['dac'][iseg,icamp] = dac.copy()
            segment['dac'].mask[iseg,icamp] = np.isnan(dac)
            # save the saturation correction
            satwfc = np.nansum(sat_corr[v] * weights)/w_sum
            segment['satwfc'][iseg,icamp] = satwfc.copy()
            segment['satwfc'].mask[iseg,icamp] = np.isnan(satwfc)
            # save the number of data points
            segment['count'][iseg,icamp] = len(v)

    # replace fill values
    for key,val in segment.items():
        if np.ma.isMaskedArray(val):
            segment[key].data[val.mask] = val.fill_value

    # open output index file
    granule = f'GLAH{PRD}_{RL}_{REGION[HEM]}_{RGT}_atc{ALONG_TRACK:0.0f}m.h5'
    output_file = base_dir.joinpath(granule)
    logging.info(output_file)
    f2 = h5py.File(output_file, mode='w')
    # add global attributes
    f2.attrs['platform'] = "Ice, Cloud, and Land Elevation Satellite (ICESat)"
    f2.attrs['instrument'] = "Geoscience Laser Altimeter System (GLAS)"
    f2.attrs['title'] = "ICESat/GLAS Antarctic and Greenland Ice Sheet altimetry data"
    f2.attrs['summary'] = ("Along-track segments from ICESat/GLAS L2 GLA12 "
        "Antarctic and Greenland Ice Sheet elevation data")
    f2.attrs['processing_level'] = "4"
    f2.attrs['Conventions'] = "CF-1.6"
    f2.attrs['featureType'] = "timeSeries"
    # add geospatial attributes
    f2.attrs['geospatial_lat_min'] = segment['latitude'].min()
    f2.attrs['geospatial_lat_max'] = segment['latitude'].max()
    f2.attrs['geospatial_lon_min'] = segment['longitude'].min()
    f2.attrs['geospatial_lon_max'] = segment['longitude'].max()
    f2.attrs['geospatial_lat_units'] = "degrees_north"
    f2.attrs['geospatial_lon_units'] = "degrees_east"
    f2.attrs['geospatial_ellipsoid'] = "WGS84"
    f2.attrs['geospatial_itrf'] = "ITRF2020"
    # add time attributes
    f2.attrs['time_type'] = "UTC"
    f2.attrs['date_type'] = "J2000"
    # convert start and end time from J2000 seconds into timescale
    tmn = segment['delta_time'].min()
    tmx = segment['delta_time'].max()
    ts = timescale.time.Timescale().from_deltatime(np.array([tmn, tmx]),
        epoch=timescale.time._j2000_epoch, standard='UTC')
    dt = np.datetime_as_string(ts.to_datetime(), unit='s')
    f2.attrs['time_coverage_start'] = str(dt[0])
    f2.attrs['time_coverage_end'] = str(dt[1])
    f2.attrs['time_coverage_duration'] = f'{tmx-tmn:0.0f}'
    today = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    f2.attrs['date_created'] = today
    # add parameter attributes
    f2.attrs['lineage'] = lineage
    f2.attrs['campaign'] = campaign
    f2.attrs['track'] = RGT
    f2.attrs['hemisphere'] = HEM
    f2.attrs['search_radius'] = SEARCH_RADIUS
    f2.attrs['x_atc'] = ALONG_TRACK
    f2.attrs['deg_x'] = ORDER_SPACE
    f2.attrs['deg_y'] = ORDER_SPACE
    f2.attrs['deg_t'] = ORDER_TIME
    # add software information
    f2.attrs['software_reference'] = gz.version.project_name
    f2.attrs['software_version'] = gz.version.full_version

    # group for campaign statistics variables
    campaign_stats = ['satwfc','icbc','tide_ocean','dac',
        'tide_earth','tide_earth_free2mean','count']
    f2.create_group('campaign_stats')
    f2['campaign_stats'].attrs['Description'] = \
        "Geophysical properties and statistics for each campaign"
    # group for reference surface variables
    ref_surf = ['x_atc','h_mean','h_sigma','h_robust_sprd',
        'misfit_RMS','dhdt','dhdt_sigma','e_slope','n_slope',
        'iterations','DOF','geoid_h','geoid_free2mean']
    f2.create_group('ref_surf')
    f2['ref_surf'].attrs['Description'] = \
        "Fit statistics and reference surface information"

    # for each output variable
    h5 = {}
    for key,val in segment.items():
        # output group
        if key in campaign_stats:
            g2 = f2['campaign_stats']
        elif key in ref_surf:
            g2 = f2['ref_surf']
        else:
            g2 = f2
        # check if HDF5 variable exists
        if key in g2:
            # overwrite HDF5 variable
            h5[key] = g2[key]
            h5[key][...] = val
        if key not in g2 and hasattr(val, 'fill_value'):
            # create HDF5 variable
            h5[key] = g2.create_dataset(key, val.shape, data=val,
                dtype=val.dtype, fillvalue=fill_value, compression='gzip')
        elif key in ('ref_pt', 'campaign'):
            # create dimensions with unlimited size
            h5[key] = g2.create_dataset(key, val.shape, data=val,
                dtype=val.dtype, maxshape=(None,))
        elif val.shape:
            h5[key] = g2.create_dataset(key, val.shape, data=val,
                dtype=val.dtype, compression='gzip')
        else:
            h5[key] = g2.create_dataset(key, val.shape,
                dtype=val.dtype)
        # add variable attributes
        for att_name,att_val in attributes[key].items():
            h5[key].attrs[att_name] = att_val
        # create or attach dimensions
        if key in ('ref_pt', 'campaign'):
            h5[key].make_scale(key)
        elif (val.ndim == 1) and (len(val) == n_seg):
            h5[key].dims[0].attach_scale(h5['ref_pt'])
        elif (val.ndim == 1) and (len(val) == n_camp):
            h5[key].dims[0].attach_scale(h5['campaign'])
        elif (val.ndim == 2):
            h5[key].dims[0].attach_scale(h5['ref_pt'])
            h5[key].dims[1].attach_scale(h5['campaign'])

    # Output HDF5 structure information
    logging.info(list(f2.keys()))
    # close the output file
    f2.close()
    # change the permissions mode of the output file
    output_file.chmod(mode=MODE)

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates along-track segments from
            ICESat/GLAS L2 GLA12 Antarctic and Greenland Ice Sheet
            elevation data
            """,
            fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('infile',
        type=pathlib.Path,
        help='ICESat track files to run')
    # region of interest to run
    parser.add_argument('--hemisphere','-H',
        type=str, default='S', choices=('N','S'),
        help='Region of interest to run')
    # filter RGTs for repeat ground-track phase
    parser.add_argument('--rgt-phase',
        type=int, default=2,
        help='Repeat ground-track phase to filter (1=8 day; 2=91 day)')
    # along-track distance for the segments
    parser.add_argument('--along-track','-A',
        type=float, default=200.0,
        help='Along-track distance for the segments (meters)')
    # search radius for the surface fit
    parser.add_argument('--search-radius','-S',
        type=float, default=600.0,
        help='Search radius for the surface fit (meters)')
    parser.add_argument('--iteration','-I',
        type=int, default=25,
        help='Number of iterations for surface fit')
    parser.add_argument('--order-spline',
        type=int, default=3,
        help='Smoothing spline order for along-track coordinates')
    parser.add_argument('--order-time',
        type=int, default=1,
        help='Temporal fit polynomial order')
    parser.add_argument('--order-space',
        type=int, default=1,
        help='Spatial fit polynomial order')
    parser.add_argument('--relative',
        type=float, default=2005.0,
        help='Relative period for time-variable fit')
    parser.add_argument('--point-threshold',
        type=int, default=8,
        help='Minimum number of points required for valid fit')
    # tide model to use
    parser.add_argument('--tide','-T',
        metavar='TIDE', type=str, default='CATS2008',
        help='Tide model to use in correction')
    # dynamic atmospheric correction
    parser.add_argument('--reanalysis','-R',
        metavar='REANALYSIS', type=str,
        help='Reanalysis model to use in inverse-barometer correction')
    # geoid height
    parser.add_argument('--geoid','-G',
        metavar='GEOID', type=str,
        help='Geoid height model to use in correction')
    # filter elevations against internal DEM
    parser.add_argument('--dem-filter',
        default=False, action='store_true',
        help='Filter elevations against internal DEM')
    # number of knots for along-track coordinates
    parser.add_argument('--knots',
        type=int, default=0,
        help='Number of knots to use for along-track coordinates')
    parser.add_argument('--runs',
        type=int, default=2000,
        help='Number of along-track coordinate Monte Carlo runs')
    # verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        action='count', default=0,
        help='Verbose output of processing run')
    # permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of directories and files created')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # create logger
    loglevels = [logging.CRITICAL, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=loglevels[args.verbose])

    # try to run tidal current program for input file
    try:
        info(args)
        # run the program with the specified arguments
        along_track_GLA12(args.infile,
            HEM=args.hemisphere,
            RGT_PHASE=args.rgt_phase,
            ALONG_TRACK=args.along_track,
            SEARCH_RADIUS=args.search_radius,
            ORDER_SPLINE=args.order_spline,
            ORDER_TIME=args.order_time,
            ORDER_SPACE=args.order_space,
            RELATIVE=args.relative,
            ITERATIONS=args.iteration,
            THRESHOLD=args.point_threshold,
            TIDE_MODEL=args.tide,
            REANALYSIS=args.reanalysis,
            GEOID=args.geoid,
            DEM_FILTER=args.dem_filter,
            KNOTS=args.knots,
            RUNS=args.runs,
            MODE=args.mode)
    except Exception as exc:
        # if there has been an error exception
        # print the type, value, and stack trace of the
        # current exception being handled
        logging.critical(f'process id {os.getpid():d} failed')
        logging.error(traceback.format_exc())

# run main program
if __name__ == '__main__':
    main()

