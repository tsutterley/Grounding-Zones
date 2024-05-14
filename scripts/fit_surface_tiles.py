#!/usr/bin/env python
u"""
fit_surface_tiles.py
Written by Tyler Sutterley (05/2024)

Fits a time-variable surface to altimetry data

INPUTS:
    infile(s): tile file(s) to run

COMMAND LINE OPTIONS:
    --help: list the command line options
    -O X, --output-directory X: output data directory
    -H X, --hemisphere X: Region of interest to run
    -W X, --width X: Output tile width
    -S X, --spacing X: Output grid spacing
    --mask-file X: Ice mask file
    --fit-type: Temporal fit type
        - polynomial
        - chebyshev
        - spline
    --iteration: Number of iterations for surface fit
    --order-time: Temporal fit polynomial order
    --order-space: Spatial fit polynomial order
    -R X, --relative X: Relative period for time-variable fit
    -K X, --knots X: Temporal knots for spline fit
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
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
    gdal: Pythonic interface to the Geospatial Data Abstraction Library (GDAL)
        https://pypi.python.org/pypi/GDAL/
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
    Updated 05/2024: switched from individual mask files to a
        common raster mask option for non-ice points
    Updated 04/2024: rewritten for python3 following new fit changes
        add spline design matrix option for time-variable fit
        add option to read from ATL06 or ATL11 datasets
        update inter-campaign bias corrections for G-C corrected data
        use functions from timescale.time for temporal operations
    Updated 04/2017: changed no input file exception to IOError
    Updated 08/2015: new definition to create the output directories
        prior to running the main program in parallel
        changed sys.exit to raise Exception
    Updated 07/2015: added support for release 34.
        added some code updates for generalization (GLAH12 or GLAH14)
    Updated 03/2015: updated for UTIG Riegl lidar.  Added error handling
    Updated 10/2014: updated for parameter files
        Updated for distributed computing using the
            multiprocessing module
        Updated for chebyshev polynomials
    Updated 07/2014: output point counts for each dataset
        Added criterion for OIB and Icesat track counts
        Output file basename to track progress
        If bad fit: output shot counts
    Updated 06/2014: improved parallelization capability
        calculate the points within patch separately
        Determine quality and fit different polynomials
    Updated 07/2014: added criterion for oib, lvis counts
    Updated 06/2014: complete rewrite of program
    Updated 05/2014: daily output, all ATM, LVIS, split
        surfaces for crossovers and along-track patches
    Forked 04/2014: test fit with campaign dates, early ATM
"""
import sys
import os
import re
import copy
import time
import logging
import pathlib
import argparse
import traceback
import numpy as np
import scipy.stats
import scipy.special
import scipy.interpolate
import grounding_zones as gz

# attempt imports
h5py = gz.utilities.import_dependency('h5py')
pyproj = gz.utilities.import_dependency('pyproj')
is2tk = gz.utilities.import_dependency('icesat2_toolkit')
timescale = gz.utilities.import_dependency('timescale')

# PURPOSE: keep track of threads
def info(args):
    logging.debug(pathlib.Path(sys.argv[0]).name)
    logging.debug(args)
    logging.debug(f'module name: {__name__}')
    if hasattr(os, 'getppid'):
        logging.debug(f'parent process: {os.getppid():d}')
    logging.debug(f'process id: {os.getpid():d}')

# PURPOSE: attempt to open an HDF5 file and wait if already open
def multiprocess_h5py(filename, *args, **kwargs):
    """
    Open an HDF5 file with a hold for already open files
    """
    # set default keyword arguments
    kwargs.setdefault('mode', 'r')
    # check that file exists if entering with read mode
    filename = pathlib.Path(filename).expanduser().absolute()
    if kwargs['mode'] in ('r','r+') and not filename.exists():
        raise FileNotFoundError(filename)
    # attempt to open HDF5 file
    while True:
        try:
            fileID = h5py.File(filename, *args, **kwargs)
            break
        except (IOError, BlockingIOError, PermissionError) as exc:
            time.sleep(1)
    # return the file access object
    return fileID

# PURPOSE: read a raster file and return the data
def read_raster_file(raster_file, **kwargs):
    """
    Read a geotiff file and prepare for use in interpolation
    """
    # read raster image for spatial coordinates and data
    dinput = is2tk.spatial.from_geotiff(raster_file, **kwargs)
    spacing = np.array(dinput['attributes']['spacing'])
    # check that x and y are strictly increasing
    if (np.sign(spacing[0]) == -1):
        dinput['x'] = dinput['x'][::-1]
        dinput['data'] = dinput['data'][:,::-1]
        spacing[0] *= -1.0
    if (np.sign(spacing[1]) == -1):
        dinput['y'] = dinput['y'][::-1]
        dinput['data'] = dinput['data'][::-1,:]
        spacing[1] *= -1.0
    # update the spacing attribute
    dinput['attributes']['spacing'] = spacing
    # return the mask object
    return dinput

def campaign_bias_correction(campaign: str):
    """
    Additive bias correction based on the laser number
        for a ICESat campaign

    Parameters
    ----------
    campaign: str
        ICESat campaign
    """
    # 
    correction = dict(laser1=0.0, laser2=-0.017, laser3=0.011)
    # find the laser number from the campaign
    laser, = re.findall(r'\d', campaign)
    try:
        return correction[f'laser{laser}']
    except KeyError as exc:
        return 0.0

def fit_surface_tiles(tile_files,
        OUTPUT_DIRECTORY=None,
        HEM='S',
        W=80e3,
        SPACING=None,
        MASK_FILE=None,
        FIT_TYPE=None,
        ITERATIONS=1,
        ORDER_TIME=0,
        ORDER_SPACE=0,
        RELATIVE=None,
        KNOTS=[],
        MODE=0o775
    ):

    # regular expression pattern for tile files
    R1 = re.compile(r'E([-+]?\d+)_N([-+]?\d+)', re.VERBOSE)
    # regular expression pattern for determining file type
    regex = {}
    regex['ATM'] = r'(BLATM2|ILATM2)_(\d+)_(\d+)_smooth_nadir(.*?)(csv|seg|pt)$'
    regex['ATM1b'] = r'(BLATM1b|ILATM1b)_(\d+)_(\d+)(.*?).(qi|TXT|h5)$'
    regex['LVIS'] = r'(BLVIS2|BVLIS2|ILVIS2)_(.*?)(\d+)_(\d+)_(R\d+)_(\d+).H5$'
    regex['LVGH'] = r'(ILVGH2)_(.*?)(\d+)_(\d+)_(R\d+)_(\d+).H5$'
    regex['ATL06'] = (r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})'
        r'(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    regex['ATL11'] = (r'(ATL\d{2})_(\d{4})(\d{2})_(\d{2})(\d{2})_'
        r'(\d{3})_(\d{2})(.*?).h5$')
    regex['GLAS'] = (r'GLAH(\d{2})_(\d{3})_(\d{1})(\d{1})(\d{2})_(\d{3})_'
        r'(\d{4})_(\d{1})_(\d{2})_(\d{4})\.H5')
    # identifier for dataset
    mission = dict(ATL06=0, ATL11=0, GLAS=1, ATM=2, ATM1b=2, LVIS=3, LVGH=3)
    mission_types = sorted(set(mission.values()))

    # extract information from input tile file
    tile_file = pathlib.Path(tile_files[0]).expanduser().absolute()
    # use first file as default output directory
    if OUTPUT_DIRECTORY is None:
        OUTPUT_DIRECTORY = tile_file.parents[1]
    # create output directory if non-existent
    OUTPUT_DIRECTORY.mkdir(mode=MODE, parents=True, exist_ok=True)
    # extract tile centers from filename
    tile_centers = R1.findall(tile_file.name).pop()
    xc, yc = 1000.0*np.array(tile_centers, dtype=np.float64)
    tile_file_formatted = f'E{xc/1e3:0.0f}_N{yc/1e3:0.0f}.h5'
    logging.info(f'Tile Center: {xc:0.1f} {yc:0.1f}')
    # grid dimensions
    xmin,xmax = xc + np.array([-0.5,0.5])*W
    ymin,ymax = yc + np.array([-0.5,0.5])*W
    dx,dy = np.broadcast_to(np.atleast_1d(SPACING),(2,))
    nx = np.int64(W//dx) + 1
    ny = np.int64(W//dy) + 1
    logging.info(f'Grid Dimensions {ny:d} {nx:d}')
    # invalid value
    FILL_VALUE = -9999.0
    # minimum number of years for valid fit
    TIME_THRESHOLD = 3.0
    # minimum number of points for valid fit
    POINT_THRESHOLD = 25

    # pyproj transformer for converting to polar stereographic
    EPSG = dict(N=3413, S=3031)[HEM]
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_epsg(EPSG)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    # dictionary of coordinate reference system variables
    cs_to_cf = crs2.cs_to_cf()
    crs_to_dict = crs2.to_dict()
    # flattening and standard parallel of datum and projection
    crs_to_cf = crs2.to_cf()
    flat = 1.0/crs_to_cf['inverse_flattening']
    reference_latitude = crs_to_cf['standard_parallel']

    # read the input file
    if MASK_FILE is not None:
        bounds = [xmin-dx, xmax+dx, ymin-dy, ymax+dy]
        m = read_raster_file(MASK_FILE, bounds=bounds)
        # calculate polar stereographic distortion
        # interpolate raster to output grid
        DX, DY = m['attributes']['spacing']
        GRIDx, GRIDy = np.meshgrid(m['x'], m['y'])
        IMx = np.clip((GRIDx - xmin)//dx, 0, nx-1).astype(int)
        IMy = np.clip((GRIDy - ymin)//dy, 0, ny-1).astype(int)
        indy, indx = np.nonzero(m['data'])
        mask = np.zeros((ny, nx))
        for i, j in zip(indy, indx):
            mask[IMy[i], IMx[j]] += DX*DY*m['data'][i,j]
        # convert to average
        mask /= (dx*dy)
        # create an interpolator for input raster data
        SPL = scipy.interpolate.RectBivariateSpline(
            m['x'], m['y'], m['data'].T, kx=1, ky=1)
        # tolerance to find valid elevations
        TOLERANCE = 0.5
    else:
        # assume all ice covered
        mask = np.ones((ny, nx))
        SPL = None

    # output coordinates (centered)
    x = np.arange(xmin + dx/2.0, xmax + dx, dx)
    y = np.arange(ymin + dx/2.0, ymax + dy, dy)
    gridx, gridy = np.meshgrid(x, y)
    # convert grid coordinates to latitude and longitude
    _, gridlat = transformer.transform(gridx, gridy, direction='inverse')
    ps_scale = is2tk.spatial.scale_factors(gridlat, flat=flat,
        reference_latitude=reference_latitude, metric='area')
    # calculate cell areas (m^2)
    cell_area = dx*dy*mask*ps_scale

    # total point count
    npts = 0
    # iterate over tile files
    for i, tile_file in enumerate(tile_files):
        # read the HDF5 tile file
        logging.info(f'Reading File: {str(tile_file)}')
        f1 = multiprocess_h5py(tile_file, mode='r')
        for short_name, val in regex.items():
            # compile regular expression opereator
            R2 = re.compile(val, re.VERBOSE)
            # find files within tile
            groups = [f for f in f1.keys() if R2.match(f)]
            # read each file group
            for group in groups:
                # check if data is multi-beam
                if short_name in ('ATL06', ):
                    # for each beam within the tile
                    # extract number of points from subgroup
                    # assuming that all data is valid
                    for gtx, subgroup in f1[group].items():
                        # indices within tile
                        indices = subgroup['index'][:].copy()
                        npts += len(indices)
                elif short_name in ('ATL11', ):
                    # for each beam pair within the tile
                    # extract number of points from subgroup
                    # assuming that all data is valid
                    for ptx, subgroup in f1[group].items():
                        # indices within tile
                        indices = subgroup['index'][:].copy()
                        ncycles = len(subgroup['cycle_number'])
                        npts += ncycles*len(indices)
                else:
                    # single beam dataset
                    # extract number of points from group
                    indices = f1[group]['index'][:].copy()
                    npts += len(indices)
        # close the tile file
        f1.close()
    # log total number of points
    logging.info(f'Total Points: {npts:d}')

    # allocate for combined variables
    d = {}
    d['lon'] = np.zeros((npts), dtype=np.float64)
    d['lat'] = np.zeros((npts), dtype=np.float64)
    d['time'] = np.zeros((npts), dtype=np.float64)
    d['data'] = np.zeros((npts), dtype=np.float64)
    d['error'] = np.zeros((npts), dtype=np.float64)
    d['mission'] = np.zeros((npts), dtype=np.int8)
    d['mask'] = np.zeros((npts), dtype=bool)
    # counter for filling arrays
    c = 0
    # iterate over tile files
    for i, tile_file in enumerate(tile_files):
        # read the HDF5 tile file
        logging.info(f'Reading File: {str(tile_file)}')
        f1 = multiprocess_h5py(tile_file, mode='r')
        d1 = tile_file.parents[1]
        for short_name, val in regex.items():
            # compile regular expression opereator
            R2 = re.compile(val, re.VERBOSE)
            # find files within tile
            groups = [f for f in f1.keys() if R2.match(f)]
            # read each file group
            for group in groups:
                # full path to data file
                FILE2 = d1.joinpath(group)
                # check if data is multi-beam
                if short_name in ('ATL06', ):
                    # multi-beam data at a single epoch
                    f2 = multiprocess_h5py(FILE2, mode='r')
                    # get the transform for converting to the latest ITRF
                    transform = gz.crs.get_itrf_transform('ITRF2014')
                    # for each beam within the tile
                    # extract number of points from subgroup
                    # assuming that all data is valid
                    for gtx, subgroup in f1[group].items():
                        # indices within tile
                        indices = subgroup['index'][:].copy()
                        file_length = len(indices)
                        # read dataset
                        mds, attrs = is2tk.io.ATL06.read_beam(f2, gtx,
                            ATTRIBUTES=True, KEEP=True)
                        # land ice segments group
                        g = 'land_ice_segments'
                        # invalid value for heights
                        invalid = attrs[gtx][g]['h_li']['_FillValue']
                        # convert time to timescale
                        ts = timescale.time.Timescale().from_deltatime(
                            mds[gtx][g]['delta_time'][indices],
                            epoch=timescale.time._atlas_sdp_epoch,
                            standard='GPS')
                        # transform the data to a common ITRF
                        lon, lat, data, tdec = transform.transform(
                            mds[gtx][g]['longitude'][indices],
                            mds[gtx][g]['latitude'][indices],
                            mds[gtx][g]['h_li'][indices],
                            ts.year)
                        # copy variables
                        d['data'][c:c+file_length] = data.copy()
                        d['lon'][c:c+file_length] = lon.copy()
                        d['lat'][c:c+file_length] = lat.copy()
                        # combine errors
                        d['error'][c:c+file_length] = np.sqrt(
                            mds[gtx][g]['h_li_sigma'][indices]**2 + 
                            mds[gtx][g]['sigma_geo_h'][indices]**2)
                        # convert timescale to J2000 seconds
                        d['time'][c:c+file_length] = ts.to_deltatime(
                            epoch=timescale.time._j2000_epoch,
                            scale=86400.0)
                        # mask for reducing to valid values
                        d['mask'][c:c+file_length] = \
                            (mds[gtx][g]['h_li'][indices] != invalid) & \
                            (mds[gtx][g]['h_li_sigma'][indices] != invalid) & \
                            (mds[gtx][g]['sigma_geo_h'][indices] != invalid) & \
                            (mds[gtx][g]['atl06_quality_summary'][indices] == 0)
                        # add to mission variable
                        d['mission'][c:c+file_length] = mission[short_name]
                        # add to counter
                        c += file_length
                    # close the input dataset
                    f2.close()
                elif short_name in ('ATL11', ):
                    # multi-beam data at multiple epochs
                    f2 = multiprocess_h5py(FILE2, mode='r')
                    # get the transform for converting to the latest ITRF
                    transform = gz.crs.get_itrf_transform('ITRF2014')
                    # for each beam pair within the tile
                    # extract number of points from subgroup
                    # assuming that all data is valid
                    for ptx, subgroup in f1[group].items():
                        # indices within tile
                        indices = subgroup['index'][:].copy()
                        cycle_number = subgroup['cycle_number'][:].copy()
                        file_length = len(indices)
                        # read dataset
                        mds, attrs = is2tk.io.ATL11.read_pair(f2, ptx,
                            ATTRIBUTES=True, KEEP=True)
                        # invalid value for heights
                        invalid = attrs[ptx]['h_corr']['_FillValue']
                        # combine errors
                        error = np.sqrt(
                            mds[ptx]['h_corr_sigma'][indices,:]**2 + 
                            mds[ptx]['h_corr_sigma_systematic'][indices,:]**2)
                        # for each cycle
                        for k, cycle in enumerate(cycle_number):
                            # convert time to timescale
                            ts = timescale.time.Timescale().from_deltatime(
                                mds[ptx]['delta_time'][indices,k],
                                epoch=timescale.time._atlas_sdp_epoch,
                                standard='GPS')
                            # transform the data to a common ITRF
                            lon, lat, data, tdec = transform.transform(
                                mds[ptx]['longitude'][indices],
                                mds[ptx]['latitude'][indices],
                                mds[ptx]['h_corr'][indices,k],
                                ts.year)
                            # copy variables
                            d['data'][c:c+file_length] = data.copy()
                            d['error'][c:c+file_length] = error[:,k]
                            # convert timescale to J2000 seconds
                            d['time'][c:c+file_length] = ts.to_deltatime(
                                epoch=timescale.time._j2000_epoch,
                                scale=86400.0)
                            d['lon'][c:c+file_length] = lon.copy()
                            d['lat'][c:c+file_length] = lat.copy()
                            # mask for reducing to valid values
                            d['mask'][c:c+file_length] = \
                                (mds[ptx]['h_corr'][indices,k] != invalid) & \
                                (mds[ptx]['h_corr_sigma'][indices,k] != invalid) & \
                                (mds[ptx]['quality_summary'][indices,k] == 0)
                            # add to mission variable
                            d['mission'][c:c+file_length] = mission[short_name]
                            # add to counter
                            c += file_length
                    # close the input dataset
                    f2.close()
                elif short_name in ('GLAS', ):
                    # single beam dataset
                    f2 = multiprocess_h5py(FILE2, mode='r')
                    # extract parameters from ICESat/GLAS HDF5 file name
                    PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE = \
                        R2.findall(group).pop()
                    # quality summary HDF5 file for NSIDC granules
                    args = (PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
                    glasmask = 'GLAH{0}_{1}_MASK_{2}{3}{4}_{5}_{6}_{7}_{8}_{9}.h5'
                    FILE3 = d1.joinpath(glasmask.format(*args))
                    f3 = multiprocess_h5py(FILE3, mode='r')
                    # extract number of points from group
                    indices = f1[group]['index'][:].copy()
                    file_length = len(indices)
                    # copy ICESat campaign name from ancillary data
                    ancillary_data = f2['ANCILLARY_DATA']
                    campaign = ancillary_data.attrs['Campaign'].decode('utf-8')
                    # get 40HZ variables
                    group = 'Data_40HZ'
                    J2000 = f2[group]['DS_UTCTime_40'][indices].copy()
                    ts = timescale.time.Timescale().from_deltatime(
                        J2000, epoch=timescale.time._j2000_epoch,
                        standard='UTC')
                    # campaign bias correction
                    bias_corr = campaign_bias_correction(campaign)
                    # saturation correction
                    sat_corr = f2[group]['Elevation_Corrections']['d_satElevCorr']
                    # get the transform for converting to the latest ITRF
                    transform = gz.crs.tp_itrf2008_to_wgs84_itrf2020()
                    # transform the data to a common ITRF
                    # Longitude (degrees East)
                    # Latitude (TOPEX/Poseidon ellipsoid degrees North)
                    # Elevation (height above TOPEX/Poseidon ellipsoid in meters)
                    subgroup = 'Elevation_Surfaces'
                    d_elev = f2[group][subgroup]['d_elev']
                    invalid = d_elev.attrs['_FillValue']
                    lon, lat, data, tdec = transform.transform(
                        f2[group]['Geolocation']['d_lon'][indices],
                        f2[group]['Geolocation']['d_lat'][indices],
                        d_elev[indices] + sat_corr[indices] + bias_corr,
                        ts.year)
                    # copy variables
                    d['data'][c:c+file_length] = data.copy()
                    d['time'][c:c+file_length] = J2000.copy()
                    d['lon'][c:c+file_length] = lon.copy()
                    d['lat'][c:c+file_length] = lat.copy()
                    # mask for reducing to valid values
                    d['mask'][c:c+file_length] = \
                        (f2[group][subgroup]['d_elev'][indices] != invalid) & \
                        (f3[group]['Quality']['quality_summary'][indices] == 0)
                    # add to mission variable
                    d['mission'][c:c+file_length] = mission[short_name]
                    # add to counter
                    c += file_length
                    # close the input datasets
                    f2.close()
                    f3.close()
                else:
                    # Operation IceBridge dataset
                    # extract number of points from group
                    indices = f1[group]['index'][:].copy()
                    mds, file_length, HEM = gz.io.icebridge.from_file(
                        FILE2, indices, format=short_name)
                    # convert the ITRF to a common reference frame
                    dt = timescale.time.Timescale().from_deltatime(
                        mds['time'][0], epoch=timescale.time._j2000_epoch,
                        standard='UTC').to_calendar()
                    ITRF = gz.io.icebridge.get_ITRF(short_name,
                        dt.year, dt.month, HEM)
                    mds = gz.io.icebridge.convert_ITRF(mds, ITRF)
                    # save to output dictionary
                    for key, val in mds.items():
                        # verify that measurement variable is in output
                        if key not in d:
                            continue
                        # save variable
                        d[key][c:c+file_length] = val.copy()
                    # use all Operation IceBridge data when available
                    d['mask'][c:c+file_length] = True
                    # add to mission variable
                    d['mission'][c:c+file_length] = mission[short_name]
                    # add to counter
                    c += file_length
        # close the tile file
        f1.close()

    # convert time into year-decimal
    ts = timescale.time.Timescale().from_deltatime(
        d['time'], epoch=timescale.time._j2000_epoch,
        standard='UTC')

    # set default knots as range of time
    if not np.any(KNOTS):
        valid, = np.nonzero(d['mask'])
        tmin = np.floor(ts.year[valid].min())
        tmax = np.ceil(ts.year[valid].max())
        KNOTS = np.arange(tmin, tmax + 1, 1)

    # output attributes
    attributes = dict(ROOT={}, x={}, y={})
    # invalid values for each variablce
    fill_value = {}
    # root group attributes
    attributes['ROOT']['x_center'] = xc
    attributes['ROOT']['y_center'] = xc
    attributes['ROOT']['tile_width'] = W
    attributes['ROOT']['spacing'] = SPACING
    attributes['ROOT']['fit_type'] = FIT_TYPE
    attributes['ROOT']['order_time'] = ORDER_TIME
    attributes['ROOT']['order_space'] = ORDER_SPACE
    if FIT_TYPE in ('polynomial', 'chebyshev') and RELATIVE is not None:
        attributes['ROOT']['time_relative'] = np.squeeze(RELATIVE)
    elif FIT_TYPE in ('spline', ):
        attributes['ROOT']['time_knots'] = np.copy(KNOTS)
    # projection attributes
    attributes['crs'] = {}
    fill_value['crs'] = None
    # add projection attributes
    attributes['crs']['standard_name'] = \
        crs_to_cf['grid_mapping_name'].title()
    attributes['crs']['spatial_epsg'] = crs2.to_epsg()
    attributes['crs']['spatial_ref'] = crs2.to_wkt()
    attributes['crs']['proj4_params'] = crs2.to_proj4()
    attributes['crs']['latitude_of_projection_origin'] = \
        crs_to_dict['lat_0']
    for att_name,att_val in crs_to_cf.items():
        attributes['crs'][att_name] = att_val
    # x and y
    attributes['x'],attributes['y'] = ({},{})
    fill_value['x'],fill_value['y'] = (None,None)
    for att_name in ['long_name', 'standard_name', 'units']:
        attributes['x'][att_name] = cs_to_cf[0][att_name]
        attributes['y'][att_name] = cs_to_cf[1][att_name]
    # time
    attributes['time'] = {}
    attributes['time']['long_name'] = 'Time'
    attributes['time']['standard_name'] = 'time'
    attributes['time']['calendar'] = 'standard'
    attributes['time']['units'] = 'Decimal Years'
    fill_value['time'] = None
    # ice area
    attributes['cell_area'] = {}
    attributes['cell_area']['long_name'] = 'Cell area'
    attributes['cell_area']['description'] = ('Area of each grid cell, '
        'accounting for polar stereographic distortion')
    attributes['cell_area']['units'] = 'm^2'
    attributes['cell_area']['coordinates'] = 'y x'
    attributes['cell_area']['grid_mapping'] = 'crs'
    fill_value['cell_area'] = 0    
    # modeled fit
    attributes['h_fit'] = {}
    attributes['h_fit']['long_name'] = 'Modeled height'
    attributes['h_fit']['units'] = 'meters'
    attributes['h_fit']['coordinates'] = 'y x'
    attributes['h_fit']['grid_mapping'] = 'crs'
    fill_value['h_fit'] = FILL_VALUE
    # beta
    attributes['beta'] = {}
    attributes['beta']['long_name'] = 'Fit coefficients'
    attributes['beta']['units'] = 'meters'
    attributes['beta']['coordinates'] = 'y x'
    attributes['beta']['grid_mapping'] = 'crs'
    fill_value['beta'] = FILL_VALUE
    # error
    attributes['error'] = {}
    attributes['error']['long_name'] = 'Error in fit coefficients'
    attributes['error']['units'] = 'meters'
    attributes['error']['coordinates'] = 'y x'
    attributes['error']['grid_mapping'] = 'crs'
    fill_value['error'] = FILL_VALUE
    # standard error
    attributes['std_error'] = {}
    attributes['std_error']['long_name'] = ('Standard error in '
        'fit coefficients')
    attributes['std_error']['units'] = 'meters'
    attributes['std_error']['coordinates'] = 'y x'
    attributes['std_error']['grid_mapping'] = 'crs'
    fill_value['std_error'] = FILL_VALUE
    # MSE
    attributes['MSE'] = {}
    attributes['MSE']['long_name'] = 'Mean square error'
    attributes['MSE']['units'] = 'meters'
    attributes['MSE']['coordinates'] = 'y x'
    attributes['MSE']['grid_mapping'] = 'crs'
    fill_value['MSE'] = FILL_VALUE
    # R2
    attributes['R2'] = {}
    attributes['R2']['long_name'] = 'Coefficient in Determination'
    attributes['R2']['units'] = 'meters'
    attributes['R2']['coordinates'] = 'y x'
    attributes['R2']['grid_mapping'] = 'crs'
    fill_value['R2'] = FILL_VALUE
    # RDE
    attributes['RDE'] = {}
    attributes['RDE']['long_name'] = 'Robust Dispersion Estimate'
    attributes['RDE']['units'] = 'meters'
    attributes['RDE']['coordinates'] = 'y x'
    attributes['RDE']['grid_mapping'] = 'crs'
    fill_value['RDE'] = FILL_VALUE
    # window
    attributes['window'] = {}
    attributes['window']['long_name'] = 'Surface window width'
    attributes['window']['description'] = ('Width of the surface window, '
        'top to bottom')
    attributes['window']['units'] = 'meters'
    attributes['window']['coordinates'] = 'y x'
    attributes['window']['grid_mapping'] = 'crs'
    fill_value['window'] = FILL_VALUE
    # iterations
    attributes['iterations'] = {}
    attributes['iterations']['long_name'] = 'Number of fit iterations'
    attributes['iterations']['units'] = '1'
    attributes['iterations']['coordinates'] = 'y x'
    attributes['iterations']['grid_mapping'] = 'crs'
    fill_value['iterations'] = 0
    # data count
    attributes['count'] = {}
    attributes['count']['long_name'] = 'Number of data points'
    attributes['count']['units'] = '1'
    attributes['count']['coordinates'] = 'y x'
    attributes['count']['grid_mapping'] = 'crs'
    fill_value['count'] = 0
    # counts for each mission
    attributes['mission'] = {}
    attributes['mission']['long_name'] = 'Number of data points for each mission'
    attributes['mission']['columns'] = ['ICESat-2', 'ICESat', 'ATM', 'LVIS']
    attributes['mission']['units'] = '1'
    attributes['mission']['coordinates'] = 'y x'
    attributes['mission']['grid_mapping'] = 'crs'
    fill_value['mission'] = 0

    # calculate coordinates in polar stereographic
    d['x'], d['y'] = transformer.transform(d['lon'], d['lat'])
    # reduce to valid surfaces with raster mask
    if SPL is not None:
        d['mask'] &= (SPL.ev(d['x'], d['y']) >= TOLERANCE)

    # number of time points in output fit
    nt = len(KNOTS)
    # design matrix to calculate results at centroid
    DMAT, _ = gz.fit._build_design_matrix(
        KNOTS, np.zeros((nt)), np.zeros((nt)),
        FIT_TYPE=FIT_TYPE,
        ORDER_TIME=ORDER_TIME,
        ORDER_SPACE=ORDER_SPACE,
        RELATIVE=RELATIVE,
        KNOTS=KNOTS)
    # number of spatial and temporal terms
    nb = gz.fit._temporal_terms(FIT_TYPE=FIT_TYPE,
        ORDER_TIME=ORDER_TIME, KNOTS=KNOTS)
    nb += gz.fit._spatial_terms(FIT_TYPE=FIT_TYPE,
        ORDER_SPACE=ORDER_SPACE)   
    # number of mission types
    nm = len(mission_types)
    # allocate for output variables
    output = {}
    # projection variable
    output['crs'] = np.empty((), dtype=np.byte)
    # coordinates
    output['x'] = np.copy(x)
    output['y'] = np.copy(y)
    output['time'] = np.copy(KNOTS)
    # cell area (possibly masked for non-ice areas)
    output['cell_area'] = np.copy(cell_area)
    # allocate for fit variables
    output['h_fit'] = np.zeros((ny, nx, nt), dtype=np.float64)
    output['beta'] = np.zeros((ny, nx, nb))
    output['error'] = np.zeros((ny, nx, nb))
    output['std_error'] = np.zeros((ny, nx, nb))
    output['MSE'] = np.zeros((ny, nx))
    output['R2'] = np.zeros((ny, nx))
    output['RDE'] = np.zeros((ny, nx))
    output['window'] = np.zeros((ny, nx))
    output['iterations'] = np.zeros((ny, nx), dtype=np.int64)
    output['count'] = np.zeros((ny, nx), dtype=np.int64)
    output['mission'] = np.zeros((ny, nx, nm), dtype=np.int64)

    # calculate fit over each raster pixel
    for xi in np.arange(xmin, xmin + W, dx):
        for yi in np.arange(ymin, ymin + W, dy):
            # grid indices
            indy = int((yi - ymin)//dy)
            indx = int((xi - xmin)//dx)
            # clip unique points to coordinates
            # buffer to improve determination at edges
            clipped = np.nonzero((d['x'] >= xi-0.1*dx) &
                (d['x'] <= xi+1.1*dx) &
                (d['y'] >= yi-0.1*dy) &
                (d['y'] <= yi+1.1*dy) &
                d['mask'])
            # skip iteration if there are no points
            if not np.any(clipped):
                continue
            # trim time in year-decimal
            tdec = ts.year[clipped]
            # check that the spread of times is valid
            if (np.ptp(tdec) < TIME_THRESHOLD):
                continue
            # create clipped data
            u = {}
            for key,val in d.items():
                u[key] = val[clipped]
            # try to fit a surface to the data
            try:
                fit = gz.fit.reduce_fit(
                    tdec, u['x'], u['y'], u['data'],
                    FIT_TYPE=FIT_TYPE,
                    ITERATIONS=ITERATIONS,
                    ORDER_TIME=ORDER_TIME,
                    ORDER_SPACE=ORDER_SPACE,
                    RELATIVE=RELATIVE,
                    KNOTS=KNOTS,
                    THRESHOLD=POINT_THRESHOLD,
                    CX=xi+dx/2.0,
                    CY=yi+dy/2.0,
                )
            except Exception as exc:
                logging.debug(traceback.format_exc())
                continue
            # save parameters for subset coordinate
            ifit = fit['indices']
            output['h_fit'][indy, indx, :] = np.dot(DMAT, fit['beta'])
            output['beta'][indy, indx, :] = fit['beta'][:nb]
            output['error'][indy, indx, :] = fit['error'][:nb]
            output['std_error'][indy, indx, :] = fit['std_error'][:nb]
            output['MSE'][indy, indx] = fit['MSE']
            output['R2'][indy, indx] = fit['R2']
            output['RDE'][indy, indx] = fit['RDE']
            output['iterations'][indy, indx] = fit['iterations']
            output['window'][indy, indx] = fit['window']
            output['count'][indy, indx] = fit['count']
            # save mission counts
            fit['mission'] = np.zeros_like(mission_types)
            for k,m in enumerate(mission_types):
                fit['mission'][k] = np.sum(u['mission'][ifit] == m)
            output['mission'][indy, indx, :] = fit['mission']

    # find and replace invalid values
    indy, indx = np.nonzero(output['count'] == 0)
    # update values for invalid points
    output['h_fit'][indy, indx, :] = FILL_VALUE
    output['beta'][indy, indx, :] = FILL_VALUE
    output['error'][indy, indx, :] = FILL_VALUE
    output['std_error'][indy, indx, :] = FILL_VALUE
    output['MSE'][indy, indx] = FILL_VALUE
    output['R2'][indy, indx] = FILL_VALUE
    output['RDE'][indy, indx] = FILL_VALUE
    output['window'][indy, indx] = FILL_VALUE

    # open output HDF5 file in append mode
    output_file = OUTPUT_DIRECTORY.joinpath(tile_file_formatted)
    logging.info(output_file)
    fileID = multiprocess_h5py(output_file, mode='a')
    # create fit_statistics group if non-existent
    group = 'fit_statistics'
    if group not in fileID:
        g1 = fileID.create_group(group)
    else:
        g1 = fileID[group]
    # add root attributes
    for att_name, att_val in attributes['ROOT'].items():
        g1.attrs[att_name] = att_val
    # for each output variable
    h5 = {}
    for key,val in output.items():
        # create or overwrite HDF5 variables
        if key not in fileID[group]:
            # create HDF5 variables
            if fill_value[key] is not None:
                h5[key] = g1.create_dataset(key, val.shape, data=val,
                    dtype=val.dtype, fillvalue=fill_value[key],
                    compression='gzip')
            elif val.shape:
                h5[key] = g1.create_dataset(key, val.shape, data=val,
                    dtype=val.dtype, compression='gzip')
            else:
                h5[key] = g1.create_dataset(key, val.shape,
                    dtype=val.dtype)
            # add variable attributes
            for att_name,att_val in attributes[key].items():
                h5[key].attrs[att_name] = att_val
        else:
            # overwrite HDF5 variables
            g1[key][...] = val.copy()
    # close the output file
    logging.info(fileID[group].keys())
    fileID.close()
    # change the permissions mode
    output_file.chmod(mode=MODE)

# PURPOSE: create arguments parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Fits a time-variable surface to altimetry data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='Tile file(s) to run')
    # directory with input/output data
    parser.add_argument('--output-directory','-O',
        type=pathlib.Path,
        help='Output data directory')
    # region of interest to run
    parser.add_argument('--hemisphere','-H',
        type=str, default='S', choices=('N','S'),
        help='Region of interest to run')
    # tile grid width
    parser.add_argument('--width','-W',
        type=float, default=80e3,
        help='Width of tile grid')
    # output grid spacing
    parser.add_argument('--spacing','-S',
        type=float, nargs='+', default=5e3,
        help='Output grid spacing')
    # mask file for non-ice points
    parser.add_argument('--mask-file',
        type=pathlib.Path,
        help='Ice mask file')
    # temporal fit parameters
    fit_types = ('polynomial', 'chebyshev', 'spline')
    parser.add_argument('--fit-type','-F',
        type=str, default='polynomial', choices=fit_types,
        help='Temporal fit type')
    parser.add_argument('--iteration','-I',
        type=int, default=25,
        help='Number of iterations for surface fit')
    parser.add_argument('--order-time',
        type=int, default=1,
        help='Temporal fit polynomial order')
    parser.add_argument('--order-space',
        type=int, default=3,
        help='Spatial fit polynomial order')
    parser.add_argument('--relative','-R',
        type=float, nargs='+',
        help='Relative period for time-variable fit')
    parser.add_argument('--knots','-K',
        type=float, nargs='+',
        help='Temporal knots for spline fit and output time series')
    # verbose output of processing run
    # print information about processing run
    parser.add_argument('--verbose','-V',
        action='count', default=0,
        help='Verbose output of processing run')
    # permissions mode of the directories and files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Local permissions mode of the output file')
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

    # run program 
    try:
        info(args)
        fit_surface_tiles(args.infile,
            OUTPUT_DIRECTORY=args.output_directory,
            HEM=args.hemisphere,
            W=args.width,
            SPACING=args.spacing,
            MASK_FILE=args.mask_file,
            FIT_TYPE=args.fit_type,
            ITERATIONS=args.iteration,
            ORDER_TIME=args.order_time,
            ORDER_SPACE=args.order_space,
            RELATIVE=args.relative,
            KNOTS=args.knots,
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
