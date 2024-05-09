#!/usr/bin/env python
u"""
interp_sea_level_ICESat2_ATL07.py
Written by Tyler Sutterley (05/2024)
Interpolates sea level anomalies (sla), absolute dynamic topography (adt) and
    mean dynamic topography (mdt) to times and locations of ICESat-2 ATL07 data

https://www.aviso.altimetry.fr/en/data/products/sea-surface-height-products/
    global/msla-h.html
ftp://ftp.sltac.cls.fr/Core/SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047/
    dataset-duacs-rep-global-merged-allsat-phy-l4-v3

Note that the AVISO sea level data are gzip compressed netCDF4 files

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -O X, --output-directory X: input/output data directory
    -V, --verbose: Output information about each created file
    -M X, --mode X: Permission mode of directories and files created

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
    scikit-learn: Machine Learning in Python
        https://scikit-learn.org/stable/index.html
        https://github.com/scikit-learn/scikit-learn
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://h5py.org
    netCDF4: Python interface to the netCDF C library
        https://unidata.github.io/netcdf4-python/netCDF4/index.html
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

PROGRAM DEPENDENCIES:
    io/ATL07.py: reads ICESat-2 sea ice height data files
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 05/2024: use wrapper to importlib for optional dependencies
    Updated 04/2024: use timescale for temporal operations
    Updated 08/2023: create s3 filesystem when using s3 urls as input
        use time functions from timescale.time
    Updated 05/2023: use timescale class for time conversion operations
        using pathlib to define and operate on paths
    Updated 12/2022: single implicit import of grounding zone tools
        refactored ICESat-2 data product read programs under io
    Updated 05/2022: use argparse descriptions within sphinx documentation
    Updated 11/2021: hemisphere flags based on ATL07 hemisphere code
    Updated 10/2021: using python logging for handling verbose output
        added parsing for converting file lines to arguments
    Updated 05/2021: print full path of output filename
    Written 03/2021
"""
from __future__ import print_function

import re
import gzip
import logging
import pathlib
import argparse
import datetime
import numpy as np
import grounding_zones as gz

# attempt imports
h5py = gz.utilities.import_dependency('h5py')
is2tk = gz.utilities.import_dependency('icesat2_toolkit')
netCDF4 = gz.utilities.import_dependency('netCDF4')
pyproj = gz.utilities.import_dependency('pyproj')
neighbors = gz.utilities.import_dependency('sklearn.neighbors')
timescale = gz.utilities.import_dependency('timescale')


# PURPOSE: set the hemisphere of interest based on ATL07 hemisphere code
# HH Hemisphere code. Northern Hemisphere = 01, Southern Hemisphere = 02
def set_hemisphere(HH):
    if (int(HH) == 1):
        projection_flag = 'N'
    elif (int(HH) == 2):
        projection_flag = 'S'
    return projection_flag

# PURPOSE: interpolates to coordinates with inverse distance weighting
def inverse_distance(x, y, z, xi, yi, SEARCH='BallTree', N=10, POWER=2.0):
    # number of output points
    npts = len(xi)
    # create neighbors object for coordinates
    if (SEARCH == 'BallTree'):
        tree = neighbors.BallTree(np.c_[x,y])
    elif (SEARCH == 'KDTree'):
        tree = neighbors.KDTree(np.c_[x,y])
    # query the search tree to find the N closest points
    dist,indices = tree.query(np.c_[xi,yi], k=N, return_distance=True)
    # normalized weights if POWER > 0 (typically between 1 and 3)
    # in the inverse distance weighting
    power_inverse_distance = dist**(-POWER)
    s = np.sum(power_inverse_distance, axis=1)
    w = power_inverse_distance/np.broadcast_to(s[:,None],(npts,N))
    # calculate interpolated fields by inverse distance weighting
    return np.sum(w*z[indices],axis=1)

# PURPOSE interpolate sea level anomalies to lat/lon and then to time
def interpolate_sea_level(base_dir, xi, yi, MJD, HEM):
    # EPSG projections for converting lat/lon to polar stereographic
    EPSG = dict(N=3413,S=3031)
    # pyproj transformer for converting to polar stereographic
    crs1 = pyproj.CRS.from_string('epsg:4326')
    crs2 = pyproj.CRS.from_string(EPSG[HEM])
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    # interpolate mean dynamic topography
    base_dir = pathlib.Path(base_dir).expanduser().absolute()
    input_file = base_dir.joinpath('mdt_cnes_cls2013_global.nc.gz')
    # read bytes from compressed file
    fd = gzip.open(input_file, mode='rb')
    # dictionary with input fields
    dinput = {}
    # read netCDF file for mean dynamic topography
    with netCDF4.Dataset('mdt', mode='r', memory=fd.read()) as fileID:
        dinput['lon'] = fileID['lon'][:].copy()
        dinput['lat'] = fileID['lat'][:].copy()
        dinput['mdt'] = np.ma.array(fileID['mdt'][0,:,:].copy(),
            fill_value=fileID['mdt']._FillValue)
        dinput['mdt'].mask = (dinput['mdt'].data == dinput['mdt'].fill_value)
    # close the compressed file objects
    fd.close()
    # create 2-D grid coordinates from longitude and latitude vectors
    gridlon,gridlat = np.meshgrid(dinput['lon'],dinput['lat'])
    # convert from latitude/longitude into polar stereographic
    xg, yg = transformer.transform(gridlon, gridlat)

    # reduce to local coordinates to improve computational time
    gridmask = np.logical_not(dinput['mdt'].mask)
    if (HEM.upper() == 'N'):
        gridmask &= (gridlat >= 50.0)
    elif (HEM.upper() == 'S'):
        gridmask &= (gridlat <= -50.0)
    indy,indx = np.nonzero(gridmask)
    # calculate mean dynamic topography by inverse distance weighting
    MDT = inverse_distance(xg[indy,indx], yg[indy,indx],
        dinput['mdt'].data[indy,indx], xi, yi)

    # scale for linearly interpolating to date
    MJD1 = np.min(np.floor(MJD))
    dt = (MJD - MJD1)
    # output sea level anomaly and absolute dynamic topography
    SLA = np.zeros_like(dt)
    ADT = np.zeros_like(dt)
    # for the range of dates
    for day in range(2):
        # convert from Modified Julian Days to calendar dates
        YY, MM, DD, HH, MN, SS = timescale.time.convert_julian(
            MJD1 + day + 2400000.5,
            format='tuple',
            astype=int
        )
        # sea level directory
        ddir = base_dir.joinpath(f'{YY:4d}')
        # input file for day before the measurement
        regex = re.compile((rf'dt_global_allsat_phy_l4_{YY:4d}{MM:02d}{DD:02d}'
            r'_(\d{4})(\d{2})(\d{2}).nc.gz'))
        input_file, = [fi for fi in ddir.iterdir() if regex.match(fi.name)]
        # dictionary with input fields
        dinput = {}
        # read bytes from compressed file
        fd = gzip.open(input_file, mode='rb')
        # read netCDF file for time
        with netCDF4.Dataset('sla', mode='r', memory=fd.read()) as fileID:
            dinput['lon'] = fileID['lon'][:].copy()
            dinput['lat'] = fileID['lat'][:].copy()
            dinput['sla'] = np.ma.array(fileID['sla'][0,:,:].copy(),
                fill_value=fileID['sla']._FillValue)
            dinput['adt'] = np.ma.array(fileID['adt'][0,:,:].copy(),
                fill_value=fileID['adt']._FillValue)
        # close the compressed file objects
        fd.close()
        # for each variable to interpolate
        out = {}
        for var in ['sla','adt']:
            # reduce to local coordinates to improve computational time
            gridmask = np.logical_not(dinput[var].mask)
            if (HEM.upper() == 'N'):
                gridmask &= (gridlat >= 50.0)
            elif (HEM.upper() == 'S'):
                gridmask &= (gridlat <= -50.0)
            indy,indx = np.nonzero(gridmask)
            # calculate variable by inverse distance weighting
            out[var] = inverse_distance(xg[indy,indx], yg[indy,indx],
                dinput[var].data[indy,indx], xi, yi)
        # linearly interpolate to date for iteration
        SLA += out['sla']*(2.0*dt*day - dt - day + 1.0)
        ADT += out['adt']*(2.0*dt*day - dt - day + 1.0)
    # return interpolated values
    return dict(h_mdt=MDT, h_sla=SLA, h_adt=ADT)

# PURPOSE: read ICESat-2 sea ice height (ATL07) from NSIDC
# interpolate AVISO sea level at points and times
def interp_sea_level_ICESat2(base_dir, INPUT_FILE,
    OUTPUT_DIRECTORY=None,
    VERBOSE=False,
    MODE=0o775):

    # create logger
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # log input file
    logging.info(f'{str(INPUT_FILE)} -->')
    # input granule basename
    GRANULE = INPUT_FILE.name
    INPUT_FILE = pathlib.Path(INPUT_FILE).expanduser().absolute()
    IS2_atl07_mds,IS2_atl07_attrs,IS2_atl07_beams = \
        is2tk.io.ATL07.read_granule(INPUT_FILE, ATTRIBUTES=True)

    # extract parameters from ICESat-2 ATLAS HDF5 sea ice file name
    rx = re.compile(r'(processed_)?(ATL\d{2})-(\d{2})_(\d{4})(\d{2})(\d{2})'
        r'(\d{2})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    try:
        SUB,PRD,HEM,YY,MM,DD,HH,MN,SS,TRK,CYCL,SN,RL,VERS,AUX = \
            rx.findall(GRANULE).pop()
    except:
        # output sea level HDF5 file (generic)
        FILENAME = f'{INPUT_FILE.stem}_AVISO_SEA_LEVEL{INPUT_FILE.suffix}'
    else:
        # output sea level HDF5 file for ASAS/NSIDC granules
        args = (PRD,HEM,'AVISO_SEA_LEVEL',YY,MM,DD,HH,MN,SS,TRK,CYCL,SN,RL,VERS,AUX)
        ff = '{0}-{1}_{2}_{3}{4}{5}{6}{7}{8}_{9}{10}{11}_{12}_{13}{14}.h5'
        FILENAME = ff.format(*args)
    # get output directory from input file
    if OUTPUT_DIRECTORY is None:
        OUTPUT_DIRECTORY = INPUT_FILE.parent
    # full path to output file
    OUTPUT_FILE = OUTPUT_DIRECTORY.joinpath(FILENAME)
    # set the hemisphere flag based on ATL07 hemisphere code
    HMN, = IS2_atl07_mds['ancillary_data']['start_region']
    HEM = set_hemisphere(HMN)

    # HDF5 file attributes
    attrib = {}
    # mean dynamic topography
    attrib['h_mdt'] = {}
    attrib['h_mdt']['long_name'] = 'Mean Dynamic Topography'
    attrib['h_mdt']['description'] = 'Sea surface height above geoid'
    attrib['h_mdt']['reference'] = ('https://www.aviso.altimetry.fr/en/data/'
        'products/sea-surface-height-products/global/msla-h.html')
    # sea level anomalies
    attrib['h_sla'] = {}
    attrib['h_sla']['long_name'] = 'Sea Level Anomaly'
    attrib['h_sla']['description'] = 'Sea surface anomalies'
    attrib['h_sla']['reference'] = ('https://www.aviso.altimetry.fr/en/data/'
        'products/sea-surface-height-products/global/msla-h.html')
    # absolute dynamic topography
    attrib['h_adt'] = {}
    attrib['h_adt']['long_name'] = 'Absolute Dynamic Topography'
    attrib['h_adt']['description'] = ('Sea surface height above geoid calculated '
        'by adding the mean dynamic topography to the sea level anomalies')
    attrib['h_adt']['reference'] = ('https://www.aviso.altimetry.fr/en/data/'
        'products/sea-surface-height-products/global/msla-h.html')

    # EPSG projections for converting lat/lon to polar stereographic
    EPSG = dict(N=3413, S=3031)
    # pyproj transformer for converting to polar stereographic
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_epsg(EPSG[HEM])
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    # copy variables for outputting to HDF5 file
    IS2_atl07_corr = {}
    IS2_atl07_fill = {}
    IS2_atl07_dims = {}
    IS2_atl07_corr_attrs = {}
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl07_corr['ancillary_data'] = {}
    IS2_atl07_corr_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl07_corr['ancillary_data'][key] = IS2_atl07_mds['ancillary_data'][key]
        # Getting attributes of group and included variables
        IS2_atl07_corr_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl07_attrs['ancillary_data'][key].items():
            IS2_atl07_corr_attrs['ancillary_data'][key][att_name] = att_val

    # for each input beam within the file
    for gtx in sorted(IS2_atl07_beams):
        # output data dictionaries for beam
        IS2_atl07_corr[gtx] = dict(sea_ice_segments={})
        IS2_atl07_fill[gtx] = dict(sea_ice_segments={})
        IS2_atl07_dims[gtx] = dict(sea_ice_segments={})
        IS2_atl07_corr_attrs[gtx] = dict(sea_ice_segments={})

        # number of segments
        val = IS2_atl07_mds[gtx]['sea_ice_segments']
        n_seg = len(val['height_segment_id'])

        # create timescale from ATLAS Standard Epoch time
        # GPS seconds since 2018-01-01 00:00:00 UTC
        ts = timescale.time.Timescale().from_deltatime(val['delta_time'],
            epoch=timescale.time._atlas_sdp_epoch, standard='GPS')

        # extract lat/lon and convert to polar stereographic
        X, Y = transformer.transform(val['longitude'], val['latitude'])

        # interpolate sea level anomalies and dynamic topographies
        interp = interpolate_sea_level(base_dir, X, Y, ts.MJD, HEM)

        # group attributes for beam
        IS2_atl07_corr_attrs[gtx]['Description'] = IS2_atl07_attrs[gtx]['Description']
        IS2_atl07_corr_attrs[gtx]['atlas_pce'] = IS2_atl07_attrs[gtx]['atlas_pce']
        IS2_atl07_corr_attrs[gtx]['atlas_beam_type'] = IS2_atl07_attrs[gtx]['atlas_beam_type']
        IS2_atl07_corr_attrs[gtx]['groundtrack_id'] = IS2_atl07_attrs[gtx]['groundtrack_id']
        IS2_atl07_corr_attrs[gtx]['atmosphere_profile'] = IS2_atl07_attrs[gtx]['atmosphere_profile']
        IS2_atl07_corr_attrs[gtx]['atlas_spot_number'] = IS2_atl07_attrs[gtx]['atlas_spot_number']
        IS2_atl07_corr_attrs[gtx]['sc_orientation'] = IS2_atl07_attrs[gtx]['sc_orientation']
        # group attributes for sea_ice_segments
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['Description'] = ("Top group for sea "
            "ice segments as computed by the ATBD algorithm.")
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['data_rate'] = ("Data within this "
            "group are stored at the variable segment rate.")

        # geolocation, time and segment ID
        # delta time
        IS2_atl07_corr[gtx]['sea_ice_segments']['delta_time'] = val['delta_time'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['delta_time'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['delta_time'] = None
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['delta_time'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['delta_time']['standard_name'] = "time"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['delta_time']['source'] = "telemetry"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['delta_time']['calendar'] = "standard"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['delta_time']['description'] = ("Number of "
            "GPS seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch "
            "offset is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS "
            "seconds between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP "
            "epoch. By adding the offset contained within atlas_sdp_gps_epoch to delta time "
            "parameters, the time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['delta_time']['coordinates'] = \
            "height_segment_id latitude longitude"
        # latitude
        IS2_atl07_corr[gtx]['sea_ice_segments']['latitude'] = val['latitude'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['latitude'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['latitude'] = ['delta_time']
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['latitude'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['latitude']['units'] = "degrees_north"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['latitude']['long_name'] = "Latitude"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['latitude']['standard_name'] = "latitude"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['latitude']['description'] = ("Latitude of "
            "segment center")
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['latitude']['valid_min'] = -90.0
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['latitude']['valid_max'] = 90.0
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['latitude']['coordinates'] = \
            "height_segment_id delta_time longitude"
        # longitude
        IS2_atl07_corr[gtx]['sea_ice_segments']['longitude'] = val['longitude'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['longitude'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['longitude'] = ['delta_time']
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['longitude'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['longitude']['units'] = "degrees_east"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['longitude']['long_name'] = "Longitude"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['longitude']['standard_name'] = "longitude"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['longitude']['description'] = ("Longitude of "
            "segment center")
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['longitude']['valid_min'] = -180.0
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['longitude']['valid_max'] = 180.0
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['longitude']['coordinates'] = \
            "height_segment_id delta_time latitude"
        # segment ID
        IS2_atl07_corr[gtx]['sea_ice_segments']['height_segment_id'] = val['height_segment_id']
        IS2_atl07_fill[gtx]['sea_ice_segments']['height_segment_id'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['height_segment_id'] = ['delta_time']
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['height_segment_id'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['height_segment_id']['units'] = "1"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['height_segment_id']['contentType'] = "referenceInformation"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['height_segment_id']['long_name'] = \
            "Identifier of each height segment"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['height_segment_id']['description'] = \
            "Identifier of each height segment"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['height_segment_id']['coordinates'] = \
            "delta_time latitude longitude"
        # geolocation segment beginning
        IS2_atl07_corr[gtx]['sea_ice_segments']['geoseg_beg'] = val['geoseg_beg'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['geoseg_beg'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['geoseg_beg'] = ['delta_time']
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_beg'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_beg']['units'] = "1"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_beg']['contentType'] = "referenceInformation"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_beg']['long_name'] = "Beginning GEOSEG"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_beg']['description'] = \
            "Geolocation segment (geoseg) ID associated with the first photon used in this sea ice segment"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_beg']['coordinates'] = \
            "height_segment_id delta_time latitude longitude"
        # geolocation segment ending
        IS2_atl07_corr[gtx]['sea_ice_segments']['geoseg_end'] = val['geoseg_end'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['geoseg_end'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['geoseg_end'] = ['delta_time']
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_end'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_end']['units'] = "1"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_end']['contentType'] = "referenceInformation"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_end']['long_name'] = "Ending GEOSEG"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_end']['description'] = \
            "Geolocation segment (geoseg) ID associated with the last photon used in this sea ice segment"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_end']['coordinates'] = \
            "height_segment_id delta_time latitude longitude"
        # along track distance
        IS2_atl07_corr[gtx]['sea_ice_segments']['seg_dist_x'] = val['seg_dist_x'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['seg_dist_x'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['seg_dist_x'] = ['delta_time']
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['seg_dist_x'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['seg_dist_x']['units'] = "meters"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['seg_dist_x']['contentType'] = "referenceInformation"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['seg_dist_x']['long_name'] = "Along track distance"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['seg_dist_x']['description'] = \
            "Along-track distance from the equator crossing to the segment center."
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['seg_dist_x']['coordinates'] = \
            "height_segment_id delta_time latitude longitude"

        # geophysical variables
        IS2_atl07_corr[gtx]['sea_ice_segments']['geophysical'] = {}
        IS2_atl07_fill[gtx]['sea_ice_segments']['geophysical'] = {}
        IS2_atl07_dims[gtx]['sea_ice_segments']['geophysical'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['Description'] = ("Contains geophysical "
            "parameters and corrections used to correct photon heights for geophysical effects, such as tides.")
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['data_rate'] = ("Data within this group "
            "are stored at the sea_ice_height segment rate.")

        # interpolated sea level products
        for key,val in interp.items():
            # copy output variables
            sea_level = np.ma.zeros((n_seg))
            sea_level.data[:] = np.copy(val)
            # replace nan values with fill value
            sea_level.mask = np.isnan(sea_level.data)
            sea_level.data[sea_level.mask] = sea_level.fill_value
            # add to output
            IS2_atl07_corr[gtx]['sea_ice_segments']['geophysical'][key] = sea_level.copy()
            IS2_atl07_fill[gtx]['sea_ice_segments']['geophysical'][key] = sea_level.fill_value
            IS2_atl07_dims[gtx]['sea_ice_segments']['geophysical'][key] = ['delta_time']
            IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical'][key] = {}
            IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical'][key]['units'] = "meters"
            IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical'][key]['contentType'] = "referenceInformation"
            IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical'][key]['long_name'] = attrib[key]['long_name']
            IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical'][key]['description'] = attrib[key]['description']
            IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical'][key]['source'] = 'AVISO/Copernicus'
            IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical'][key]['reference'] = attrib[key]['reference']
            IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical'][key]['coordinates'] = \
                "../height_segment_id ../delta_time ../latitude ../longitude"

    # print file information
    logging.info(f'\t{str(OUTPUT_FILE)}')
    HDF5_ATL07_corr_write(IS2_atl07_corr, IS2_atl07_corr_attrs,
        FILENAME=OUTPUT_FILE,
        INPUT=GRANULE,
        FILL_VALUE=IS2_atl07_fill,
        DIMENSIONS=IS2_atl07_dims,
        CLOBBER=True)
    # change the permissions mode
    OUTPUT_FILE.chmod(mode=MODE)

# PURPOSE: outputting the correction values for ICESat-2 data to HDF5
def HDF5_ATL07_corr_write(IS2_atl07_corr, IS2_atl07_attrs, INPUT=None,
    FILENAME='', FILL_VALUE=None, DIMENSIONS=None, CLOBBER=False):
    # setting HDF5 clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'w-'

    # open output HDF5 file
    FILENAME = pathlib.Path(FILENAME).expanduser().absolute()
    fileID = h5py.File(FILENAME, clobber)

    # create HDF5 records
    h5 = {}

    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    h5['ancillary_data'] = {}
    for k,v in IS2_atl07_corr['ancillary_data'].items():
        # Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS2_atl07_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    # write each output beam
    beams = [k for k in IS2_atl07_corr.keys() if bool(re.match(r'gt\d[lr]',k))]
    for gtx in beams:
        fileID.create_group(gtx)
        # add HDF5 group attributes for beam
        for att_name in ['Description','atlas_pce','atlas_beam_type',
            'groundtrack_id','atmosphere_profile','atlas_spot_number',
            'sc_orientation']:
            fileID[gtx].attrs[att_name] = IS2_atl07_attrs[gtx][att_name]
        # create sea_ice_segments group
        fileID[gtx].create_group('sea_ice_segments')
        h5[gtx] = dict(sea_ice_segments={})
        for att_name in ['Description','data_rate']:
            att_val = IS2_atl07_attrs[gtx]['sea_ice_segments'][att_name]
            fileID[gtx]['sea_ice_segments'].attrs[att_name] = att_val

        # delta_time, geolocation and segment identification variables
        for k in ['delta_time','latitude','longitude','height_segment_id',
            'geoseg_beg','geoseg_end','seg_dist_x']:
            # values and attributes
            v = IS2_atl07_corr[gtx]['sea_ice_segments'][k]
            attrs = IS2_atl07_attrs[gtx]['sea_ice_segments'][k]
            fillvalue = FILL_VALUE[gtx]['sea_ice_segments'][k]
            # Defining the HDF5 dataset variables
            val = '{0}/{1}/{2}'.format(gtx,'sea_ice_segments',k)
            if fillvalue:
                h5[gtx]['sea_ice_segments'][k] = fileID.create_dataset(val,
                    np.shape(v), data=v, dtype=v.dtype, fillvalue=fillvalue,
                    compression='gzip')
            else:
                h5[gtx]['sea_ice_segments'][k] = fileID.create_dataset(val,
                    np.shape(v), data=v, dtype=v.dtype, compression='gzip')
            # create or attach dimensions for HDF5 variable
            if DIMENSIONS[gtx]['sea_ice_segments'][k]:
                # attach dimensions
                for i,dim in enumerate(DIMENSIONS[gtx]['sea_ice_segments'][k]):
                    h5[gtx]['sea_ice_segments'][k].dims[i].attach_scale(
                        h5[gtx]['sea_ice_segments'][dim])
            else:
                # make dimension
                h5[gtx]['sea_ice_segments'][k].make_scale(k)
            # add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5[gtx]['sea_ice_segments'][k].attrs[att_name] = att_val

        # add to geophysical corrections
        key = 'geophysical'
        fileID[gtx]['sea_ice_segments'].create_group(key)
        h5[gtx]['sea_ice_segments'][key] = {}
        for att_name in ['Description','data_rate']:
            att_val=IS2_atl07_attrs[gtx]['sea_ice_segments'][key][att_name]
            fileID[gtx]['sea_ice_segments'][key].attrs[att_name] = att_val
        for k,v in IS2_atl07_corr[gtx]['sea_ice_segments'][key].items():
            # attributes
            attrs = IS2_atl07_attrs[gtx]['sea_ice_segments'][key][k]
            fillvalue = FILL_VALUE[gtx]['sea_ice_segments'][key][k]
            # Defining the HDF5 dataset variables
            val = '{0}/{1}/{2}/{3}'.format(gtx,'sea_ice_segments',key,k)
            if fillvalue:
                h5[gtx]['sea_ice_segments'][key][k] = \
                    fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, fillvalue=fillvalue, compression='gzip')
            else:
                h5[gtx]['sea_ice_segments'][key][k] = \
                    fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, compression='gzip')
            # attach dimensions
            for i,dim in enumerate(DIMENSIONS[gtx]['sea_ice_segments'][key][k]):
                h5[gtx]['sea_ice_segments'][key][k].dims[i].attach_scale(
                    h5[gtx]['sea_ice_segments'][dim])
            # add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5[gtx]['sea_ice_segments'][key][k].attrs[att_name] = att_val

    # HDF5 file title
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'ATLAS/ICESat-2 L3A Sea Ice Height'
    fileID.attrs['summary'] = ('Estimates of the sea ice correction parameters '
        'needed to interpret and assess the quality of the height estimates.')
    fileID.attrs['description'] = ('The data set (ATL07) contains along-track '
        'heights for sea ice and open water leads (at varying length scales) '
        'relative to the WGS84 ellipsoid (ITRF2014 reference frame) after '
        'adjustment for geoidal and tidal variations, and inverted barometer '
        'effects.')
    date_created = datetime.datetime.today()
    fileID.attrs['date_created'] = date_created.isoformat()
    project = 'ICESat-2 > Ice, Cloud, and land Elevation Satellite-2'
    fileID.attrs['project'] = project
    platform = 'ICESat-2 > Ice, Cloud, and land Elevation Satellite-2'
    fileID.attrs['project'] = platform
    # add attribute for elevation instrument and designated processing level
    instrument = 'ATLAS > Advanced Topographic Laser Altimeter System'
    fileID.attrs['instrument'] = instrument
    fileID.attrs['source'] = 'Spacecraft'
    fileID.attrs['references'] = 'https://nsidc.org/data/icesat-2'
    fileID.attrs['processing_level'] = '4'
    # add attributes for input ATL07 file
    fileID.attrs['lineage'] = pathlib.Path(INPUT).name
    # find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for gtx in beams:
        lon = IS2_atl07_corr[gtx]['sea_ice_segments']['longitude']
        lat = IS2_atl07_corr[gtx]['sea_ice_segments']['latitude']
        delta_time = IS2_atl07_corr[gtx]['sea_ice_segments']['delta_time']
        # setting the geospatial and temporal ranges
        lnmn = lon.min() if (lon.min() < lnmn) else lnmn
        lnmx = lon.max() if (lon.max() > lnmx) else lnmx
        ltmn = lat.min() if (lat.min() < ltmn) else ltmn
        ltmx = lat.max() if (lat.max() > ltmx) else ltmx
        tmn = delta_time.min() if (delta_time.min() < tmn) else tmn
        tmx = delta_time.max() if (delta_time.max() > tmx) else tmx
    # add geospatial and temporal attributes
    fileID.attrs['geospatial_lat_min'] = ltmn
    fileID.attrs['geospatial_lat_max'] = ltmx
    fileID.attrs['geospatial_lon_min'] = lnmn
    fileID.attrs['geospatial_lon_max'] = lnmx
    fileID.attrs['geospatial_lat_units'] = "degrees_north"
    fileID.attrs['geospatial_lon_units'] = "degrees_east"
    fileID.attrs['geospatial_ellipsoid'] = "WGS84"
    fileID.attrs['date_type'] = 'UTC'
    fileID.attrs['time_type'] = 'CCSDS UTC-A'
    # convert start and end time from ATLAS SDP seconds into timescale
    ts = timescale.time.Timescale().from_deltatime(np.array([tmn,tmx]),
        epoch=timescale.time._atlas_sdp_epoch, standard='GPS')
    dt = np.datetime_as_string(ts.to_datetime(), unit='s')
    # add attributes with measurement date start, end and duration
    fileID.attrs['time_coverage_start'] = str(dt[0])
    fileID.attrs['time_coverage_end'] = str(dt[1])
    fileID.attrs['time_coverage_duration'] = f'{tmx-tmn:0.0f}'
    # add software information
    fileID.attrs['software_reference'] = gz.version.project_name
    fileID.attrs['software_version'] = gz.version.full_version
    # Closing the HDF5 file
    fileID.close()

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Interpolates AVISO sea level anomalies, absolute
            dynamic topography and mean dynamic topography to ICESat-2
            ATL07 sea ice height data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='ICESat-2 ATL07 file to run')
    # directory with sea level data
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=pathlib.Path.cwd(),
        help='Working data directory')
    # directory with output data
    parser.add_argument('--output-directory','-O',
        type=pathlib.Path,
        help='Output data directory')
    # verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Output information about each created file')
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

    # run for each input ATL07 file
    for FILE in args.infile:
        interp_sea_level_ICESat2(args.directory, FILE,
            OUTPUT_DIRECTORY=args.output_directory,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()