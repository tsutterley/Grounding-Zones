#!/usr/bin/env python
u"""
compute_tides_ICESat2_ATL07.py
Written by Tyler Sutterley (07/2024)
Calculates tidal elevations for correcting ICESat-2 sea ice height data

Uses OTIS format tidal solutions provided by Ohio State University and ESR
    http://volkov.oce.orst.edu/tides/region.html
    https://www.esr.org/research/polar-tide-models/list-of-polar-tide-models/
    ftp://ftp.esr.org/pub/datasets/tmd/
Global Tide Model (GOT) solutions provided by Richard Ray at GSFC
or Finite Element Solution (FES) models provided by AVISO

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -O X, --output-directory X: input/output data directory
    -T X, --tide X: Tide model to use in correction
    -I X, --interpolate X: Interpolation method
        spline
        linear
        nearest
        bilinear
    -E X, --extrapolate X: Extrapolate with nearest-neighbors
    -c X, --cutoff X: Extrapolation cutoff in kilometers
        set to inf to extrapolate for all points
    --infer-minor: Infer the height values for minor constituents
    -M X, --mode X: Permission mode of directories and files created
    -V, --verbose: Output information about each created file

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
    pyTMD: Python-based tidal prediction software
        https://pypi.org/project/pyTMD/
        https://pytmd.readthedocs.io/en/latest/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

PROGRAM DEPENDENCIES:
    io/ATL07.py: reads ICESat-2 sea ice height data files
    utilities.py: download and management utilities for syncing files
    astro.py: computes the basic astronomical mean longitudes
    crs.py: Coordinate Reference System (CRS) routines
    load_constituent.py: loads parameters for a given tidal constituent
    arguments.py: load the nodal corrections for tidal constituents
    io/model.py: retrieves tide model parameters for named tide models
    io/OTIS.py: extract tidal harmonic constants from OTIS tide models
    io/ATLAS.py: extract tidal harmonic constants from netcdf models
    io/GOT.py: extract tidal harmonic constants from GSFC GOT models
    io/FES.py: extract tidal harmonic constants from FES tide models
    interpolate.py: interpolation routines for spatial data
    predict.py: predict tidal values using harmonic constants

UPDATE HISTORY:
    Updated 07/2024: added option to crop to the domain of the input data
        added option to use JSON format definition files
    Updated 05/2024: use wrapper to importlib for optional dependencies
    Updated 04/2024: use timescale for temporal operations
    Updated 01/2024: made the inferrence of minor constituents an option
    Updated 08/2023: create s3 filesystem when using s3 urls as input
        changed ESR netCDF4 format to TMD3 format
    Updated 05/2023: use timescale class for time conversion operations
        using pathlib to define and operate on paths
    Updated 12/2022: single implicit import of grounding zone tools
        refactored ICESat-2 data product read programs under io
        use read and interpolation scheme for tidal constituents
    Updated 07/2022: place some imports within try/except statements
    Updated 05/2022: added ESR netCDF4 formats to list of model types
        updated keyword arguments to read tide model programs
    Updated 04/2022: use argparse descriptions within documentation
    Updated 03/2022: using static decorators to define available models
    Updated 02/2022: added Arctic 2km model (Arc2kmTM) to list of models
    Updated 12/2021: added TPXO9-atlas-v5 to list of available tide models
    Updated 10/2021: using python logging for handling verbose output
    Updated 09/2021: refactor to use model class for files and attributes
    Updated 07/2021: can use prefix files to define command line arguments
    Updated 06/2021: added new Gr1km-v2 1km Greenland model from ESR
    Updated 05/2021: added option for extrapolation cutoff in kilometers
    Updated 04/2021: can use a generically named ATL07 file as input
    Updated 03/2021: added TPXO9-atlas-v4 in binary OTIS format
        simplified netcdf inputs to be similar to binary OTIS read program
        replaced numpy bool/int to prevent deprecation warnings
    Updated 12/2020: H5py deprecation warning change to use make_scale
        added valid data extrapolation with nearest_extrap
        merged time conversion routines into module
    Updated 11/2020: added model constituents from TPXO9-atlas-v3
    Updated 10/2020: using argparse to set command line parameters
    Updated 08/2020: using builtin time operations.  python3 regular expressions
    Updated 07/2020: added FES2014 and FES2014_load.  use merged delta times
    Updated 06/2020: added version 2 of TPXO9-atlas (TPXO9-atlas-v2)
    Updated 03/2020: use read_ICESat2_ATL07.py from read-ICESat-2 repository
    Updated 02/2020: changed CATS2008 grid to match version on U.S. Antarctic
        Program Data Center http://www.usap-dc.org/view/dataset/601235
    Updated 11/2019: added AOTIM-5-2018 tide model (2018 update to 2004 model)
    Forked 11/2019 from compute_tides_ICESat2_ATL06.py
    Updated 10/2019: external read functions.  adjust regex for processed files
        changing Y/N flags to True/False
    Updated 09/2019: using date functions paralleling public repository
        add option for TPXO9-atlas.  add OTIS netcdf tide option
    Updated 05/2019: check if beam exists in a try except else clause
    Updated 04/2019: check if subsetted beam contains sea ice data
    Written 04/2019
"""
from __future__ import print_function

import sys
import re
import logging
import pathlib
import argparse
import datetime
import numpy as np
import grounding_zones as gz

# attempt imports
h5py = gz.utilities.import_dependency('h5py')
is2tk = gz.utilities.import_dependency('icesat2_toolkit')
pyTMD = gz.utilities.import_dependency('pyTMD')
timescale = gz.utilities.import_dependency('timescale')

# PURPOSE: read ICESat-2 sea ice height (ATL07)
# compute tides at points and times using tidal model driver algorithms
def compute_tides_ICESat2(tide_dir, INPUT_FILE,
        OUTPUT_DIRECTORY=None,
        TIDE_MODEL=None,
        ATLAS_FORMAT=None,
        GZIP=True,
        DEFINITION_FILE=None,
        DEFINITION_FORMAT='ascii',
        CROP=False,
        METHOD='spline',
        EXTRAPOLATE=False,
        CUTOFF=None,
        INFER_MINOR=False,
        VERBOSE=False,
        MODE=0o775
    ):

    # create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logger = pyTMD.utilities.build_logger('pytmd', level=loglevel)

    # get parameters for tide model
    if DEFINITION_FILE is not None:
        model = pyTMD.io.model(tide_dir).from_file(DEFINITION_FILE,
            format=DEFINITION_FORMAT)
    else:
        model = pyTMD.io.model(tide_dir, format=ATLAS_FORMAT,
            compressed=GZIP).elevation(TIDE_MODEL)

    # log input file
    logging.info(f'{str(INPUT_FILE)} -->')
    # input granule basename
    GRANULE = INPUT_FILE.name

    # extract parameters from ICESat-2 ATLAS HDF5 sea ice file name
    rx = re.compile(r'(processed_)?(ATL\d{2})-(\d{2})_(\d{4})(\d{2})(\d{2})'
        r'(\d{2})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    try:
        SUB,PRD,HEM,YY,MM,DD,HH,MN,SS,TRK,CYCL,SN,RL,VERS,AUX = \
            rx.findall(GRANULE).pop()
    except:
        # output tide HDF5 file (generic)
        args = (INPUT_FILE.stem,model.name,INPUT_FILE.suffix)
        FILENAME = '{0}_{1}_TIDES{2}'.format(*args)
    else:
        # output tide HDF5 file for ASAS/NSIDC granules
        args = (PRD,HEM,model.name,YY,MM,DD,HH,MN,SS,TRK,CYCL,SN,RL,VERS,AUX)
        ff = '{0}-{1}_{2}_TIDES_{3}{4}{5}{6}{7}{8}_{9}{10}{11}_{12}_{13}{14}.h5'
        FILENAME = ff.format(*args)
    # get output directory from input file
    if OUTPUT_DIRECTORY is None:
        OUTPUT_DIRECTORY = INPUT_FILE.parent
    # full path to output file
    OUTPUT_FILE = OUTPUT_DIRECTORY.joinpath(FILENAME)

    # check if data is an s3 presigned url
    if str(INPUT_FILE).startswith('s3:'):
        client = is2tk.utilities.attempt_login('urs.earthdata.nasa.gov',
            authorization_header=True)
        session = is2tk.utilities.s3_filesystem()
        INPUT_FILE = session.open(INPUT_FILE, mode='rb')
    else:
        INPUT_FILE = pathlib.Path(INPUT_FILE).expanduser().absolute()
    # read data from input ATL07 file
    IS2_atl07_mds,IS2_atl07_attrs,IS2_atl07_beams = \
        is2tk.io.ATL07.read_granule(INPUT_FILE, ATTRIBUTES=True)

    # find geospatial ranges for bounding box
    BOUNDS = [np.inf, -np.inf, np.inf, -np.inf]
    for gtx in IS2_atl07_beams:
        lon = IS2_atl07_mds[gtx]['sea_ice_segments']['longitude']
        lat = IS2_atl07_mds[gtx]['sea_ice_segments']['latitude']
        BOUNDS[0] = np.minimum(BOUNDS[0], np.min(lon))
        BOUNDS[1] = np.maximum(BOUNDS[1], np.max(lon))
        BOUNDS[2] = np.minimum(BOUNDS[2], np.min(lat))
        BOUNDS[3] = np.maximum(BOUNDS[3], np.max(lat))

    # read tidal constants
    if model.format in ('OTIS','ATLAS','TMD3'):
        constituents = pyTMD.io.OTIS.read_constants(model.grid_file,
            model.model_file, model.projection, type=model.type,
            grid=model.format, crop=CROP, bounds=BOUNDS)
        # available model constituents
        c = constituents.fields
    elif (model.format == 'netcdf'):
        constituents = pyTMD.io.ATLAS.read_constants(model.grid_file,
            model.model_file, type=model.type, compressed=model.compressed,
            crop=CROP, bounds=BOUNDS)
        # available model constituents
        c = constituents.fields
    elif (model.format == 'GOT'):
        constituents = pyTMD.io.GOT.read_constants(model.model_file,
            compressed=model.compressed, crop=CROP, bounds=BOUNDS)
        # available model constituents
        c = constituents.fields
    elif (model.format == 'FES'):
        constituents = pyTMD.io.FES.read_constants(model.model_file,
            type=model.type, version=model.version, compressed=model.compressed,
            crop=CROP, bounds=BOUNDS)
        # available model constituents
        c = model.constituents

    # copy variables for outputting to HDF5 file
    IS2_atl07_tide = {}
    IS2_atl07_fill = {}
    IS2_atl07_dims = {}
    IS2_atl07_tide_attrs = {}
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl07_tide['ancillary_data'] = {}
    IS2_atl07_tide_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl07_tide['ancillary_data'][key] = IS2_atl07_mds['ancillary_data'][key]
        # Getting attributes of group and included variables
        IS2_atl07_tide_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl07_attrs['ancillary_data'][key].items():
            IS2_atl07_tide_attrs['ancillary_data'][key][att_name] = att_val

    # for each input beam within the file
    for gtx in sorted(IS2_atl07_beams):
        # output data dictionaries for beam
        IS2_atl07_tide[gtx] = dict(sea_ice_segments={})
        IS2_atl07_fill[gtx] = dict(sea_ice_segments={})
        IS2_atl07_dims[gtx] = dict(sea_ice_segments={})
        IS2_atl07_tide_attrs[gtx] = dict(sea_ice_segments={})

        # number of segments
        val = IS2_atl07_mds[gtx]['sea_ice_segments']
        n_seg = len(val['height_segment_id'])

        # create timescale from ATLAS Standard Epoch time
        # GPS seconds since 2018-01-01 00:00:00 UTC
        ts = timescale.time.Timescale().from_deltatime(val['delta_time'],
            epoch=timescale.time._atlas_sdp_epoch, standard='GPS')

        # interpolate tidal constants to grid points
        if model.format in ('OTIS','ATLAS','TMD3'):
            amp,ph,D = pyTMD.io.OTIS.interpolate_constants(val['longitude'],
                val['latitude'], constituents, model.projection, type=model.type,
                method=METHOD, extrapolate=EXTRAPOLATE, cutoff=CUTOFF)
            # use delta time at 2000.0 to match TMD outputs
            deltat = np.zeros((n_seg))
        elif (model.format == 'netcdf'):
            amp,ph,D = pyTMD.io.ATLAS.interpolate_constants(val['longitude'],
                val['latitude'], constituents, type=model.type, method=METHOD,
                extrapolate=EXTRAPOLATE, cutoff=CUTOFF, scale=model.scale)
            # use delta time at 2000.0 to match TMD outputs
            deltat = np.zeros((n_seg))
        elif (model.format == 'GOT'):
            amp,ph = pyTMD.io.GOT.interpolate_constants(val['longitude'],
                val['latitude'], constituents, method=METHOD,
                extrapolate=EXTRAPOLATE, cutoff=CUTOFF, scale=model.scale)
            # delta time (TT - UT1)
            deltat = ts.tt_ut1
        elif (model.format == 'FES'):
            amp,ph = pyTMD.io.FES.interpolate_constants(val['longitude'],
                val['latitude'], constituents, method=METHOD,
                extrapolate=EXTRAPOLATE, cutoff=CUTOFF, scale=model.scale)
            # delta time (TT - UT1)
            deltat = ts.tt_ut1

        # calculate complex phase in radians for Euler's
        cph = -1j*ph*np.pi/180.0
        # calculate constituent oscillation
        hc = amp*np.exp(cph)

        # predict tidal elevations at time
        tide = np.ma.empty((n_seg))
        tide.mask = np.any(hc.mask,axis=1)
        tide.data[:] = pyTMD.predict.drift(ts.tide, hc, c,
            deltat=deltat, corrections=model.format)
        # calculate values for minor constituents by inferrence
        if INFER_MINOR:
            minor = pyTMD.predict.infer_minor(ts.tide, hc, c,
                deltat=deltat, corrections=model.format)
            tide.data[:] += minor.data[:]
        # replace masked and nan values with fill value
        invalid, = np.nonzero(np.isnan(tide.data) | tide.mask)
        tide.data[invalid] = tide.fill_value
        tide.mask[invalid] = True

        # group attributes for beam
        IS2_atl07_tide_attrs[gtx]['Description'] = IS2_atl07_attrs[gtx]['Description']
        IS2_atl07_tide_attrs[gtx]['atlas_pce'] = IS2_atl07_attrs[gtx]['atlas_pce']
        IS2_atl07_tide_attrs[gtx]['atlas_beam_type'] = IS2_atl07_attrs[gtx]['atlas_beam_type']
        IS2_atl07_tide_attrs[gtx]['groundtrack_id'] = IS2_atl07_attrs[gtx]['groundtrack_id']
        IS2_atl07_tide_attrs[gtx]['atmosphere_profile'] = IS2_atl07_attrs[gtx]['atmosphere_profile']
        IS2_atl07_tide_attrs[gtx]['atlas_spot_number'] = IS2_atl07_attrs[gtx]['atlas_spot_number']
        IS2_atl07_tide_attrs[gtx]['sc_orientation'] = IS2_atl07_attrs[gtx]['sc_orientation']
        # group attributes for sea_ice_segments
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['Description'] = ("Top group for sea "
            "ice segments as computed by the ATBD algorithm.")
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['data_rate'] = ("Data within this "
            "group are stored at the variable segment rate.")

        # geolocation, time and segment ID
        # delta time
        IS2_atl07_tide[gtx]['sea_ice_segments']['delta_time'] = val['delta_time'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['delta_time'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['delta_time'] = None
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['delta_time'] = {}
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['delta_time']['standard_name'] = "time"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['delta_time']['source'] = "telemetry"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['delta_time']['calendar'] = "standard"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['delta_time']['description'] = ("Number of "
            "GPS seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch "
            "offset is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS "
            "seconds between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP "
            "epoch. By adding the offset contained within atlas_sdp_gps_epoch to delta time "
            "parameters, the time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['delta_time']['coordinates'] = \
            "height_segment_id latitude longitude"
        # latitude
        IS2_atl07_tide[gtx]['sea_ice_segments']['latitude'] = val['latitude'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['latitude'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['latitude'] = ['delta_time']
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['latitude'] = {}
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['latitude']['units'] = "degrees_north"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['latitude']['long_name'] = "Latitude"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['latitude']['standard_name'] = "latitude"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['latitude']['description'] = ("Latitude of "
            "segment center")
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['latitude']['valid_min'] = -90.0
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['latitude']['valid_max'] = 90.0
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['latitude']['coordinates'] = \
            "height_segment_id delta_time longitude"
        # longitude
        IS2_atl07_tide[gtx]['sea_ice_segments']['longitude'] = val['longitude'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['longitude'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['longitude'] = ['delta_time']
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['longitude'] = {}
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['longitude']['units'] = "degrees_east"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['longitude']['long_name'] = "Longitude"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['longitude']['standard_name'] = "longitude"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['longitude']['description'] = ("Longitude of "
            "segment center")
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['longitude']['valid_min'] = -180.0
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['longitude']['valid_max'] = 180.0
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['longitude']['coordinates'] = \
            "height_segment_id delta_time latitude"
        # segment ID
        IS2_atl07_tide[gtx]['sea_ice_segments']['height_segment_id'] = val['height_segment_id']
        IS2_atl07_fill[gtx]['sea_ice_segments']['height_segment_id'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['height_segment_id'] = ['delta_time']
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['height_segment_id'] = {}
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['height_segment_id']['units'] = "1"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['height_segment_id']['contentType'] = "referenceInformation"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['height_segment_id']['long_name'] = \
            "Identifier of each height segment"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['height_segment_id']['description'] = \
            "Identifier of each height segment"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['height_segment_id']['coordinates'] = \
            "delta_time latitude longitude"
        # geolocation segment beginning
        IS2_atl07_tide[gtx]['sea_ice_segments']['geoseg_beg'] = val['geoseg_beg'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['geoseg_beg'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['geoseg_beg'] = ['delta_time']
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geoseg_beg'] = {}
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geoseg_beg']['units'] = "1"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geoseg_beg']['contentType'] = "referenceInformation"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geoseg_beg']['long_name'] = "Beginning GEOSEG"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geoseg_beg']['description'] = \
            "Geolocation segment (geoseg) ID associated with the first photon used in this sea ice segment"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geoseg_beg']['coordinates'] = \
            "height_segment_id delta_time latitude longitude"
        # geolocation segment ending
        IS2_atl07_tide[gtx]['sea_ice_segments']['geoseg_end'] = val['geoseg_end'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['geoseg_end'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['geoseg_end'] = ['delta_time']
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geoseg_end'] = {}
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geoseg_end']['units'] = "1"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geoseg_end']['contentType'] = "referenceInformation"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geoseg_end']['long_name'] = "Ending GEOSEG"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geoseg_end']['description'] = \
            "Geolocation segment (geoseg) ID associated with the last photon used in this sea ice segment"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geoseg_end']['coordinates'] = \
            "height_segment_id delta_time latitude longitude"
        # along track distance
        IS2_atl07_tide[gtx]['sea_ice_segments']['seg_dist_x'] = val['seg_dist_x'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['seg_dist_x'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['seg_dist_x'] = ['delta_time']
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['seg_dist_x'] = {}
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['seg_dist_x']['units'] = "meters"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['seg_dist_x']['contentType'] = "referenceInformation"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['seg_dist_x']['long_name'] = "Along track distance"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['seg_dist_x']['description'] = \
            "Along-track distance from the equator crossing to the segment center."
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['seg_dist_x']['coordinates'] = \
            "height_segment_id delta_time latitude longitude"

        # geophysical variables
        IS2_atl07_tide[gtx]['sea_ice_segments']['geophysical'] = {}
        IS2_atl07_fill[gtx]['sea_ice_segments']['geophysical'] = {}
        IS2_atl07_dims[gtx]['sea_ice_segments']['geophysical'] = {}
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geophysical'] = {}
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geophysical']['Description'] = ("Contains geophysical "
            "parameters and corrections used to correct photon heights for geophysical effects, such as tides.")
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geophysical']['data_rate'] = ("Data within this group "
            "are stored at the sea_ice_height segment rate.")

        # computed tide
        IS2_atl07_tide[gtx]['sea_ice_segments']['geophysical'][model.atl07] = tide.copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['geophysical'][model.atl07] = tide.fill_value
        IS2_atl07_dims[gtx]['sea_ice_segments']['geophysical'][model.atl07] = ['delta_time']
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geophysical'][model.atl07] = {}
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geophysical'][model.atl07]['units'] = "meters"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geophysical'][model.atl07]['contentType'] = "referenceInformation"
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geophysical'][model.atl07]['long_name'] = model.long_name
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geophysical'][model.atl07]['description'] = model.description
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geophysical'][model.atl07]['source'] = model.name
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geophysical'][model.atl07]['reference'] = model.reference
        IS2_atl07_tide_attrs[gtx]['sea_ice_segments']['geophysical'][model.atl07]['coordinates'] = \
            "../height_segment_id ../delta_time ../latitude ../longitude"

    # print file information
    logger.info(f'\t{str(OUTPUT_FILE)}')
    HDF5_ATL07_tide_write(IS2_atl07_tide, IS2_atl07_tide_attrs,
        FILENAME=OUTPUT_FILE,
        INPUT=GRANULE,
        FILL_VALUE=IS2_atl07_fill,
        DIMENSIONS=IS2_atl07_dims,
        CLOBBER=True)
    # change the permissions mode
    OUTPUT_FILE.chmod(mode=MODE)

# PURPOSE: outputting the tide values for ICESat-2 data to HDF5
def HDF5_ATL07_tide_write(IS2_atl07_tide, IS2_atl07_attrs, INPUT=None,
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
    for k,v in IS2_atl07_tide['ancillary_data'].items():
        # Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS2_atl07_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    # write each output beam
    beams = [k for k in IS2_atl07_tide.keys() if bool(re.match(r'gt\d[lr]',k))]
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
            v = IS2_atl07_tide[gtx]['sea_ice_segments'][k]
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
        for k,v in IS2_atl07_tide[gtx]['sea_ice_segments'][key].items():
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
    fileID.attrs['summary'] = ('Estimates of the sea ice tidal parameters '
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
        lon = IS2_atl07_tide[gtx]['sea_ice_segments']['longitude']
        lat = IS2_atl07_tide[gtx]['sea_ice_segments']['latitude']
        delta_time = IS2_atl07_tide[gtx]['sea_ice_segments']['delta_time']
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
    fileID.attrs['software_reference'] = pyTMD.version.project_name
    fileID.attrs['software_version'] = pyTMD.version.full_version
    # Closing the HDF5 file
    fileID.close()

# PURPOSE: create a list of available ocean and load tide models
def get_available_models():
    """Create a list of available tide models
    """
    try:
        return sorted(pyTMD.io.model.ocean_elevation() + pyTMD.io.model.load_elevation())
    except (NameError, AttributeError):
        return None

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates tidal elevations for correcting ICESat-2 ATL07
            sea ice height data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    group = parser.add_mutually_exclusive_group(required=True)
    # input ICESat-2 sea ice height files
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='ICESat-2 ATL07 file to run')
    # directory with tide data
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=pathlib.Path.cwd(),
        help='Working data directory')
    # directory with output data
    parser.add_argument('--output-directory','-O',
        type=pathlib.Path,
        help='Output data directory')
    # tide model to use
    group.add_argument('--tide','-T',
        metavar='TIDE', type=str,
        choices=get_available_models(),
        help='Tide model to use in correction')
    parser.add_argument('--atlas-format',
        type=str, choices=('OTIS','netcdf'), default='netcdf',
        help='ATLAS tide model format')
    parser.add_argument('--gzip','-G',
        default=False, action='store_true',
        help='Tide model files are gzip compressed')
    # tide model definition file to set an undefined model
    group.add_argument('--definition-file',
        type=pathlib.Path,
        help='Tide model definition file')
    parser.add_argument('--definition-format',
        type=str, default='ascii', choices=('ascii', 'json'),
        help='Format for model definition file')
    # crop tide model to (buffered) bounds of data
    parser.add_argument('--crop', '-C',
        default=False, action='store_true',
        help='Crop tide model to bounds of data')
    # interpolation method
    parser.add_argument('--interpolate','-I',
        metavar='METHOD', type=str, default='spline',
        choices=('spline','linear','nearest','bilinear'),
        help='Spatial interpolation method')
    # extrapolate with nearest-neighbors
    parser.add_argument('--extrapolate','-E',
        default=False, action='store_true',
        help='Extrapolate with nearest-neighbors')
    # extrapolation cutoff in kilometers
    # set to inf to extrapolate over all points
    parser.add_argument('--cutoff','-c',
        type=np.float64, default=10.0,
        help='Extrapolation cutoff in kilometers')
    # infer minor constituents from major
    parser.add_argument('--infer-minor',
        default=False, action='store_true',
        help='Infer the height values for minor constituents')
    # verbosity settings
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
        compute_tides_ICESat2(args.directory, FILE,
            OUTPUT_DIRECTORY=args.output_directory,
            TIDE_MODEL=args.tide,
            ATLAS_FORMAT=args.atlas_format,
            GZIP=args.gzip,
            DEFINITION_FILE=args.definition_file,
            DEFINITION_FORMAT=args.definition_format,
            CROP=args.crop,
            METHOD=args.interpolate,
            EXTRAPOLATE=args.extrapolate,
            CUTOFF=args.cutoff,
            INFER_MINOR=args.infer_minor,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
