#!/usr/bin/env python
u"""
compute_tides_ICESat2_ATL10.py
Written by Tyler Sutterley (12/2022)
Calculates tidal elevations for correcting ICESat-2 sea ice height data

Uses OTIS format tidal solutions provided by Ohio State University and ESR
    http://volkov.oce.orst.edu/tides/region.html
    https://www.esr.org/research/polar-tide-models/list-of-polar-tide-models/
    ftp://ftp.esr.org/pub/datasets/tmd/
Global Tide Model (GOT) solutions provided by Richard Ray at GSFC
or Finite Element Solution (FES) models provided by AVISO

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -T X, --tide X: Tide model to use in correction
    -I X, --interpolate X: Interpolation method
        spline
        linear
        nearest
        bilinear
    -E X, --extrapolate X: Extrapolate with nearest-neighbors
    -c X, --cutoff X: Extrapolation cutoff in kilometers
        set to inf to extrapolate for all points
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

PROGRAM DEPENDENCIES:
    io/ATL10.py: reads ICESat-2 sea ice freeboard data files
    time.py: utilities for calculating time operations
    utilities.py: download and management utilities for syncing files
    calc_astrol_longitudes.py: computes the basic astronomical mean longitudes
    convert_ll_xy.py: convert lat/lon points to and from projected coordinates
    load_constituent.py: loads parameters for a given tidal constituent
    load_nodal_corrections.py: load the nodal corrections for tidal constituents
    io.model.py: retrieves tide model parameters for named tide models
    io/OTIS.py: extract tidal harmonic constants from OTIS tide models
    io/ATLAS.py: extract tidal harmonic constants from netcdf models
    io/GOT.py: extract tidal harmonic constants from GSFC GOT models
    io/FES.py: extract tidal harmonic constants from FES tide models
    interpolate.py: interpolation routines for spatial data
    predict.py: predict tidal values using harmonic constants

UPDATE HISTORY:
    Updated 12/2022: single implicit import of grounding zone tools
        refactored ICESat-2 data product read programs under io
        use read and interpolation scheme for tidal constituents
    Updated 07/2022: place some imports within try/except statements
    Updated 05/2022: added ESR netCDF4 formats to list of model types
        updated keyword arguments to read tide model programs
    Updated 04/2022: use argparse descriptions within documentation
    Updated 03/2022: using static decorators to define available models
    Updated 02/2022: added Arctic 2km model (Arc2kmTM) to list of models
    Forked 12/2021 from compute_tides_ICESat2_ATL07.py
    Updated 12/2021: added TPXO9-atlas-v5 to list of available tide models
    Updated 10/2021: using python logging for handling verbose output
    Updated 09/2021: refactor to use model class for files and attributes
    Updated 10/2021: can use prefix files to define command line arguments
    Updated 06/2021: added new Gr1km-v2 1km Greenland model from ESR
    Updated 05/2021: added option for extrapolation cutoff in kilometers
    Updated 04/2021: can use a generically named ATL10 file as input
    Updated 03/2021: added TPXO9-atlas-v4 in binary OTIS format
        simplified netcdf inputs to be similar to binary OTIS read program
        replaced numpy bool/int to prevent deprecation warnings
    Updated 12/2020: H5py deprecation warning change to use make_scale
        added valid data extrapolation with nearest_extrap
        merged time conversion routines into module
    Updated 11/2020: added model constituents from TPXO9-atlas-v3
    Updated 10/2020: using argparse to set command line parameters
    Updated 08/2020: using builtin time operations.  python3 regular expressions
    Updated 10/2020: added FES2014 and FES2014_load.  use merged delta times
    Updated 06/2020: added version 2 of TPXO9-atlas (TPXO9-atlas-v2)
    Updated 03/2020: use read_ICESat2_ATL10.py from read-ICESat-2 repository
    Updated 02/2020: changed CATS2008 grid to match version on U.S. Antarctic
        Program Data Center http://www.usap-dc.org/view/dataset/601235
    Updated 11/2019: added AOTIM-5-2018 tide model (2018 update to 2004 model)
    Forked 11/2019 from compute_tides_ICESat2_atl06.py
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
import os
import re
import logging
import argparse
import datetime
import warnings
import numpy as np
import grounding_zones as gz

# attempt imports
try:
    import h5py
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("h5py not available")
try:
    import icesat2_toolkit as is2tk
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("icesat2_toolkit not available")
try:
    import pyTMD
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("pyTMD not available")
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: read ICESat-2 sea ice freeboard (ATL10) from NSIDC
# compute tides at points and times using tidal model driver algorithms
def compute_tides_ICESat2(tide_dir, INPUT_FILE, TIDE_MODEL=None,
    ATLAS_FORMAT=None, GZIP=True, DEFINITION_FILE=None, METHOD='spline',
    EXTRAPOLATE=False, CUTOFF=None, VERBOSE=False, MODE=0o775):

    # create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logger = pyTMD.utilities.build_logger('pytmd',level=loglevel)

    # get parameters for tide model
    if DEFINITION_FILE is not None:
        model = pyTMD.io.model(tide_dir).from_file(DEFINITION_FILE)
    else:
        model = pyTMD.io.model(tide_dir, format=ATLAS_FORMAT,
            compressed=GZIP).elevation(TIDE_MODEL)

    # read data from input file
    logger.info(f'{INPUT_FILE} -->')
    IS2_atl10_mds,IS2_atl10_attrs,IS2_atl10_beams = \
        is2tk.io.ATL10.read_granule(INPUT_FILE, ATTRIBUTES=True)
    DIRECTORY = os.path.dirname(INPUT_FILE)
    # extract parameters from ICESat-2 ATLAS HDF5 sea ice file name
    rx = re.compile(r'(processed_)?(ATL\d{2})-(\d{2})_(\d{4})(\d{2})(\d{2})'
        r'(\d{2})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    try:
        SUB,PRD,HEM,YY,MM,DD,HH,MN,SS,TRK,CYCL,SN,RL,VERS,AUX=rx.findall(INPUT_FILE).pop()
    except:
        # output tide HDF5 file (generic)
        fileBasename,fileExtension = os.path.splitext(INPUT_FILE)
        args = (fileBasename,model.name,fileExtension)
        OUTPUT_FILE = '{0}_{1}_TIDES{2}'.format(*args)
    else:
        # output tide HDF5 file for ASAS/NSIDC granules
        args = (PRD,HEM,model.name,YY,MM,DD,HH,MN,SS,TRK,CYCL,SN,RL,VERS,AUX)
        ff = '{0}-{1}_{2}_TIDES_{3}{4}{5}{6}{7}{8}_{9}{10}{11}_{12}_{13}{14}.h5'
        OUTPUT_FILE = ff.format(*args)

    # number of GPS seconds between the GPS epoch
    # and ATLAS Standard Data Product (SDP) epoch
    atlas_sdp_gps_epoch = IS2_atl10_mds['ancillary_data']['atlas_sdp_gps_epoch']
    # delta time (TT - UT1) file
    delta_file = pyTMD.utilities.get_data_path(['data','merged_deltat.data'])

    # read tidal constants
    if model.format in ('OTIS','ATLAS','ESR'):
        constituents = pyTMD.io.OTIS.read_constants(model.grid_file,
            model.model_file, model.projection, type=model.type,
            grid=model.format)
        # available model constituents
        c = constituents.fields
    elif (model.format == 'netcdf'):
        constituents = pyTMD.io.ATLAS.read_constants(model.grid_file,
            model.model_file, type=model.type, compressed=model.compressed)
        # available model constituents
        c = constituents.fields
    elif (model.format == 'GOT'):
        constituents = pyTMD.io.GOT.read_constants(model.model_file,
            compressed=model.compressed)
        # available model constituents
        c = constituents.fields
    elif (model.format == 'FES'):
        constituents = pyTMD.io.FES.read_constants(model.model_file,
            type=model.type, version=model.version, compressed=model.compressed)
        # available model constituents
        c = model.constituents

    # copy variables for outputting to HDF5 file
    IS2_atl10_tide = {}
    IS2_atl10_fill = {}
    IS2_atl10_dims = {}
    IS2_atl10_tide_attrs = {}
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl10_tide['ancillary_data'] = {}
    IS2_atl10_tide_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl10_tide['ancillary_data'][key] = IS2_atl10_mds['ancillary_data'][key]
        # Getting attributes of group and included variables
        IS2_atl10_tide_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl10_attrs['ancillary_data'][key].items():
            IS2_atl10_tide_attrs['ancillary_data'][key][att_name] = att_val

    # for each input beam within the file
    for gtx in sorted(IS2_atl10_beams):
        # output data dictionaries for beam
        IS2_atl10_tide[gtx] = dict(freeboard_beam_segment={},leads={})
        IS2_atl10_fill[gtx] = dict(freeboard_beam_segment={},leads={})
        IS2_atl10_dims[gtx] = dict(freeboard_beam_segment={},leads={})
        IS2_atl10_tide_attrs[gtx] = dict(freeboard_beam_segment={},leads={})

        # group attributes for beam
        IS2_atl10_tide_attrs[gtx]['Description'] = IS2_atl10_attrs[gtx]['Description']
        IS2_atl10_tide_attrs[gtx]['atlas_pce'] = IS2_atl10_attrs[gtx]['atlas_pce']
        IS2_atl10_tide_attrs[gtx]['atlas_beam_type'] = IS2_atl10_attrs[gtx]['atlas_beam_type']
        IS2_atl10_tide_attrs[gtx]['groundtrack_id'] = IS2_atl10_attrs[gtx]['groundtrack_id']
        IS2_atl10_tide_attrs[gtx]['atmosphere_profile'] = IS2_atl10_attrs[gtx]['atmosphere_profile']
        IS2_atl10_tide_attrs[gtx]['atlas_spot_number'] = IS2_atl10_attrs[gtx]['atlas_spot_number']
        IS2_atl10_tide_attrs[gtx]['sc_orientation'] = IS2_atl10_attrs[gtx]['sc_orientation']

        # group attributes for freeboard_beam_segment
        IS2_atl10_tide_attrs[gtx]['freeboard_beam_segment']['Description'] = ("Contains freeboard "
            "estimate and associated height segment parameters for only the sea ice segments by beam.")
        IS2_atl10_tide_attrs[gtx]['freeboard_beam_segment']['data_rate'] = ("Data within this "
            "group are stored at the freeboard swath segment rate.")
        # group attributes for leads
        IS2_atl10_tide_attrs[gtx]['leads']['Description'] = ("Contains parameters relating "
            "to the freeboard values.")
        IS2_atl10_tide_attrs[gtx]['leads']['data_rate'] = ("Data within this "
            "group are stored at the lead index rate.")

        # for each ATL10 group
        for group in ['freeboard_beam_segment','leads']:
            # number of segments
            val = IS2_atl10_mds[gtx][group]
            n_seg = len(val['delta_time'])

            # convert time from ATLAS SDP to days relative to Jan 1, 1992
            gps_seconds = atlas_sdp_gps_epoch + val['delta_time']
            leap_seconds = pyTMD.time.count_leap_seconds(gps_seconds)
            tide_time = pyTMD.time.convert_delta_time(gps_seconds-leap_seconds,
                epoch1=pyTMD.time._gps_epoch, epoch2=pyTMD.time._tide_epoch,
                scale=1.0/86400.0)

            # interpolate tidal constants to grid points
            if model.format in ('OTIS','ATLAS','ESR'):
                amp,ph,D = pyTMD.io.OTIS.interpolate_constants(val['longitude'],
                    val['latitude'], constituents, model.projection, type=model.type,
                    method=METHOD, extrapolate=EXTRAPOLATE, cutoff=CUTOFF)
                # use delta time at 2000.0 to match TMD outputs
                deltat = np.zeros_like(tide_time)
            elif (model.format == 'netcdf'):
                amp,ph,D = pyTMD.io.ATLAS.interpolate_constants(val['longitude'],
                    val['latitude'], constituents, type=model.type, method=METHOD,
                    extrapolate=EXTRAPOLATE, cutoff=CUTOFF, scale=model.scale)
                # use delta time at 2000.0 to match TMD outputs
                deltat = np.zeros_like(tide_time)
            elif (model.format == 'GOT'):
                amp,ph = pyTMD.io.GOT.interpolate_constants(val['longitude'],
                    val['latitude'], constituents, method=METHOD,
                    extrapolate=EXTRAPOLATE, cutoff=CUTOFF, scale=model.scale)
                # interpolate delta times from calendar dates to tide time
                deltat = pyTMD.time.interpolate_delta_time(delta_file, tide_time)
            elif (model.format == 'FES'):
                amp,ph = pyTMD.io.FES.interpolate_constants(val['longitude'],
                    val['latitude'], constituents, method=METHOD,
                    extrapolate=EXTRAPOLATE, cutoff=CUTOFF, scale=model.scale)
                # interpolate delta times from calendar dates to tide time
                deltat = pyTMD.time.interpolate_delta_time(delta_file, tide_time)

            # calculate complex phase in radians for Euler's
            cph = -1j*ph*np.pi/180.0
            # calculate constituent oscillation
            hc = amp*np.exp(cph)

            # predict tidal elevations at time and infer minor corrections
            tide = np.ma.empty((n_seg))
            tide.mask = np.any(hc.mask,axis=1)
            tide.data[:] = pyTMD.predict.drift(tide_time, hc, c,
                deltat=deltat, corrections=model.format)
            minor = pyTMD.predict.infer_minor(tide_time, hc, c,
                deltat=deltat, corrections=model.format)
            tide.data[:] += minor.data[:]
            # replace masked and nan values with fill value
            invalid, = np.nonzero(np.isnan(tide.data) | tide.mask)
            tide.data[invalid] = tide.fill_value
            tide.mask[invalid] = True

            # geolocation, time and segment ID
            # delta time
            IS2_atl10_tide[gtx][group]['delta_time'] = val['delta_time'].copy()
            IS2_atl10_fill[gtx][group]['delta_time'] = None
            IS2_atl10_dims[gtx][group]['delta_time'] = None
            IS2_atl10_tide_attrs[gtx][group]['delta_time'] = {}
            IS2_atl10_tide_attrs[gtx][group]['delta_time']['units'] = "seconds since 2018-01-01"
            IS2_atl10_tide_attrs[gtx][group]['delta_time']['long_name'] = "Elapsed GPS seconds"
            IS2_atl10_tide_attrs[gtx][group]['delta_time']['standard_name'] = "time"
            IS2_atl10_tide_attrs[gtx][group]['delta_time']['source'] = "telemetry"
            IS2_atl10_tide_attrs[gtx][group]['delta_time']['calendar'] = "standard"
            IS2_atl10_tide_attrs[gtx][group]['delta_time']['description'] = ("Number of "
                "GPS seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch "
                "offset is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS "
                "seconds between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP "
                "epoch. By adding the offset contained within atlas_sdp_gps_epoch to delta time "
                "parameters, the time in gps_seconds relative to the GPS epoch can be computed.")
            IS2_atl10_tide_attrs[gtx][group]['delta_time']['coordinates'] = \
                "latitude longitude"
            # latitude
            IS2_atl10_tide[gtx][group]['latitude'] = val['latitude'].copy()
            IS2_atl10_fill[gtx][group]['latitude'] = None
            IS2_atl10_dims[gtx][group]['latitude'] = ['delta_time']
            IS2_atl10_tide_attrs[gtx][group]['latitude'] = {}
            IS2_atl10_tide_attrs[gtx][group]['latitude']['units'] = "degrees_north"
            IS2_atl10_tide_attrs[gtx][group]['latitude']['contentType'] = "physicalMeasurement"
            IS2_atl10_tide_attrs[gtx][group]['latitude']['long_name'] = "Latitude"
            IS2_atl10_tide_attrs[gtx][group]['latitude']['standard_name'] = "latitude"
            IS2_atl10_tide_attrs[gtx][group]['latitude']['description'] = ("Latitude of "
                "segment center")
            IS2_atl10_tide_attrs[gtx][group]['latitude']['valid_min'] = -90.0
            IS2_atl10_tide_attrs[gtx][group]['latitude']['valid_max'] = 90.0
            IS2_atl10_tide_attrs[gtx][group]['latitude']['coordinates'] = \
                "delta_time longitude"
            # longitude
            IS2_atl10_tide[gtx][group]['longitude'] = val['longitude'].copy()
            IS2_atl10_fill[gtx][group]['longitude'] = None
            IS2_atl10_dims[gtx][group]['longitude'] = ['delta_time']
            IS2_atl10_tide_attrs[gtx][group]['longitude'] = {}
            IS2_atl10_tide_attrs[gtx][group]['longitude']['units'] = "degrees_east"
            IS2_atl10_tide_attrs[gtx][group]['longitude']['contentType'] = "physicalMeasurement"
            IS2_atl10_tide_attrs[gtx][group]['longitude']['long_name'] = "Longitude"
            IS2_atl10_tide_attrs[gtx][group]['longitude']['standard_name'] = "longitude"
            IS2_atl10_tide_attrs[gtx][group]['longitude']['description'] = ("Longitude of "
                "segment center")
            IS2_atl10_tide_attrs[gtx][group]['longitude']['valid_min'] = -180.0
            IS2_atl10_tide_attrs[gtx][group]['longitude']['valid_max'] = 180.0
            IS2_atl10_tide_attrs[gtx][group]['longitude']['coordinates'] = \
                "delta_time latitude"

            # geophysical variables
            IS2_atl10_tide[gtx][group]['geophysical'] = {}
            IS2_atl10_fill[gtx][group]['geophysical'] = {}
            IS2_atl10_dims[gtx][group]['geophysical'] = {}
            IS2_atl10_tide_attrs[gtx][group]['geophysical'] = {}
            IS2_atl10_tide_attrs[gtx][group]['geophysical']['Description'] = ("Contains geophysical "
                "parameters and corrections used to correct photon heights for geophysical effects, "
                "such as tides.")
            IS2_atl10_tide_attrs[gtx][group]['geophysical']['data_rate'] = ("Data within this group "
                "are stored at the variable segment rate.")

            # computed tide
            IS2_atl10_tide[gtx][group]['geophysical'][model.atl10] = tide.copy()
            IS2_atl10_fill[gtx][group]['geophysical'][model.atl10] = tide.fill_value
            IS2_atl10_dims[gtx][group]['geophysical'][model.atl10] = ['delta_time']
            IS2_atl10_tide_attrs[gtx][group]['geophysical'][model.atl10] = {}
            IS2_atl10_tide_attrs[gtx][group]['geophysical'][model.atl10]['units'] = "meters"
            IS2_atl10_tide_attrs[gtx][group]['geophysical'][model.atl10]['contentType'] = \
                "referenceInformation"
            IS2_atl10_tide_attrs[gtx][group]['geophysical'][model.atl10]['long_name'] = model.long_name
            IS2_atl10_tide_attrs[gtx][group]['geophysical'][model.atl10]['description'] = model.description
            IS2_atl10_tide_attrs[gtx][group]['geophysical'][model.atl10]['source'] = model.name
            IS2_atl10_tide_attrs[gtx][group]['geophysical'][model.atl10]['reference'] = model.reference
            IS2_atl10_tide_attrs[gtx][group]['geophysical'][model.atl10]['coordinates'] = \
                "../delta_time ../latitude ../longitude"

    # print file information
    logger.info(f'\t{OUTPUT_FILE}')
    HDF5_ATL10_tide_write(IS2_atl10_tide, IS2_atl10_tide_attrs,
        CLOBBER=True, INPUT=os.path.basename(INPUT_FILE),
        FILL_VALUE=IS2_atl10_fill, DIMENSIONS=IS2_atl10_dims,
        FILENAME=os.path.join(DIRECTORY,OUTPUT_FILE))
    # change the permissions mode
    os.chmod(os.path.join(DIRECTORY,OUTPUT_FILE), MODE)

# PURPOSE: outputting the tide values for ICESat-2 data to HDF5
def HDF5_ATL10_tide_write(IS2_atl10_tide, IS2_atl10_attrs, INPUT=None,
    FILENAME='', FILL_VALUE=None, DIMENSIONS=None, CLOBBER=False):
    # setting HDF5 clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'w-'

    # open output HDF5 file
    fileID = h5py.File(os.path.expanduser(FILENAME), clobber)

    # create HDF5 records
    h5 = {}

    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    h5['ancillary_data'] = {}
    for k,v in IS2_atl10_tide['ancillary_data'].items():
        # Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS2_atl10_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    # write each output beam
    beams = [k for k in IS2_atl10_tide.keys() if bool(re.match(r'gt\d[lr]',k))]
    for gtx in beams:
        fileID.create_group(gtx)
        # add HDF5 group attributes for beam
        for att_name in ['Description','atlas_pce','atlas_beam_type',
            'groundtrack_id','atmosphere_profile','atlas_spot_number',
            'sc_orientation']:
            fileID[gtx].attrs[att_name] = IS2_atl10_attrs[gtx][att_name]
        # create freeboard_beam_segment and leads groups
        h5[gtx] = dict(freeboard_beam_segment={},leads={})
        for group in ['freeboard_beam_segment','leads']:
            fileID[gtx].create_group(group)
            for att_name in ['Description','data_rate']:
                att_val = IS2_atl10_attrs[gtx][group][att_name]
                fileID[gtx][group].attrs[att_name] = att_val

            # delta_time and geolocation variables
            for k in ['delta_time','latitude','longitude']:
                # values and attributes
                v = IS2_atl10_tide[gtx][group][k]
                attrs = IS2_atl10_attrs[gtx][group][k]
                fillvalue = FILL_VALUE[gtx][group][k]
                # Defining the HDF5 dataset variables
                val = '{0}/{1}/{2}'.format(gtx,group,k)
                if fillvalue:
                    h5[gtx][group][k] = fileID.create_dataset(val,
                        np.shape(v), data=v, dtype=v.dtype, fillvalue=fillvalue,
                        compression='gzip')
                else:
                    h5[gtx][group][k] = fileID.create_dataset(val,
                        np.shape(v), data=v, dtype=v.dtype, compression='gzip')
                # create or attach dimensions for HDF5 variable
                if DIMENSIONS[gtx][group][k]:
                    # attach dimensions
                    for i,dim in enumerate(DIMENSIONS[gtx][group][k]):
                        h5[gtx][group][k].dims[i].attach_scale(
                            h5[gtx][group][dim])
                else:
                    # make dimension
                    h5[gtx][group][k].make_scale(k)
                # add HDF5 variable attributes
                for att_name,att_val in attrs.items():
                    h5[gtx][group][k].attrs[att_name] = att_val

            # add to geophysical corrections
            key = 'geophysical'
            fileID[gtx][group].create_group(key)
            h5[gtx][group][key] = {}
            for att_name in ['Description','data_rate']:
                att_val=IS2_atl10_attrs[gtx][group][key][att_name]
                fileID[gtx][group][key].attrs[att_name] = att_val
            for k,v in IS2_atl10_tide[gtx][group][key].items():
                # attributes
                attrs = IS2_atl10_attrs[gtx][group][key][k]
                fillvalue = FILL_VALUE[gtx][group][key][k]
                # Defining the HDF5 dataset variables
                val = '{0}/{1}/{2}/{3}'.format(gtx,group,key,k)
                if fillvalue:
                    h5[gtx][group][key][k] = \
                        fileID.create_dataset(val, np.shape(v), data=v,
                        dtype=v.dtype, fillvalue=fillvalue, compression='gzip')
                else:
                    h5[gtx][group][key][k] = \
                        fileID.create_dataset(val, np.shape(v), data=v,
                        dtype=v.dtype, compression='gzip')
                # attach dimensions
                for i,dim in enumerate(DIMENSIONS[gtx][group][key][k]):
                    h5[gtx][group][key][k].dims[i].attach_scale(
                        h5[gtx][group][dim])
                # add HDF5 variable attributes
                for att_name,att_val in attrs.items():
                    h5[gtx][group][key][k].attrs[att_name] = att_val

    # HDF5 file title
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'ATLAS/ICESat-2 L3A Sea Ice Freeboard'
    fileID.attrs['summary'] = ('Estimates of the sea ice tidal parameters '
        'needed to interpret and assess the quality of the freeboard estimates.')
    fileID.attrs['description'] = ('The data set (ATL10) contains estimates '
        'of sea ice freeboard, calculated using three different approaches. '
        'Sea ice leads used to establish the reference sea surface and '
        'descriptive statistics used in the height estimates are also provided')
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
    # add attributes for input ATL10 file
    fileID.attrs['input_files'] = os.path.basename(INPUT)
    # find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for gtx in beams:
        # for each ATL10 group
        for group in ['freeboard_beam_segment','leads']:
            lon = IS2_atl10_tide[gtx][group]['longitude']
            lat = IS2_atl10_tide[gtx][group]['latitude']
            delta_time = IS2_atl10_tide[gtx][group]['delta_time']
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
    # convert start and end time from ATLAS SDP seconds into GPS seconds
    atlas_sdp_gps_epoch=IS2_atl10_tide['ancillary_data']['atlas_sdp_gps_epoch']
    gps_seconds = atlas_sdp_gps_epoch + np.array([tmn,tmx])
    # calculate leap seconds
    leaps = pyTMD.time.count_leap_seconds(gps_seconds)
    # convert from seconds since 1980-01-06T00:00:00 to Julian days
    time_julian = 2400000.5 + pyTMD.time.convert_delta_time(gps_seconds - leaps,
        epoch1=pyTMD.time._gps_epoch, epoch2=(1858,11,17,0,0,0), scale=1.0/86400.0)
    # convert to calendar date
    YY,MM,DD,HH,MN,SS = pyTMD.time.convert_julian(time_julian,format='tuple')
    # add attributes with measurement date start, end and duration
    tcs = datetime.datetime(int(YY[0]), int(MM[0]), int(DD[0]),
        int(HH[0]), int(MN[0]), int(SS[0]), int(1e6*(SS[0] % 1)))
    fileID.attrs['time_coverage_start'] = tcs.isoformat()
    tce = datetime.datetime(int(YY[1]), int(MM[1]), int(DD[1]),
        int(HH[1]), int(MN[1]), int(SS[1]), int(1e6*(SS[1] % 1)))
    fileID.attrs['time_coverage_end'] = tce.isoformat()
    fileID.attrs['time_coverage_duration'] = f'{tmx-tmn:0.0f}'
    # add software information
    fileID.attrs['software_reference'] = pyTMD.version.project_name
    fileID.attrs['software_version'] = pyTMD.version.full_version
    fileID.attrs['software_revision'] = pyTMD.utilities.get_git_revision_hash()
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
        description="""Calculates tidal elevations for correcting ICESat-2 ATL10
            sea ice freeboard data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    group = parser.add_mutually_exclusive_group(required=True)
    # input ICESat-2 sea ice height files
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat-2 ATL10 file to run')
    # directory with tide data
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory')
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
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        help='Tide model definition file for use as correction')
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

    # run for each input ATL10 file
    for FILE in args.infile:
        compute_tides_ICESat2(args.directory, FILE, TIDE_MODEL=args.tide,
            ATLAS_FORMAT=args.atlas_format, GZIP=args.gzip,
            DEFINITION_FILE=args.definition_file, METHOD=args.interpolate,
            EXTRAPOLATE=args.extrapolate, CUTOFF=args.cutoff,
            VERBOSE=args.verbose, MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
