#!/usr/bin/env python
u"""
compute_tides_ICESat2_ATL03.py
Written by Tyler Sutterley (05/2023)
Calculates tidal elevations for correcting ICESat-2 photon height data
Calculated at ATL03 segment level using reference photon geolocation and time
Segment level corrections can be applied to the individual photon events (PEs)

Uses OTIS format tidal solutions provided by Ohio State University and ESR
    http://volkov.oce.orst.edu/tides/region.html
    https://www.esr.org/research/polar-tide-models/list-of-polar-tide-models/
    ftp://ftp.esr.org/pub/datasets/tmd/
Global Tide Model (GOT) solutions provided by Richard Ray at GSFC
or Finite Element Solution (FES) models provided by AVISO

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -T X, --tide X: Tide model to use in correction
    --atlas-format X: ATLAS tide model format (OTIS, netcdf)
    --gzip, -G: Tide model files are gzip compressed
    --definition-file X: Model definition file for use as correction
    -I X, --interpolate X: Interpolation method
        spline
        linear
        nearest
        bilinear
    -E X, --extrapolate X: Extrapolate with nearest-neighbors
    -c X, --cutoff X: Extrapolation cutoff in kilometers
        set to inf to extrapolate for all points
    --apply-flexure: Apply ice flexure scaling factor to height constituents
        Only valid for models containing flexure fields
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
    io/ATL03.py: reads ICESat-2 global geolocated photon data files
    time.py: utilities for calculating time operations
    utilities.py: download and management utilities for syncing files
    astro.py: computes the basic astronomical mean longitudes
    convert_crs.py: convert points to and from Coordinates Reference Systems
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
    Updated 05/2023: use timescale class for time conversion operations
        using pathlib to define and operate on paths
    Updated 12/2022: single implicit import of grounding zone tools
        refactored ICESat-2 data product read programs under io
        use read and interpolation scheme for tidal constituents
    Updated 07/2022: place some imports within try/except statements
    Updated 05/2022: added ESR netCDF4 formats to list of model types
        updated keyword arguments to read tide model programs
        added command line option to apply flexure for applicable models
    Updated 04/2022: use argparse descriptions within documentation
    Updated 03/2022: using static decorators to define available models
    Updated 02/2022: added Arctic 2km model (Arc2kmTM) to list of models
    Updated 12/2021: added TPXO9-atlas-v5 to list of available tide models
    Updated 10/2021: using python logging for handling verbose output
    Updated 09/2021: refactor to use model class for files and attributes
    Updated 07/2021: can use prefix files to define command line arguments
    Updated 06/2021: added new Gr1km-v2 1km Greenland model from ESR
    Updated 05/2021: added option for extrapolation cutoff in kilometers
    Updated 04/2021: can use a generically named ATL03 file as input
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
    Updated 03/2020: use read_ICESat2_ATL03.py from read-ICESat-2 repository
    Updated 02/2020: changed CATS2008 grid to match version on U.S. Antarctic
        Program Data Center http://www.usap-dc.org/view/dataset/601235
    Updated 11/2019: calculate minor constituents as separate variable
        added AOTIM-5-2018 tide model (2018 update to 2004 model)
    Updated 10/2019: external read functions.  adjust regex for processed files
        changing Y/N flags to True/False
    Updated 09/2019: using date functions paralleling public repository
        add option for TPXO9-atlas.  add OTIS netcdf tide option
    Written 04/2019
"""
from __future__ import print_function

import sys
import re
import logging
import pathlib
import argparse
import datetime
import warnings
import numpy as np
import grounding_zones as gz

# attempt imports
try:
    import h5py
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("h5py not available", ImportWarning)
try:
    import icesat2_toolkit as is2tk
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("icesat2_toolkit not available", ImportWarning)
try:
    import pyTMD
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("pyTMD not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: read ICESat-2 geolocated photon data (ATL03) from NSIDC
# compute tides at points and times using tidal model driver algorithms
def compute_tides_ICESat2(tide_dir, INPUT_FILE,
    TIDE_MODEL=None,
    ATLAS_FORMAT=None,
    GZIP=True,
    DEFINITION_FILE=None,
    METHOD='spline',
    EXTRAPOLATE=False,
    CUTOFF=None,
    APPLY_FLEXURE=False,
    VERBOSE=False,
    MODE=0o775):

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
    logger.info(f'{str(INPUT_FILE)} -->')
    INPUT_FILE = pathlib.Path(INPUT_FILE).expanduser().absolute()
    IS2_atl03_mds,IS2_atl03_attrs,IS2_atl03_beams = \
        is2tk.io.ATL03.read_main(INPUT_FILE, ATTRIBUTES=True)

    # flexure flag if being applied
    flexure_flag = '_FLEXURE' if APPLY_FLEXURE and model.flexure else ''
    # extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})'
        r'(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    try:
        SUB,PRD,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX = \
            rx.findall(INPUT_FILE.name).pop()
    except:
        # output tide HDF5 file (generic)
        args = (INPUT_FILE.stem,model.name,flexure_flag,INPUT_FILE.suffix)
        FILENAME = '{0}_{1}{2}_TIDES{3}'.format(*args)
    else:
        # output tide HDF5 file for ASAS/NSIDC granules
        args = (PRD,model.name,flexure_flag,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
        file_format = '{0}_{1}{2}_TIDES_{3}{4}{5}{6}{7}{8}_{9}{10}{11}_{12}_{13}{14}.h5'
        FILENAME = file_format.format(*args)
    # full path to output file
    OUTPUT_FILE = INPUT_FILE.with_name(FILENAME)

    # read tidal constants
    if model.format in ('OTIS','ATLAS','ESR'):
        constituents = pyTMD.io.OTIS.read_constants(model.grid_file,
            model.model_file, model.projection, type=model.type,
            grid=model.format, apply_flexure=APPLY_FLEXURE)
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
    IS2_atl03_tide = {}
    IS2_atl03_fill = {}
    IS2_atl03_dims = {}
    IS2_atl03_tide_attrs = {}
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl03_tide['ancillary_data'] = {}
    IS2_atl03_tide_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl03_tide['ancillary_data'][key] = IS2_atl03_mds['ancillary_data'][key]
        # Getting attributes of group and included variables
        IS2_atl03_tide_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl03_attrs['ancillary_data'][key].items():
            IS2_atl03_tide_attrs['ancillary_data'][key][att_name] = att_val

    # for each input beam within the file
    for gtx in sorted(IS2_atl03_beams):
        # output data dictionaries for beam
        IS2_atl03_tide[gtx] = dict(geolocation={}, geophys_corr={})
        IS2_atl03_fill[gtx] = dict(geolocation={}, geophys_corr={})
        IS2_atl03_dims[gtx] = dict(geolocation={}, geophys_corr={})
        IS2_atl03_tide_attrs[gtx] = dict(geolocation={}, geophys_corr={})

        # read data and attributes for beam
        val,attrs = is2tk.io.ATL03.read_beam(INPUT_FILE, gtx,
                                             ATTRIBUTES=True)
        # number of segments
        n_seg = len(val['geolocation']['segment_id'])
        # extract variables for computing tides
        segment_id = val['geolocation']['segment_id'].copy()
        delta_time = val['geolocation']['delta_time'].copy()
        lon = val['geolocation']['reference_photon_lon'].copy()
        lat = val['geolocation']['reference_photon_lat'].copy()
        # invalid value
        fv = attrs['geolocation']['sigma_h']['_FillValue']

        # create timescale from ATLAS Standard Epoch time
        # GPS seconds since 2018-01-01 00:00:00 UTC
        timescale = pyTMD.time.timescale().from_deltatime(delta_time,
            epoch=pyTMD.time._atlas_sdp_epoch, standard='GPS')

        # interpolate tidal constants to grid points
        if model.format in ('OTIS','ATLAS','ESR'):
            amp,ph,D = pyTMD.io.OTIS.interpolate_constants(lon, lat,
                constituents, model.projection, type=model.type,
                method=METHOD, extrapolate=EXTRAPOLATE, cutoff=CUTOFF)
            # use delta time at 2000.0 to match TMD outputs
            deltat = np.zeros((n_seg))
        elif (model.format == 'netcdf'):
            amp,ph,D = pyTMD.io.ATLAS.interpolate_constants(lon, lat,
                constituents, type=model.type, method=METHOD,
                extrapolate=EXTRAPOLATE, cutoff=CUTOFF, scale=model.scale)
            # use delta time at 2000.0 to match TMD outputs
            deltat = np.zeros((n_seg))
        elif (model.format == 'GOT'):
            amp,ph = pyTMD.io.GOT.interpolate_constants(lon, lat,
                constituents, method=METHOD, extrapolate=EXTRAPOLATE,
                cutoff=CUTOFF, scale=model.scale)
            # delta time (TT - UT1)
            deltat = timescale.tt_ut1
        elif (model.format == 'FES'):
            amp,ph = pyTMD.io.FES.interpolate_constants(lon, lat,
                constituents, method=METHOD, extrapolate=EXTRAPOLATE,
                cutoff=CUTOFF, scale=model.scale)
            # delta time (TT - UT1)
            deltat = timescale.tt_ut1

        # calculate complex phase in radians for Euler's
        cph = -1j*ph*np.pi/180.0
        # calculate constituent oscillation
        hc = amp*np.exp(cph)

        # predict tidal elevations at time and infer minor corrections
        tide = np.ma.empty((n_seg),fill_value=fv)
        tide.mask = np.any(hc.mask,axis=1)
        tide.data[:] = pyTMD.predict.drift(timescale.tide, hc, c,
            deltat=deltat, corrections=model.format)
        minor = pyTMD.predict.infer_minor(timescale.tide, hc, c,
            deltat=deltat, corrections=model.format)
        tide.data[:] += minor.data[:]
        # replace masked and nan values with fill value
        invalid, = np.nonzero(np.isnan(tide.data) | tide.mask)
        tide.data[invalid] = tide.fill_value
        tide.mask[invalid] = True

        # group attributes for beam
        IS2_atl03_tide_attrs[gtx]['Description'] = attrs['Description']
        IS2_atl03_tide_attrs[gtx]['atlas_pce'] = attrs['atlas_pce']
        IS2_atl03_tide_attrs[gtx]['atlas_beam_type'] = attrs['atlas_beam_type']
        IS2_atl03_tide_attrs[gtx]['groundtrack_id'] = attrs['groundtrack_id']
        IS2_atl03_tide_attrs[gtx]['atmosphere_profile'] = attrs['atmosphere_profile']
        IS2_atl03_tide_attrs[gtx]['atlas_spot_number'] = attrs['atlas_spot_number']
        IS2_atl03_tide_attrs[gtx]['sc_orientation'] = attrs['sc_orientation']

        # group attributes for geolocation
        IS2_atl03_tide_attrs[gtx]['geolocation']['Description'] = ("Contains parameters related to "
            "geolocation.  The rate of all of these parameters is at the rate corresponding to the "
            "ICESat-2 Geolocation Along Track Segment interval (nominally 20 m along-track).")
        IS2_atl03_tide_attrs[gtx]['geolocation']['data_rate'] = ("Data within this group are "
            "stored at the ICESat-2 20m segment rate.")
        # group attributes for geophys_corr
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['Description'] = ("Contains parameters used to "
            "correct photon heights for geophysical effects, such as tides.  These parameters are "
            "posted at the same interval as the ICESat-2 Geolocation Along-Track Segment interval "
            "(nominally 20m along-track).")
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['data_rate'] = ("These parameters are stored at "
            "the ICESat-2 Geolocation Along Track Segment rate (nominally every 20 m along-track).")

        # geolocation, time and segment ID
        # delta time in geolocation group
        IS2_atl03_tide[gtx]['geolocation']['delta_time'] = delta_time
        IS2_atl03_fill[gtx]['geolocation']['delta_time'] = None
        IS2_atl03_dims[gtx]['geolocation']['delta_time'] = None
        IS2_atl03_tide_attrs[gtx]['geolocation']['delta_time'] = {}
        IS2_atl03_tide_attrs[gtx]['geolocation']['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl03_tide_attrs[gtx]['geolocation']['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl03_tide_attrs[gtx]['geolocation']['delta_time']['standard_name'] = "time"
        IS2_atl03_tide_attrs[gtx]['geolocation']['delta_time']['calendar'] = "standard"
        IS2_atl03_tide_attrs[gtx]['geolocation']['delta_time']['description'] = ("Elapsed seconds "
            "from the ATLAS SDP GPS Epoch, corresponding to the transmit time of the reference "
            "photon. The ATLAS Standard Data Products (SDP) epoch offset is defined within "
            "/ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds between the GPS epoch "
            "(1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By adding the offset "
            "contained within atlas_sdp_gps_epoch to delta time parameters, the time in gps_seconds "
            "relative to the GPS epoch can be computed.")
        IS2_atl03_tide_attrs[gtx]['geolocation']['delta_time']['coordinates'] = \
            "segment_id reference_photon_lat reference_photon_lon"
        # delta time in geophys_corr group
        IS2_atl03_tide[gtx]['geophys_corr']['delta_time'] = delta_time
        IS2_atl03_fill[gtx]['geophys_corr']['delta_time'] = None
        IS2_atl03_dims[gtx]['geophys_corr']['delta_time'] = None
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['delta_time'] = {}
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['delta_time']['standard_name'] = "time"
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['delta_time']['calendar'] = "standard"
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['delta_time']['description'] = ("Elapsed seconds "
            "from the ATLAS SDP GPS Epoch, corresponding to the transmit time of the reference "
            "photon. The ATLAS Standard Data Products (SDP) epoch offset is defined within "
            "/ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds between the GPS epoch "
            "(1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By adding the offset "
            "contained within atlas_sdp_gps_epoch to delta time parameters, the time in gps_seconds "
            "relative to the GPS epoch can be computed.")
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['delta_time']['coordinates'] = ("../geolocation/segment_id "
            "../geolocation/reference_photon_lat ../geolocation/reference_photon_lon")

        # latitude
        IS2_atl03_tide[gtx]['geolocation']['reference_photon_lat'] = lat
        IS2_atl03_fill[gtx]['geolocation']['reference_photon_lat'] = None
        IS2_atl03_dims[gtx]['geolocation']['reference_photon_lat'] = ['delta_time']
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lat'] = {}
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lat']['units'] = "degrees_north"
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lat']['contentType'] = "physicalMeasurement"
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lat']['long_name'] = "Latitude"
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lat']['standard_name'] = "latitude"
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lat']['description'] = ("Latitude of each "
            "reference photon. Computed from the ECF Cartesian coordinates of the bounce point.")
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lat']['valid_min'] = -90.0
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lat']['valid_max'] = 90.0
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lat']['coordinates'] = \
            "segment_id delta_time reference_photon_lon"
        # longitude
        IS2_atl03_tide[gtx]['geolocation']['reference_photon_lon'] = lon
        IS2_atl03_fill[gtx]['geolocation']['reference_photon_lon'] = None
        IS2_atl03_dims[gtx]['geolocation']['reference_photon_lon'] = ['delta_time']
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lon'] = {}
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lon']['units'] = "degrees_east"
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lon']['contentType'] = "physicalMeasurement"
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lon']['long_name'] = "Longitude"
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lon']['standard_name'] = "longitude"
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lon']['description'] = ("Longitude of each "
            "reference photon. Computed from the ECF Cartesian coordinates of the bounce point.")
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lon']['valid_min'] = -180.0
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lon']['valid_max'] = 180.0
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lon']['coordinates'] = \
            "segment_id delta_time reference_photon_lat"
        # segment ID
        IS2_atl03_tide[gtx]['geolocation']['segment_id'] = segment_id
        IS2_atl03_fill[gtx]['geolocation']['segment_id'] = None
        IS2_atl03_dims[gtx]['geolocation']['segment_id'] = ['delta_time']
        IS2_atl03_tide_attrs[gtx]['geolocation']['segment_id'] = {}
        IS2_atl03_tide_attrs[gtx]['geolocation']['segment_id']['units'] = "1"
        IS2_atl03_tide_attrs[gtx]['geolocation']['segment_id']['contentType'] = "referenceInformation"
        IS2_atl03_tide_attrs[gtx]['geolocation']['segment_id']['long_name'] = "Along-track segment ID number"
        IS2_atl03_tide_attrs[gtx]['geolocation']['segment_id']['description'] = ("A 7 digit number "
            "identifying the along-track geolocation segment number.  These are sequential, starting with "
            "1 for the first segment after an ascending equatorial crossing node.")
        IS2_atl03_tide_attrs[gtx]['geolocation']['segment_id']['coordinates'] = \
            "delta_time reference_photon_lat reference_photon_lon"

        # computed tide
        IS2_atl03_tide[gtx]['geophys_corr'][model.atl03] = tide
        IS2_atl03_fill[gtx]['geophys_corr'][model.atl03] = tide.fill_value
        IS2_atl03_dims[gtx]['geophys_corr'][model.atl03] = ['delta_time']
        IS2_atl03_tide_attrs[gtx]['geophys_corr'][model.atl03] = {}
        IS2_atl03_tide_attrs[gtx]['geophys_corr'][model.atl03]['units'] = "meters"
        IS2_atl03_tide_attrs[gtx]['geophys_corr'][model.atl03]['contentType'] = "referenceInformation"
        IS2_atl03_tide_attrs[gtx]['geophys_corr'][model.atl03]['long_name'] = model.long_name
        IS2_atl03_tide_attrs[gtx]['geophys_corr'][model.atl03]['description'] = model.description
        IS2_atl03_tide_attrs[gtx]['geophys_corr'][model.atl03]['source'] = model.name
        IS2_atl03_tide_attrs[gtx]['geophys_corr'][model.atl03]['reference'] = model.reference
        IS2_atl03_tide_attrs[gtx]['geophys_corr'][model.atl03]['coordinates'] = \
            ("../geolocation/segment_id ../geolocation/delta_time "
            "../geolocation/reference_photon_lat ../geolocation/reference_photon_lon")

    # print file information
    logger.info(f'\t{str(OUTPUT_FILE)}')
    HDF5_ATL03_tide_write(IS2_atl03_tide, IS2_atl03_tide_attrs,
        CLOBBER=True, INPUT=INPUT_FILE.name,
        FILL_VALUE=IS2_atl03_fill, DIMENSIONS=IS2_atl03_dims,
        FILENAME=OUTPUT_FILE)
    # change the permissions mode
    OUTPUT_FILE.chmod(mode=MODE)

# PURPOSE: outputting the tide values for ICESat-2 data to HDF5
def HDF5_ATL03_tide_write(IS2_atl03_tide, IS2_atl03_attrs, INPUT=None,
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
    for k,v in IS2_atl03_tide['ancillary_data'].items():
        # Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS2_atl03_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    # write each output beam
    beams = [k for k in IS2_atl03_tide.keys() if bool(re.match(r'gt\d[lr]',k))]
    for gtx in beams:
        fileID.create_group(gtx)
        h5[gtx] = {}
        # add HDF5 group attributes for beam
        for att_name in ['Description','atlas_pce','atlas_beam_type',
            'groundtrack_id','atmosphere_profile','atlas_spot_number',
            'sc_orientation']:
            fileID[gtx].attrs[att_name] = IS2_atl03_attrs[gtx][att_name]
        # create geolocation and geophys_corr groups
        for key in ['geolocation','geophys_corr']:
            fileID[gtx].create_group(key)
            h5[gtx][key] = {}
            for att_name in ['Description','data_rate']:
                att_val = IS2_atl03_attrs[gtx][key][att_name]
                fileID[gtx][key].attrs[att_name] = att_val

            # all variables for group
            groupkeys = set(IS2_atl03_tide[gtx][key].keys())-set(['delta_time'])
            for k in ['delta_time',*sorted(groupkeys)]:
                # values and attributes
                v = IS2_atl03_tide[gtx][key][k]
                attrs = IS2_atl03_attrs[gtx][key][k]
                fillvalue = FILL_VALUE[gtx][key][k]
                # Defining the HDF5 dataset variables
                val = '{0}/{1}/{2}'.format(gtx,key,k)
                if fillvalue:
                    h5[gtx][key][k] = fileID.create_dataset(val, np.shape(v),
                        data=v, dtype=v.dtype, fillvalue=fillvalue,
                        compression='gzip')
                else:
                    h5[gtx][key][k] = fileID.create_dataset(val, np.shape(v),
                        data=v, dtype=v.dtype, compression='gzip')
                # create or attach dimensions for HDF5 variable
                if DIMENSIONS[gtx][key][k]:
                    # attach dimensions
                    for i,dim in enumerate(DIMENSIONS[gtx][key][k]):
                        h5[gtx][key][k].dims[i].attach_scale(h5[gtx][key][dim])
                else:
                    # make dimension
                    h5[gtx][key][k].make_scale(k)
                # add HDF5 variable attributes
                for att_name,att_val in attrs.items():
                    h5[gtx][key][k].attrs[att_name] = att_val

    # HDF5 file title
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'ATLAS/ICESat-2 L2A Global Geolocated Photon Data'
    fileID.attrs['summary'] = ('The purpose of ATL03 is to provide along-track '
        'photon data for all 6 ATLAS beams and associated statistics')
    fileID.attrs['description'] = ('Photon heights determined by ATBD '
        'Algorithm using POD and PPD. All photon events per transmit pulse '
        'per beam. Includes POD and PPD vectors. Classification of each '
        'photon by several ATBD Algorithms.')
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
    # add attributes for input ATL03 file
    fileID.attrs['lineage'] = pathlib.Path(INPUT).name
    # find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for gtx in beams:
        lon = IS2_atl03_tide[gtx]['geolocation']['reference_photon_lon']
        lat = IS2_atl03_tide[gtx]['geolocation']['reference_photon_lat']
        delta_time = IS2_atl03_tide[gtx]['geolocation']['delta_time']
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
    timescale = pyTMD.time.timescale().from_deltatime(np.array([tmn,tmx]),
        epoch=pyTMD.time._atlas_sdp_epoch, standard='GPS')
    dt = np.datetime_as_string(timescale.to_datetime(), unit='s')
    # add attributes with measurement date start, end and duration
    fileID.attrs['time_coverage_start'] = dt[0]
    fileID.attrs['time_coverage_end'] = dt[1]
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
        description="""Calculates tidal elevations for correcting ICESat-2 ATL03
            geolocated photon height data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    group = parser.add_mutually_exclusive_group(required=True)
    # input ICESat-2 geolocated photon height files
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='ICESat-2 ATL03 file to run')
    # directory with tide data
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=pathlib.Path.cwd()
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
        type=pathlib.Path,
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
    # apply flexure scaling factors to height constituents
    parser.add_argument('--apply-flexure',
        default=False, action='store_true',
        help='Apply ice flexure scaling factor to height constituents')
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

    # run for each input ATL03 file
    for FILE in args.infile:
        compute_tides_ICESat2(args.directory, FILE,
            TIDE_MODEL=args.tide,
            ATLAS_FORMAT=args.atlas_format,
            GZIP=args.gzip,
            DEFINITION_FILE=args.definition_file,
            METHOD=args.interpolate,
            EXTRAPOLATE=args.extrapolate,
            CUTOFF=args.cutoff,
            APPLY_FLEXURE=args.apply_flexure,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
