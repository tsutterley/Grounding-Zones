#!/usr/bin/env python
u"""
compute_tides_ICESat2_ATL11.py
Written by Tyler Sutterley (08/2024)
Calculates tidal elevations for correcting ICESat-2 annual land ice height data

Uses OTIS format tidal solutions provided by Oregon State University and ESR
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
    --infer-minor: Infer values for minor constituents
    --minor-constituents: Minor constituents to infer
    --apply-flexure: Apply ice flexure scaling factor to height values
        Only valid for models containing flexure fields
    -C, --crossovers: Run ATL11 Crossovers
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
    io/ATL11.py: reads ICESat-2 annual land ice height data files
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
    Updated 08/2024: project bounds for cropping non-geographic OTIS models
        added option to allow inferring only specific minor constituents
        added option to try automatic detection of definition file format
    Updated 07/2024: added option to crop to the domain of the input data
        added option to use JSON format definition files
        renamed format for ATLAS to ATLAS-compact
        renamed format for netcdf to ATLAS-netcdf
        renamed format for FES to FES-netcdf and added FES-ascii
        renamed format for GOT to GOT-ascii and added GOT-netcdf
        only append crossovers group if there are valid crossovers
    Updated 06/2024: added option to not run with crossover measurements
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
    Updated 04/2021: can use a generically named ATL11 file as input
    Updated 03/2021: added TPXO9-atlas-v4 in binary OTIS format
        simplified netcdf inputs to be similar to binary OTIS read program
        replaced numpy bool/int to prevent deprecation warnings
    Updated 02/2021: additionally calculate tides for crossing track data
    Updated 01/2021: using standalone ATL11 reader
    Updated 12/2020: merged time conversion routines into module
    Written 12/2020
"""
from __future__ import print_function

import sys
import re
import logging
import pathlib
import argparse
import datetime
import numpy as np
import collections
import grounding_zones as gz

# attempt imports
h5py = gz.utilities.import_dependency('h5py')
is2tk = gz.utilities.import_dependency('icesat2_toolkit')
pyTMD = gz.utilities.import_dependency('pyTMD')
timescale = gz.utilities.import_dependency('timescale')

# PURPOSE: read ICESat-2 annual land ice height data (ATL11)
# compute tides at points and times using tidal model driver algorithms
def compute_tides_ICESat2(tide_dir, INPUT_FILE,
        OUTPUT_DIRECTORY=None,
        TIDE_MODEL=None,
        ATLAS_FORMAT=None,
        GZIP=True,
        DEFINITION_FILE=None,
        DEFINITION_FORMAT='auto',
        CROP=False,
        METHOD='spline',
        EXTRAPOLATE=False,
        CUTOFF=None,
        INFER_MINOR=False,
        MINOR_CONSTITUENTS=None,
        APPLY_FLEXURE=False,
        CROSSOVERS=False,
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

    # flexure flag if being applied
    flexure_flag = '_FLEXURE' if APPLY_FLEXURE and model.flexure else ''
    # extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})_(\d{2})(\d{2})_'
        r'(\d{3})_(\d{2})(.*?).h5$')
    try:
        SUB,PRD,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX = \
            rx.findall(GRANULE).pop()
    except:
        # output tide HDF5 file (generic)
        args = (INPUT_FILE.stem,model.name,flexure_flag,INPUT_FILE.suffix)
        FILENAME = '{0}_{1}{2}_TIDES{3}'.format(*args)
    else:
        # output tide HDF5 file for ASAS/NSIDC granules
        args = (PRD,model.name,flexure_flag,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
        file_format = '{0}_{1}{2}_TIDES_{3}{4}_{5}{6}_{7}_{8}{9}.h5'
        FILENAME = file_format.format(*args)
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
    # read data from input file
    IS2_atl11_mds,IS2_atl11_attrs,IS2_atl11_pairs = \
        is2tk.io.ATL11.read_granule(INPUT_FILE,
                                    ATTRIBUTES=True,
                                    CROSSOVERS=True)

    # transform bounding box coordinates
    if model.projection:
        transformer = pyTMD.crs().get(model.projection)
    # find geospatial ranges for bounding box
    BOUNDS = [np.inf, -np.inf, np.inf, -np.inf]
    for ptx in IS2_atl11_pairs:
        lon = IS2_atl11_mds[ptx]['longitude']
        lat = IS2_atl11_mds[ptx]['latitude']
        if model.projection:
            x, y = transformer.transform(lon, lat)
            BOUNDS[0] = np.minimum(BOUNDS[0], np.min(x))
            BOUNDS[1] = np.maximum(BOUNDS[1], np.max(x))
            BOUNDS[2] = np.minimum(BOUNDS[2], np.min(y))
            BOUNDS[3] = np.maximum(BOUNDS[3], np.max(y))
        else:
            BOUNDS[0] = np.minimum(BOUNDS[0], np.min(lon))
            BOUNDS[1] = np.maximum(BOUNDS[1], np.max(lon))
            BOUNDS[2] = np.minimum(BOUNDS[2], np.min(lat))
            BOUNDS[3] = np.maximum(BOUNDS[3], np.max(lat))

    # read tidal constants
    corrections, _, grid = model.format.partition('-')
    if model.format in ('OTIS','ATLAS-compact','TMD3'):
        constituents = pyTMD.io.OTIS.read_constants(model.grid_file,
            model.model_file, model.projection, type=model.type,
            grid=corrections, crop=CROP, bounds=BOUNDS,
            apply_flexure=APPLY_FLEXURE)
        # available model constituents
        c = constituents.fields
    elif model.format in ('ATLAS-netcdf',):
        constituents = pyTMD.io.ATLAS.read_constants(model.grid_file,
            model.model_file, type=model.type, compressed=model.compressed,
            crop=CROP, bounds=BOUNDS)
        # available model constituents
        c = constituents.fields
    elif model.format in ('GOT-ascii','GOT-netcdf'):
        constituents = pyTMD.io.GOT.read_constants(model.model_file,
            compressed=model.compressed, grid=grid, crop=CROP, bounds=BOUNDS)
        # available model constituents
        c = constituents.fields
    elif model.format in ('FES-ascii','FES-netcdf'):
        constituents = pyTMD.io.FES.read_constants(model.model_file,
            type=model.type, version=model.version, compressed=model.compressed,
            crop=CROP, bounds=BOUNDS)
        # available model constituents
        c = model.constituents

    # copy variables for outputting to HDF5 file
    IS2_atl11_tide = {}
    IS2_atl11_fill = {}
    IS2_atl11_dims = {}
    IS2_atl11_tide_attrs = {}
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl11_tide['ancillary_data'] = {}
    IS2_atl11_tide_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl11_tide['ancillary_data'][key] = IS2_atl11_mds['ancillary_data'][key]
        # Getting attributes of group and included variables
        IS2_atl11_tide_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl11_attrs['ancillary_data'][key].items():
            IS2_atl11_tide_attrs['ancillary_data'][key][att_name] = att_val
    # HDF5 group name for across-track data
    XT = 'crossing_track_data'

    # for each input beam pair within the file
    for ptx in sorted(IS2_atl11_pairs):
        # output data dictionaries for beam
        IS2_atl11_tide[ptx] = dict(cycle_stats=collections.OrderedDict(),
            crossing_track_data=collections.OrderedDict())
        IS2_atl11_fill[ptx] = dict(cycle_stats={},crossing_track_data={})
        IS2_atl11_dims[ptx] = dict(cycle_stats={},crossing_track_data={})
        IS2_atl11_tide_attrs[ptx] = dict(cycle_stats={},crossing_track_data={})

        # extract along-track and across-track variables
        ref_pt = {}
        latitude = {}
        longitude = {}
        delta_time = {}
        groups = ['AT']
        # allocate for output tidal variables
        tide = {}
        # along-track (AT) reference point, latitude, longitude and time
        ref_pt['AT'] = IS2_atl11_mds[ptx]['ref_pt'].copy()
        latitude['AT'] = np.ma.array(IS2_atl11_mds[ptx]['latitude'],
            fill_value=IS2_atl11_attrs[ptx]['latitude']['_FillValue'])
        longitude['AT'] = np.ma.array(IS2_atl11_mds[ptx]['longitude'],
            fill_value=IS2_atl11_attrs[ptx]['longitude']['_FillValue'])
        delta_time['AT'] = np.ma.array(IS2_atl11_mds[ptx]['delta_time'],
            fill_value=IS2_atl11_attrs[ptx]['delta_time']['_FillValue'])
        # number of average segments and number of included cycles
        # fill_value for invalid heights and corrections
        fv = IS2_atl11_attrs[ptx]['h_corr']['_FillValue']
        # shape of along-track data
        n_points,n_cycles = delta_time['AT'].shape
        # along-track (AT) tides
        tide['AT'] = np.ma.empty((n_points,n_cycles),fill_value=fv)
        tide['AT'].mask = (delta_time['AT'] == delta_time['AT'].fill_value)

        # if running ATL11 crossovers
        if CROSSOVERS:
            # across-track (XT) reference point, latitude, longitude and time
            ref_pt['XT'] = IS2_atl11_mds[ptx][XT]['ref_pt'].copy()
            latitude['XT'] = np.ma.array(IS2_atl11_mds[ptx][XT]['latitude'],
                fill_value=IS2_atl11_attrs[ptx][XT]['latitude']['_FillValue'])
            longitude['XT'] = np.ma.array(IS2_atl11_mds[ptx][XT]['longitude'],
                fill_value=IS2_atl11_attrs[ptx][XT]['longitude']['_FillValue'])
            delta_time['XT'] = np.ma.array(IS2_atl11_mds[ptx][XT]['delta_time'],
                fill_value=IS2_atl11_attrs[ptx][XT]['delta_time']['_FillValue'])
            # shape of across-track data
            n_cross, = delta_time['XT'].shape
            # across-track (XT) tides
            tide['XT'] = np.ma.empty((n_cross),fill_value=fv)
            tide['XT'].mask = (delta_time['XT'] == delta_time['XT'].fill_value)
            # add to group
            if np.any(n_cross):
                groups.append('XT')

        # calculate tides for along-track and across-track data
        for track in groups:
            # create timescale from ATLAS Standard Epoch time
            # GPS seconds since 2018-01-01 00:00:00 UTC
            ts = timescale.time.Timescale().from_deltatime(delta_time[track],
                epoch=timescale.time._atlas_sdp_epoch, standard='GPS')
            nt = len(ts)

            # interpolate tidal constants to grid points
            if model.format in ('OTIS','ATLAS-compact','TMD3'):
                amp,ph,D = pyTMD.io.OTIS.interpolate_constants(longitude[track],
                    latitude[track], constituents, model.projection, type=model.type,
                    method=METHOD, extrapolate=EXTRAPOLATE, cutoff=CUTOFF)
                # use delta time at 2000.0 to match TMD outputs
                deltat = np.zeros_like(ts.tt_ut1)
            elif model.format in ('ATLAS-netcdf',):
                amp,ph,D = pyTMD.io.ATLAS.interpolate_constants(longitude[track],
                    latitude[track], constituents, type=model.type, method=METHOD,
                    extrapolate=EXTRAPOLATE, cutoff=CUTOFF, scale=model.scale)
                # use delta time at 2000.0 to match TMD outputs
                deltat = np.zeros_like(ts.tt_ut1)
            elif model.format in ('GOT-ascii','GOT-netcdf'):
                amp,ph = pyTMD.io.GOT.interpolate_constants(longitude[track],
                    latitude[track], constituents, method=METHOD,
                    extrapolate=EXTRAPOLATE, cutoff=CUTOFF, scale=model.scale)
                # delta time (TT - UT1)
                deltat = ts.tt_ut1
            elif model.format in ('FES-ascii','FES-netcdf'):
                amp,ph = pyTMD.io.FES.interpolate_constants(longitude[track],
                    latitude[track], constituents, method=METHOD,
                    extrapolate=EXTRAPOLATE, cutoff=CUTOFF, scale=model.scale)
                # delta time (TT - UT1)
                deltat = ts.tt_ut1

            # calculate complex phase in radians for Euler's
            cph = -1j*ph*np.pi/180.0
            # calculate constituent oscillation
            hc = amp*np.exp(cph)

            # calculate tides for track type
            minor_constituents = model.minor or MINOR_CONSTITUENTS
            if (track == 'AT'):
                # calculate tides for each cycle if along-track
                for cycle in range(n_cycles):
                    # find valid time and spatial points for cycle
                    tide[track].mask[:,cycle] |= np.any(hc.mask,axis=1)
                    valid, = np.nonzero(~tide[track].mask[:,cycle])
                    # predict tidal elevations and infer minor corrections
                    tide[track].data[valid,cycle] = pyTMD.predict.drift(
                        ts.tide[valid,cycle], hc[valid,:], c,
                        deltat=deltat[valid,cycle], corrections=corrections)
                    # calculate values for minor constituents by inferrence
                    if INFER_MINOR:
                        minor = pyTMD.predict.infer_minor(
                            ts.tide[valid,cycle], hc[valid,:], c,
                            deltat=deltat[valid,cycle],
                            corrections=corrections,
                            minor=minor_constituents)
                        tide[track].data[valid,cycle] += minor.data[:]
            elif (track == 'XT'):
                # find valid time and spatial points
                tide[track].mask[:] |= np.any(hc.mask,axis=1)
                valid, = np.nonzero(~tide[track].mask[:])
                # predict tidal elevations and infer minor corrections
                tide[track].data[valid] = pyTMD.predict.drift(
                    ts.tide[valid], hc[valid,:], c,
                    deltat=deltat[valid], corrections=corrections)
                # calculate values for minor constituents by inferrence
                if INFER_MINOR:
                    minor = pyTMD.predict.infer_minor(
                        ts.tide[valid], hc[valid,:], c,
                        deltat=deltat[valid],
                        corrections=corrections,
                        minor=minor_constituents)
                    tide[track].data[valid] += minor.data[:]

            # replace masked and nan values with fill value
            invalid = np.nonzero(np.isnan(tide[track].data) | tide[track].mask)
            tide[track].data[invalid] = tide[track].fill_value
            tide[track].mask[invalid] = True

        # group attributes for beam
        IS2_atl11_tide_attrs[ptx]['description'] = ('Contains the primary science parameters '
            'for this data set')
        IS2_atl11_tide_attrs[ptx]['beam_pair'] = IS2_atl11_attrs[ptx]['beam_pair']
        IS2_atl11_tide_attrs[ptx]['ReferenceGroundTrack'] = IS2_atl11_attrs[ptx]['ReferenceGroundTrack']
        IS2_atl11_tide_attrs[ptx]['first_cycle'] = IS2_atl11_attrs[ptx]['first_cycle']
        IS2_atl11_tide_attrs[ptx]['last_cycle'] = IS2_atl11_attrs[ptx]['last_cycle']
        IS2_atl11_tide_attrs[ptx]['equatorial_radius'] = IS2_atl11_attrs[ptx]['equatorial_radius']
        IS2_atl11_tide_attrs[ptx]['polar_radius'] = IS2_atl11_attrs[ptx]['polar_radius']

        # geolocation, time and reference point
        # reference point
        IS2_atl11_tide[ptx]['ref_pt'] = ref_pt['AT'].copy()
        IS2_atl11_fill[ptx]['ref_pt'] = None
        IS2_atl11_dims[ptx]['ref_pt'] = None
        IS2_atl11_tide_attrs[ptx]['ref_pt'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx]['ref_pt']['units'] = "1"
        IS2_atl11_tide_attrs[ptx]['ref_pt']['contentType'] = "referenceInformation"
        IS2_atl11_tide_attrs[ptx]['ref_pt']['long_name'] = "Reference point number"
        IS2_atl11_tide_attrs[ptx]['ref_pt']['source'] = "ATL06"
        IS2_atl11_tide_attrs[ptx]['ref_pt']['description'] = ("The reference point is the "
            "7 digit segment_id number corresponding to the center of the ATL06 data used "
            "for each ATL11 point.  These are sequential, starting with 1 for the first "
            "segment after an ascending equatorial crossing node.")
        IS2_atl11_tide_attrs[ptx]['ref_pt']['coordinates'] = \
            "delta_time latitude longitude"
        # cycle_number
        IS2_atl11_tide[ptx]['cycle_number'] = IS2_atl11_mds[ptx]['cycle_number'].copy()
        IS2_atl11_fill[ptx]['cycle_number'] = None
        IS2_atl11_dims[ptx]['cycle_number'] = None
        IS2_atl11_tide_attrs[ptx]['cycle_number'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx]['cycle_number']['units'] = "1"
        IS2_atl11_tide_attrs[ptx]['cycle_number']['long_name'] = "Orbital cycle number"
        IS2_atl11_tide_attrs[ptx]['cycle_number']['source'] = "ATL06"
        IS2_atl11_tide_attrs[ptx]['cycle_number']['description'] = ("Number of 91-day periods "
            "that have elapsed since ICESat-2 entered the science orbit. Each of the 1,387 "
            "reference ground track (RGTs) is targeted in the polar regions once "
            "every 91 days.")
        # delta time
        IS2_atl11_tide[ptx]['delta_time'] = delta_time['AT'].copy()
        IS2_atl11_fill[ptx]['delta_time'] = delta_time['AT'].fill_value
        IS2_atl11_dims[ptx]['delta_time'] = ['ref_pt','cycle_number']
        IS2_atl11_tide_attrs[ptx]['delta_time'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx]['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl11_tide_attrs[ptx]['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl11_tide_attrs[ptx]['delta_time']['standard_name'] = "time"
        IS2_atl11_tide_attrs[ptx]['delta_time']['calendar'] = "standard"
        IS2_atl11_tide_attrs[ptx]['delta_time']['source'] = "ATL06"
        IS2_atl11_tide_attrs[ptx]['delta_time']['description'] = ("Number of GPS "
            "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
            "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
            "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
            "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
            "time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl11_tide_attrs[ptx]['delta_time']['coordinates'] = \
            "ref_pt cycle_number latitude longitude"
        # latitude
        IS2_atl11_tide[ptx]['latitude'] = latitude['AT'].copy()
        IS2_atl11_fill[ptx]['latitude'] = latitude['AT'].fill_value
        IS2_atl11_dims[ptx]['latitude'] = ['ref_pt']
        IS2_atl11_tide_attrs[ptx]['latitude'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx]['latitude']['units'] = "degrees_north"
        IS2_atl11_tide_attrs[ptx]['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_tide_attrs[ptx]['latitude']['long_name'] = "Latitude"
        IS2_atl11_tide_attrs[ptx]['latitude']['standard_name'] = "latitude"
        IS2_atl11_tide_attrs[ptx]['latitude']['source'] = "ATL06"
        IS2_atl11_tide_attrs[ptx]['latitude']['description'] = ("Center latitude of "
            "selected segments")
        IS2_atl11_tide_attrs[ptx]['latitude']['valid_min'] = -90.0
        IS2_atl11_tide_attrs[ptx]['latitude']['valid_max'] = 90.0
        IS2_atl11_tide_attrs[ptx]['latitude']['coordinates'] = \
            "ref_pt delta_time longitude"
        # longitude
        IS2_atl11_tide[ptx]['longitude'] = longitude['AT'].copy()
        IS2_atl11_fill[ptx]['longitude'] = longitude['AT'].fill_value
        IS2_atl11_dims[ptx]['longitude'] = ['ref_pt']
        IS2_atl11_tide_attrs[ptx]['longitude'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx]['longitude']['units'] = "degrees_east"
        IS2_atl11_tide_attrs[ptx]['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_tide_attrs[ptx]['longitude']['long_name'] = "Longitude"
        IS2_atl11_tide_attrs[ptx]['longitude']['standard_name'] = "longitude"
        IS2_atl11_tide_attrs[ptx]['longitude']['source'] = "ATL06"
        IS2_atl11_tide_attrs[ptx]['longitude']['description'] = ("Center longitude of "
            "selected segments")
        IS2_atl11_tide_attrs[ptx]['longitude']['valid_min'] = -180.0
        IS2_atl11_tide_attrs[ptx]['longitude']['valid_max'] = 180.0
        IS2_atl11_tide_attrs[ptx]['longitude']['coordinates'] = \
            "ref_pt delta_time latitude"

        # cycle statistics variables
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['Description'] = ("The cycle_stats subgroup "
            "contains summary information about segments for each reference point, including "
            "the uncorrected mean heights for reference surfaces, blowing snow and cloud "
            "indicators, and geolocation and height misfit statistics.")
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['data_rate'] = ("Data within this group "
            "are stored at the average segment rate.")
        # computed tide
        IS2_atl11_tide[ptx]['cycle_stats'][model.atl11] = tide['AT'].copy()
        IS2_atl11_fill[ptx]['cycle_stats'][model.atl11] = tide['AT'].fill_value
        IS2_atl11_dims[ptx]['cycle_stats'][model.atl11] = ['ref_pt','cycle_number']
        IS2_atl11_tide_attrs[ptx]['cycle_stats'][model.atl11] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx]['cycle_stats'][model.atl11]['units'] = "meters"
        IS2_atl11_tide_attrs[ptx]['cycle_stats'][model.atl11]['contentType'] = "referenceInformation"
        IS2_atl11_tide_attrs[ptx]['cycle_stats'][model.atl11]['long_name'] = model.long_name
        IS2_atl11_tide_attrs[ptx]['cycle_stats'][model.atl11]['description'] = model.description
        IS2_atl11_tide_attrs[ptx]['cycle_stats'][model.atl11]['source'] = model.name
        IS2_atl11_tide_attrs[ptx]['cycle_stats'][model.atl11]['reference'] = model.reference
        IS2_atl11_tide_attrs[ptx]['cycle_stats'][model.atl11]['coordinates'] = \
            "../ref_pt ../cycle_number ../delta_time ../latitude ../longitude"

        # if crossover measurements were calculated
        if CROSSOVERS:
            # crossing track variables
            IS2_atl11_tide_attrs[ptx][XT]['Description'] = ("The crossing_track_data "
                "subgroup contains elevation data at crossover locations. These are "
                "locations where two ICESat-2 pair tracks cross, so data are available "
                "from both the datum track, for which the granule was generated, and "
                "from the crossing track.")
            IS2_atl11_tide_attrs[ptx][XT]['data_rate'] = ("Data within this group are "
                "stored at the average segment rate.")

            # reference point
            IS2_atl11_tide[ptx][XT]['ref_pt'] = ref_pt['XT'].copy()
            IS2_atl11_fill[ptx][XT]['ref_pt'] = None
            IS2_atl11_dims[ptx][XT]['ref_pt'] = None
            IS2_atl11_tide_attrs[ptx][XT]['ref_pt'] = collections.OrderedDict()
            IS2_atl11_tide_attrs[ptx][XT]['ref_pt']['units'] = "1"
            IS2_atl11_tide_attrs[ptx][XT]['ref_pt']['contentType'] = "referenceInformation"
            IS2_atl11_tide_attrs[ptx][XT]['ref_pt']['long_name'] = ("fit center reference point number, "
                "segment_id")
            IS2_atl11_tide_attrs[ptx][XT]['ref_pt']['source'] = "derived, ATL11 algorithm"
            IS2_atl11_tide_attrs[ptx][XT]['ref_pt']['description'] = ("The reference-point number of the "
                "fit center for the datum track. The reference point is the 7 digit segment_id number "
                "corresponding to the center of the ATL06 data used for each ATL11 point.  These are "
                "sequential, starting with 1 for the first segment after an ascending equatorial "
                "crossing node.")
            IS2_atl11_tide_attrs[ptx][XT]['ref_pt']['coordinates'] = \
                "delta_time latitude longitude"

            # reference ground track of the crossing track
            IS2_atl11_tide[ptx][XT]['rgt'] = IS2_atl11_mds[ptx][XT]['rgt'].copy()
            IS2_atl11_fill[ptx][XT]['rgt'] = IS2_atl11_attrs[ptx][XT]['rgt']['_FillValue']
            IS2_atl11_dims[ptx][XT]['rgt'] = None
            IS2_atl11_tide_attrs[ptx][XT]['rgt'] = collections.OrderedDict()
            IS2_atl11_tide_attrs[ptx][XT]['rgt']['units'] = "1"
            IS2_atl11_tide_attrs[ptx][XT]['rgt']['contentType'] = "referenceInformation"
            IS2_atl11_tide_attrs[ptx][XT]['rgt']['long_name'] = "crossover reference ground track"
            IS2_atl11_tide_attrs[ptx][XT]['rgt']['source'] = "ATL06"
            IS2_atl11_tide_attrs[ptx][XT]['rgt']['description'] = "The RGT number for the crossing data."
            IS2_atl11_tide_attrs[ptx][XT]['rgt']['coordinates'] = \
                "ref_pt delta_time latitude longitude"
            # cycle_number of the crossing track
            IS2_atl11_tide[ptx][XT]['cycle_number'] = IS2_atl11_mds[ptx][XT]['cycle_number'].copy()
            IS2_atl11_fill[ptx][XT]['cycle_number'] = IS2_atl11_attrs[ptx][XT]['cycle_number']['_FillValue']
            IS2_atl11_dims[ptx][XT]['cycle_number'] = None
            IS2_atl11_tide_attrs[ptx][XT]['cycle_number'] = collections.OrderedDict()
            IS2_atl11_tide_attrs[ptx][XT]['cycle_number']['units'] = "1"
            IS2_atl11_tide_attrs[ptx][XT]['cycle_number']['long_name'] = "crossover cycle number"
            IS2_atl11_tide_attrs[ptx][XT]['cycle_number']['source'] = "ATL06"
            IS2_atl11_tide_attrs[ptx][XT]['cycle_number']['description'] = ("Cycle number for the "
                "crossing data. Number of 91-day periods that have elapsed since ICESat-2 entered "
                "the science orbit. Each of the 1,387 reference ground track (RGTs) is targeted "
                "in the polar regions once every 91 days.")
            # delta time of the crossing track
            IS2_atl11_tide[ptx][XT]['delta_time'] = delta_time['XT'].copy()
            IS2_atl11_fill[ptx][XT]['delta_time'] = delta_time['XT'].fill_value
            IS2_atl11_dims[ptx][XT]['delta_time'] = ['ref_pt']
            IS2_atl11_tide_attrs[ptx][XT]['delta_time'] = {}
            IS2_atl11_tide_attrs[ptx][XT]['delta_time']['units'] = "seconds since 2018-01-01"
            IS2_atl11_tide_attrs[ptx][XT]['delta_time']['long_name'] = "Elapsed GPS seconds"
            IS2_atl11_tide_attrs[ptx][XT]['delta_time']['standard_name'] = "time"
            IS2_atl11_tide_attrs[ptx][XT]['delta_time']['calendar'] = "standard"
            IS2_atl11_tide_attrs[ptx][XT]['delta_time']['source'] = "ATL06"
            IS2_atl11_tide_attrs[ptx][XT]['delta_time']['description'] = ("Number of GPS "
                "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
                "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
                "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
                "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
                "time in gps_seconds relative to the GPS epoch can be computed.")
            IS2_atl11_tide_attrs[ptx]['delta_time']['coordinates'] = \
                "ref_pt latitude longitude"
            # latitude of the crossover measurement
            IS2_atl11_tide[ptx][XT]['latitude'] = latitude['XT'].copy()
            IS2_atl11_fill[ptx][XT]['latitude'] = latitude['XT'].fill_value
            IS2_atl11_dims[ptx][XT]['latitude'] = ['ref_pt']
            IS2_atl11_tide_attrs[ptx][XT]['latitude'] = collections.OrderedDict()
            IS2_atl11_tide_attrs[ptx][XT]['latitude']['units'] = "degrees_north"
            IS2_atl11_tide_attrs[ptx][XT]['latitude']['contentType'] = "physicalMeasurement"
            IS2_atl11_tide_attrs[ptx][XT]['latitude']['long_name'] = "crossover latitude"
            IS2_atl11_tide_attrs[ptx][XT]['latitude']['standard_name'] = "latitude"
            IS2_atl11_tide_attrs[ptx][XT]['latitude']['source'] = "ATL06"
            IS2_atl11_tide_attrs[ptx][XT]['latitude']['description'] = ("Center latitude of "
                "selected segments")
            IS2_atl11_tide_attrs[ptx][XT]['latitude']['valid_min'] = -90.0
            IS2_atl11_tide_attrs[ptx][XT]['latitude']['valid_max'] = 90.0
            IS2_atl11_tide_attrs[ptx][XT]['latitude']['coordinates'] = \
                "ref_pt delta_time longitude"
            # longitude of the crossover measurement
            IS2_atl11_tide[ptx][XT]['longitude'] = longitude['XT'].copy()
            IS2_atl11_fill[ptx][XT]['longitude'] = longitude['XT'].fill_value
            IS2_atl11_dims[ptx][XT]['longitude'] = ['ref_pt']
            IS2_atl11_tide_attrs[ptx][XT]['longitude'] = collections.OrderedDict()
            IS2_atl11_tide_attrs[ptx][XT]['longitude']['units'] = "degrees_east"
            IS2_atl11_tide_attrs[ptx][XT]['longitude']['contentType'] = "physicalMeasurement"
            IS2_atl11_tide_attrs[ptx][XT]['longitude']['long_name'] = "crossover longitude"
            IS2_atl11_tide_attrs[ptx][XT]['longitude']['standard_name'] = "longitude"
            IS2_atl11_tide_attrs[ptx][XT]['longitude']['source'] = "ATL06"
            IS2_atl11_tide_attrs[ptx][XT]['longitude']['description'] = ("Center longitude of "
                "selected segments")
            IS2_atl11_tide_attrs[ptx][XT]['longitude']['valid_min'] = -180.0
            IS2_atl11_tide_attrs[ptx][XT]['longitude']['valid_max'] = 180.0
            IS2_atl11_tide_attrs[ptx][XT]['longitude']['coordinates'] = \
                "ref_pt delta_time latitude"
            # computed tide for the crossover measurement
            IS2_atl11_tide[ptx][XT][model.atl11] = tide['XT'].copy()
            IS2_atl11_fill[ptx][XT][model.atl11] = tide['XT'].fill_value
            IS2_atl11_dims[ptx][XT][model.atl11] = ['ref_pt']
            IS2_atl11_tide_attrs[ptx][XT][model.atl11] = collections.OrderedDict()
            IS2_atl11_tide_attrs[ptx][XT][model.atl11]['units'] = "meters"
            IS2_atl11_tide_attrs[ptx][XT][model.atl11]['contentType'] = "referenceInformation"
            IS2_atl11_tide_attrs[ptx][XT][model.atl11]['long_name'] = model.long_name
            IS2_atl11_tide_attrs[ptx][XT][model.atl11]['description'] = model.description
            IS2_atl11_tide_attrs[ptx][XT][model.atl11]['source'] = model.name
            IS2_atl11_tide_attrs[ptx][XT][model.atl11]['reference'] = model.reference
            IS2_atl11_tide_attrs[ptx][XT][model.atl11]['coordinates'] = \
                "ref_pt delta_time latitude longitude"

    # print file information
    logger.info(f'\t{str(OUTPUT_FILE)}')
    HDF5_ATL11_tide_write(IS2_atl11_tide, IS2_atl11_tide_attrs,
        FILENAME=OUTPUT_FILE,
        INPUT=GRANULE,
        CROSSOVERS=CROSSOVERS,
        FILL_VALUE=IS2_atl11_fill,
        DIMENSIONS=IS2_atl11_dims,
        CLOBBER=True)
    # change the permissions mode
    OUTPUT_FILE.chmod(mode=MODE)

# PURPOSE: outputting the tide values for ICESat-2 data to HDF5
def HDF5_ATL11_tide_write(IS2_atl11_tide, IS2_atl11_attrs, INPUT=None,
    FILENAME='', FILL_VALUE=None, DIMENSIONS=None, CROSSOVERS=False,
    CLOBBER=False):
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
    for k,v in IS2_atl11_tide['ancillary_data'].items():
        # Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS2_atl11_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    # write each output beam pair
    pairs = [k for k in IS2_atl11_tide.keys() if bool(re.match(r'pt\d',k))]
    for ptx in pairs:
        fileID.create_group(ptx)
        h5[ptx] = {}
        # add HDF5 group attributes for beam
        for att_name in ['description','beam_pair','ReferenceGroundTrack',
            'first_cycle','last_cycle','equatorial_radius','polar_radius']:
            fileID[ptx].attrs[att_name] = IS2_atl11_attrs[ptx][att_name]

        # ref_pt, cycle number, geolocation and delta_time variables
        for k in ['ref_pt','cycle_number','delta_time','latitude','longitude']:
            # values and attributes
            v = IS2_atl11_tide[ptx][k]
            attrs = IS2_atl11_attrs[ptx][k]
            fillvalue = FILL_VALUE[ptx][k]
            # Defining the HDF5 dataset variables
            val = '{0}/{1}'.format(ptx,k)
            if fillvalue:
                h5[ptx][k] = fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, fillvalue=fillvalue, compression='gzip')
            else:
                h5[ptx][k] = fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, compression='gzip')
            # create or attach dimensions for HDF5 variable
            if DIMENSIONS[ptx][k]:
                # attach dimensions
                for i,dim in enumerate(DIMENSIONS[ptx][k]):
                    h5[ptx][k].dims[i].attach_scale(h5[ptx][dim])
            else:
                # make dimension
                h5[ptx][k].make_scale(k)
            # add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5[ptx][k].attrs[att_name] = att_val

        # add to cycle_stats variables
        groups = ['cycle_stats']
        # if running crossovers: add to crossing_track_data variables
        if CROSSOVERS:
            groups.append('crossing_track_data')
        for key in groups:
            fileID[ptx].create_group(key)
            h5[ptx][key] = {}
            for att_name in ['Description','data_rate']:
                att_val=IS2_atl11_attrs[ptx][key][att_name]
                fileID[ptx][key].attrs[att_name] = att_val
            for k,v in IS2_atl11_tide[ptx][key].items():
                # attributes
                attrs = IS2_atl11_attrs[ptx][key][k]
                fillvalue = FILL_VALUE[ptx][key][k]
                # Defining the HDF5 dataset variables
                val = '{0}/{1}/{2}'.format(ptx,key,k)
                if fillvalue:
                    h5[ptx][key][k] = fileID.create_dataset(val, np.shape(v), data=v,
                        dtype=v.dtype, fillvalue=fillvalue, compression='gzip')
                else:
                    h5[ptx][key][k] = fileID.create_dataset(val, np.shape(v), data=v,
                        dtype=v.dtype, compression='gzip')
                # create or attach dimensions for HDF5 variable
                if DIMENSIONS[ptx][key][k]:
                    # attach dimensions
                    for i,dim in enumerate(DIMENSIONS[ptx][key][k]):
                        if (key == 'cycle_stats'):
                            h5[ptx][key][k].dims[i].attach_scale(h5[ptx][dim])
                        else:
                            h5[ptx][key][k].dims[i].attach_scale(h5[ptx][key][dim])
                else:
                    # make dimension
                    h5[ptx][key][k].make_scale(k)
                # add HDF5 variable attributes
                for att_name,att_val in attrs.items():
                    h5[ptx][key][k].attrs[att_name] = att_val

    # HDF5 file title
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'ATLAS/ICESat-2 Annual Land Ice Height'
    fileID.attrs['summary'] = ('The purpose of ATL11 is to provide an ICESat-2 '
        'satellite cycle summary of heights and height changes of land-based '
        'ice and will be provided as input to ATL15 and ATL16, gridded '
        'estimates of heights and height-changes.')
    fileID.attrs['description'] = ('Land ice parameters for each beam pair. '
        'All parameters are calculated for the same along-track increments '
        'for each beam pair and repeat.')
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
    # add attributes for input ATL11 files
    fileID.attrs['lineage'] = pathlib.Path(INPUT).name
    # find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for ptx in pairs:
        lon = IS2_atl11_tide[ptx]['longitude']
        lat = IS2_atl11_tide[ptx]['latitude']
        delta_time = IS2_atl11_tide[ptx]['delta_time']
        valid = np.nonzero(delta_time != FILL_VALUE[ptx]['delta_time'])
        # setting the geospatial and temporal ranges
        lnmn = lon.min() if (lon.min() < lnmn) else lnmn
        lnmx = lon.max() if (lon.max() > lnmx) else lnmx
        ltmn = lat.min() if (lat.min() < ltmn) else ltmn
        ltmx = lat.max() if (lat.max() > ltmx) else ltmx
        tmn = delta_time[valid].min() if (delta_time[valid].min() < tmn) else tmn
        tmx = delta_time[valid].max() if (delta_time[valid].max() > tmx) else tmx
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
        description="""Calculates tidal elevations for correcting ICESat-2 ATL11
            annual land ice height data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    group = parser.add_mutually_exclusive_group(required=True)
    # input ICESat-2 annual land ice height files
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='ICESat-2 ATL11 file to run')
    # directory with tide data
    parser.add_argument('--directory','-D',
        type=pathlib.Path,
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
        type=str, default='auto', choices=('ascii','json','auto'),
        help='Format for model definition file')
    # crop tide model to (buffered) bounds of data
    parser.add_argument('--crop',
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
        help='Infer values for minor constituents')
    # specify minor constituents to infer
    parser.add_argument('--minor-constituents',
        metavar='MINOR', type=str, nargs='+',
        help='Minor constituents to infer')
    # apply flexure scaling factors to height constituents
    parser.add_argument('--apply-flexure',
        default=False, action='store_true',
        help='Apply ice flexure scaling factor to height values')
    # run with ATL11 crossovers
    parser.add_argument('--crossovers','-C',
        default=False, action='store_true',
        help='Run ATL11 Crossovers')
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

    # run for each input ATL11 file
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
            MINOR_CONSTITUENTS=args.minor_constituents,
            APPLY_FLEXURE=args.apply_flexure,
            CROSSOVERS=args.crossovers,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
