#!/usr/bin/env python
u"""
compute_tides_icebridge_data.py
Written by Tyler Sutterley (08/2023)
Calculates tidal elevations for correcting Operation IceBridge elevation data

Uses OTIS format tidal solutions provided by Ohio State University and ESR
    http://volkov.oce.orst.edu/tides/region.html
    https://www.esr.org/research/polar-tide-models/list-of-polar-tide-models/
    ftp://ftp.esr.org/pub/datasets/tmd/
Global Tide Model (GOT) solutions provided by Richard Ray at GSFC
or Finite Element Solution (FES) models provided by AVISO

INPUTS:
    ATM1B, ATM icessn or LVIS file from NSIDC

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
    netCDF4: Python interface to the netCDF C library
         https://unidata.github.io/netcdf4-python/netCDF4/index.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/

PROGRAM DEPENDENCIES:
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
    read_ATM1b_QFIT_binary.py: read ATM1b QFIT binary files (NSIDC version 1)

UPDATE HISTORY:
    Updated 08/2023: changed ESR netCDF4 format to TMD3 format
    Updated 05/2023: use timescale class for time conversion operations
        using pathlib to define and operate on paths
        move icebridge data inputs to a separate module in io
    Updated 12/2022: single implicit import of grounding zone tools
        refactored pyTMD tide model structure
    Updated 07/2022: update imports of ATM1b QFIT functions to released version
        place some imports within try/except statements
    Updated 05/2022: added ESR netCDF4 formats to list of model types
        updated keyword arguments to read tide model programs
        added command line option to apply flexure for applicable models
    Updated 04/2022: include utf-8 encoding in reads to be windows compliant
        use argparse descriptions within sphinx documentation
    Updated 03/2022: using static decorators to define available models
    Updated 02/2022: added Arctic 2km model (Arc2kmTM) to list of models
    Updated 12/2021: added TPXO9-atlas-v5 to list of available tide models
    Updated 10/2021: using python logging for handling verbose output
        using collections to store attributes in order of creation
    Updated 09/2021: refactor to use model class for files and attributes
    Updated 07/2021: can use prefix files to define command line arguments
    Updated 06/2021: added new Gr1km-v2 1km Greenland model from ESR
    Updated 05/2021: added option for extrapolation cutoff in kilometers
        modified import of ATM1b QFIT reader
    Updated 03/2021: added TPXO9-atlas-v4 in binary OTIS format
        simplified netcdf inputs to be similar to binary OTIS read program
        replaced numpy bool/int to prevent deprecation warnings
    Updated 12/2020: added valid data extrapolation with nearest_extrap
         merged time conversion routines into module
    Updated 11/2020: added model constituents from TPXO9-atlas-v3
    Updated 10/2020: using argparse to set command line parameters
    Updated 09/2020: output ocean and load tide as tide_ocean and tide_load
    Updated 08/2020: using builtin time operations.  python3 regular expressions
    Updated 07/2020: added FES2014 and FES2014_load.  use merged delta times
    Updated 06/2020: added version 2 of TPXO9-atlas (TPXO9-atlas-v2)
    Updated 03/2020: use read_ATM1b_QFIT_binary from repository
    Updated 02/2020: changed CATS2008 grid to match version on U.S. Antarctic
        Program Data Center http://www.usap-dc.org/view/dataset/601235
    Updated 11/2019: added AOTIM-5-2018 tide model (2018 update to 2004 model)
    Updated 09/2019: added TPXO9_atlas reading from netcdf4 tide files
    Updated 05/2019: added option interpolate to choose the interpolation method
    Updated 02/2019: using range for python3 compatibility
    Updated 10/2018: updated GPS time calculation for calculating leap seconds
    Updated 07/2018: added GSFC Global Ocean Tides (GOT) models
    Written 06/2018
"""
from __future__ import print_function

import sys
import re
import time
import logging
import pathlib
import argparse
import warnings
import collections
import numpy as np
import grounding_zones as gz

# attempt imports
try:
    import h5py
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.warn("h5py not available", ImportWarning)
try:
    import pyTMD
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.warn("pyTMD not available", ImportWarning)

# PURPOSE: read Operation IceBridge data from NSIDC
# compute tides at points and times using tidal model driver algorithms
def compute_tides_icebridge_data(tide_dir, arg, TIDE_MODEL,
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
    logger = pyTMD.utilities.build_logger('pytmd', level=loglevel)

    # get parameters for tide model
    if DEFINITION_FILE is not None:
        model = pyTMD.io.model(tide_dir).from_file(DEFINITION_FILE)
    else:
        model = pyTMD.io.model(tide_dir, format=ATLAS_FORMAT,
            compressed=GZIP).elevation(TIDE_MODEL)

    # extract file name and subsetter indices lists
    match_object = re.match(r'(.*?)(\[(.*?)\])?$', arg)
    input_file = pathlib.Path(match_object.group(1)).expanduser().absolute()
    # subset input file to indices
    if match_object.group(2):
        # decompress ranges and add to list
        input_subsetter = []
        for i in re.findall(r'((\d+)-(\d+)|(\d+))',match_object.group(3)):
            input_subsetter.append(int(i[3])) if i[3] else \
                input_subsetter.extend(range(int(i[1]),int(i[2])+1))
    else:
        input_subsetter = None

    # calculate if input files are from ATM or LVIS (+GH)
    regex = {}
    regex['ATM'] = r'(BLATM2|ILATM2)_(\d+)_(\d+)_smooth_nadir(.*?)(csv|seg|pt)$'
    regex['ATM1b'] = r'(BLATM1b|ILATM1b)_(\d+)_(\d+)(.*?).(qi|TXT|h5)$'
    regex['LVIS'] = r'(BLVIS2|BVLIS2|ILVIS2)_(.*?)(\d+)_(\d+)_(R\d+)_(\d+).H5$'
    regex['LVGH'] = r'(ILVGH2)_(.*?)(\d+)_(\d+)_(R\d+)_(\d+).H5$'
    for key,val in regex.items():
        if re.match(val, input_file.name):
            OIB = key

    # HDF5 file attributes
    attrib = collections.OrderedDict()
    # time
    attrib['time'] = {}
    attrib['time']['long_name'] = 'Time'
    attrib['time']['description'] = ('Time_corresponding_to_the_measurement_'
        'position')
    attrib['time']['units'] = 'Days since 1992-01-01T00:00:00'
    attrib['time']['standard_name'] = 'time'
    attrib['time']['calendar'] = 'standard'
    # latitude
    attrib['lat'] = {}
    attrib['lat']['long_name'] = 'Latitude_of_measurement'
    attrib['lat']['description'] = ('Corresponding_to_the_measurement_'
        'position_at_the_acquisition_time')
    attrib['lat']['units'] = 'Degrees_North'
    # longitude
    attrib['lon'] = {}
    attrib['lon']['long_name'] = 'Longitude_of_measurement'
    attrib['lon']['description'] = ('Corresponding_to_the_measurement_'
        'position_at_the_acquisition_time')
    attrib['lon']['units'] = 'Degrees_East'
    # tides
    attrib[model.variable] = {}
    attrib[model.variable]['description'] = model.description
    attrib[model.variable]['reference'] = model.reference
    attrib[model.variable]['model'] = model.name
    attrib[model.variable]['units'] = 'meters'
    attrib[model.variable]['long_name'] = model.long_name

    # extract information from input file
    # acquisition year, month and day
    # number of points
    # instrument (PRE-OIB ATM or LVIS, OIB ATM or LVIS)
    if OIB in ('ATM','ATM1b'):
        M1,YYMMDD1,HHMMSS1,AX1,SF1 = re.findall(regex[OIB], input_file.name).pop()
        # early date strings omitted century and millenia (e.g. 93 for 1993)
        if (len(YYMMDD1) == 6):
            year_two_digit,MM1,DD1 = YYMMDD1[:2],YYMMDD1[2:4],YYMMDD1[4:]
            year_two_digit = float(year_two_digit)
            if (year_two_digit >= 90):
                YY1 = f'{1900.0+year_two_digit:4.0f}'
            else:
                YY1 = f'{2000.0+year_two_digit:4.0f}'
        elif (len(YYMMDD1) == 8):
            YY1,MM1,DD1 = YYMMDD1[:4],YYMMDD1[4:6],YYMMDD1[6:]
    elif OIB in ('LVIS','LVGH'):
        M1,RG1,YY1,MMDD1,RLD1,SS1 = re.findall(regex[OIB], input_file.name).pop()
        MM1,DD1 = MMDD1[:2],MMDD1[2:]

    # read data from input_file
    logger.info(f'{str(input_file)} -->')
    if (OIB == 'ATM'):
        # load IceBridge ATM data from input_file
        dinput, file_lines, HEM = gz.io.icebridge.read_ATM_icessn_file(
            input_file, input_subsetter)
    elif (OIB == 'ATM1b'):
        # load IceBridge Level-1b ATM data from input_file
        dinput, file_lines, HEM = gz.io.icebridge.read_ATM_qfit_file(
            input_file, input_subsetter)
    elif OIB in ('LVIS','LVGH'):
        # load IceBridge LVIS data from input_file
        dinput, file_lines, HEM = gz.io.icebridge.read_LVIS_HDF5_file(
            input_file, input_subsetter)

    # create timescale from J2000: seconds since 2000-01-01 12:00:00 UTC
    timescale = pyTMD.time.timescale().from_deltatime(dinput['time'],
        epoch=pyTMD.time._j2000_epoch, standard='UTC')

    # read tidal constants and interpolate to grid points
    if model.format in ('OTIS','ATLAS','TMD3'):
        amp,ph,D,c = pyTMD.io.OTIS.extract_constants(dinput['lon'], dinput['lat'],
            model.grid_file, model.model_file, model.projection,
            type=model.type, method=METHOD, extrapolate=EXTRAPOLATE,
            cutoff=CUTOFF, grid=model.format, apply_flexure=APPLY_FLEXURE)
        deltat = np.zeros((file_lines))
    elif model.format in ('netcdf'):
        amp,ph,D,c = pyTMD.io.ATLAS.extract_constants(dinput['lon'], dinput['lat'],
            model.grid_file, model.model_file, type=model.type, method=METHOD,
            extrapolate=EXTRAPOLATE, cutoff=CUTOFF, scale=model.scale,
            compressed=model.compressed)
        deltat = np.zeros((file_lines))
    elif (model.format == 'GOT'):
        amp,ph,c = pyTMD.io.GOT.extract_constants(dinput['lon'], dinput['lat'],
            model.model_file, method=METHOD, extrapolate=EXTRAPOLATE,
            cutoff=CUTOFF, scale=model.scale, compressed=model.compressed)
        # delta time (TT - UT1)
        deltat = timescale.tt_ut1
    elif (model.format == 'FES'):
        amp,ph = pyTMD.io.FES.extract_constants(dinput['lon'], dinput['lat'],
            model.model_file, type=model.type, version=model.version,
            method=METHOD, extrapolate=EXTRAPOLATE, cutoff=CUTOFF,
            scale=model.scale, compressed=model.compressed)
        # available model constituents
        c = model.constituents
        # delta time (TT - UT1)
        deltat = timescale.tt_ut1

    # calculate complex phase in radians for Euler's
    cph = -1j*ph*np.pi/180.0
    # calculate constituent oscillation
    hc = amp*np.exp(cph)

    # output tidal HDF5 file
    # form: rg_NASA_model_TIDES_WGS84_fl1yyyymmddjjjjj.H5
    # where rg is the hemisphere flag (GR or AN) for the region
    # model is the tidal model name flag (e.g. CATS0201)
    # fl1 and fl2 are the data flags (ATM, LVIS, GLAS)
    # yymmddjjjjj is the year, month, day and second of the input file
    # output region flags: GR for Greenland and AN for Antarctica
    hem_flag = {'N':'GR','S':'AN'}
    # use starting second to distinguish between files for the day
    JJ1 = np.min(dinput['time']) % 86400
    # flexure flag if being applied
    flexure_flag = '_FLEXURE' if APPLY_FLEXURE and model.flexure else ''
    # output file format
    args = (hem_flag[HEM],model.name,flexure_flag,OIB,YY1,MM1,DD1,JJ1)
    FILENAME = '{0}_NASA_{1}{2}_TIDES_WGS84_{3}{4}{5}{6}{7:05.0f}.H5'.format(*args)
    # print file information
    output_file = input_file.with_name(FILENAME)
    logger.info(f'\t{str(output_file)}')

    # open output HDF5 file
    fid = h5py.File(output_file, mode='w')

    # predict tidal elevations at time and infer minor corrections
    fill_value = -9999.0
    tide = np.ma.empty((file_lines),fill_value=fill_value)
    tide.mask = np.any(hc.mask,axis=1)
    tide.data[:] = pyTMD.predict.drift(timescale.tide, hc, c,
        deltat=deltat, corrections=model.format)
    minor = pyTMD.predict.infer_minor(timescale.tide, hc, c,
        deltat=deltat, corrections=model.format)
    tide.data[:] += minor.data[:]
    # replace invalid values with fill value
    tide.data[tide.mask] = tide.fill_value
    # copy tide to output variable
    dinput[model.variable] = tide.copy()

    # output dictionary with HDF5 variables
    h5 = {}
    # add variables to output file
    for key,attributes in attrib.items():
        # Defining the HDF5 dataset variables for lat/lon
        h5[key] = fid.create_dataset(key, (file_lines,),
            data=dinput[key][:], dtype=dinput[key].dtype,
            compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in attributes.items():
            h5[key].attrs[att_name] = att_val
        # attach dimensions
        if key not in ('time',):
            for i,dim in enumerate(['time']):
                h5[key].dims[i].label = 'RECORD_SIZE'
                h5[key].dims[i].attach_scale(h5[dim])

    # HDF5 file attributes
    fid.attrs['featureType'] = 'trajectory'
    fid.attrs['title'] = 'Tidal_correction_for_elevation_measurements'
    fid.attrs['summary'] = ('Tidal_correction_computed_at_elevation_'
        'measurements_using_a_tidal_model_driver.')
    fid.attrs['project'] = 'NASA_Operation_IceBridge'
    fid.attrs['processing_level'] = '4'
    fid.attrs['date_created'] = time.strftime('%Y-%m-%d',time.localtime())
    # add attributes for input file
    fid.attrs['lineage'] = input_file.name
    fid.attrs['tide_model'] = model.name
    # add geospatial and temporal attributes
    fid.attrs['geospatial_lat_min'] = dinput['lat'].min()
    fid.attrs['geospatial_lat_max'] = dinput['lat'].max()
    fid.attrs['geospatial_lon_min'] = dinput['lon'].min()
    fid.attrs['geospatial_lon_max'] = dinput['lon'].max()
    fid.attrs['geospatial_lat_units'] = "degrees_north"
    fid.attrs['geospatial_lon_units'] = "degrees_east"
    fid.attrs['geospatial_ellipsoid'] = "WGS84"
    fid.attrs['time_type'] = 'UTC'
    # add attributes with measurement date start, end and duration
    dt = np.datetime_as_string(timescale.to_datetime(), unit='s')
    duration = timescale.day*(np.max(timescale.MJD) - np.min(timescale.MJD))
    fid.attrs['time_coverage_start'] = str(dt[0])
    fid.attrs['time_coverage_end'] = dt[-1]
    fid.attrs['time_coverage_duration'] = f'{duration:0.0f}'
    # add software information
    fid.attrs['software_reference'] = pyTMD.version.project_name
    fid.attrs['software_version'] = pyTMD.version.full_version
    # close the output HDF5 dataset
    fid.close()
    # change the permissions level to MODE
    output_file.chmod(mode=MODE)

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
        description="""Calculates tidal elevations for correcting Operation
            IceBridge elevation data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    group = parser.add_mutually_exclusive_group(required=True)
    # input operation icebridge files
    parser.add_argument('infile',
        type=str, nargs='+',
        help='Input Operation IceBridge file to run')
    # directory with tide data
    parser.add_argument('--directory','-D',
        type=pathlib.Path,
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

    # run for each input Operation IceBridge file
    for arg in args.infile:
        compute_tides_icebridge_data(args.directory, arg,
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
