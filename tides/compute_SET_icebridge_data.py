#!/usr/bin/env python
u"""
compute_SET_icebridge_data.py
Written by Tyler Sutterley (01/2024)
Calculates radial solid Earth tide displacements for correcting Operation
    IceBridge elevation data following IERS Convention (2010) guidelines
    http://maia.usno.navy.mil/conventions/2010officialinfo.php
    http://maia.usno.navy.mil/conventions/chapter7.php

INPUTS:
    ATM1B, ATM icessn or LVIS file from NSIDC

COMMAND LINE OPTIONS:
    -p X, --tide-system X: Permanent tide system for output values
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

PROGRAM DEPENDENCIES:
    time.py: utilities for calculating time operations
    spatial.py: utilities for reading, writing and operating on spatial data
    utilities.py: download and management utilities for syncing files
    predict.py: calculates solid Earth tides
    read_ATM1b_QFIT_binary.py: read ATM1b QFIT binary files (NSIDC version 1)

UPDATE HISTORY:
    Updated 01/2024: refactored lunisolar ephemerides functions
    Updated 05/2023: use timescale class for time conversion operations
        add option for using higher resolution ephemerides from JPL
        using pathlib to define and operate on paths
        move icebridge data inputs to a separate module in io
    Updated 04/2023: added permanent tide system offset (free-to-mean)
    Written 03/2023
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
# compute solid earth tide radial displacements at data points and times
def compute_SET_icebridge_data(arg, TIDE_SYSTEM=None, EPHEMERIDES=None,
    VERBOSE=False, MODE=0o775):

    # create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logger = pyTMD.utilities.build_logger('pytmd', level=loglevel)

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

    # invalid value
    fill_value = -9999.0
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
    # solid earth tides
    attrib['tide_earth'] = {}
    attrib['tide_earth']['long_name'] = 'Solid_Earth_Tide'
    attrib['tide_earth']['description'] = ('Solid_earth_tides_in_the_'
        f'{TIDE_SYSTEM}_system')
    attrib['tide_earth']['reference'] = 'https://doi.org/10.1029/97JB01515'
    attrib['tide_earth']['units'] = 'meters'
    attrib['tide_earth']['_FillValue'] = fill_value
    # solid earth permanent tide offset
    attrib['tide_earth_free2mean'] = {}
    attrib['tide_earth_free2mean']['long_name'] = \
        'Solid_Earth_Tide_Free-to-Mean_conversion'
    attrib['tide_earth_free2mean']['description'] = ('Additive_value_to_convert_'
        'solid_earth_tide_from_the_tide_free_system_to_the_mean_tide_system')
    attrib['tide_earth_free2mean']['reference'] = 'https://doi.org/10.1029/97JB01515'
    attrib['tide_earth_free2mean']['units'] = 'meters'
    attrib['tide_earth_free2mean']['_FillValue'] = fill_value

    # extract information from input file
    # acquisition year, month and day
    # number of points
    # instrument (PRE-OIB ATM or LVIS, OIB ATM or LVIS)
    if OIB in ('ATM','ATM1b'):
        M1,YYMMDD1,HHMMSS1,AX1,SF1 = re.findall(regex[OIB], input_file.name).pop()
        # early date strings omitted century and millennia (e.g. 93 for 1993)
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

    # earth and physical parameters for WGS84 ellipsoid
    units = pyTMD.datum('WGS84')
    # create timescale from J2000: seconds since 2000-01-01 12:00:00 UTC
    timescale = pyTMD.time.timescale().from_deltatime(dinput['time'],
        epoch=pyTMD.time._j2000_epoch, standard='UTC')
    # convert tide times to dynamical time
    tide_time = timescale.tide + timescale.tt_ut1

    # convert input coordinates to cartesian
    X, Y, Z = pyTMD.spatial.to_cartesian(dinput['lon'], dinput['lat'],
        h=dinput['data'], a_axis=units.a_axis, flat=units.flat)
    # compute ephemerides for lunisolar coordinates
    SX, SY, SZ = pyTMD.astro.solar_ecef(timescale.MJD, ephemerides=EPHEMERIDES)
    LX, LY, LZ = pyTMD.astro.lunar_ecef(timescale.MJD, ephemerides=EPHEMERIDES)
    # convert coordinates to column arrays
    XYZ = np.c_[X, Y, Z]
    SXYZ = np.c_[SX, SY, SZ]
    LXYZ = np.c_[LX, LY, LZ]
    # predict solid earth tides (cartesian)
    dxi = pyTMD.predict.solid_earth_tide(tide_time,
        XYZ, SXYZ, LXYZ, a_axis=units.a_axis,
        tide_system=TIDE_SYSTEM)
    # calculate radial component of solid earth tides
    dln, dlt, drad = pyTMD.spatial.to_geodetic(
        X + dxi[:,0], Y + dxi[:,1], Z + dxi[:,2],
        a_axis=units.a_axis, flat=units.flat)
    # remove effects of original topography
    dinput['tide_earth'] = drad - dinput['data']
    # calculate permanent tide offset (meters)
    dinput['tide_earth_free2mean'] = 0.06029 - \
        0.180873*np.sin(dinput['lat']*np.pi/180.0)**2

    # output solid earth tide HDF5 file
    # form: rg_NASA_SOLID_EARTH_TIDE_WGS84_fl1yyyymmddjjjjj.H5
    # where rg is the hemisphere flag (GR or AN) for the region
    # fl1 and fl2 are the data flags (ATM, LVIS, GLAS)
    # yymmddjjjjj is the year, month, day and second of the input file
    # output region flags: GR for Greenland and AN for Antarctica
    hem_flag = {'N':'GR','S':'AN'}
    # use starting second to distinguish between files for the day
    JJ1 = np.min(dinput['time']) % 86400
    # output file format
    args = (hem_flag[HEM],'SOLID_EARTH_TIDE',OIB,YY1,MM1,DD1,JJ1)
    FILENAME = '{0}_NASA_{1}_WGS84_{2}{3}{4}{5}{6:05.0f}.H5'.format(*args)
    # print file information
    output_file = input_file.with_name(FILENAME)
    logger.info(f'\t{str(output_file)}')

    # open output HDF5 file
    fid = h5py.File(output_file, mode='w')

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
    fid.attrs['summary'] = ('Solid_Earth_tide_radial_displacements_'
        'computed_at_elevation_measurements.')
    fid.attrs['project'] = 'NASA_Operation_IceBridge'
    fid.attrs['processing_level'] = '4'
    fid.attrs['date_created'] = time.strftime('%Y-%m-%d',time.localtime())
    # add attributes for input file
    fid.attrs['lineage'] = input_file.name
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

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates radial solid earth tide displacements for
            correcting Operation IceBridge elevation data following IERS
            Convention (2010) guidelines
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line options
    parser.add_argument('infile',
        type=str, nargs='+',
        help='Input Operation IceBridge file to run')
    # permanent tide system for output values
    parser.add_argument('--tide-system','-p',
        type=str, choices=('tide_free','mean_tide'), default='tide_free',
        help='Permanent tide system for output values')
    # method for calculating lunisolar ephemerides
    parser.add_argument('--ephemerides','-c',
        type=str, choices=('approximate','JPL'), default='approximate',
        help='Method for calculating lunisolar ephemerides')
    # verbosity settings
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Output information about each created file')
    # permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of output file')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # run for each input file
    for arg in args.infile:
        compute_SET_icebridge_data(arg,
            TIDE_SYSTEM=args.tide_system,
            EPHEMERIDES=args.ephemerides,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
