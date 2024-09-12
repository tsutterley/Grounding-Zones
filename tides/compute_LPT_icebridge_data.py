#!/usr/bin/env python
u"""
compute_LPT_icebridge_data.py
Written by Tyler Sutterley (08/2024)
Calculates load pole tide displacements for correcting Operation IceBridge
    elevation data following IERS Convention (2010) guidelines
    http://maia.usno.navy.mil/conventions/2010officialinfo.php
    http://maia.usno.navy.mil/conventions/chapter7.php

INPUTS:
    ATM1B, ATM icessn or LVIS file

COMMAND LINE OPTIONS:
    -c X, --convention X: IERS mean or secular pole convention
        2003
        2010
        2015
        2018
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
    pyTMD: Python-based tidal prediction software
        https://pypi.org/project/pyTMD/
        https://pytmd.readthedocs.io/en/latest/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

PROGRAM DEPENDENCIES:
    io/icebridge.py: reads NASA Operation IceBridge data files
    spatial.py: utilities for reading, writing and operating on spatial data
    utilities.py: download and management utilities for syncing files
    eop.py: utilities for calculating Earth Orientation Parameters (EOP)
    read_ATM1b_QFIT_binary.py: read ATM1b QFIT binary files (NSIDC version 1)

UPDATE HISTORY:
    Updated 08/2024: use prediction function for cartesian tidal displacements
        use rotation matrix to convert from cartesian to spherical
    Updated 05/2024: use wrapper to importlib for optional dependencies
    Updated 04/2024: use timescale for temporal operations
    Updated 05/2023: use timescale class for time conversion operations
        use defaults from eop module for pole tide and EOP files
        using pathlib to define and operate on paths
        move icebridge data inputs to a separate module in io
    Updated 03/2023: added option for changing the IERS mean pole convention
    Updated 12/2022: single implicit import of grounding zone tools
        use constants class from pyTMD for ellipsoidal parameters
        refactored pyTMD tide model structure
    Updated 07/2022: update imports of ATM1b QFIT functions to released version
        place some imports within try/except statements
    Updated 04/2022: include utf-8 encoding in reads to be windows compliant
        use argparse descriptions within sphinx documentation
    Updated 10/2021: using python logging for handling verbose output
        using collections to store attributes in order of creation
    Updated 07/2021: can use prefix files to define command line arguments
    Updated 05/2021: modified import of ATM1b QFIT reader
    Updated 03/2021: use cartesian coordinate conversion routine in spatial
        replaced numpy bool/int to prevent deprecation warnings
    Updated 12/2020: merged time conversion routines into module
    Updated 11/2020: use internal mean pole and finals EOP files
    Updated 10/2020: using argparse to set command line parameters
    Updated 09/2020: output modified julian days as time variable
    Updated 08/2020: using builtin time operations.  python3 regular expressions
    Updated 03/2020: use read_ATM1b_QFIT_binary from repository
    Updated 02/2019: using range for python3 compatibility
    Updated 10/2018: updated GPS time calculation for calculating leap seconds
    Written 06/2018
"""
from __future__ import print_function

import sys
import re
import time
import logging
import pathlib
import argparse
import collections
import numpy as np
import scipy.interpolate
import grounding_zones as gz

# attempt imports
h5py = gz.utilities.import_dependency('h5py')
pyTMD = gz.utilities.import_dependency('pyTMD')
timescale = gz.utilities.import_dependency('timescale')

# PURPOSE: read Operation IceBridge data
# compute load pole tide radial displacements at data points and times
def compute_LPT_icebridge_data(arg,
        CONVENTION='2018',
        VERBOSE=False,
        MODE=0o775
    ):

    # create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logger = pyTMD.utilities.build_logger('pytmd',level=loglevel)

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
    # Modified Julian Days
    attrib['time'] = {}
    attrib['time']['long_name'] = 'Time'
    attrib['time']['units'] = 'days since 1858-11-17T00:00:00'
    attrib['time']['description'] = 'Modified Julian Days'
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
    # load pole tides
    attrib['tide_pole'] = {}
    attrib['tide_pole']['long_name'] = 'Solid_Earth_Pole_Tide'
    attrib['tide_pole']['description'] = ('Solid_Earth_pole_tide_radial_'
        'displacements_at_the_measurement_position_at_the_acquisition_'
        'time_due_to_polar_motion')
    attrib['tide_pole']['reference'] = ('ftp://tai.bipm.org/iers/conv2010/'
        'chapter7/tn36_c7.pdf')
    attrib['tide_pole']['units'] = 'meters'

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

    # extract lat/lon
    lon = dinput['lon'][:]
    lat = dinput['lat'][:]
    # create timescale from J2000: seconds since 2000-01-01 12:00:00 UTC
    ts = timescale.time.Timescale().from_deltatime(dinput['time'],
        epoch=timescale.time._j2000_epoch, standard='UTC')

    # degrees to radians
    dtr = np.pi/180.0
    # earth and physical parameters for ellipsoid
    wgs84 = pyTMD.datum(ellipsoid='WGS84', units='MKS')
    # tidal love/shida numbers appropriate for the load tide
    hb2 = 0.6207
    lb2 = 0.0847
    # bad value
    fill_value = -9999.0

    # convert from geodetic latitude to geocentric latitude
    # calculate X, Y and Z from geodetic latitude and longitude
    X,Y,Z = pyTMD.spatial.to_cartesian(lon, lat,
        a_axis=wgs84.a_axis, flat=wgs84.flat)
    rr = np.sqrt(X**2.0 + Y**2.0 + Z**2.0)
    # geocentric latitude (radians)
    latitude_geocentric = np.arctan(Z / np.sqrt(X**2.0 + Y**2.0))
    # geocentric colatitude (radians)
    theta = (np.pi/2.0 - latitude_geocentric)
    # calculate longitude (radians)
    phi = np.arctan2(Y, X)

    # compute normal gravity at spatial location
    # p. 80, Eqn.(2-199)
    gamma_0 = wgs84.gamma_0(theta)

    # rotation matrix for converting from cartesian coordinates
    R = np.zeros((file_lines, 3, 3))
    R[:,0,0] = np.cos(phi)*np.cos(theta)
    R[:,1,0] = -np.sin(phi)
    R[:,2,0] = np.cos(phi)*np.sin(theta)
    R[:,0,1] = np.sin(phi)*np.cos(theta)
    R[:,1,1] = np.cos(phi)
    R[:,2,1] = np.sin(phi)*np.sin(theta)
    R[:,0,2] = -np.sin(theta)
    R[:,2,2] = np.cos(theta)

    # calculate load pole tides in cartesian coordinates
    XYZ = np.c_[X, Y, Z]
    dxi = pyTMD.predict.load_pole_tide(ts.tide, XYZ,
        deltat=ts.tt_ut1,
        gamma_0=gamma_0,
        omega=wgs84.omega,
        h2=hb2,
        l2=lb2,
        convention=CONVENTION
    )
    # calculate components of load pole tides
    S = np.einsum('ti...,tji...->tj...', dxi, R)

    # output load pole tide HDF5 file
    # form: rg_NASA_LOAD_POLE_TIDE_WGS84_fl1yyyymmddjjjjj.H5
    # where rg is the hemisphere flag (GR or AN) for the region
    # fl1 and fl2 are the data flags (ATM, LVIS, GLAS)
    # yymmddjjjjj is the year, month, day and second of the input file
    # output region flags: GR for Greenland and AN for Antarctica
    hem_flag = {'N':'GR','S':'AN'}
    # use starting second to distinguish between files for the day
    JJ1 = np.min(dinput['time']) % 86400
    # output file format
    args = (hem_flag[HEM],'LOAD_POLE_TIDE',OIB,YY1,MM1,DD1,JJ1)
    FILENAME = '{0}_NASA_{1}_WGS84_{2}{3}{4}{5}{6:05.0f}.H5'.format(*args)
    # print file information
    output_file = input_file.with_name(FILENAME)
    logger.info(f'\t{str(output_file)}')

    # open output HDF5 file
    fid = h5py.File(output_file, mode='w')

    # convert to masked array
    Srad = np.ma.zeros((file_lines),fill_value=fill_value)
    Srad.data[:] = S[:,2].copy()
    # replace fill values
    Srad.mask = np.isnan(Srad.data)
    Srad.data[Srad.mask] = Srad.fill_value
    # copy radial displacement to output dictionary
    dinput['tide_pole'] = Srad.copy()

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
    fid.attrs['title'] = 'Load_Pole_Tide_correction'
    fid.attrs['summary'] = ('Solid_Earth_pole_tide_radial_displacements_'
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
    dt = np.datetime_as_string(ts.to_datetime(), unit='s')
    duration = ts.day*(np.max(ts.MJD) - np.min(ts.MJD))
    fid.attrs['time_coverage_start'] = str(dt[0])
    fid.attrs['time_coverage_end'] = str(dt[-1])
    fid.attrs['time_coverage_duration'] = f'{duration:0.0f}'
    # add software information
    fid.attrs['software_reference'] = pyTMD.version.project_name
    fid.attrs['software_version'] = pyTMD.version.full_version
    # close the output HDF5 dataset
    fid.close()
    # change the permissions level to MODE
    output_file.chmod(mode=MODE)

# PURPOSE: create a list of available EOP conventions
def get_available_conventions():
    """Create a list of available EOP conventions
    """
    try:
        return timescale.eop._conventions
    except (NameError, AttributeError):
        return ('2003', '2010', '2015', '2018')

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates radial load pole tide displacements for
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
    # Earth orientation parameters
    parser.add_argument('--convention','-c',
        type=str, choices=get_available_conventions(), default='2018',
        help='IERS mean or secular pole convention')
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
        compute_LPT_icebridge_data(arg,
            CONVENTION=args.convention,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
