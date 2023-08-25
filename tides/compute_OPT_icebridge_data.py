#!/usr/bin/env python
u"""
compute_OPT_icebridge_data.py
Written by Tyler Sutterley (05/2023)
Calculates radial ocean pole tide displacements for correcting Operation
    IceBridge elevation data following IERS Convention (2010) guidelines
    http://maia.usno.navy.mil/conventions/2010officialinfo.php
    http://maia.usno.navy.mil/conventions/chapter7.php

INPUTS:
    ATM1B, ATM icessn or LVIS file from NSIDC

COMMAND LINE OPTIONS:
    -c X, --convention X: IERS mean or secular pole convention
        2003
        2010
        2015
        2018
    -I X, --interpolate X: Interpolation method
        spline
        linear
        nearest
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
    eop.py: utilities for calculating Earth Orientation Parameters (EOP)
    io/ocean_pole_tide.py: read ocean pole load tide map from IERS
    read_ATM1b_QFIT_binary.py: read ATM1b QFIT binary files (NSIDC version 1)

UPDATE HISTORY:
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
        use longcomplex data format to be windows compliant
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
        replaced griddata interpolation with scipy regular grid interpolators
    Updated 03/2020: use read_ATM1b_QFIT_binary from repository
    Updated 05/2019: added option interpolate to choose the interpolation method
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
import warnings
import collections
import numpy as np
import scipy.interpolate
import grounding_zones as gz

# attempt imports
try:
    import h5py
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("h5py not available", ImportWarning)
try:
    import pyTMD
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("pyTMD not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: read Operation IceBridge data from NSIDC
# compute ocean pole tide radial displacements at data points and times
def compute_OPT_icebridge_data(arg, CONVENTION='2018', METHOD=None,
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
    # ocean pole tides
    attrib['tide_oc_pole'] = {}
    attrib['tide_oc_pole']['long_name'] = 'Ocean_Pole_Tide'
    attrib['tide_oc_pole']['description'] = ('Ocean_pole_tide_radial_'
        'displacements_at_the_measurement_position_at_the_acquisition_time_due_'
        'to_polar_motion')
    attrib['tide_oc_pole']['reference'] = ('ftp://tai.bipm.org/iers/conv2010/'
        'chapter7/opoleloadcoefcmcor.txt.gz')
    attrib['tide_oc_pole']['units'] = 'meters'

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

    # extract lat/lon
    lon = dinput['lon'][:]
    lat = dinput['lat'][:]
    # create timescale from J2000: seconds since 2000-01-01 12:00:00 UTC
    timescale = pyTMD.time.timescale().from_deltatime(dinput['time'],
        epoch=pyTMD.time._j2000_epoch, standard='UTC')
    # convert dynamic time to Modified Julian Days (MJD)
    MJD = timescale.tt - 2400000.5
    # convert Julian days to calendar dates
    Y,M,D,h,m,s = pyTMD.time.convert_julian(timescale.tt, format='tuple')
    # calculate time in year-decimal format
    time_decimal = pyTMD.time.convert_calendar_decimal(Y,M,day=D,
        hour=h,minute=m,second=s)
    # elevation
    h1 = dinput['data'][:]

    # degrees to radians and arcseconds to radians
    dtr = np.pi/180.0
    atr = np.pi/648000.0
    # earth and physical parameters for ellipsoid
    units = pyTMD.constants('WGS84')
    # mean equatorial gravitational acceleration [m/s^2]
    ge = 9.7803278
    # density of sea water [kg/m^3]
    rho_w = 1025.0
    # tidal love number differential (1 + kl - hl) for pole tide frequencies
    gamma = 0.6870 + 0.0036j
    # bad value
    fill_value = -9999.0

    # convert from geodetic latitude to geocentric latitude
    # calculate X, Y and Z from geodetic latitude and longitude
    X,Y,Z = pyTMD.spatial.to_cartesian(lon, lat, h=h1,
        a_axis=units.a_axis, flat=units.flat)
    # calculate geocentric latitude and convert to degrees
    latitude_geocentric = np.arctan(Z / np.sqrt(X**2.0 + Y**2.0))/dtr

    # pole tide displacement scale factor
    Hp = np.sqrt(8.0*np.pi/15.0)*(units.omega**2*units.a_axis**4)/units.GM
    K = 4.0*np.pi*units.G*rho_w*Hp*units.a_axis/(3.0*ge)
    K1 = 4.0*np.pi*units.G*rho_w*Hp*units.a_axis**3/(3.0*units.GM)

    # output ocean pole tide HDF5 file
    # form: rg_NASA_OCEAN_POLE_TIDE_WGS84_fl1yyyymmddjjjjj.H5
    # where rg is the hemisphere flag (GR or AN) for the region
    # fl1 and fl2 are the data flags (ATM, LVIS, GLAS)
    # yymmddjjjjj is the year, month, day and second of the input file
    # output region flags: GR for Greenland and AN for Antarctica
    hem_flag = {'N':'GR','S':'AN'}
    # use starting second to distinguish between files for the day
    JJ1 = np.min(dinput['time']) % 86400
    # output file format
    args = (hem_flag[HEM],'OCEAN_POLE_TIDE',OIB,YY1,MM1,DD1,JJ1)
    FILENAME = '{0}_NASA_{1}_WGS84_{2}{3}{4}{5}{6:05.0f}.H5'.format(*args)
    # print file information
    output_file = input_file.with_name(FILENAME)
    logger.info(f'\t{str(output_file)}')

    # open output HDF5 file
    fid = h5py.File(output_file, mode='w')

    # read ocean pole tide map from Desai (2002)
    iur, iun, iue, ilon, ilat = pyTMD.io.ocean_pole_tide()
    # interpolate ocean pole tide map from Desai (2002)
    if (METHOD == 'spline'):
        # use scipy bivariate splines to interpolate to output points
        f1 = scipy.interpolate.RectBivariateSpline(ilon, ilat[::-1],
            iur[:,::-1].real, kx=1, ky=1)
        f2 = scipy.interpolate.RectBivariateSpline(ilon, ilat[::-1],
            iur[:,::-1].imag, kx=1, ky=1)
        UR = np.zeros((file_lines),dtype=np.longcomplex)
        UR.real = f1.ev(lon,latitude_geocentric)
        UR.imag = f2.ev(lon,latitude_geocentric)
    else:
        # use scipy regular grid to interpolate values for a given method
        r1 = scipy.interpolate.RegularGridInterpolator((ilon,ilat[::-1]),
            iur[:,::-1], method=METHOD)
        UR = r1.__call__(np.c_[lon,latitude_geocentric])

    # calculate angular coordinates of mean/secular pole at time
    mpx, mpy, fl = pyTMD.eop.iers_mean_pole(time_decimal, convention=CONVENTION)
    # read and interpolate IERS daily polar motion values
    px, py = pyTMD.eop.iers_polar_motion(MJD, k=3, s=0)
    # calculate differentials from mean/secular pole positions
    mx = px - mpx
    my = -(py - mpy)
    # calculate radial displacement at time
    Urad = np.ma.zeros((file_lines),fill_value=fill_value)
    Urad.data[:] = K*atr*np.real((mx*gamma.real + my*gamma.imag)*UR.real +
        (my*gamma.real - mx*gamma.imag)*UR.imag)
    # replace fill values
    Urad.mask = np.isnan(Urad.data)
    Urad.data[Urad.mask] = Urad.fill_value
    # copy radial displacement to output variable
    dinput['tide_oc_pole'] = Urad.copy()

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
    fid.attrs['summary'] = ('Ocean_pole_tide_radial_displacements_'
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

# PURPOSE: create a list of available EOP conventions
def get_available_conventions():
    """Create a list of available EOP conventions
    """
    try:
        return pyTMD.eop._conventions
    except (NameError, AttributeError):
        return ('2003', '2010', '2015', '2018')

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates radial ocean pole tide displacements for
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
    # interpolation method
    parser.add_argument('--interpolate','-I',
        metavar='METHOD', type=str, default='spline',
        choices=('spline','linear','nearest'),
        help='Spatial interpolation method')
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
        compute_OPT_icebridge_data(arg,
            CONVENTION=args.convention,
            METHOD=args.interpolate,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
