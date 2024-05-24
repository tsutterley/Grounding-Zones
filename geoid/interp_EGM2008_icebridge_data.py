#!/usr/bin/env python
u"""
interp_EGM2008_icebridge_data.py
Written by Tyler Sutterley (05/2024)
Reads EGM2008 geoid height spatial grids from unformatted binary files
provided by the National Geospatial-Intelligence Agency and interpolates
to Operation IceBridge elevation data

NGA Office of Geomatics
    https://earth-info.nga.mil/

INPUTS:
    ATM1B, ATM icessn or LVIS file

COMMAND LINE OPTIONS:
    -O X, --output-directory X: input/output data directory
    -G X, --gravity X: 2.5x2.5 arcminute geoid height spatial grid
    -n X, --love X: Degree 2 load Love number (default EGM2008 value)
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
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/
        
PROGRAM DEPENDENCIES:
    io/icebridge.py: reads NASA Operation IceBridge data files
    ref_ellipsoid.py: Computes parameters for a reference ellipsoid

UPDATE HISTORY:
    Written 05/2024
"""
from __future__ import print_function

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
geoidtk = gz.utilities.import_dependency('geoid_toolkit')
h5py = gz.utilities.import_dependency('h5py')
timescale = gz.utilities.import_dependency('timescale')

# PURPOSE: read Operation IceBridge data
# and interpolates EGM2008 geoid undulation at points
def interp_EGM2008_icebridge_data(model_file, arg,
    LOVE=0.3,
    VERBOSE=False,
    MODE=0o775):

    # create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # extract file name and subsetter indices lists
    match_object = re.match(r'(.*?)(\[(.*?)\])?$',arg)
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

    # extract information from first input file
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
    logging.info(f'{input_file} -->')
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

    # set grid parameters
    dlon,dlat = (2.5/60.0), (2.5/60.0)
    latlimit_north, latlimit_south = (90.0, -90.0)
    longlimit_west, longlimit_east = (0.0, 360.0)
    # boundary parameters
    nlat = np.abs((latlimit_north - latlimit_south)/dlat).astype('i') + 1
    nlon = np.abs((longlimit_west - longlimit_east)/dlon).astype('i') + 1
    # grid coordinates (degrees)
    lon = longlimit_west + np.arange(nlon)*dlon
    lat = latlimit_south + np.arange(nlat)*dlat

    # check that EGM2008 data file is present in file system
    model_file = pathlib.Path(model_file).expanduser().absolute()
    if not model_file.exists():
        raise FileNotFoundError(f'{str(model_file)} not found')
    # open input file and read contents
    GRAVITY = np.fromfile(model_file, dtype='<f4').reshape(nlat,nlon+1)
    # Earth Gravitational Model 2008 parameters
    GM = 0.3986004415E+15
    R = 0.63781363E+07
    LMAX = 2190
    model = 'EGM2008'

  # HDF5 file attributes
    attrib = collections.OrderedDict()
    # J2000 time
    attrib['time'] = {}
    attrib['time']['long_name'] = 'Transmit time in J2000 seconds'
    attrib['time']['units'] = 'seconds since 2000-01-01 12:00:00 UTC'
    attrib['time']['description'] = ('The transmit time of each shot in '
        'the 1 second frame measured as UTC seconds elapsed since Jan 1 '
        '2000 12:00:00 UTC.')
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
    # geoid undulation
    attrib['geoid_h'] = {}
    attrib['geoid_h']['units'] = 'm'
    attrib['geoid_h']['long_name'] = 'Geoidal_Undulation'
    attrib['geoid_h']['description'] = ('Geoidal_undulation_with_'
        'respect_to_WGS84_ellipsoid')
    attrib['geoid_h']['source'] = 'EGM2008'
    attrib['geoid_h']['tide_system'] = 'tide_free'
    attrib['geoid_h']['earth_gravity_constant'] = GM
    attrib['geoid_h']['radius'] = R
    attrib['geoid_h']['degree_of_truncation'] = LMAX
    attrib['geoid_h']['coordinates'] = 'lat lon'
    # geoid conversion
    attrib['geoid_free2mean'] = {}
    attrib['geoid_free2mean']['units'] = 'm'
    attrib['geoid_free2mean']['long_name'] = 'Geoid_Free-to-Mean_conversion'
    attrib['geoid_free2mean']['description'] = ('Additive_value_to_convert_'
        'geoid_heights_from_the_tide-free_system_to_the_mean-tide_system')
    attrib['geoid_free2mean']['earth_gravity_constant'] = GM
    attrib['geoid_free2mean']['radius'] = R
    attrib['geoid_free2mean']['coordinates'] = 'lat lon'

    # geoid undulation (wrapped to 360 degrees)
    geoid_h = np.zeros((nlat, nlon), dtype=np.float32)
    geoid_h[:,:-1] = GRAVITY[::-1,1:-1]
    # repeat values for 360
    geoid_h[:,-1] = geoid_h[:,0]
    # create interpolator for geoid height
    SPL = scipy.interpolate.RectBivariateSpline(lon, lat, geoid_h.T,
        kx=1, ky=1)
    # interpolate geoid height to elevation coordinates
    dinput['geoid_h'] = SPL.ev(dinput['lon'], dinput['lat'])

    # colatitude in radians
    theta = (90.0 - dinput['lat'])*np.pi/180.0
    # calculate offset for converting from tide_free to mean_tide
    # legendre polynomial of degree 2 (unnormalized)
    P2 = 0.5*(3.0*np.cos(theta)**2 - 1.0)
    # from Rapp 1991 (Consideration of Permanent Tidal Deformation)
    dinput['geoid_free2mean'] = -0.198*P2*(1.0 + LOVE)

    # output geoid HDF5 file
    # form: rg_NASA_model_GEOID_WGS84_fl1yyyymmddjjjjj.H5
    # where rg is the hemisphere flag (GR or AN) for the region
    # model is the geoid model name flag
    # fl1 and fl2 are the data flags (ATM, LVIS, GLAS)
    # yymmddjjjjj is the year, month, day and second of the input file
    # output region flags: GR for Greenland and AN for Antarctica
    hem_flag = {'N':'GR','S':'AN'}
    # use starting second to distinguish between files for the day
    JJ1 = np.min(dinput['time']) % 86400
    # output file format
    args = (hem_flag[HEM],model,OIB,YY1,MM1,DD1,JJ1)
    FILENAME = '{0}_NASA_{1}_GEOID_WGS84_{2}{3}{4}{5}{6:05.0f}.H5'.format(*args)
    output_file = input_file.with_name(FILENAME)
    # print file information
    logging.info(f'\t{str(output_file)}')

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
    fid.attrs['title'] = 'Geoid_height_for_elevation_measurements'
    fid.attrs['summary'] = ('EGM2008_geoid_undulation_interpolated_to_'
        'elevation_measurements')
    fid.attrs['project'] = 'NASA_Operation_IceBridge'
    fid.attrs['processing_level'] = '4'
    fid.attrs['date_created'] = time.strftime('%Y-%m-%d',time.localtime())
    # add attributes for input file
    fid.attrs['lineage'] = input_file.name
    fid.attrs['gravity_model'] = model
    # add geospatial and temporal attributes
    fid.attrs['geospatial_lat_min'] = dinput['lat'].min()
    fid.attrs['geospatial_lat_max'] = dinput['lat'].max()
    fid.attrs['geospatial_lon_min'] = dinput['lon'].min()
    fid.attrs['geospatial_lon_max'] = dinput['lon'].max()
    fid.attrs['geospatial_lat_units'] = "degrees_north"
    fid.attrs['geospatial_lon_units'] = "degrees_east"
    fid.attrs['geospatial_ellipsoid'] = "WGS84"
    fid.attrs['time_type'] = 'UTC'
    # convert start and end time from J2000 seconds into timescale
    tmn, tmx = np.min(dinput['time']), np.max(dinput['time'])
    ts = timescale.time.Timescale().from_deltatime(np.array([tmn,tmx]),
        epoch=timescale.time._j2000_epoch, standard='UTC')
    duration = ts.day*(np.max(ts.MJD) - np.min(ts.MJD))
    dt = np.datetime_as_string(ts.to_datetime(), unit='s')
    # add attributes with measurement date start, end and duration
    fid.attrs['time_coverage_start'] = str(dt[0])
    fid.attrs['time_coverage_end'] = str(dt[1])
    fid.attrs['time_coverage_duration'] = f'{duration:0.0f}'
    # add software information
    fid.attrs['software_reference'] = gz.version.project_name
    fid.attrs['software_version'] = gz.version.full_version
    # close the output HDF5 dataset
    fid.close()
    # change the permissions level to MODE
    output_file.chmod(mode=MODE)

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Reads EGM2008 geoid height spatial grids and
            interpolates to Operation IceBridge elevation data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    # input operation icebridge files
    parser.add_argument('infile',
        type=str, nargs='+',
        help='Input Operation IceBridge file to run')
    # set gravity model file to use
    parser.add_argument('--gravity','-G',
        type=pathlib.Path, required=True,
        help='Gravity model file to use')
    # load love number of degree 2 (default EGM2008 value)
    parser.add_argument('--love','-n',
        type=float, default=0.3,
        help='Degree 2 load Love number')
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

    # run for each input GLAH12 file
    for FILE in args.infile:
        interp_EGM2008_icebridge_data(args.gravity, FILE,
            LOVE=args.love,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
