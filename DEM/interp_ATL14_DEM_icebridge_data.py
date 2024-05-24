#!/usr/bin/env python
u"""
interp_ATL14_DEM_icebridge_data.py
Written by Tyler Sutterley (05/2024)
Interpolates ATL14 elevations to locations of Operation IceBridge data

INPUTS:
    ATM1B, ATM icessn or LVIS file

COMMAND LINE OPTIONS:
    -m X, --model X: path to ATL14 model file
    -V, --verbose: Output information about each created file
    -M X, --mode X: Permission mode of directories and files created

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org
    netCDF4: Python interface to the netCDF C library
        https://unidata.github.io/netcdf4-python/netCDF4/index.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
    pyTMD: Python-based tidal prediction software
        https://pypi.org/project/pyTMD/
        https://pytmd.readthedocs.io/en/latest/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

PROGRAM DEPENDENCIES:
    io/icebridge.py: reads NASA Operation IceBridge data files
    utilities.py: download and management utilities for syncing files
    read_ATM1b_QFIT_binary.py: read ATM1b QFIT binary files (NSIDC version 1)

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
h5py = gz.utilities.import_dependency('h5py')
pyproj = gz.utilities.import_dependency('pyproj')
pyTMD = gz.utilities.import_dependency('pyTMD')
timescale = gz.utilities.import_dependency('timescale')

# PURPOSE: read ICESat ice sheet HDF5 elevation data (GLAH12)
# interpolate DEM data to x and y coordinates
def interp_ATL14_DEM_icebridge_data(arg,
    DEM_MODEL=None,
    VERBOSE=False,
    MODE=None):

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
    # ATL14 DEM elevation
    attrib['dem_h'] = {}
    attrib['dem_h']['units'] = "meters"
    attrib['dem_h']['long_name'] = "DEM Height"
    attrib['dem_h']['description'] = \
        "Surface height of the digital elevation model (DEM)"
    attrib['dem_h']['source'] = 'ATL14'
    attrib['dem_h']['coordinates'] = 'lat lon'
    # ATL14 DEM elevation uncertainty
    attrib['dem_h_sigma'] = {}
    attrib['dem_h_sigma']['units'] = "meters"
    attrib['dem_h_sigma']['long_name'] = "DEM Uncertainty"
    attrib['dem_h_sigma']['description'] = \
        "Uncertainty in the DEM surface height"
    attrib['dem_h_sigma']['source'] = 'ATL14'
    attrib['dem_h_sigma']['coordinates'] = 'lat lon'

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

    # pyproj transformer for converting from latitude/longitude
    EPSG = dict(N=3413, S=3031)
    SIGN = dict(N=1.0, S=-1.0)
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_epsg(EPSG[HEM])
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    # convert from latitude/longitude to polar stereographic
    X,Y = transformer.transform(dinput['lon'], dinput['lat'])

    # extract valid data
    valid, = np.nonzero(np.sign(dinput['lat']) == SIGN[HEM])
    # check if there are valid points for the hemisphere
    if not valid.any():
        return
    # get bounds of valid points within hemisphere
    xmin, xmax = np.min(X[valid]), np.max(X[valid])
    ymin, ymax = np.min(Y[valid]), np.max(Y[valid])
    # read ATL14 model for the hemisphere
    DEM = gz.io.ATL14(DEM_MODEL, BOUNDS=[xmin, xmax, ymin, ymax])
    # verify coordinate reference systems match
    assert pyproj.CRS.is_exact_same(crs2, DEM.crs), \
        'Inconsistent coordinate reference systems'
    # create 2D interpolation of DEM data
    R1 = scipy.interpolate.RegularGridInterpolator((DEM.y, DEM.x),
        DEM.h, bounds_error=False)
    R2 = scipy.interpolate.RegularGridInterpolator((DEM.y, DEM.x),
        DEM.h_sigma**2, bounds_error=False)
    R3 = scipy.interpolate.RegularGridInterpolator((DEM.y, DEM.x),
        DEM.ice_area, bounds_error=False)
    # clear DEM variable
    DEM = None

    # invalid value
    fill_value = -9999.0
    # output interpolated digital elevation model
    dinput['dem_h'] = np.ma.zeros((file_lines),
        fill_value=fill_value, dtype=np.float32)
    dinput['dem_h'].mask = np.ma.zeros((file_lines), dtype=bool)
    dinput['dem_h_sigma'] = np.ma.zeros((file_lines),
        fill_value=fill_value, dtype=np.float32)
    dinput['dem_h_sigma'].mask = np.ma.zeros((file_lines), dtype=bool)
    dem_ice_area = np.zeros((file_lines))

    # interpolate DEM to GLA12 locations
    dinput['dem_h'].data[valid] = R1.__call__(np.c_[Y[valid], X[valid]])
    dem_h_sigma2 = R2.__call__(np.c_[Y[valid], X[valid]])
    dinput['dem_h_sigma'].data[valid] = np.sqrt(dem_h_sigma2)
    dem_ice_area[valid] = R3.__call__(np.c_[Y[valid], X[valid]])

    # update masks and replace fill values
    for key in ['dem_h', 'dem_h_sigma']:
        dinput[key].mask[:] = (dem_ice_area <= 0.0) | \
            (np.abs(dinput['dem_h'].data) >= 1e4)
        dinput[key].data[dinput[key].mask] = dinput[key].fill_value

    # output DEM HDF5 file
    # form: rg_NASA_model_ATL14_DEM_WGS84_fl1yyyymmddjjjjj.H5
    # where rg is the hemisphere flag (GR or AN) for the region
    # fl1 and fl2 are the data flags (ATM, LVIS, GLAS)
    # yymmddjjjjj is the year, month, day and second of the input file
    # output region flags: GR for Greenland and AN for Antarctica
    hem_flag = {'N':'GR','S':'AN'}
    # use starting second to distinguish between files for the day
    JJ1 = np.min(dinput['time']) % 86400
    # output file format
    file_format = '{0}_NASA_ATL14_DEM_WGS84_{1}{2}{3}{4}{5:05.0f}.H5'
    FILENAME = file_format.format(hem_flag[HEM],OIB,YY1,MM1,DD1,JJ1)
    # print file information
    output_file = input_file.with_name(FILENAME)
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
    fid.attrs['title'] = 'NASA_ATL14_DEM_WGS84'
    fid.attrs['summary'] = ('NASA_ICESat-2_Gridded_Land_Ice_Height_'
        '(ATL14)_interpolated_to_elevation_measurements.')
    fid.attrs['project'] = 'NASA_Operation_IceBridge'
    fid.attrs['processing_level'] = '4'
    fid.attrs['date_created'] = time.strftime('%Y-%m-%d',time.localtime())
    # add attributes for input file
    lineage = [input_file, *DEM_MODEL]
    fid.attrs['lineage'] = [pathlib.Path(i).name for i in lineage]
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
    fid.attrs['time_coverage_end'] = str(dt[-1])
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
        description="""Interpolate DEMs to Operation IceBridge elevation data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    # input operation icebridge files
    parser.add_argument('infile',
        type=str, nargs='+',
        help='Input Operation IceBridge file to run')
    # full path to ATL14 digital elevation file
    parser.add_argument('--dem-model','-m',
        type=pathlib.Path, nargs='+', required=True,
        help='ICESat-2 ATL14 DEM file to run')
    # verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of run')
    # permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permissions mode of output files')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # run for each input file
    for arg in args.infile:
        # run program with parameters
        interp_ATL14_DEM_icebridge_data(arg,
            DEM_MODEL=args.dem_model,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
