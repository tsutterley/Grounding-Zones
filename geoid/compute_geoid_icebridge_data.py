#!/usr/bin/env python
u"""
compute_geoid_icebridge_data.py
Written by Tyler Sutterley (12/2022)
Calculates geoid undulations for correcting Operation IceBridge elevation data

INPUTS:
    ATM1B, ATM icessn or LVIS file from NSIDC

COMMAND LINE OPTIONS:
    -G X, --gravity X: Gravity model file to use (.gfc format)
    -l X, --lmax X: maximum spherical harmonic degree (level of truncation)
    -n X, --love X: Degree 2 load Love number (default EGM2008 value)
    -M X, --mode X: Permission mode of directories and files created
    -V, --verbose: Output information about each created file

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/

PROGRAM DEPENDENCIES:
    utilities.py: download and management utilities for syncing files
    geoid_undulation.py: geoidal undulation at a given latitude and longitude
    read_ICGEM_harmonics.py: reads the coefficients for a given gravity model file
    real_potential.py: real potential at a latitude and height for gravity model
    norm_potential.py: normal potential of an ellipsoid at a latitude and height
    norm_gravity.py: normal gravity of an ellipsoid at a latitude and height
    ref_ellipsoid.py: Computes parameters for a reference ellipsoid
    gauss_weights.py: Computes Gaussian weights as a function of degree
    read_ATM1b_QFIT_binary.py: read ATM1b QFIT binary files (NSIDC version 1)

UPDATE HISTORY:
    Updated 12/2022: single implicit import of grounding zone tools
    Updated 07/2022: update imports of ATM1b QFIT functions to released version
        place some imports within try/except statements
    Updated 05/2022: use argparse descriptions within documentation
    Updated 10/2021: using python logging for handling verbose output
        using collections to store attributes in order of creation
        additionally output conversion between tide free and mean tide values
    Updated 07/2021: can use prefix files to define command line arguments
    Updated 05/2021: modified import of ATM1b QFIT reader
    Updated 03/2021: replaced numpy bool/int to prevent deprecation warnings
    Updated 12/2020: merged time conversion routines into module
    Updated 10/2020: using argparse to set command line parameters
    Updated 08/2020: using builtin time operations.  python3 regular expressions
    Updated 03/2020: use read_ATM1b_QFIT_binary from repository
    Updated 02/2019: using range for python3 compatibility
    Updated 10/2018: updated GPS time calculation for calculating leap seconds
    Written 07/2017
"""
from __future__ import print_function

import sys
import os
import re
import time
import logging
import argparse
import warnings
import collections
import numpy as np
import grounding_zones as gz

# attempt imports
try:
    import ATM1b_QFIT.read_ATM1b_QFIT_binary
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("ATM1b_QFIT not available", ImportWarning)
try:
    import h5py
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("h5py not available", ImportWarning)
try:
    import geoid_toolkit as geoidtk
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("geoid_toolkit not available", ImportWarning)
try:
    import icesat2_toolkit as is2tk
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("icesat2_toolkit not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: reading the number of file lines removing commented lines
def file_length(input_file, input_subsetter, HDF5=False, QFIT=False):
    # subset the data to indices if specified
    if input_subsetter:
        file_lines = len(input_subsetter)
    elif HDF5:
        # read the size of an input variable within a HDF5 file
        with h5py.File(input_file,'r') as fileID:
            file_lines, = fileID[HDF5].shape
    elif QFIT:
        # read the size of a QFIT binary file
        file_lines = ATM1b_QFIT.ATM1b_QFIT_shape(input_file)
    else:
        # read the input file, split at lines and remove all commented lines
        with open(input_file, mode='r', encoding='utf8') as f:
            i = [i for i in f.readlines() if re.match(r'^(?!\#|\n)',i)]
        file_lines = len(i)
    # return the number of lines
    return file_lines

## PURPOSE: read the ATM Level-1b data file for variables of interest
def read_ATM_qfit_file(input_file, input_subsetter):
    # regular expression pattern for extracting parameters
    mission_flag = r'(BLATM1B|ILATM1B|ILNSA1B)'
    regex_pattern = rf'{mission_flag}_(\d+)_(\d+)(.*?).(qi|TXT|h5)'
    # extract mission and other parameters from filename
    MISSION,YYMMDD,HHMMSS,AUX,SFX = re.findall(regex_pattern,input_file).pop()
    # early date strings omitted century and millenia (e.g. 93 for 1993)
    if (len(YYMMDD) == 6):
        yr2d,month,day = np.array([YYMMDD[:2],YYMMDD[2:4],YYMMDD[4:]],dtype='i')
        year = (yr2d + 1900.0) if (yr2d >= 90) else (yr2d + 2000.0)
    elif (len(YYMMDD) == 8):
        year,month,day = np.array([YYMMDD[:4],YYMMDD[4:6],YYMMDD[6:]],dtype='i')
    # output python dictionary with variables
    ATM_L1b_input = {}
    # Version 1 of ATM QFIT files (ascii)
    # output text file from qi2txt with proper filename format
    # do not use the shortened output format from qi2txt
    if (SFX == 'TXT'):
        # compile regular expression operator for reading lines
        regex_pattern = r'[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?'
        rx = re.compile(regex_pattern, re.VERBOSE)
        # read the input file, split at lines and remove all commented lines
        with open(input_file, mode='r', encoding='utf8') as f:
            file_contents = [i for i in f.read().splitlines() if
                re.match(r'^(?!\#|\n)',i)]
        # number of lines of data within file
        file_lines = file_length(input_file,input_subsetter)
        # create output variables with length equal to the number of lines
        ATM_L1b_input['lat'] = np.zeros_like(file_contents,dtype=np.float64)
        ATM_L1b_input['lon'] = np.zeros_like(file_contents,dtype=np.float64)
        ATM_L1b_input['data'] = np.zeros_like(file_contents,dtype=np.float64)
        hour = np.zeros_like(file_contents,dtype=np.float64)
        minute = np.zeros_like(file_contents,dtype=np.float64)
        second = np.zeros_like(file_contents,dtype=np.float64)
        # for each line within the file
        for i,line in enumerate(file_contents):
            # find numerical instances within the line
            line_contents = rx.findall(line)
            ATM_L1b_input['lat'][i] = np.float64(line_contents[1])
            ATM_L1b_input['lon'][i] = np.float64(line_contents[2])
            ATM_L1b_input['data'][i] = np.float64(line_contents[3])
            hour[i] = np.float64(line_contents[-1][:2])
            minute[i] = np.float64(line_contents[-1][2:4])
            second[i] = np.float64(line_contents[-1][4:])
    # Version 1 of ATM QFIT files (binary)
    elif (SFX == 'qi'):
        # read input QFIT data file and subset if specified
        fid,h = ATM1b_QFIT.read_ATM1b_QFIT_binary(input_file)
        # number of lines of data within file
        file_lines = file_length(input_file,input_subsetter,QFIT=True)
        ATM_L1b_input['lat'] = fid['latitude'][:]
        ATM_L1b_input['lon'] = fid['longitude'][:]
        ATM_L1b_input['data'] = fid['elevation'][:]
        time_hhmmss = fid['time_hhmmss'][:]
        # extract hour, minute and second from time_hhmmss
        hour = np.zeros_like(time_hhmmss,dtype=np.float64)
        minute = np.zeros_like(time_hhmmss,dtype=np.float64)
        second = np.zeros_like(time_hhmmss,dtype=np.float64)
        # for each line within the file
        for i,packed_time in enumerate(time_hhmmss):
            # convert to zero-padded string with 3 decimal points
            line_contents = f'{packed_time:010.3f}'
            hour[i] = np.float64(line_contents[:2])
            minute[i] = np.float64(line_contents[2:4])
            second[i] = np.float64(line_contents[4:])
    # Version 2 of ATM QFIT files (HDF5)
    elif (SFX == 'h5'):
        # Open the HDF5 file for reading
        fileID = h5py.File(os.path.expanduser(input_file), 'r')
        # number of lines of data within file
        file_lines = file_length(input_file,input_subsetter,HDF5='elevation')
        # create output variables with length equal to input elevation
        ATM_L1b_input['lat'] = fileID['latitude'][:]
        ATM_L1b_input['lon'] = fileID['longitude'][:]
        ATM_L1b_input['data'] = fileID['elevation'][:]
        time_hhmmss = fileID['instrument_parameters']['time_hhmmss'][:]
        # extract hour, minute and second from time_hhmmss
        hour = np.zeros_like(time_hhmmss,dtype=np.float64)
        minute = np.zeros_like(time_hhmmss,dtype=np.float64)
        second = np.zeros_like(time_hhmmss,dtype=np.float64)
        # for each line within the file
        for i,packed_time in enumerate(time_hhmmss):
            # convert to zero-padded string with 3 decimal points
            line_contents = f'{packed_time:010.3f}'
            hour[i] = np.float64(line_contents[:2])
            minute[i] = np.float64(line_contents[2:4])
            second[i] = np.float64(line_contents[4:])
        # close the input HDF5 file
        fileID.close()
    # calculate the number of leap seconds between GPS time (seconds
    # since Jan 6, 1980 00:00:00) and UTC
    gps_seconds = is2tk.time.convert_calendar_dates(
        year,month,day,hour=hour,minute=minute,second=second,
        epoch=(1980,1,6,0,0,0),scale=86400.0)
    leap_seconds = is2tk.time.count_leap_seconds(gps_seconds)
    # calculation of Julian day taking into account leap seconds
    # converting to J2000 seconds
    ATM_L1b_input['time'] = is2tk.time.convert_calendar_dates(
        year,month,day,hour=hour,minute=minute,second=second-leap_seconds,
        epoch=(2000,1,1,12,0,0,0),scale=86400.0)
    # subset the data to indices if specified
    if input_subsetter:
        for key,val in ATM_L1b_input.items():
            ATM_L1b_input[key] = val[input_subsetter]
    # hemispheric shot count
    count = {}
    count['N'] = np.count_nonzero(ATM_L1b_input['lat'] >= 0.0)
    count['S'] = np.count_nonzero(ATM_L1b_input['lat'] < 0.0)
    # determine hemisphere with containing shots in file
    HEM, = [key for key, val in count.items() if val]
    # return the output variables
    return ATM_L1b_input,file_lines,HEM

# PURPOSE: read the ATM Level-2 data file for variables of interest
def read_ATM_icessn_file(input_file, input_subsetter):
    # regular expression pattern for extracting parameters
    mission_flag = r'(BLATM2|ILATM2)'
    regex_pattern = rf'{mission_flag}_(\d+)_(\d+)_smooth_nadir(.*?)(csv|seg|pt)$'
    # extract mission and other parameters from filename
    MISSION,YYMMDD,HHMMSS,AUX,SFX = re.findall(regex_pattern,input_file).pop()
    # early date strings omitted century and millenia (e.g. 93 for 1993)
    if (len(YYMMDD) == 6):
        yr2d,month,day = np.array([YYMMDD[:2],YYMMDD[2:4],YYMMDD[4:]],dtype='i')
        year = (yr2d + 1900.0) if (yr2d >= 90) else (yr2d + 2000.0)
    elif (len(YYMMDD) == 8):
        year,month,day = np.array([YYMMDD[:4],YYMMDD[4:6],YYMMDD[6:]],dtype='i')
    # input file column names for variables of interest with column indices
    # variables not used: (SNslope:4, WEslope:5, npt_used:7, npt_edit:8, d:9)
    file_dtype = {'seconds':0, 'lat':1, 'lon':2, 'data':3, 'RMS':6, 'track':-1}
    # compile regular expression operator for reading lines (extracts numbers)
    regex_pattern = r'[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?'
    rx = re.compile(regex_pattern, re.VERBOSE)
    # read the input file, split at lines and remove all commented lines
    with open(input_file, mode='r', encoding='utf8') as f:
        file_contents = [i for i in f.read().splitlines() if
            re.match(r'^(?!\#|\n)',i)]
    # number of lines of data within file
    file_lines = file_length(input_file,input_subsetter)
    # output python dictionary with variables
    ATM_L2_input = {}
    # create output variables with length equal to the number of file lines
    for key in file_dtype.keys():
        ATM_L2_input[key] = np.zeros_like(file_contents, dtype=np.float64)
    # for each line within the file
    for line_number,line_entries in enumerate(file_contents):
        # find numerical instances within the line
        line_contents = rx.findall(line_entries)
        # for each variable of interest: save to dinput as float
        for key,val in file_dtype.items():
            ATM_L2_input[key][line_number] = np.float64(line_contents[val])
    # convert shot time (seconds of day) to J2000
    hour = np.floor(ATM_L2_input['seconds']/3600.0)
    minute = np.floor((ATM_L2_input['seconds'] % 3600)/60.0)
    second = ATM_L2_input['seconds'] % 60.0
    # First column in Pre-IceBridge and ICESSN Version 1 files is GPS time
    if (MISSION == 'BLATM2') or (SFX != 'csv'):
        # calculate the number of leap seconds between GPS time (seconds
        # since Jan 6, 1980 00:00:00) and UTC
        gps_seconds = is2tk.time.convert_calendar_dates(
            year,month,day,hour=hour,minute=minute,second=second,
            epoch=(1980,1,6,0,0,0),scale=86400.0)
        leap_seconds = is2tk.time.count_leap_seconds(gps_seconds)
    else:
        leap_seconds = 0.0
    # calculation of Julian day
    # converting to J2000 seconds
    ATM_L2_input['time'] = is2tk.time.convert_calendar_dates(
        year,month,day,hour=hour,minute=minute,second=second-leap_seconds,
        epoch=(2000,1,1,12,0,0,0),scale=86400.0)
    # convert RMS from centimeters to meters
    ATM_L2_input['error'] = ATM_L2_input['RMS']/100.0
    # subset the data to indices if specified
    if input_subsetter:
        for key,val in ATM_L2_input.items():
            ATM_L2_input[key] = val[input_subsetter]
    # hemispheric shot count
    count = {}
    count['N'] = np.count_nonzero(ATM_L2_input['lat'] >= 0.0)
    count['S'] = np.count_nonzero(ATM_L2_input['lat'] < 0.0)
    # determine hemisphere with containing shots in file
    HEM, = [key for key, val in count.items() if val]
    # return the output variables
    return ATM_L2_input,file_lines,HEM

# PURPOSE: read the LVIS Level-2 data file for variables of interest
def read_LVIS_HDF5_file(input_file, input_subsetter):
    # LVIS region flags: GL for Greenland and AQ for Antarctica
    lvis_flag = {'GL':'N','AQ':'S'}
    # regular expression pattern for extracting parameters from HDF5 files
    # computed in read_icebridge_lvis.py
    mission_flag = r'(BLVIS2|BVLIS2|ILVIS2|ILVGH2)'
    regex_pattern = rf'{mission_flag}_(.*?)(\d+)_(\d+)_(R\d+)_(\d+).H5'
    # extract mission, region and other parameters from filename
    MISSION,REGION,YY,MMDD,RLD,SS = re.findall(regex_pattern,input_file).pop()
    LDS_VERSION = '2.0.2' if (int(RLD[1:3]) >= 18) else '1.04'
    # input and output python dictionaries with variables
    file_input = {}
    LVIS_L2_input = {}
    fileID = h5py.File(input_file,'r')
    # create output variables with length equal to input shot number
    file_lines = file_length(input_file,input_subsetter,HDF5='Shot_Number')
    # https://lvis.gsfc.nasa.gov/Data/Data_Structure/DataStructure_LDS104.html
    # https://lvis.gsfc.nasa.gov/Data/Data_Structure/DataStructure_LDS202.html
    if (LDS_VERSION == '1.04'):
        # elevation surfaces
        file_input['elev'] = fileID['Elevation_Surfaces/Elevation_Centroid'][:]
        file_input['elev_low'] = fileID['Elevation_Surfaces/Elevation_Low'][:]
        file_input['elev_high'] = fileID['Elevation_Surfaces/Elevation_High'][:]
        # latitude
        file_input['lat'] = fileID['Geolocation/Latitude_Centroid'][:]
        file_input['lat_low'] = fileID['Geolocation/Latitude_Low'][:]
        # longitude
        file_input['lon'] = fileID['Geolocation/Longitude_Centroid'][:]
        file_input['lon_low'] = fileID['Geolocation/Longitude_Low'][:]
    elif (LDS_VERSION == '2.0.2'):
        # elevation surfaces
        file_input['elev_low'] = fileID['Elevation_Surfaces/Elevation_Low'][:]
        file_input['elev_high'] = fileID['Elevation_Surfaces/Elevation_High'][:]
        # heights above lowest detected mode
        file_input['RH50'] = fileID['Waveform/RH50'][:]
        file_input['RH100'] = fileID['Waveform/RH100'][:]
        # calculate centroidal elevation using 50% of waveform energy
        file_input['elev'] = file_input['elev_low'] + file_input['RH50']
        # latitude
        file_input['lat_top'] = fileID['Geolocation/Latitude_Top'][:]
        file_input['lat_low'] = fileID['Geolocation/Latitude_Low'][:]
        # longitude
        file_input['lon_top'] = fileID['Geolocation/Longitude_Top'][:]
        file_input['lon_low'] = fileID['Geolocation/Longitude_Low'][:]
        # linearly interpolate latitude and longitude to RH50
        file_input['lat'] = file_input['lat_low'] + file_input['RH50'] * \
            (file_input['lat_top'] - file_input['lat_low'])/file_input['RH100']
        file_input['lon'] = file_input['lon_low'] + file_input['RH50'] * \
            (file_input['lon_top'] - file_input['lon_low'])/file_input['RH100']
    # J2000 seconds
    LVIS_L2_input['time'] = fileID['Time/J2000'][:]
    # close the input HDF5 file
    fileID.close()
    # output combined variables
    LVIS_L2_input['data'] = np.zeros_like(file_input['elev'],dtype=np.float64)
    LVIS_L2_input['lon'] = np.zeros_like(file_input['elev'],dtype=np.float64)
    LVIS_L2_input['lat'] = np.zeros_like(file_input['elev'],dtype=np.float64)
    LVIS_L2_input['error'] = np.zeros_like(file_input['elev'],dtype=np.float64)
    # find where elev high is equal to elev low
    # see note about using LVIS centroid elevation product
    # http://lvis.gsfc.nasa.gov/OIBDataStructure.html
    ii = np.nonzero(file_input['elev_low'] == file_input['elev_high'])
    jj = np.nonzero(file_input['elev_low'] != file_input['elev_high'])
    # where lowest point of waveform is equal to highest point -->
    # using the elev_low elevation
    LVIS_L2_input['data'][ii] = file_input['elev_low'][ii]
    # for other locations use the centroid elevation
    # as the centroid is a useful product over rough terrain
    # when you are calculating ice volume change
    LVIS_L2_input['data'][jj] = file_input['elev'][jj]
    # latitude and longitude for each case
    # elevation low == elevation high
    LVIS_L2_input['lon'][ii] = file_input['lon_low'][ii]
    LVIS_L2_input['lat'][ii] = file_input['lat_low'][ii]
    # centroid elevations
    LVIS_L2_input['lon'][jj] = file_input['lon'][jj]
    LVIS_L2_input['lat'][jj] = file_input['lat'][jj]
    # estimated uncertainty for both cases
    LVIS_variance_low = (file_input['elev_low'] - file_input['elev'])**2
    LVIS_variance_high = (file_input['elev_high'] - file_input['elev'])**2
    LVIS_L2_input['error']=np.sqrt((LVIS_variance_low + LVIS_variance_high)/2.0)
    # subset the data to indices if specified
    if input_subsetter:
        for key,val in LVIS_L2_input.items():
            LVIS_L2_input[key] = val[input_subsetter]
    # return the output variables
    return LVIS_L2_input,file_lines,lvis_flag[REGION]

# PURPOSE: read Operation IceBridge data from NSIDC
# and computes geoid undulation at points
def compute_geoid_icebridge_data(model_file, arg, LMAX=None, LOVE=None,
    VERBOSE=False, MODE=0o775):

    # create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # extract file name and subsetter indices lists
    match_object = re.match(r'(.*?)(\[(.*?)\])?$',arg)
    input_file = os.path.expanduser(match_object.group(1))
    # subset input file to indices
    if match_object.group(2):
        # decompress ranges and add to list
        input_subsetter = []
        for i in re.findall(r'((\d+)-(\d+)|(\d+))',match_object.group(3)):
            input_subsetter.append(int(i[3])) if i[3] else \
                input_subsetter.extend(range(int(i[1]),int(i[2])+1))
    else:
        input_subsetter = None

    # read gravity model Ylms and change tide to tide free
    Ylms = geoidtk.read_ICGEM_harmonics(model_file, LMAX=LMAX, TIDE='tide_free')
    R = np.float64(Ylms['radius'])
    GM = np.float64(Ylms['earth_gravity_constant'])
    LMAX = np.int64(Ylms['max_degree'])
    # reference to WGS84 ellipsoid
    REFERENCE = 'WGS84'

    # output directory for input_file
    DIRECTORY = os.path.dirname(input_file)
    # calculate if input files are from ATM or LVIS (+GH)
    regex = {}
    regex['ATM'] = r'(BLATM2|ILATM2)_(\d+)_(\d+)_smooth_nadir(.*?)(csv|seg|pt)$'
    regex['ATM1b'] = r'(BLATM1b|ILATM1b)_(\d+)_(\d+)(.*?).(qi|TXT|h5)$'
    regex['LVIS'] = r'(BLVIS2|BVLIS2|ILVIS2)_(.*?)(\d+)_(\d+)_(R\d+)_(\d+).H5$'
    regex['LVGH'] = r'(ILVGH2)_(.*?)(\d+)_(\d+)_(R\d+)_(\d+).H5$'
    for key,val in regex.items():
        if re.match(val, os.path.basename(input_file)):
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
    # geoid undulation
    attrib['geoid_h'] = {}
    attrib['geoid_h']['units'] = 'm'
    attrib['geoid_h']['long_name'] = 'Geoidal_Undulation'
    args = (Ylms['modelname'], Ylms['max_degree'])
    attrib['geoid_h']['description'] = ('{0}_geoidal_undulation_'
        'computed_from_degree_{1}_gravity_model.').format(*args)
    attrib['geoid_h']['tide_system'] = Ylms['tide_system']
    attrib['geoid_h']['earth_gravity_constant'] = GM
    attrib['geoid_h']['radius'] = R
    attrib['geoid_h']['degree_of_truncation'] = LMAX
    attrib['geoid_h']['coordinates'] = 'lat lon'
    # geoid conversion
    attrib['geoid_free2mean'] = {}
    attrib['geoid_free2mean']['units'] = 'm'
    attrib['geoid_free2mean']['long_name'] = 'Geoid_Free-to-Mean_conversion'
    args = (Ylms['modelname'],Ylms['max_degree'])
    attrib['geoid_free2mean']['description'] = ('Additive value to convert '
        'geoid heights from the tide-free system to the mean-tide system')
    attrib['geoid_free2mean']['tide_system'] = Ylms['tide_system']
    attrib['geoid_free2mean']['earth_gravity_constant'] = GM
    attrib['geoid_free2mean']['radius'] = R
    attrib['geoid_free2mean']['coordinates'] = 'lat lon'

    # extract information from first input file
    # acquisition year, month and day
    # number of points
    # instrument (PRE-OIB ATM or LVIS, OIB ATM or LVIS)
    if OIB in ('ATM','ATM1b'):
        M1,YYMMDD1,HHMMSS1,AX1,SF1 = re.findall(regex[OIB], input_file).pop()
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
        M1,RG1,YY1,MMDD1,RLD1,SS1 = re.findall(regex[OIB], input_file).pop()
        MM1,DD1 = MMDD1[:2],MMDD1[2:]

    # read data from input_file
    logging.info(f'{input_file} -->')
    if (OIB == 'ATM'):
        # load IceBridge ATM data from input_file
        dinput,file_lines,HEM = read_ATM_icessn_file(input_file,input_subsetter)
    elif (OIB == 'ATM1b'):
        # load IceBridge Level-1b ATM data from input_file
        dinput,file_lines,HEM = read_ATM_qfit_file(input_file,input_subsetter)
    elif OIB in ('LVIS','LVGH'):
        # load IceBridge LVIS data from input_file
        dinput,file_lines,HEM = read_LVIS_HDF5_file(input_file,input_subsetter)

    # output tidal HDF5 file
    # form: rg_NASA_model_GEOID_WGS84_fl1yyyymmddjjjjj.H5
    # where rg is the hemisphere flag (GR or AN) for the region
    # model is the tidal model name flag (e.g. CATS0201)
    # fl1 and fl2 are the data flags (ATM, LVIS, GLAS)
    # yymmddjjjjj is the year, month, day and second of the input file
    # output region flags: GR for Greenland and AN for Antarctica
    hem_flag = {'N':'GR','S':'AN'}
    # use starting second to distinguish between files for the day
    JJ1 = np.min(dinput['time']) % 86400
    # output file format
    args = (hem_flag[HEM],Ylms['modelname'],OIB,YY1,MM1,DD1,JJ1)
    FILENAME = '{0}_NASA_{1}_GEOID_WGS84_{2}{3}{4}{5}{6:05.0f}.H5'.format(*args)
    # print file information
    logging.info(f'\t{os.path.join(DIRECTORY,FILENAME)}')

    # open output HDF5 file
    fid = h5py.File(os.path.join(DIRECTORY,FILENAME), 'w')

    # colatitude in radians
    theta = (90.0 - dinput['lat'])*np.pi/180.0
    # calculate geoid at coordinates
    dinput['geoid_h'] = geoidtk.geoid_undulation(dinput['lat'], dinput['lon'],
        REFERENCE, Ylms['clm'], Ylms['slm'], LMAX, R, GM).astype(np.float64)
    # calculate offset for converting from tide_free to mean_tide
    # legendre polynomial of degree 2 (unnormalized)
    P2 = 0.5*(3.0*np.cos(theta)**2 - 1.0)
    # from Rapp 1991 (Consideration of Permanent Tidal Deformation)
    dinput['geoid_free2mean'] = -0.198*P2*(1.0 + LOVE)

    # output dictionary with HDF5 variables
    h5 = {}
    # add variables to output file
    for key,attributes in attrib.items():
        # Defining the HDF5 dataset variables for lat/lon
        h5[key] = fid.create_dataset(key, (file_lines,),
            data=dinput[key][:], dtype=dinput[key].dtype,
            compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in attributes:
            h5.attrs[att_name] = att_val
        # attach dimensions
        if key not in ('time',):
            for i,dim in enumerate(['time']):
                h5[key].dims[i].label = 'RECORD_SIZE'
                h5[key].dims[i].attach_scale(h5[dim])

    # HDF5 file attributes
    fid.attrs['featureType'] = 'trajectory'
    fid.attrs['title'] = 'Geoid_height_for_elevation_measurements'
    fid.attrs['summary'] = ('Geoid_undulation_computed_at_elevation_'
        'measurements_using_a_tidal_model_driver.')
    fid.attrs['project'] = 'NASA_Operation_IceBridge'
    fid.attrs['processing_level'] = '4'
    fid.attrs['date_created'] = time.strftime('%Y-%m-%d',time.localtime())
    # add attributes for input file
    fid.attrs['elevation_file'] = os.path.basename(input_file)
    fid.attrs['gravity_model'] = Ylms['modelname']
    # add geospatial and temporal attributes
    fid.attrs['geospatial_lat_min'] = dinput['lat'].min()
    fid.attrs['geospatial_lat_max'] = dinput['lat'].max()
    fid.attrs['geospatial_lon_min'] = dinput['lon'].min()
    fid.attrs['geospatial_lon_max'] = dinput['lon'].max()
    fid.attrs['geospatial_lat_units'] = "degrees_north"
    fid.attrs['geospatial_lon_units'] = "degrees_east"
    fid.attrs['geospatial_ellipsoid'] = "WGS84"
    fid.attrs['time_type'] = 'UTC'
    # convert start/end time from J2000 dates into Julian days
    # J2000: seconds since 2000-01-01 12:00:00 UTC
    time_range = np.array([np.min(dinput['time']),np.max(dinput['time'])])
    time_julian = 2400000.5 + is2tk.time.convert_delta_time(time_range,
        epoch1=(2000,1,1,12,0,0), epoch2=(1858,11,17,0,0,0), scale=1.0/86400.0)
    # convert to calendar date
    cal = is2tk.time.convert_julian(time_julian,ASTYPE=int)
    # add attributes with measurement date start, end and duration
    args = (cal['hour'][0],cal['minute'][0],cal['second'][0])
    fid.attrs['RangeBeginningTime'] = '{0:02d}:{1:02d}:{2:02d}'.format(*args)
    args = (cal['hour'][-1],cal['minute'][-1],cal['second'][-1])
    fid.attrs['RangeEndingTime'] = '{0:02d}:{1:02d}:{2:02d}'.format(*args)
    args = (cal['year'][0],cal['month'][0],cal['day'][0])
    fid.attrs['RangeBeginningDate'] = '{0:4d}-{1:02d}-{2:02d}'.format(*args)
    args = (cal['year'][-1],cal['month'][-1],cal['day'][-1])
    fid.attrs['RangeEndingDate'] = '{0:4d}-{1:02d}-{2:02d}'.format(*args)
    duration = np.round(time_julian[-1]*86400.0 - time_julian[0]*86400.0)
    fid.attrs['DurationTimeSeconds'] = f'{duration:0.0f}'
    # add software information
    fid.attrs['software_reference'] = gz.version.project_name
    fid.attrs['software_version'] = gz.version.full_version
    # close the output HDF5 dataset
    fid.close()
    # change the permissions level to MODE
    os.chmod(os.path.join(DIRECTORY,FILENAME), MODE)

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates geoid undulations for correcting Operation
            IceBridge elevation data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    # input operation icebridge files
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='Input Operation IceBridge file')
    # set gravity model file to use
    parser.add_argument('--gravity','-G',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Gravity model file to use')
    # maximum spherical harmonic degree (level of truncation)
    parser.add_argument('--lmax','-l',
        type=int, help='Maximum spherical harmonic degree')
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

    # run for each input GLA12 file
    for arg in args.infile:
        compute_geoid_icebridge_data(args.gravity, arg, LMAX=args.lmax,
            LOVE=args.love, VERBOSE=args.verbose, MODE=args.mode)


# run main program
if __name__ == '__main__':
    main()
