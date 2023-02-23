#!/usr/bin/env python
u"""
tile_icebridge_data.py
Written by Tyler Sutterley (12/2022)
Creates tile index files of Operation IceBridge elevation data

INPUTS:
    ATM1B, ATM icessn or LVIS file from NSIDC

COMMAND LINE OPTIONS:
    --help: list the command line options
    -S X, --spacing X: Output grid spacing
    -V, --verbose: Verbose output of run
    -M X, --mode X: Permissions mode of the directories and files

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://www.numpy.org
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/
    netCDF4: Python interface to the netCDF C library
        https://unidata.github.io/netcdf4-python/
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/

PROGRAM DEPENDENCIES:
    read_ATM1b_QFIT_binary.py: read ATM1b QFIT binary files (NSIDC version 1)

UPDATE HISTORY:
    Updated 12/2022: check that file exists within multiprocess HDF5 function
        single implicit import of grounding zone tools
    Updated 07/2022: update imports of ATM1b QFIT functions to released version
        place some imports within try/except statements
    Updated 06/2022: add checks if variables and groups already exist
    Updated 05/2022: use argparse descriptions within documentation
    Updated 11/2021: adjust tiling to index by center coordinates
        wait if merged HDF5 tile file is unavailable
    Written 10/2021
"""
import sys
import os
import re
import time
import pyproj
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
    import icesat2_toolkit as is2tk
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("icesat2_toolkit not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: attempt to open an HDF5 file and wait if already open
def multiprocess_h5py(filename, *args, **kwargs):
    # check that file exists if entering with read mode
    if kwargs['mode'] in ('r','r+') and not os.access(filename, os.F_OK):
        raise FileNotFoundError(filename)
    # attempt to open HDF5 file
    while True:
        try:
            fileID = h5py.File(filename, *args, **kwargs)
            break
        except (IOError, BlockingIOError, PermissionError) as exc:
            time.sleep(1)
    # return the file access object
    return fileID

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

# PURPOSE: create tile index files of Operation IceBridge data
def tile_icebridge_data(arg,
    SPACING=None,
    VERBOSE=False,
    MODE=0o775):

    # create logger
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

    # calculate if input files are from ATM or LVIS (+GH)
    regex = {}
    regex['ATM'] = r'(BLATM2|ILATM2)_(\d+)_(\d+)_smooth_nadir(.*?)(csv|seg|pt)$'
    regex['ATM1b'] = r'(BLATM1b|ILATM1b)_(\d+)_(\d+)(.*?).(qi|TXT|h5)$'
    regex['LVIS'] = r'(BLVIS2|BVLIS2|ILVIS2)_(.*?)(\d+)_(\d+)_(R\d+)_(\d+).H5$'
    regex['LVGH'] = r'(ILVGH2)_(.*?)(\d+)_(\d+)_(R\d+)_(\d+).H5$'
    for key,val in regex.items():
        if re.match(val, os.path.basename(input_file)):
            OIB = key

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

    # track file progress
    logging.info(input_file)
    # read data from input_file
    if (OIB == 'ATM'):
        # load IceBridge ATM data from input_file
        dinput,_,HEM = read_ATM_icessn_file(input_file,input_subsetter)
    elif (OIB == 'ATM1b'):
        # load IceBridge Level-1b ATM data from input_file
        dinput,_,HEM = read_ATM_qfit_file(input_file,input_subsetter)
    elif OIB in ('LVIS','LVGH'):
        # load IceBridge LVIS data from input_file
        dinput,_,HEM = read_LVIS_HDF5_file(input_file,input_subsetter)

    # pyproj transformer for converting to polar stereographic
    EPSG = dict(N=3413,S=3031)
    SIGN = dict(N=1.0,S=-1.0)
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_epsg(EPSG[HEM])
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    # dictionary of coordinate reference system variables
    cs_to_cf = crs2.cs_to_cf()
    crs_to_dict = crs2.to_dict()

    # attributes for each output item
    attributes = collections.OrderedDict()
    # time
    attributes['time'] = {}
    attributes['time']['long_name'] = 'time'
    attributes['time']['standard_name'] = 'time'
    attributes['time']['description'] = ('The transmit time of each shot in '
        'the 1 second frame measured as UTC seconds elapsed since Jan 1 '
        '2000 12:00:00 UTC.')
    attributes['time']['units'] = 'seconds since 2000-01-01 12:00:00 UTC'
    attributes['time']['calendar'] = 'standard'
    # x and y
    attributes['x'],attributes['y'] = ({},{})
    for att_name in ['long_name','standard_name','units']:
        attributes['x'][att_name] = cs_to_cf[0][att_name]
        attributes['y'][att_name] = cs_to_cf[1][att_name]
    # index
    attributes['index'] = {}
    attributes['index']['long_name'] = 'Index'
    attributes['index']['grid_mapping'] = 'Polar_Stereographic'
    attributes['index']['units'] = '1'
    attributes['index']['coordinates'] = 'x y'

    # index directory for hemisphere
    index_directory = 'north' if (HEM == 'N') else 'south'
    # output directory and index file
    DIRECTORY = os.path.dirname(input_file)
    BASENAME = os.path.basename(input_file)
    fileBasename,_ = os.path.splitext(BASENAME)
    output_file = os.path.join(DIRECTORY, index_directory,
        f'{fileBasename}.h5')

    # create index directory for hemisphere
    if not os.access(os.path.join(DIRECTORY,index_directory),os.F_OK):
        os.makedirs(os.path.join(DIRECTORY,index_directory),
            mode=MODE, exist_ok=True)

    # indices of points in hemisphere
    valid, = np.nonzero(np.sign(dinput['lat']) == SIGN[HEM])
    # convert latitude and longitude to regional projection
    x,y = transformer.transform(dinput['lon'],dinput['lat'])
    # large-scale tiles
    xtile = (x-0.5*SPACING)//SPACING
    ytile = (y-0.5*SPACING)//SPACING

    # open output index file
    f2 = h5py.File(output_file,'w')
    f2.attrs['featureType'] = 'trajectory'
    f2.attrs['GDAL_AREA_OR_POINT'] = 'Point'
    f2.attrs['time_type'] = 'UTC'
    today = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    f2.attrs['date_created'] = today
    # add software information
    f2.attrs['software_reference'] = gz.version.project_name
    f2.attrs['software_version'] = gz.version.full_version
    # create projection variable
    h5 = f2.create_dataset('Polar_Stereographic', (), dtype=np.byte)
    # add projection attributes
    h5.attrs['standard_name'] = 'Polar_Stereographic'
    h5.attrs['spatial_epsg'] = crs2.to_epsg()
    h5.attrs['spatial_ref'] = crs2.to_wkt()
    h5.attrs['proj4_params'] = crs2.to_proj4()
    h5.attrs['latitude_of_projection_origin'] = crs_to_dict['lat_0']
    for att_name,att_val in crs2.to_cf().items():
        h5.attrs[att_name] = att_val
    # for each valid tile pair
    for xp,yp in set(zip(xtile[valid],ytile[valid])):
        # center of each tile (adjust due to integer truncation)
        xc = (xp+1)*SPACING
        yc = (yp+1)*SPACING
        # create group
        tile_group = f'E{xc/1e3:0.0f}_N{yc/1e3:0.0f}'
        if tile_group not in f2:
            g2 = f2.create_group(tile_group)
        else:
            g2 = f2[tile_group]
        # add group attributes
        g2.attrs['x_center'] = xc
        g2.attrs['y_center'] = yc
        g2.attrs['spacing'] = SPACING

        # create merged tile file if not existing
        tile_file = os.path.join(DIRECTORY,index_directory,
            f'{tile_group}.h5')
        clobber = 'a' if os.access(tile_file, os.F_OK) else 'w'
        # open output merged tile file
        f3 = multiprocess_h5py(tile_file, mode=clobber)
        if BASENAME not in f3:
            g3 = f3.create_group(BASENAME)
        else:
            g3 = f3[BASENAME]
        # add file-level variables and attributes
        if (clobber == 'w'):
            # create projection variable
            h5 = f3.create_dataset('Polar_Stereographic', (),
                dtype=np.byte)
            # add projection attributes
            h5.attrs['standard_name'] = 'Polar_Stereographic'
            h5.attrs['spatial_epsg'] = crs2.to_epsg()
            h5.attrs['spatial_ref'] = crs2.to_wkt()
            for att_name,att_val in crs2.to_cf().items():
                h5.attrs[att_name] = att_val
            # add file attributes
            f3.attrs['featureType'] = 'trajectory'
            f3.attrs['x_center'] = xc
            f3.attrs['y_center'] = yc
            f3.attrs['spacing'] = SPACING
            f3.attrs['GDAL_AREA_OR_POINT'] = 'Point'
            f3.attrs['time_type'] = 'UTC'
            f3.attrs['date_created'] = today
            # add software information
            f3.attrs['software_reference'] = gz.version.project_name
            f3.attrs['software_version'] = gz.version.full_version

        # indices of points within tile
        indices, = np.nonzero((xtile == xp) & (ytile == yp))
        # output variables for index file
        output = collections.OrderedDict()
        output['time'] = dinput['time'][indices].copy()
        output['x'] = x[indices].copy()
        output['y'] = y[indices].copy()
        output['index'] = indices.copy()
        # for each output group
        for g in [g2,g3]:
            # for each output variable
            h5 = {}
            for key,val in output.items():
                # check if HDF5 variable exists
                if key not in g:
                    # create HDF5 variable
                    h5[key] = g.create_dataset(key, val.shape, data=val,
                        dtype=val.dtype, compression='gzip')
                else:
                    # overwrite HDF5 variable
                    h5[key] = g[key]
                    h5[key][...] = val
                # add variable attributes
                for att_name,att_val in attributes[key].items():
                    h5[key].attrs[att_name] = att_val
                # create or attach dimensions
                if key not in ('time',):
                    for i,dim in enumerate(['time']):
                        h5[key].dims[i].attach_scale(h5[dim])
                else:
                    h5[key].make_scale(key)
        # close the merged tile file
        f3.close()

    # Output HDF5 structure information
    logging.info(list(f2.keys()))
    # close the output file
    f2.close()
    # change the permissions mode of the output file
    os.chmod(output_file, mode=MODE)

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Creates tile index files of Operation
            IceBridge elevation data
            """
    )
    # command line parameters
    # input operation icebridge files
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='Input Operation IceBridge file')
    # output grid spacing
    parser.add_argument('--spacing','-S',
        type=float, default=10e3,
        help='Output grid spacing')
    # verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of run')
    # permissions mode of the directories and files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of directories and files')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # run program for each file
    for arg in args.infile:
        tile_icebridge_data(arg,
            SPACING=args.spacing,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
