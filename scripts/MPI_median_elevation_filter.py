#!/usr/bin/env python
u"""
MPI_median_elevation_filter.py
Written by Tyler Sutterley (06/2022)

Filters elevation change rates from triangulated Operation IceBridge data
    using an interquartile range algorithm described by Pritchard (2009)
    and a robust dispersion estimator (RDE) filter described in Smith (2017)

INPUTS:
    triangulated ATM, LVIS or GLAS files to be read
    first file is the file to be filtered (input_files[0])
    all other files are added for use in the filter (input_files[1:])

COMMAND LINE OPTIONS:
    -V, --verbose: output MPI rank and size for job
    -D X, --distance=X: radial distance for determining points to median filter
    --count=X: minimum number of points within radial distance to be valid
    -M X, --mode=X: Permission mode of files created

REQUIRES MPI PROGRAM
    MPI: standardized and portable message-passing system
        https://www.open-mpi.org/
        http://mpitutorial.com/

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    mpi4py: MPI for Python
        http://pythonhosted.org/mpi4py/
        http://mpi4py.readthedocs.org/en/stable/
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://h5py.org
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/

PROGRAM DEPENDENCIES:
    time.py: Utilities for calculating time operations
    spatial.py: utilities for reading and writing spatial data
    read_ATM1b_QFIT_binary.py: read ATM1b QFIT binary files (NSIDC version 1)

REFERENCE:
    H D Pritchard, R J Arthern, D G Vaughan and L A Edwards, "Extensive dynamic
    thinning on the margins of the Greenland and Antarctic ice sheets",
    Nature, 461(7266), pp. 971-975 (2009).  https://doi.org/10.1038/nature08471

    B E Smith, N Gourmelen, A Huth and I Joughin, "Connected subglacial lake
    drainage beneath Thwaites Glacier, West Antarctica", The Cryosphere, 11,
    pp. 451-467 (2017).  https://doi.org/10.5194/tc-11-451-2017

UPDATE HISTORY:
    Updated 07/2022: place some imports within try/except statements
    Updated 06/2022: updated ATM1b read functions for distributed version
        use argparse descriptions within documentation
    Updated 01/2022: use argparse to set command line options
        use pyproj for converting to polar stereographic coordinates
        use logging for verbose output of processing run
        use icesat2_toolkit time utilities for converting GPS to UTC
    Updated 10/2019: changing Y/N flags to True/False
    Updated 09/2019: decode elevation file attributes for python3 compatibility
    Updated 02/2019: using range for python3 compatibility
    Updated 10/2018: updated GPS time calculation for calculating leap seconds
    Updated 06/2018: can read LVIS LDS version 2.0.2 (2017 campaign onward)
    Updated 01/2018: updated inputs to process triangulated lagrangian files
    Updated 10/2017: format of YYMMDD from ATM1b qfit filenames
    Updated 06/2017: outputs of QFIT binary read program now includes headers
        read subset indices if a single value instead of a range
    Updated 05/2017: updated references for robust dispersion estimator (RDE)
        print input filename if using verbose output.  added --mode option
        added some descriptive comments of the input and output files
        using the FORMAT option with map_ll for case with a single input file
        using reformatted HDF5 files from read_icebridge_lvis.py
        added input data subsetters for reducing auxiliary file data size
        (should be backwards compatible with shells without the subsetter)
        added function for reading Level-1b ATM QFIT binary files
    Forked 04/2017: new processing chain (MPI_triangulate_elevation.py)
        indices map to valid points versus all points (iterates only over valid)
        outputs masks from two dispersion filters: IQR (Pritchard, 2009) and RDE
    Updated 12/2016: using MJD strings in output filenames versus floats
    Updated 09/2016: fileID.filepath() for netCDF4, axes_grid1 in mpl_toolkits
    Updated 07/2016: using netCDF4-python
    Updated 06/2016: using __future__ print function. convert_julian with ASTYPE
        adjust output figure parameters and size, made point count a variable
    Updated 05/2016: using getopt to set verbose parameter and radial distance
    Updated 04/2016: renamed IQR_filter_triangulated.py. inputs to only file
    Updated 10/2015: output index to mask netcdf file and not elevation points
        added comm.Barrier() to wait for all processes to finish
        can run files from icebridge_triangulate and icebridge_lvis_triangulate
        verifies that points within range in filtering function are finite
    Written 09/2015
"""
from __future__ import print_function

import sys
import os
import re
import time
import pyproj
import logging
import argparse
import warnings
import numpy as np
import grounding_zones as gz

# attempt imports
try:
    import ATM1b_QFIT.read_ATM1b_QFIT_binary
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("ATM1b_QFIT not available")
try:
    import h5py
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("h5py not available")
try:
    import icesat2_toolkit as is2tk
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("icesat2_toolkit not available")
try:
    from mpi4py import MPI
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("mpi4py not available")
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: keep track of MPI threads
def info(rank, size):
    logging.info(f'Rank {rank+1:d} of {size:d}')
    logging.info(f'module name: {__name__}')
    if hasattr(os, 'getppid'):
        logging.info(f'parent process: {os.getppid():d}')
    logging.info(f'process id: {os.getpid():d}')

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Filters elevation change rates from triangulated
            Operation IceBridge dat using an interquartile range algorithm
            described by Pritchard (2009) and a robust dispersion estimator
            (RDE) filter described in Smith (2017)
            """,
    )
    # command line parameters
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='Input files')
    # radial distance to determine points to filter (25km)
    parser.add_argument('--distance','-D',
        type=float, default=25e3,
        help='Radial distance (m) for determining filters')
    # minimum number of points within radial distance for validity
    parser.add_argument('--count','-C',
        type=int, default=10,
        help='Minimum allowable points within radial distance')
    # output module information for process
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of run')
    # permissions mode of the output files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='permissions mode of output files')
    # return the parser
    return parser

# PURPOSE: set the hemisphere flag from an input region flag
def set_hemisphere(REGION):
    projection_flag = {'AN':'S', 'GR':'N'}
    return projection_flag[REGION]

# PURPOSE: reading the number of file lines removing commented lines
def file_length(input_file,input_subsetter,HDF5=False,QFIT=False):
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

# PURPOSE: read input HDF5 data from MPI_triangulate_elevation.py
def read_HDF5_triangle_data(input_file, input_subsetter):
    # Open the HDF5 file for reading
    fileID = h5py.File(input_file, 'r')
    # Output HDF5 file information
    logging.info(fileID.filename)
    logging.info(list(fileID.keys()))
    # allocate python dictionaries for input variables and attributes
    HDF5_data = {}
    HDF5_attributes = {}
    # get each HDF5 variable
    for key,val in fileID.items():
        HDF5_data[key] = val[input_subsetter] if input_subsetter else val[:]
        # allocate python dictionary for specific variable attributes
        HDF5_attributes[key] = {}
        for att_name,att_val in val.attrs.items():
            HDF5_attributes[key][att_name] = att_val
    # get global attributes
    for att_name,att_val in fileID.attrs.items():
        HDF5_attributes[att_name] = att_val
    # close the HDF5 file
    fileID.close()
    # return data and attributes
    return HDF5_data,HDF5_attributes

# PURPOSE: read the ATM Level-1b data file for variables of interest
def read_ATM_qfit_file(input_file, s):
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
        f = open(input_file,'r')
        file_contents=[i for i in f.readlines() if re.match(r'^(?!\#|\n)',i)]
        f.close()
        # number of lines of data within file
        file_lines = file_length(input_file,s)
        # subset the data to indices if specified
        file_contents = [file_contents[i] for i in s] if s else file_contents
        # create output variables with length equal to the number of lines
        ATM_L1b_input['lat'] = np.zeros((file_lines),dtype=np.float64)
        ATM_L1b_input['lon'] = np.zeros((file_lines),dtype=np.float64)
        ATM_L1b_input['data'] = np.zeros((file_lines),dtype=np.float64)
        hour = np.zeros((file_lines),dtype=np.float64)
        minute = np.zeros((file_lines),dtype=np.float64)
        second = np.zeros((file_lines),dtype=np.float64)
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
        fid,h = ATM1b_QFIT.read_ATM1b_QFIT_binary(input_file, SUBSETTER=s)
        file_lines, = fid['elevation'].shape
        ATM_L1b_input['lat'] = fid['latitude'][:]
        ATM_L1b_input['lon'] = fid['longitude'][:]
        ATM_L1b_input['data'] = fid['elevation'][:]
        time_hhmmss = fid['time_hhmmss'][:]
        # extract hour, minute and second from time_hhmmss
        hour = np.zeros((file_lines),dtype=np.float64)
        minute = np.zeros((file_lines),dtype=np.float64)
        second = np.zeros((file_lines),dtype=np.float64)
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
        fileID = h5py.File(input_file, 'r')
        # if not using the input subsetter
        if s is None:
            # create output variables with length equal to input elevation
            file_lines, = fileID['elevation'].shape
            ATM_L1b_input['lat'] = fileID['latitude'][:]
            ATM_L1b_input['lon'] = fileID['longitude'][:]
            ATM_L1b_input['data'] = fileID['elevation'][:]
            time_hhmmss = fileID['instrument_parameters']['time_hhmmss'][:]
        else:
            # use subsetted data
            file_lines = len(s)
            ATM_L1b_input['lat'] = fileID['latitude'][s]
            ATM_L1b_input['lon'] = fileID['longitude'][s]
            ATM_L1b_input['data'] = fileID['elevation'][s]
            time_hhmmss = fileID['instrument_parameters']['time_hhmmss'][s]
        # extract hour, minute and second from time_hhmmss
        hour = np.zeros((file_lines),dtype=np.float64)
        minute = np.zeros((file_lines),dtype=np.float64)
        second = np.zeros((file_lines),dtype=np.float64)
        # for each line within the file
        for i,packed_time in enumerate(time_hhmmss):
            # convert to zero-padded string with 3 decimal points
            line_contents = f'{packed_time:010.3f}'
            hour[i] = np.float64(line_contents[:2])
            minute[i] = np.float64(line_contents[2:4])
            second[i] = np.float64(line_contents[4:])
        fileID.close()
    # leap seconds for converting from GPS time to UTC
    S = calc_GPS_to_UTC(year,month,day,hour,minute,second)
    # calculation of Julian day taking into account leap seconds
    JD = calc_julian_day(year,month,day,HOUR=hour,MINUTE=minute,SECOND=second-S)
    # converting to J2000 seconds
    ATM_L1b_input['time'] = (JD - 2451545.0)*86400.0
    # return the output variables
    return ATM_L1b_input,file_lines

# PURPOSE: read the ATM Level-2 data file for variables of interest
def read_ATM_icessn_file(input_file, s):
    # regular expression pattern for extracting parameters
    regex_pattern = r'(BLATM2|ILATM2)_(\d+)_(\d+)_smooth_nadir(.*?)(csv|seg|pt)$'
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
        file_contents=[i for i in f.readlines() if re.match(r'^(?!\#|\n)',i)]
    # number of lines of data within file
    file_lines = file_length(input_file,s)
    # subset the data to indices if specified
    file_contents = [file_contents[i] for i in s] if s else file_contents
    # output python dictionary with variables
    ATM_L2_input = {}
    # create output variables with length equal to the number of file lines
    for key in file_dtype.keys():
        ATM_L2_input[key] = np.zeros((file_lines), dtype=np.float64)
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
        # leap seconds for converting from GPS time to UTC
        S = calc_GPS_to_UTC(year,month,day,hour,minute,second)
    else:
        S = 0.0
    # calculation of Julian day
    JD = calc_julian_day(year,month,day,HOUR=hour,MINUTE=minute,SECOND=second-S)
    # converting to J2000 seconds
    ATM_L2_input['time'] = (JD - 2451545.0)*86400.0
    # convert RMS from centimeters to meters
    ATM_L2_input['error'] = ATM_L2_input['RMS']/100.0
    # return the output variables
    return ATM_L2_input,file_lines

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

# PURPOSE: calculate the Julian day from calendar date
# http://scienceworld.wolfram.com/astronomy/JulianDate.html
def calc_julian_day(YEAR, MONTH, DAY, HOUR=0, MINUTE=0, SECOND=0):
    JD = 367.*YEAR - np.floor(7.*(YEAR + np.floor((MONTH+9.)/12.))/4.) - \
        np.floor(3.*(np.floor((YEAR + (MONTH - 9.)/7.)/100.) + 1.)/4.) + \
        np.floor(275.*MONTH/9.) + DAY + 1721028.5 + HOUR/24. + MINUTE/1440. + \
        SECOND/86400.
    return np.array(JD,dtype=np.float64)

# PURPOSE: calculate the number of leap seconds between GPS time (seconds
# since Jan 6, 1980 00:00:00) and UTC
def calc_GPS_to_UTC(YEAR, MONTH, DAY, HOUR, MINUTE, SECOND):
    GPS = 367.*YEAR - np.floor(7.*(YEAR + np.floor((MONTH+9.)/12.))/4.) - \
        np.floor(3.*(np.floor((YEAR + (MONTH - 9.)/7.)/100.) + 1.)/4.) + \
        np.floor(275.*MONTH/9.) + DAY + 1721028.5 - 2444244.5
    GPS_Time = GPS*86400.0 + HOUR*3600.0 + MINUTE*60.0 + SECOND
    return is2tk.time.count_leap_seconds(GPS_Time)

# PURPOSE: convert time from delta seconds into Julian and year-decimal
def convert_delta_time(delta_time, epoch=(2000,1,1,12,0,0)):
    """
    converts delta times into into Julian and year-decimal

    Arguments
    ---------
    delta_time: seconds since epoch

    Keyword arguments
    -----------------
    epoch: epoch for delta times (default 2000-01-01T12:00:00)

    Returns
    -------
    julian: time in Julian days
    decimal: time in year-decimal
    """
    # convert to array if single value
    delta_time = np.atleast_1d(delta_time)
    # calculate Julian time (UTC) by converting to MJD and then adding offset
    time_julian = 2400000.5 + is2tk.time.convert_delta_time(delta_time,
        epoch1=epoch, epoch2=(1858,11,17,0,0,0), scale=1.0/86400.0)
    # convert to calendar date
    Y,M,D,h,m,s = is2tk.time.convert_julian(time_julian,FORMAT='tuple')
    # calculate year-decimal time (UTC)
    time_decimal = is2tk.time.convert_calendar_decimal(Y,M,day=D,
        hour=h,minute=m,second=s)
    # return both the Julian and year-decimal formatted dates
    return dict(julian=np.squeeze(time_julian),decimal=np.squeeze(time_decimal))

# PURPOSE: check if dh/dt is valid by checking the interquartile range from
# Pritchard (2009) and the robust dispersion estimator (RDE) from Smith (2017)
def filter_dhdt(xpt, ypt, hpt, X, Y, H, COUNT=10, DISTANCE=25e3):
    # find valid, finite points within DISTANCE radius
    dd = (xpt-X)**2 + (ypt-Y)**2
    valid_count = np.count_nonzero((dd <= (DISTANCE**2)) & np.isfinite(H))
    # create valid flags
    IQR_valid = False
    RDE_valid = False
    # if there are COUNT or more valid points within DISTANCE in meters
    if (valid_count >= COUNT):
        # indices for finite points within the distance
        ind = np.nonzero((dd <= (DISTANCE**2)) & np.isfinite(H))
        h0 = H[ind]
        # calculate percentiles for IQR, MDE and median
        # IQR: first and third quartiles (25th and 75th percentiles)
        # MDE: 16th and 84th percentiles
        # median: 50th percentile
        Q1,Q3,P16,P84,MEDIAN = np.percentile(h0,[25,75,16,84,50])
        # calculate interquartile range
        IQR = Q3 - Q1
        # calculate robust dispersion estimator (RDE)
        RDE = P84 - P16
        # IQR pass: dh/dt of point-(median value) is within 75% of IQR
        IQR_valid = (np.abs(hpt-MEDIAN) <= (0.75*IQR))
        # RDE pass: dh/dt of point-(median value) is within 50% of P84-P16
        RDE_valid = (np.abs(hpt-MEDIAN) <= (0.50*RDE))
    # return the valid flags
    return (IQR_valid,RDE_valid)

# PURPOSE: calculate rates of elevation change and filters results based on
# Pritchard et al. (2009) IQR filter and a robust dispersion estimator (RDE)
# from Smith et al. (2017)
def main():
    # start MPI communicator
    comm = MPI.COMM_WORLD

    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # create logger
    loglevel = logging.INFO if args.verbose else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # list of input files for median filtering (tilde-expand paths)
    # can a single file or a range of files
    # secondary files will be added in to the filter but excluded from output
    input_files = []
    # list of indices for subsetting input auxiliary files
    input_subsetter = []
    # for each input argument (input file)
    for arg in args.infile:
        # extract file name and subsetter indices lists
        match_object = re.match(r'(.*?)(\[(.*?)\])?$',arg)
        input_files.append(os.path.expanduser(match_object.group(1)))
        # subset auxiliary files to indices
        if match_object.group(2):
            # decompress ranges and add to list
            file_indices = []
            for i in re.findall(r'((\d+)-(\d+)|(\d+))',match_object.group(3)):
                file_indices.append(int(i[3])) if i[3] else \
                    file_indices.extend(range(int(i[1]),int(i[2])+1))
            input_subsetter.append(file_indices)
        else:
            input_subsetter.append(None)

    # output module information for process
    logging.info(input_files[0]) if (comm.rank == 0) else None
    info(comm.rank,comm.size)

    # compile regular expression operator for finding files
    # and extracting region, instrument and date information
    f1 = r'TRIANGULATED|LAGRANGIAN|OVERLAPPING_FOOTPRINTS|LAGRANGIAN_FOOTPRINTS'
    p1 = r'(AN|GR)_NASA_({0})_WGS84_(.*?)(\d+)-(.*?)(\d+)\.H5$'
    rx1 = re.compile(p1.format(f1), re.VERBOSE)
    # REG1: hemisphere flag (GR or AN) for the region
    # TYPE1: Triangulated data or Lagrangian data
    # OIB1 and OIB2: data flags (ATM, LVIS)
    # YMDS1: acquisition year, month, day and second
    # YMD2: triangulated year, month and day
    REG1,TYPE1,OIB1,YMDS1,OIB2,YMD2 = rx1.findall(input_files[0]).pop()
    HEM = set_hemisphere(REG1)
    # get the dimensions of the input HDF5 file (sum auxiliary files)
    n_1 = file_length(input_files[0], None, HDF5='data')
    n_2 = np.sum([file_length(f,s,HDF5='data') for f,s in
        zip(input_files[1:],input_subsetter[1:])], dtype=np.int64)

    # output mask format for each type
    file_format = '{0}_NASA_{1}_MASK_{2}{3}-{4}{5}.H5'

    # full path for ATM/LVIS directory
    DIRECTORY = os.path.dirname(input_files[0])

    # lists with input files
    original_files = []
    # input data on MPI rank 0 (parent process)
    if (comm.rank == 0):
        # reading input datasets
        # dinput1: triangulated data (from OIB2)
        # dinput2: original data (from OIB1)
        # dinput3: triangulated data within DISTANCE (from OIB2)
        # dinput4: original data within DISTANCE (from OIB1)
        # dinput3 and dinput4 can be from multiple files (data is merged)
        #     and are also optional if no other data files are within DISTANCE
        # dinput1 and dinput3 will be combined to form dh1 and dt1
        # dinput2 and dinput4 will be combined to form dh2 and dt2
        # dh/dt will be calculated as (dh2 - dh1)/(dt2 - dh1)

        # read the input file on rank 0 (parent process)
        dinput1,att1 = read_HDF5_triangle_data(input_files[0],None)

        # pyproj transformer for converting lat/lon to polar stereographic
        EPSG = dict(N=3413,S=3031)
        crs1 = pyproj.CRS.from_epsg(4326)
        crs2 = pyproj.CRS.from_epsg(EPSG[HEM])
        transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

        # original data file used in the triangulation
        elevation_file = att1['elevation_file']#.decode('utf-8')
        FILENAME = os.path.join(DIRECTORY,os.path.basename(elevation_file))
        original_files.append(elevation_file)
        # check that original data file exists locally
        if not os.access(FILENAME, os.F_OK):
            raise FileNotFoundError(f'{FILENAME} not found in filesystem')
        # read the original data file
        if (OIB1 == 'ATM'):
            # load IceBridge ATM data
            dinput2,file_lines = read_ATM_icessn_file(FILENAME, None)
        elif (OIB1 == 'ATM1b'):
            # load IceBridge Level-1b ATM data
            dinput2,file_lines = read_ATM_qfit_file(FILENAME, None)
        elif OIB1 in ('LVIS','LVGH'):
            # load IceBridge LVIS data
            dinput2,file_lines = read_LVIS_HDF5_file(FILENAME, None)
        # check that data sizes match
        if (file_lines != n_1):
            logging.critical(os.path.basename(input_files[0]),
                os.path.basename(FILENAME))
            raise RuntimeError(f'Mismatch ({n_1:d} {file_lines:d})')

        # if there are secondary files to add to filter (data within DISTANCE)
        if (n_2 > 0):
            # create dictionary for input data for files (merged)
            dinput3 = dict(data=np.zeros((n_2)), time=np.zeros((n_2)),
                error=np.zeros((n_2)), lon=np.zeros((n_2)), lat=np.zeros((n_2)))
            dinput4 = dict(data=np.zeros((n_2)), time=np.zeros((n_2)),
                error=np.zeros((n_2)), lon=np.zeros((n_2)), lat=np.zeros((n_2)))
            # read data from input_files[1:] and combine into single array
            c = 0
            for fi,s in zip(input_files[1:],input_subsetter[1:]):
                file_input,att3 = read_HDF5_triangle_data(fi,s)
                n_3 = file_length(fi,s,HDF5='data')
                # iterate through input keys of iterest
                for key in ['data','lon','lat','time','error']:
                    dinput3[key][c:c+n_3] = file_input.get(key,None)
                # original data file used in the triangulation
                elevation_file = att3['elevation_file']#.decode('utf-8')
                FILENAME = os.path.join(DIRECTORY,os.path.basename(elevation_file))
                original_files.append(elevation_file)
                # check that original data file exists locally
                if not os.access(FILENAME, os.F_OK):
                    raise FileNotFoundError(f'{FILENAME} not found in filesystem')
                # read the original data file
                if (OIB1 == 'ATM'):
                    # load IceBridge ATM data from FILENAME
                    file_input,n_4 = read_ATM_icessn_file(FILENAME,s)
                elif (OIB1 == 'ATM1b'):
                    # load IceBridge Level-1b ATM data from FILENAME
                    file_input,n_4 = read_ATM_qfit_file(FILENAME,s)
                elif OIB1 in ('LVIS','LVGH'):
                    # load IceBridge LVIS data from FILENAME
                    file_input,n_4 = read_LVIS_HDF5_file(FILENAME,s)
                # check that data sizes match
                if (n_3 != n_4):
                    logging.critical(os.path.basename(fi),
                        os.path.basename(FILENAME))
                    raise RuntimeError(f'Mismatch ({n_3:d} {n_4:d})')
                # iterate through input keys of iterest
                for key in ['data','lon','lat','time','error']:
                    dinput4[key][c:c+n_4] = file_input.get(key,None)
                # add file lines to counter
                c += n_4
            # check that final data sizes match
            if (c != n_2):
                raise RuntimeError(f'Total Mismatch ({c:d} {n_2:d})')

            # convert from latitude/longitude into polar stereographic
            X1,Y1 = transformer.transform(dinput1['lon'], dinput1['lat'])
            X2,Y2 = transformer.transform(dinput3['lon'], dinput3['lat'])
            # extract values and combine into single arrays
            X = np.concatenate((X1, X2), axis=0)
            Y = np.concatenate((Y1, Y2), axis=0)
            # elevation and time
            dh1 = np.concatenate((dinput1['data'], dinput3['data']), axis=0)
            dh2 = np.concatenate((dinput2['data'], dinput4['data']), axis=0)
            J1 = np.concatenate((dinput1['time'], dinput3['time']), axis=0)
            J2 = np.concatenate((dinput2['time'], dinput4['time']), axis=0)
        else:
            # convert from latitude/longitude into polar stereographic
            X,Y = transformer.transform(dinput1['lon'], dinput1['lat'])
            # elevation and time
            dh1 = dinput1['data']
            dh2 = dinput2['data']
            J1 = dinput1['time']
            J2 = dinput2['time']

        # mask for points not equal to fill value
        mask = (dh1 != att1['data']['_FillValue'])
        # convert from J2000 into decimal years for dh/dt
        # this should prevent prevent underflow errors
        dt1 = convert_delta_time(J1)['decimal']
        dt2 = convert_delta_time(J2)['decimal']
        # calculate dhdt from input triangulated file(s) and original file(s)
        dhdt = (dh2 - dh1)/(dt2 - dt1)
    else:
        # create X, Y and dhdt arrays (combined for all input HDF5 files)
        X = np.empty((n_1 + n_2))
        Y = np.empty((n_1 + n_2))
        dhdt = np.empty((n_1 + n_2))
        mask = np.empty((n_1 + n_2), dtype=bool)

    # Broadcast triangulation data from rank 0 to all other ranks
    comm.Bcast([X, MPI.DOUBLE])
    comm.Bcast([Y, MPI.DOUBLE])
    comm.Bcast([dhdt, MPI.DOUBLE])
    comm.Bcast([mask, MPI.BOOL])

    # indices of valid points
    indices, = np.nonzero(mask)
    # number of points to iterate through in file 1
    iteration_count = np.count_nonzero(mask[:n_1])
    # IQR filter from Pritchard et al. (2009)
    distributed_IQR = np.zeros((n_1), dtype=bool)
    # robust dispersion estimator (RDE) from Smith et al. (2017)
    distributed_RDE = np.zeros((n_1), dtype=bool)
    # only iterate through valid points from first file
    for iteration in range(comm.rank, iteration_count, comm.size):
        # indice for iteration (maps from valid points to original file)
        i = indices[iteration]
        # list of complementary indices.  includes indices from all files
        ii = list(set(indices)-set([i]))
        # check if dh/dt point is valid using X, Y and dhdt for point i
        distributed_IQR[i],distributed_RDE[i] = filter_dhdt(X[i], Y[i], dhdt[i],
            X[ii], Y[ii], dhdt[ii], COUNT=args.count, DISTANCE=args.distance)

    # create matrices for valid reduced data output
    associated_IQR = np.zeros((n_1),dtype=bool)
    associated_RDE = np.zeros((n_1),dtype=bool)
    # communicate output MPI matrices to zero rank
    # operation is a logical "or" across the elements.
    comm.Reduce(sendbuf=[distributed_IQR, MPI.BOOL], \
        recvbuf=[associated_IQR, MPI.BOOL], op=MPI.LOR, root=0)
    comm.Reduce(sendbuf=[distributed_RDE, MPI.BOOL], \
        recvbuf=[associated_RDE, MPI.BOOL], op=MPI.LOR, root=0)
    # wait for all distributed processes to finish
    comm.Barrier()

    # output data on MPI rank 0 (parent process)
    if (comm.rank == 0):
        # form: rg_NASA_TRIANGULATED_MASK_fl1yyyymmddjjjjj-fl2yyyymmdd.H5
        # where rg is the hemisphere flag (GR or AN) for the region
        # fl1 and fl2 are the data flags (ATM, LVIS)
        # yymmddjjjjj is the year, month, day and second of input file 1
        # yymmdd is the year, month and day of the triangulated files
        FILE = file_format.format(REG1,TYPE1,OIB1,YMDS1,OIB2,YMD2)
        median_masks = dict(IQR=associated_IQR, RDE=associated_RDE)
        HDF5_triangulated_mask(median_masks, DISTANCE=args.distance,
            COUNT=args.count, INPUT=input_files, ORIGINAL=original_files,
            FILENAME=os.path.join(DIRECTORY,FILE))
        # print file name if verbose output is specified
        logging.info(os.path.join(DIRECTORY,FILE))
        # change the permissions level to MODE
        os.chmod(os.path.join(DIRECTORY,FILE), args.mode)

# PURPOSE: outputting the data mask for reducing input data as HDF5 file
def HDF5_triangulated_mask(valid_mask, DISTANCE=0, COUNT=0, FILENAME='',
    INPUT=None, ORIGINAL=None, CLOBBER=True):
    # setting HDF5 clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'w-'

    # description and references for each median filter
    description = {}
    reference = {}
    description['IQR'] = ('an_interquartile_range_algorithm_(IQR)_described_by_'
        'Pritchard_et_al.,_Nature_(2009)')
    reference['IQR'] = 'http://dx.doi.org/10.1038/nature08471'
    description['RDE'] = ('a_robust_dispersion_estimator_(RDE)_algorithm_'
        'described_by_Smith_et_al.,_The_Cryosphere_(2017)')
    reference['RDE'] = 'http://dx.doi.org/10.5194/tc-11-451-2017'
    # open output HDF5 file
    fileID = h5py.File(os.path.expanduser(FILENAME), clobber)
    # Defining the HDF5 dataset variables
    h5 = {}
    for key, val in valid_mask.items():
        h5[key] = fileID.create_dataset(key, val.shape, data=val,
            dtype=val.dtype, compression='gzip')
        # add HDF5 variable attributes
        h5[key].attrs['long_name'] = '{0}_filter'.format(key)
        h5[key].attrs['description'] = ('Elevation_change_mask_calculated_'
            'using_{0}').format(description[key])
        h5[key].attrs['reference'] = reference[key]
        h5[key].attrs['passing_count'] = np.count_nonzero(val)
        h5[key].attrs['search_radius'] = '{0:0.0f}km'.format(DISTANCE/1e3)
        h5[key].attrs['threshold_count'] = COUNT
        # attach dimensions
        h5[key].dims[0].label = 'RECORD_SIZE'
    # HDF5 file attributes
    fileID.attrs['date_created'] = time.strftime('%Y-%m-%d',time.localtime())
    # add attributes for input elevation file and files triangulated
    input_elevation_files = ','.join([os.path.basename(f) for f in ORIGINAL])
    fileID.attrs['elevation_files'] = input_elevation_files
    input_triangulated_files = ','.join([os.path.basename(f) for f in INPUT])
    fileID.attrs['triangulated_files'] = input_triangulated_files
    # add software information
    fileID.attrs['software_reference'] = gz.version.project_name
    fileID.attrs['software_version'] = gz.version.full_version
    fileID.attrs['software_revision'] = gz.utilities.get_git_revision_hash()
    # Closing the HDF5 file
    fileID.close()

# run main program
if __name__ == '__main__':
    main()
