#!/usr/bin/env python
u"""
MPI_triangulate_elevation.py
Written by Tyler Sutterley (07/2022)

Calculates interpolated elevations by triangulated irregular
    network meshing (TINs) to compare with an input file

INPUTS:
    ATM or LVIS files to be read
    first file is the file to be compared with (input_files[0])
    all other files are to be triangulated to the first file (input_files[1:])

COMMAND LINE OPTIONS:
    -V, --verbose: output MPI rank and size for job
    -D X, --distance=X: radial distance for determining points to triangulate
        Default: 300 meters
    -A X, --angle=X: maximum angle of valid triangles for calculating elevations
        Default: 120 degrees
    -M X, --mode=X: Permission mode of files created

REQUIRES MPI PROGRAM
    MPI: standardized and portable message-passing system
        https://www.open-mpi.org/
        http://mpitutorial.com/

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
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

UPDATE HISTORY:
    Updated 07/2022: place some imports within try/except statements
    Updated 06/2022: updated ATM1b read functions for distributed version
    	use argparse descriptions within documentation
    Updated 01/2022: use argparse to set command line options
        use pyproj for converting to polar stereographic coordinates
        use logging for verbose output of processing run
        use icesat2_toolkit time utilities for converting GPS to UTC
    Updated 10/2019: changing Y/N flags to True/False
    Updated 07/2019: use numpy array for delaunay triangulation (python3)
    Updated 02/2019: using range for python3 compatibility
    Updated 10/2018: updated GPS time calculation for calculating leap seconds
    Updated 06/2018: can read LVIS LDS version 2.0.2 (2017 campaign onward)
    Updated 02/2018: can reduce input file to run for a subset of points
    Updated 01/2018: calculate attempts to create a Delaunay trianglulation
    Updated 10/2017: format of YYMMDD from ATM1b qfit filenames
    Updated 06/2017: use actual geospatial lat/lon min and max in attributes
        outputs of QFIT binary read program now includes headers
        read subset indices if a single value instead of a range
    Updated 05/2017: print input filename if using verbose output
        added some descriptive comments of the input and output files
        using reformatted HDF5 files from read_icebridge_lvis.py
        added input data subsetters for reducing triangulated file data size
        (should be backwards compatible with shells without the subsetter)
        added function for reading Level-1b ATM QFIT binary files
    Forked 04/2017: read raw data and compare with other raw datasets
        output as HDF5 file with more specific attributes
    Updated 10/2016: version update of IMBIE-2 basins
        cull triangles that are overly obtuse (with max angle >= 120 degrees)
    Updated 07/2016: using netCDF4-python
    Updated 06/2016: using __future__ print function
    Updated 05/2016: can set radial distance with getopt
    Updated 04/2016: calculate x and y components of surface slopes
        update using getopt for setting parameters and triangulate ATM or LVIS
    Updated 10/2015: broadcasting ATM/LVIS data to be triangulated
    Written 08/2015-09/2015
"""
from __future__ import print_function

import sys
import os
import re
import time
import h5py
import pyproj
import logging
import argparse
import warnings
import numpy as np
import scipy.spatial
import icesat2_toolkit.time
import icesat2_toolkit.spatial
#-- attempt imports
try:
    import ATM1b_QFIT.read_ATM1b_QFIT_binary
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("ATM1b_QFIT not available")
try:
    from mpi4py import MPI
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("mpi4py not available")
#-- ignore warnings
warnings.filterwarnings("ignore")

#-- PURPOSE: keep track of MPI threads
def info(rank, size):
    logging.info(f'Rank {rank+1:d} of {size:d}')
    logging.info(f'module name: {__name__}')
    if hasattr(os, 'getppid'):
        logging.info(f'parent process: {os.getppid():d}')
    logging.info(f'process id: {os.getpid():d}')

#-- PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates interpolated elevations by
            triangulated irregular network meshing (TINs) to compare
            with an input file
            """,
    )
    #-- command line parameters
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='Input files')
    #-- radial distance to determine points to triangulate (meters)
    parser.add_argument('--distance','-D',
        type=float, default=300.0,
        help='Radial distance (m) for calculating triangulations')
    #-- maximum angle within triangle for calculating elevations (degrees)
    parser.add_argument('--angle','-A',
        type=float, default=120.0,
        help='Maximum angle allowed within a Delaunay triangle')
    #-- output module information for process
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of run')
    #-- permissions mode of the output files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='permissions mode of output files')
    #-- return the parser
    return parser

#-- PURPOSE: reading the number of file lines removing commented lines
def file_length(input_file,input_subsetter,HDF5=False,QFIT=False):
    #-- subset the data to indices if specified
    if input_subsetter:
        file_lines = len(input_subsetter)
    elif HDF5:
        #-- read the size of an input variable within a HDF5 file
        with h5py.File(input_file,'r') as fileID:
            file_lines, = fileID[HDF5].shape
    elif QFIT:
        #-- read the size of a QFIT binary file
        file_lines = ATM1b_QFIT.ATM1b_QFIT_shape(input_file)
    else:
        #-- read the input file, split at lines and remove all commented lines
        with open(input_file, mode='r', encoding='utf8') as f:
            i = [i for i in f.readlines() if re.match(r'^(?!\#|\n)',i)]
        file_lines = len(i)
    #-- return the number of lines
    return file_lines

#-- PURPOSE: read the ATM Level-1b data file for variables of interest
def read_ATM_qfit_file(input_file, input_subsetter):
    #-- regular expression pattern for extracting parameters
    mission_flag = r'(BLATM1B|ILATM1B|ILNSA1B)'
    regex_pattern = rf'{mission_flag}_(\d+)_(\d+)(.*?).(qi|TXT|h5)'
    #-- extract mission and other parameters from filename
    MISSION,YYMMDD,HHMMSS,AUX,SFX = re.findall(regex_pattern,input_file).pop()
    #-- early date strings omitted century and millenia (e.g. 93 for 1993)
    if (len(YYMMDD) == 6):
        yr2d,month,day = np.array([YYMMDD[:2],YYMMDD[2:4],YYMMDD[4:]],dtype='i')
        year = (yr2d + 1900.0) if (yr2d >= 90) else (yr2d + 2000.0)
    elif (len(YYMMDD) == 8):
        year,month,day = np.array([YYMMDD[:4],YYMMDD[4:6],YYMMDD[6:]],dtype='i')
    #-- output python dictionary with variables
    ATM_L1b_input = {}
    #-- Version 1 of ATM QFIT files (ascii)
    #-- output text file from qi2txt with proper filename format
    #-- do not use the shortened output format from qi2txt
    if (SFX == 'TXT'):
        #-- compile regular expression operator for reading lines
        regex_pattern = r'[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?'
        rx = re.compile(regex_pattern, re.VERBOSE)
        #-- read the input file, split at lines and remove all commented lines
        f = open(input_file,'r')
        file_contents=[i for i in f.readlines() if re.match(r'^(?!\#|\n)',i)]
        f.close()
        #-- number of lines of data within file
        file_lines = file_length(input_file,input_subsetter)
        #-- create output variables with length equal to the number of lines
        ATM_L1b_input['lat'] = np.zeros_like(file_contents,dtype=np.float64)
        ATM_L1b_input['lon'] = np.zeros_like(file_contents,dtype=np.float64)
        ATM_L1b_input['data'] = np.zeros_like(file_contents,dtype=np.float64)
        hour = np.zeros_like(file_contents,dtype=np.float64)
        minute = np.zeros_like(file_contents,dtype=np.float64)
        second = np.zeros_like(file_contents,dtype=np.float64)
        #-- for each line within the file
        for i,line in enumerate(file_contents):
            #-- find numerical instances within the line
            line_contents = rx.findall(line)
            ATM_L1b_input['lat'][i] = np.float64(line_contents[1])
            ATM_L1b_input['lon'][i] = np.float64(line_contents[2])
            ATM_L1b_input['data'][i] = np.float64(line_contents[3])
            hour[i] = np.float64(line_contents[-1][:2])
            minute[i] = np.float64(line_contents[-1][2:4])
            second[i] = np.float64(line_contents[-1][4:])
    #-- Version 1 of ATM QFIT files (binary)
    elif (SFX == 'qi'):
        #-- read input QFIT data file and subset if specified
        fid,h = ATM1b_QFIT.read_ATM1b_QFIT_binary(input_file)
        #-- number of lines of data within file
        file_lines = file_length(input_file,input_subsetter,QFIT=True)
        ATM_L1b_input['lat'] = fid['latitude'][:]
        ATM_L1b_input['lon'] = fid['longitude'][:]
        ATM_L1b_input['data'] = fid['elevation'][:]
        time_hhmmss = fid['time_hhmmss'][:]
        #-- extract hour, minute and second from time_hhmmss
        hour = np.zeros_like(time_hhmmss,dtype=np.float64)
        minute = np.zeros_like(time_hhmmss,dtype=np.float64)
        second = np.zeros_like(time_hhmmss,dtype=np.float64)
        #-- for each line within the file
        for i,packed_time in enumerate(time_hhmmss):
            #-- convert to zero-padded string with 3 decimal points
            line_contents = f'{packed_time:010.3f}'
            hour[i] = np.float64(line_contents[:2])
            minute[i] = np.float64(line_contents[2:4])
            second[i] = np.float64(line_contents[4:])
    #-- Version 2 of ATM QFIT files (HDF5)
    elif (SFX == 'h5'):
        #-- Open the HDF5 file for reading
        fileID = h5py.File(input_file, 'r')
        #-- number of lines of data within file
        file_lines = file_length(input_file,input_subsetter,HDF5='elevation')
        #-- create output variables with length equal to input elevation
        ATM_L1b_input['lat'] = fileID['latitude'][:]
        ATM_L1b_input['lon'] = fileID['longitude'][:]
        ATM_L1b_input['data'] = fileID['elevation'][:]
        time_hhmmss = fileID['instrument_parameters']['time_hhmmss'][:]
        #-- extract hour, minute and second from time_hhmmss
        hour = np.zeros_like(time_hhmmss,dtype=np.float64)
        minute = np.zeros_like(time_hhmmss,dtype=np.float64)
        second = np.zeros_like(time_hhmmss,dtype=np.float64)
        #-- for each line within the file
        for i,packed_time in enumerate(time_hhmmss):
            #-- convert to zero-padded string with 3 decimal points
            line_contents = f'{packed_time:010.3f}'
            hour[i] = np.float64(line_contents[:2])
            minute[i] = np.float64(line_contents[2:4])
            second[i] = np.float64(line_contents[4:])
        #-- close the input HDF5 file
        fileID.close()
    #-- leap seconds for converting from GPS time to UTC
    S = calc_GPS_to_UTC(year,month,day,hour,minute,second)
    #-- calculation of Julian day taking into account leap seconds
    JD = calc_julian_day(year,month,day,HOUR=hour,MINUTE=minute,SECOND=second-S)
    #-- converting to J2000 seconds
    ATM_L1b_input['time'] = (JD - 2451545.0)*86400.0
    #-- subset the data to indices if specified
    if input_subsetter:
        for key,val in ATM_L1b_input.items():
            ATM_L1b_input[key] = val[input_subsetter]
    #-- hemispheric shot count
    count = {}
    count['N'] = np.count_nonzero(ATM_L1b_input['lat'] >= 0.0)
    count['S'] = np.count_nonzero(ATM_L1b_input['lat'] < 0.0)
    #-- determine hemisphere with containing shots in file
    HEM, = [key for key, val in count.items() if val]
    #-- return the output variables
    return ATM_L1b_input,file_lines,HEM

#-- PURPOSE: read the ATM Level-2 data file for variables of interest
def read_ATM_icessn_file(input_file, input_subsetter):
    #-- regular expression pattern for extracting parameters
    regex_pattern = r'(BLATM2|ILATM2)_(\d+)_(\d+)_smooth_nadir(.*?)(csv|seg|pt)$'
    #-- extract mission and other parameters from filename
    MISSION,YYMMDD,HHMMSS,AUX,SFX = re.findall(regex_pattern,input_file).pop()
    #-- early date strings omitted century and millenia (e.g. 93 for 1993)
    if (len(YYMMDD) == 6):
        yr2d,month,day = np.array([YYMMDD[:2],YYMMDD[2:4],YYMMDD[4:]],dtype='i')
        year = (yr2d + 1900.0) if (yr2d >= 90) else (yr2d + 2000.0)
    elif (len(YYMMDD) == 8):
        year,month,day = np.array([YYMMDD[:4],YYMMDD[4:6],YYMMDD[6:]],dtype='i')
    #-- input file column names for variables of interest with column indices
    #-- variables not used: (SNslope:4, WEslope:5, npt_used:7, npt_edit:8, d:9)
    file_dtype = {'seconds':0, 'lat':1, 'lon':2, 'data':3, 'RMS':6, 'track':-1}
    #-- compile regular expression operator for reading lines (extracts numbers)
    regex_pattern = r'[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?'
    rx = re.compile(regex_pattern, re.VERBOSE)
    #-- read the input file, split at lines and remove all commented lines
    with open(input_file, mode='r', encoding='utf8') as f:
        file_contents=[i for i in f.readlines() if re.match(r'^(?!\#|\n)',i)]
    #-- number of lines of data within file
    file_lines = file_length(input_file,input_subsetter)
    #-- output python dictionary with variables
    ATM_L2_input = {}
    #-- create output variables with length equal to the number of file lines
    for key in file_dtype.keys():
        ATM_L2_input[key] = np.zeros_like(file_contents, dtype=np.float64)
    #-- for each line within the file
    for line_number,line_entries in enumerate(file_contents):
        #-- find numerical instances within the line
        line_contents = rx.findall(line_entries)
        #-- for each variable of interest: save to dinput as float
        for key,val in file_dtype.items():
            ATM_L2_input[key][line_number] = np.float64(line_contents[val])
    #-- convert shot time (seconds of day) to J2000
    hour = np.floor(ATM_L2_input['seconds']/3600.0)
    minute = np.floor((ATM_L2_input['seconds'] % 3600)/60.0)
    second = ATM_L2_input['seconds'] % 60.0
    #-- First column in Pre-IceBridge and ICESSN Version 1 files is GPS time
    if (MISSION == 'BLATM2') or (SFX != 'csv'):
        #-- leap seconds for converting from GPS time to UTC
        S = calc_GPS_to_UTC(year,month,day,hour,minute,second)
    else:
        S = 0.0
    #-- calculation of Julian day
    JD = calc_julian_day(year,month,day,HOUR=hour,MINUTE=minute,SECOND=second-S)
    #-- converting to J2000 seconds
    ATM_L2_input['time'] = (JD - 2451545.0)*86400.0
    #-- convert RMS from centimeters to meters
    ATM_L2_input['error'] = ATM_L2_input['RMS']/100.0
    #-- subset the data to indices if specified
    if input_subsetter:
        for key,val in ATM_L2_input.items():
            ATM_L2_input[key] = val[input_subsetter]
    #-- hemispheric shot count
    count = {}
    count['N'] = np.count_nonzero(ATM_L2_input['lat'] >= 0.0)
    count['S'] = np.count_nonzero(ATM_L2_input['lat'] < 0.0)
    #-- determine hemisphere with containing shots in file
    HEM, = [key for key, val in count.items() if val]
    #-- return the output variables
    return ATM_L2_input,file_lines,HEM

#-- PURPOSE: read the LVIS Level-2 data file for variables of interest
def read_LVIS_HDF5_file(input_file, input_subsetter):
    #-- LVIS region flags: GL for Greenland and AQ for Antarctica
    lvis_flag = {'GL':'N','AQ':'S'}
    #-- regular expression pattern for extracting parameters from HDF5 files
    #-- computed in read_icebridge_lvis.py
    mission_flag = r'(BLVIS2|BVLIS2|ILVIS2|ILVGH2)'
    regex_pattern = rf'{mission_flag}_(.*?)(\d+)_(\d+)_(R\d+)_(\d+).H5'
    #-- extract mission, region and other parameters from filename
    MISSION,REGION,YY,MMDD,RLD,SS = re.findall(regex_pattern,input_file).pop()
    LDS_VERSION = '2.0.2' if (np.int64(RLD[1:3]) >= 18) else '1.04'
    #-- input and output python dictionaries with variables
    file_input = {}
    LVIS_L2_input = {}
    fileID = h5py.File(input_file,'r')
    #-- create output variables with length equal to input shot number
    file_lines = file_length(input_file,input_subsetter,HDF5='Shot_Number')
    #-- https://lvis.gsfc.nasa.gov/Data/Data_Structure/DataStructure_LDS104.html
    #-- https://lvis.gsfc.nasa.gov/Data/Data_Structure/DataStructure_LDS202.html
    if (LDS_VERSION == '1.04'):
        #-- elevation surfaces
        file_input['elev'] = fileID['Elevation_Surfaces/Elevation_Centroid'][:]
        file_input['elev_low'] = fileID['Elevation_Surfaces/Elevation_Low'][:]
        file_input['elev_high'] = fileID['Elevation_Surfaces/Elevation_High'][:]
        #-- latitude
        file_input['lat'] = fileID['Geolocation/Latitude_Centroid'][:]
        file_input['lat_low'] = fileID['Geolocation/Latitude_Low'][:]
        #-- longitude
        file_input['lon'] = fileID['Geolocation/Longitude_Centroid'][:]
        file_input['lon_low'] = fileID['Geolocation/Longitude_Low'][:]
    elif (LDS_VERSION == '2.0.2'):
        #-- elevation surfaces
        file_input['elev_low'] = fileID['Elevation_Surfaces/Elevation_Low'][:]
        file_input['elev_high'] = fileID['Elevation_Surfaces/Elevation_High'][:]
        #-- heights above lowest detected mode
        file_input['RH50'] = fileID['Waveform/RH50'][:]
        file_input['RH100'] = fileID['Waveform/RH100'][:]
        #-- calculate centroidal elevation using 50% of waveform energy
        file_input['elev'] = file_input['elev_low'] + file_input['RH50']
        #-- latitude
        file_input['lat_top'] = fileID['Geolocation/Latitude_Top'][:]
        file_input['lat_low'] = fileID['Geolocation/Latitude_Low'][:]
        #-- longitude
        file_input['lon_top'] = fileID['Geolocation/Longitude_Top'][:]
        file_input['lon_low'] = fileID['Geolocation/Longitude_Low'][:]
        #-- linearly interpolate latitude and longitude to RH50
        file_input['lat'] = file_input['lat_low'] + file_input['RH50'] * \
            (file_input['lat_top'] - file_input['lat_low'])/file_input['RH100']
        file_input['lon'] = file_input['lon_low'] + file_input['RH50'] * \
            (file_input['lon_top'] - file_input['lon_low'])/file_input['RH100']
    #-- J2000 seconds
    LVIS_L2_input['time'] = fileID['Time/J2000'][:]
    #-- close the input HDF5 file
    fileID.close()
    #-- output combined variables
    LVIS_L2_input['data'] = np.zeros_like(file_input['elev'],dtype=np.float64)
    LVIS_L2_input['lon'] = np.zeros_like(file_input['elev'],dtype=np.float64)
    LVIS_L2_input['lat'] = np.zeros_like(file_input['elev'],dtype=np.float64)
    LVIS_L2_input['error'] = np.zeros_like(file_input['elev'],dtype=np.float64)
    #-- find where elev high is equal to elev low
    #-- see note about using LVIS centroid elevation product
    #-- http://lvis.gsfc.nasa.gov/OIBDataStructure.html
    ii = np.nonzero(file_input['elev_low'] == file_input['elev_high'])
    jj = np.nonzero(file_input['elev_low'] != file_input['elev_high'])
    #-- where lowest point of waveform is equal to highest point -->
    #-- using the elev_low elevation
    LVIS_L2_input['data'][ii] = file_input['elev_low'][ii]
    #-- for other locations use the centroid elevation
    #-- as the centroid is a useful product over rough terrain
    #-- when you are calculating ice volume change
    LVIS_L2_input['data'][jj] = file_input['elev'][jj]
    #-- latitude and longitude for each case
    #-- elevation low == elevation high
    LVIS_L2_input['lon'][ii] = file_input['lon_low'][ii]
    LVIS_L2_input['lat'][ii] = file_input['lat_low'][ii]
    #-- centroid elevations
    LVIS_L2_input['lon'][jj] = file_input['lon'][jj]
    LVIS_L2_input['lat'][jj] = file_input['lat'][jj]
    #-- estimated uncertainty for both cases
    LVIS_variance_low = (file_input['elev_low'] - file_input['elev'])**2
    LVIS_variance_high = (file_input['elev_high'] - file_input['elev'])**2
    LVIS_L2_input['error']=np.sqrt((LVIS_variance_low + LVIS_variance_high)/2.0)
    #-- subset the data to indices if specified
    if input_subsetter:
        for key,val in LVIS_L2_input.items():
            LVIS_L2_input[key] = val[input_subsetter]
    #-- return the output variables
    return LVIS_L2_input,file_lines,lvis_flag[REGION]

#-- PURPOSE: calculate the Julian day from calendar date
#-- http://scienceworld.wolfram.com/astronomy/JulianDate.html
def calc_julian_day(YEAR, MONTH, DAY, HOUR=0, MINUTE=0, SECOND=0):
    JD = 367.*YEAR - np.floor(7.*(YEAR + np.floor((MONTH+9.)/12.))/4.) - \
        np.floor(3.*(np.floor((YEAR + (MONTH - 9.)/7.)/100.) + 1.)/4.) + \
        np.floor(275.*MONTH/9.) + DAY + 1721028.5 + HOUR/24. + MINUTE/1440. + \
        SECOND/86400.
    return np.array(JD,dtype=np.float64)

#-- PURPOSE: calculate the number of leap seconds between GPS time (seconds
#-- since Jan 6, 1980 00:00:00) and UTC
def calc_GPS_to_UTC(YEAR, MONTH, DAY, HOUR, MINUTE, SECOND):
    GPS = 367.*YEAR - np.floor(7.*(YEAR + np.floor((MONTH+9.)/12.))/4.) - \
        np.floor(3.*(np.floor((YEAR + (MONTH - 9.)/7.)/100.) + 1.)/4.) + \
        np.floor(275.*MONTH/9.) + DAY + 1721028.5 - 2444244.5
    GPS_Time = GPS*86400.0 + HOUR*3600.0 + MINUTE*60.0 + SECOND
    return icesat2_toolkit.time.count_leap_seconds(GPS_Time)

#-- Module for outputting the individual interpolated elevations
def main():
    #-- start MPI communicator
    comm = MPI.COMM_WORLD

    #-- Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()
    #-- value to replace invalid triangulation points
    FILL_VALUE = -9999.0

    #-- create logger
    loglevel = logging.INFO if args.verbose else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    #-- list of input files for triangulation (tilde-expand paths)
    #-- first file listed contains the data to be compared with
    #-- all secondary files contain the data to be triangulated
    input_files = []
    #-- list of indices for subsetting input datasets
    input_subsetter = []
    #-- for each input argument (input file)
    for arg in args.infile:
        #-- extract file name and subsetter indices lists
        match_object = re.match(r'(.*?)(\[(.*?)\])?$',arg)
        input_files.append(os.path.expanduser(match_object.group(1)))
        #-- subset triangulated files to indices
        if match_object.group(2):
            #-- decompress ranges and add to list
            file_indices = []
            for i in re.findall(r'((\d+)-(\d+)|(\d+))',match_object.group(3)):
                file_indices.append(int(i[3])) if i[3] else \
                    file_indices.extend(range(int(i[1]),int(i[2])+1))
            input_subsetter.append(file_indices)
        else:
            input_subsetter.append(None)

    #-- output module information for process
    logging.info(input_files[0]) if (comm.rank == 0) else None
    info(comm.rank,comm.size)

    #-- output directory for input_files[0]
    icebridge_dir = os.path.dirname(input_files[0])
    #-- calculate if input files are from ATM or LVIS (+GH)
    regex = {}
    regex['ATM'] = r'(BLATM2|ILATM2)_(\d+)_(\d+)_smooth_nadir(.*?)(csv|seg|pt)$'
    regex['ATM1b'] = r'(BLATM1b|ILATM1b)_(\d+)_(\d+)(.*?).(qi|TXT|h5)$'
    regex['LVIS'] = r'(BLVIS2|BVLIS2|ILVIS2)_(.*?)(\d+)_(\d+)_(R\d+)_(\d+).H5$'
    regex['LVGH'] = r'(ILVGH2)_(.*?)(\d+)_(\d+)_(R\d+)_(\d+).H5$'
    #-- output file formats
    file_format = '{0}_NASA_{1}_WGS84_{2}{3}{4}{5}{6:05.0f}-{7}{8}{9}{10}.H5'

    for key,val in regex.items():
        if re.match(val, os.path.basename(input_files[0])):
            OIB1 = key
        if re.match(val, os.path.basename(input_files[1])):
            OIB2 = key

    #-- extract information from first input file
    #-- acquisition year, month and day
    #-- number of points
    #-- mission (PRE-OIB ATM or LVIS, OIB ATM or LVIS)
    if OIB1 in ('ATM','ATM1b'):
        M1,YYMMDD1,HHMMSS1,AX1,SF1=re.findall(regex[OIB1], input_files[0]).pop()
        #-- check if file is csv/ascii, binary (qi) or HDF5 format
        if (SF1 == 'h5'):
            n_1 = file_length(input_files[0],None,HDF5='elevation')
        elif (SF1 == 'qi'):
            n_1 = file_length(input_files[0],None,QFIT=True)
        else:
            n_1 = file_length(input_files[0],None)
        #-- early date strings omitted century and millenia (e.g. 93 for 1993)
        if (len(YYMMDD1) == 6):
            SD1 = os.path.basename(os.path.dirname(input_files[0]))
            YY1,MM1,DD1 = re.findall('(\d+)\.(\d+)\.(\d+)',SD1).pop()
        elif (len(YYMMDD1) == 8):
            YY1,MM1,DD1 = YYMMDD1[:4],YYMMDD1[4:6],YYMMDD1[6:]
    elif OIB1 in ('LVIS','LVGH'):
        n_1 = file_length(input_files[0],None,HDF5='Shot_Number')
        M1,RG1,YY1,MMDD1,RLD1,SS1=re.findall(regex[OIB1], input_files[0]).pop()
        MM1,DD1 = MMDD1[:2],MMDD1[2:]

    #-- if there are no data points in the input file (primary)
    if (n_1 == 0):
        file1 = os.path.basename(input_files[0])
        raise ValueError(f'No data points found in {file1}')

    #-- extract information from second set of input files
    #-- acquisition year, month and day
    #-- total number of points
    #-- mission (PRE-OIB ATM or LVIS, OIB ATM or LVIS)
    if OIB2 in ('ATM','ATM1b'):
        M2,YYMMDD2,HHMMSS2,AX2,SF2=re.findall(regex[OIB2], input_files[1]).pop()
        #-- check if file is csv/ascii, binary (qi) or HDF5 format
        if (SF2 == 'h5'):
            n_2 = np.sum([file_length(f,input_subsetter[i+1],HDF5='elevation')
                for i,f in enumerate(input_files[1:])])
        elif (SF2 == 'qi'):
            n_2 = np.sum([file_length(f,input_subsetter[i+1],QFIT=True)
                for i,f in enumerate(input_files[1:])])
        else:
            n_2 = np.sum([file_length(f,input_subsetter[i+1])
                for i,f in enumerate(input_files[1:])])
        #-- early date strings omitted century and millenia (e.g. 93 for 1993)
        if (len(YYMMDD2) == 6):
            SD2 = os.path.basename(os.path.dirname(input_files[1]))
            YY2,MM2,DD2 = re.findall('(\d+)\.(\d+)\.(\d+)',SD2).pop()
        elif (len(YYMMDD2) == 8):
            YY2,MM2,DD2 = YYMMDD2[:4],YYMMDD2[4:6],YYMMDD2[6:]
    elif OIB2 in ('LVIS','LVGH'):
        n_2=np.sum([file_length(f,input_subsetter[i+1],HDF5='Shot_Number')
            for i,f in enumerate(input_files[1:])])
        M2,RG2,YY2,MMDD2,RLD2,SS2=re.findall(regex[OIB2], input_files[1]).pop()
        MM2,DD2 = MMDD2[:2],MMDD2[2:]

    #-- if there are no data points in the input files to be triangulated:
    #-- end program with error for set of input files
    if (n_2 == 0):
        file2 = ','.join([os.path.basename(f) for f in input_files[1:]])
        raise ValueError(f'No data points found in {file2}')

    #-- read all input data on rank 0 (parent process)
    if (comm.rank == 0):
        #-- read data from input_files[0] (data to be compared with)
        if (OIB1 == 'ATM') and (n_1 > 0):
            #-- load IceBridge ATM data from input_files[0]
            dinput1,file_lines,HEM = read_ATM_icessn_file(input_files[0],None)
        elif (OIB1 == 'ATM1b') and (n_1 > 0):
            #-- load IceBridge Level-1b ATM data from input_files[0]
            dinput1,file_lines,HEM = read_ATM_qfit_file(input_files[0],None)
        elif OIB1 in ('LVIS','LVGH') and (n_1 > 0):
            #-- load IceBridge LVIS data from input_files[0]
            dinput1,file_lines,HEM = read_LVIS_HDF5_file(input_files[0],None)

        #-- create dictionary for input data for files to be triangulated
        dinput2 = dict(data=np.zeros((n_2)), time=np.zeros((n_2)),
            error=np.zeros((n_2)), lon=np.zeros((n_2)), lat=np.zeros((n_2)))
        #-- counter variable for filling arrays
        c = 0
        #-- read data from input_files[1:] and combine into single array
        for fi,s in zip(input_files[1:],input_subsetter[1:]):
            if (OIB2 == 'ATM') and (n_2 > 0):
                #-- load IceBridge ATM data from fi
                file_input,file_lines,HEM2 = read_ATM_icessn_file(fi,s)
            elif (OIB2 == 'ATM1b') and (n_2 > 0):
                #-- load IceBridge Level-1b ATM data from fi
                file_input,file_lines,HEM2 = read_ATM_qfit_file(fi,s)
            elif OIB2 in ('LVIS','LVGH') and (n_2 > 0):
                #-- load IceBridge LVIS data from fi
                file_input,file_lines,HEM2 = read_LVIS_HDF5_file(fi,s)
            #-- iterate through input keys of iterest
            for key in ['data','lon','lat','time','error']:
                dinput2[key][c:c+file_lines] = file_input.get(key,None)
            #-- add file lines to counter
            c += file_lines
        #-- check that hemispheres are matching for both files
        #-- if not: exist with error for input files
        if (HEM != HEM2):
            raise RuntimeError(f'Hemisphere Mismatch ({HEM} {HEM2})')

        #-- pyproj transformer for converting lat/lon to polar stereographic
        EPSG = dict(N=3413,S=3031)
        crs1 = pyproj.CRS.from_epsg(4326)
        crs2 = pyproj.CRS.from_epsg(EPSG[HEM])
        transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
        #-- convert from latitude/longitude to polar stereographic coordinates
        X1,Y1 = transformer.transform(dinput1['lon'], dinput1['lat'])
        X2,Y2 = transformer.transform(dinput2['lon'], dinput2['lat'])
    else:
        #-- create dictionaries for input data to be broadcast from rank 0
        dinput2 = dict(data=np.empty((n_2)), time=np.empty((n_2)),
            error=np.empty((n_2)))
        #-- coordinates of input and triangulated data
        X1 = np.empty((n_1))
        Y1 = np.empty((n_1))
        X2 = np.empty((n_2))
        Y2 = np.empty((n_2))

    #-- Broadcast OIB1 data from rank 0 to all other ranks
    comm.Bcast([X1, MPI.DOUBLE])
    comm.Bcast([Y1, MPI.DOUBLE])

    #-- Broadcast OIB2 data from rank 0 to all other ranks
    comm.Bcast([X2, MPI.DOUBLE])
    comm.Bcast([Y2, MPI.DOUBLE])
    comm.Bcast([dinput2['data'], MPI.DOUBLE])
    comm.Bcast([dinput2['time'], MPI.DOUBLE])
    #-- data error for ATM/LVIS, estimated error for ATM1b
    if OIB2 in ('ATM1b',):
        dinput2['error'] = None
    else:
        comm.Bcast([dinput2['error'], MPI.DOUBLE])

    #-- number of points to iterate through in file 1
    if input_subsetter[0]:
        #-- using a subset of points from the original file
        indices = input_subsetter[0]
        iteration_count = len(input_subsetter[0])
    else:
        #-- using all data points in the original file
        indices = np.arange(n_1)
        iteration_count = n_1

    #-- allocate for interpolated elevation, error and slopes
    distributed = {}
    distributed['data'] = np.zeros((n_1))
    distributed['error'] = np.zeros((n_1))
    distributed['time'] = np.zeros((n_1))
    distributed['x_slope'] = np.zeros((n_1))
    distributed['y_slope'] = np.zeros((n_1))
    #-- mask determining valid triangulation
    distributed['mask'] = np.zeros((n_1), dtype=bool)
    #-- run for each input point (distributed over comm.size # of processes)
    for iteration in range(comm.rank, iteration_count, comm.size):
        #-- indice for iteration (can run through a subset of points)
        i = indices[iteration]
        #-- calculate interpolated elevation, error and time
        #-- calculate x and y slope parameters
        elev_interp = triangulate_elevation(X2, Y2, dinput2['data'],
            dinput2['time'], X1[i], Y1[i], args.distance,
            RMS=dinput2['error'], ANGLE=args.angle)
        #-- check output flag to determine if valid triangulation was found
        if elev_interp['flag']:
            distributed['data'][i] = elev_interp['data']
            distributed['error'][i] = elev_interp['error']
            distributed['time'][i] = elev_interp['time']
            distributed['x_slope'][i] = elev_interp['slope'][0]
            distributed['y_slope'][i] = elev_interp['slope'][1]
            distributed['mask'][i] = True

    #-- create matrices for associated data output
    associated = {}
    associated['data'] = np.zeros((n_1))
    associated['time'] = np.zeros((n_1))
    associated['error'] = np.zeros((n_1))
    associated['x_slope'] = np.zeros((n_1))
    associated['y_slope'] = np.zeros((n_1))
    associated['mask'] = np.zeros((n_1),dtype=bool)
    #-- communicate output MPI matrices to zero rank
    #-- operation is a element summation.
    comm.Reduce(sendbuf=[distributed['data'], MPI.DOUBLE], \
        recvbuf=[associated['data'], MPI.DOUBLE], \
        op=MPI.SUM, root=0)
    comm.Reduce(sendbuf=[distributed['error'], MPI.DOUBLE], \
        recvbuf=[associated['error'], MPI.DOUBLE], \
        op=MPI.SUM, root=0)
    comm.Reduce(sendbuf=[distributed['time'], MPI.DOUBLE], \
        recvbuf=[associated['time'], MPI.DOUBLE], \
        op=MPI.SUM, root=0)
    comm.Reduce(sendbuf=[distributed['x_slope'], MPI.DOUBLE], \
        recvbuf=[associated['x_slope'], MPI.DOUBLE], \
        op=MPI.SUM, root=0)
    comm.Reduce(sendbuf=[distributed['y_slope'], MPI.DOUBLE], \
        recvbuf=[associated['y_slope'], MPI.DOUBLE], \
        op=MPI.SUM, root=0)
    #-- operation is a logical "or" across the elements.
    comm.Reduce(sendbuf=[distributed['mask'], MPI.BOOL], \
        recvbuf=[associated['mask'], MPI.BOOL], \
        op=MPI.LOR, root=0)

    #-- wait for all distributed processes to finish
    comm.Barrier()

    #-- output data if MPI rank is 0 (parent process)
    if (comm.rank == 0):
        #-- if there are valid points
        if associated['mask'].any():
            #-- create mask to determine valid points
            #-- find indices of invalid points
            invalid_indices, = np.nonzero(~associated['mask'])
            #-- create combined dictionary for output variables
            #-- replacing invalid indices with FILL_VALUE
            combined = {}
            for key in ['data','error','time','x_slope','y_slope']:
                #-- replace zeros in associated with fill values
                combined[key] = associated[key].copy()
                combined[key][invalid_indices] = FILL_VALUE
            #-- add coordinates to output dictionary from data input
            combined['lon'] = dinput1['lon'].copy()
            combined['lat'] = dinput1['lat'].copy()
            #-- output region flags: GR for Greenland and AN for Antarctica
            hem_flag = {'N':'GR','S':'AN'}
            #-- output HDF5 file with triangulated ATM/LVIS data
            #-- use starting second to distinguish between files for the day
            JJ1 = np.min(dinput1['time']) % 86400
            #-- rg_NASA_TRIANGULATED_WGS84_fl1yyyymmddjjjjj-fl2yyyymmdd.H5
            #-- where rg is the hemisphere flag (GR or AN) for the region
            #-- fl1 and fl2 are the data flags (ATM, LVIS)
            #-- yymmddjjjjj is the year, month, day and second of input file
            #-- yymmdd is the year, month and day of the triangulated files
            FILE = file_format.format(hem_flag[HEM],'TRIANGULATED',OIB1,
                YY1,MM1,DD1,JJ1,OIB2,YY2,MM2,DD2)
            HDF5_triangulated_data(combined, MISSION=M2, INPUT=input_files,
                FILENAME=os.path.join(icebridge_dir,FILE),
                FILL_VALUE=FILL_VALUE, CLOBBER=True)
            #-- print file name if verbose output is specified
            logging.info(os.path.join(icebridge_dir,FILE))
            #-- change the permissions level to MODE
            os.chmod(os.path.join(icebridge_dir,FILE), args.mode)

#-- PURPOSE: triangulate elevation points to xpt and ypt
def triangulate_elevation(X, Y, H, T, xpt, ypt, DISTANCE, RMS=None, ANGLE=120.):
    #-- find points where the distance is less than DISTANCE (in meters)
    #-- and the points are not at the same coordinates
    dd = (xpt-X)**2 + (ypt-Y)**2
    indices, = np.nonzero(dd <= (DISTANCE**2))
    valid_indices = find_valid_indices(indices, X[indices], Y[indices])
    #-- create valid triangulation flag
    Flag = False
    #-- if there are only 3 non-duplicate points within DISTANCE in meters
    #-- create a triangle with points, and check if (xpt,ypt) is within triangle
    #-- if so: check that triangle is valid (distance and maximum angle)
    #-- and calculate equivalent elevation and slope at (xpt,ypt)
    if (valid_indices['count'] == 3):
        #-- never attempted a Delaunay triangulation
        delaunay_attempt = 0
        #-- indices for points within the distance
        ind = valid_indices['indices']
        x0 = X[ind]
        y0 = Y[ind]
        h0 = H[ind]
        t0 = T[ind]
        #-- for ATM points: use RMS from icessn fit
        #-- for LVIS points: use Gaussian-Centroid
        if RMS is not None:
            r0 = RMS[ind]
        #-- verify that original point is within triangle
        if icesat2_toolkit.spatial.inside_polygon(xpt, ypt, x0, y0):
            #-- vertices for triangle
            #-- x coordinates for each vertice
            x1 = x0[0]
            x2 = x0[1]
            x3 = x0[2]
            #-- y coordinates for each vertice
            y1 = y0[0]
            y2 = y0[1]
            y3 = y0[2]
            #-- calculate maximum angle within triangle
            #-- using 120 degrees as maximum criterion as default
            #-- http://imr.sandia.gov/papers/abstracts/Hi79.html
            angle_max = triangle_maximum_angle(x1, y1, x2, y2, x3, y3)
            #-- verify that triangle vertices are within the specified distance
            #-- of each other and triangle is not overly obtuse
            if ((x1-x2)**2+(y1-y2)**2 <= (DISTANCE**2)) & \
                ((x2-x3)**2+(y2-y3)**2 <= (DISTANCE**2)) & \
                ((x3-x1)**2+(y3-y1)**2 <= (DISTANCE**2)) & (angle_max < ANGLE):
                #-- set valid flag as True
                Flag = True
                #-- use cross-product method to calculate surface slopes
                V1 = np.array([x2-x1, y2-y1, h0[1]-h0[0]])
                V2 = np.array([x3-x2, y3-y2, h0[2]-h0[1]])
                V_norm = np.cross(V1, V2)#-- Normal Vector
                #-- slopes = differentials of plane formula coefficients
                x_slope = -V_norm[0]/V_norm[2]
                y_slope = -V_norm[1]/V_norm[2]
                #-- use Barycentric interpolation for triangle
                #-- first calculate area of the triangle
                area = 0.5*np.abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                #-- now calculate areas of 3 triangles connecting to point
                #-- area_1 is the area of the triangle connecting
                #-- point 2 and point 3 with the original point
                #-- (replace x1 and y1 with xpt and ypt in area equation)
                #-- the larger the area: the closer the point is to point 1
                area_1 = 0.5*np.abs(xpt*(y2-y3) + x2*(y3-ypt) + x3*(ypt-y2))
                area_2 = 0.5*np.abs(x1*(ypt-y3) + xpt*(y3-y1) + x3*(y1-ypt))
                area_3 = 0.5*np.abs(x1*(y2-ypt) + x2*(ypt-y1) + xpt*(y1-y2))
                #-- calculate interpolated elevation point, time and error
                #-- hpt = area_1*h1 + area_2*h2 + area_3*h3
                #-- NOTE: area_1 + area_2 + area_3 = area
                elev = (area_1*h0[0] + area_2*h0[1] + \
                    area_3*h0[2])/(area_1 + area_2 + area_3)
                time = (area_1*t0[0] + area_2*t0[1] + \
                    area_3*t0[2])/(area_1 + area_2 + area_3)
                if RMS is not None:
                    #-- for ATM data: use the icessn RMS values
                    #-- for LVIS data: use Gaussian - Centroid
                    total_error = 0.0
                    total_error += (area_1*r0[0]/area)**2
                    total_error += (area_2*r0[1]/area)**2
                    total_error += (area_3*r0[2]/area)**2
                    error = np.sqrt(total_error)
                else:
                    #-- for ATM1b data: use the variance off the mean
                    total_variance = 0.0
                    total_variance += (h0[0] - elev)**2
                    total_variance += (h0[1] - elev)**2
                    total_variance += (h0[2] - elev)**2
                    error = np.sqrt(total_variance/3.0)

    #-- if there are more than 3 non-duplicate points within DISTANCE in meters
    #-- create Delaunay triangulation with points
    #-- for each triangle in the mesh: check if (xpt,ypt) is within a triangle
    #-- if so: check that found triangle is valid (distance and maximum angle)
    #-- and calculate equivalent elevation and slope at (xpt,ypt)
    elif (valid_indices['count'] >= 4):
        #-- indices for points within the distance
        ind = valid_indices['indices']
        x0 = X[ind]
        y0 = Y[ind]
        h0 = H[ind]
        t0 = T[ind]
        #-- for ATM points: use RMS from icessn fit
        #-- for LVIS points: use Gaussian-Centroid
        if RMS is not None:
            r0 = RMS[ind]

        #-- calculate Delaunay triangulation with points
        delaunay_attempt,delaunay_vertices = find_valid_triangulation(x0, y0)

        #-- for each Delaunay triangle (vertices is indices for each vertex)
        for vert in delaunay_vertices:
            #-- verify that original point is within triangle
            #-- if not will go to next set of vertices
            if icesat2_toolkit.spatial.inside_polygon(xpt, ypt, x0[vert], y0[vert]):
                #-- vertices for triangle
                #-- x coordinates for each vertice
                x1 = x0[vert[0]]
                x2 = x0[vert[1]]
                x3 = x0[vert[2]]
                #-- y coordinates for each vertice
                y1 = y0[vert[0]]
                y2 = y0[vert[1]]
                y3 = y0[vert[2]]
                #-- calculate maximum angle within triangle
                #-- using 120 degrees as maximum criterion as default
                #-- http://imr.sandia.gov/papers/abstracts/Hi79.html
                angle_max = triangle_maximum_angle(x1, y1, x2, y2, x3, y3)
                #-- verify that triangle vertices are within the specified
                #-- distance of each other and triangle is not overly obtuse
                if ((x1-x2)**2+(y1-y2)**2 <= (DISTANCE**2)) & \
                    ((x2-x3)**2+(y2-y3)**2 <= (DISTANCE**2)) & \
                    ((x3-x1)**2+(y3-y1)**2 <= (DISTANCE**2)) & \
                    (angle_max < ANGLE):
                    #-- set valid flag as True
                    Flag = True
                    #-- use cross-product method to calculate surface slopes
                    V1 = np.array([x2-x1, y2-y1, h0[vert[1]]-h0[vert[0]]])
                    V2 = np.array([x3-x2, y3-y2, h0[vert[2]]-h0[vert[1]]])
                    V_norm = np.cross(V1, V2)#-- Normal Vector
                    #-- slopes = differentials of plane formula coefficients
                    x_slope = -V_norm[0]/V_norm[2]
                    y_slope = -V_norm[1]/V_norm[2]
                    #-- use Barycentric interpolation for triangle
                    #-- first calculate area of the triangle
                    area = 0.5*np.abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    #-- now calculate areas of 3 triangles connecting to point
                    #-- area_1 is the area of the triangle connecting
                    #-- point 2 and point 3 with the original point
                    #-- (replace x1 and y1 with xpt and ypt in area equation)
                    #-- the larger the area: the closer the point is to point 1
                    area_1 = 0.5*np.abs(xpt*(y2-y3) + x2*(y3-ypt) + x3*(ypt-y2))
                    area_2 = 0.5*np.abs(x1*(ypt-y3) + xpt*(y3-y1) + x3*(y1-ypt))
                    area_3 = 0.5*np.abs(x1*(y2-ypt) + x2*(ypt-y1) + xpt*(y1-y2))
                    #-- calculate interpolated elevation point, time and error
                    #-- hpt = area_1*h1 + area_2*h2 + area_3*h3
                    #-- NOTE: area_1 + area_2 + area_3 = area
                    elev = (area_1*h0[vert[0]] + area_2*h0[vert[1]] + \
                        area_3*h0[vert[2]])/(area_1 + area_2 + area_3)
                    time = (area_1*t0[vert[0]] + area_2*t0[vert[1]] + \
                        area_3*t0[vert[2]])/(area_1 + area_2 + area_3)
                    if RMS is not None:
                        #-- for ATM data: use the icessn RMS values
                        #-- for LVIS data: use Gaussian - Centroid
                        total_error = 0.0
                        total_error += (area_1*r0[vert[0]]/area)**2
                        total_error += (area_2*r0[vert[1]]/area)**2
                        total_error += (area_3*r0[vert[2]]/area)**2
                        error = np.sqrt(total_error)
                    else:
                        #-- for ATM1b data: use the variance off the mean
                        total_variance = 0.0
                        total_variance += (h0[vert[0]] - elev)**2
                        total_variance += (h0[vert[1]] - elev)**2
                        total_variance += (h0[vert[2]] - elev)**2
                        error = np.sqrt(total_variance/3.0)

    #-- if there are NOT more than 3 points within DISTANCE in meters
    #-- or if a triangulation cannot be made with the points within range
    #-- or a triangulation cannot be found with the point inside
    #-- or if triangle housing the point is invalid (distance or angle)
    if not Flag:
        #-- output elevation, error, time and slope variables as nan
        elev = np.nan
        error = np.nan
        time = np.nan
        x_slope = np.nan
        y_slope = np.nan
        delaunay_attempt = None

    #-- return data, error, time, slopes and valid flag variables
    return {'data':elev, 'error':error, 'time': time, 'slope':[x_slope,y_slope],
        'flag':Flag, 'attempt':delaunay_attempt}

#-- PURPOSE: find indices of valid points (no duplicates)
def find_valid_indices(indices, X, Y):
    #-- create dictionary with valid count and valid indices
    valid = {}
    valid['count'] = 0
    valid['indices'] = []
    #-- create list objects for X and Y coordinates of valid points
    x0 = []
    y0 = []
    for i, val in enumerate(indices):
        #-- if the first indice: add to arrays
        if (i == 0):
            valid['count'] += 1
            valid['indices'].append(val)
            x0.append(X[i])
            y0.append(Y[i])
        else:
            #-- for all other indices
            #-- find distance between point and all other points
            dist = (Y[i]-np.array(y0))**2 + (X[i]-np.array(x0))**2
            #-- if distance is greater than 0 for all points
            if (dist > 0.0).all():
                valid['count'] += 1
                valid['indices'].append(val)
                x0.append(X[i])
                y0.append(Y[i])
    #-- return the valid indices dictionary
    return valid

#-- PURPOSE: find a valid Delaunay triangulation for coordinates x0 and y0
#-- http://www.qhull.org/html/qhull.htm#options
#-- Attempt 1: standard qhull options Qt Qbb Qc Qz
#-- Attempt 2: rescale and center the inputs with option QbB
#-- Attempt 3: joggle the inputs to find a triangulation with option QJ
#-- if no passing triangulations: exit with empty list
def find_valid_triangulation(x0, y0):
    """
    Attempt to find a valid Delaunay triangulation for coordinates

    - Attempt 1: ``Qt Qbb Qc Qz``
    - Attempt 2: ``Qt Qc QbB``
    - Attempt 3: ``QJ QbB``

    Parameters
    ----------
    x0: float
        x-coordinates
    y0: float
        y-coordinates
    """
    #-- Attempt 1: try with standard options Qt Qbb Qc Qz
    #-- Qt: triangulated output, all facets will be simplicial
    #-- Qbb: scale last coordinate to [0,m] for Delaunay triangulations
    #-- Qc: keep coplanar points with nearest facet
    #-- Qz: add point-at-infinity to Delaunay triangulation

    #-- Attempt 2 in case of qhull error from Attempt 1 try Qt Qc QbB
    #-- Qt: triangulated output, all facets will be simplicial
    #-- Qc: keep coplanar points with nearest facet
    #-- QbB: scale input to unit cube centered at the origin

    #-- Attempt 3 in case of qhull error from Attempt 2 try QJ QbB
    #-- QJ: joggle input instead of merging facets
    #-- QbB: scale input to unit cube centered at the origin

    #-- try each set of qhull_options
    points = np.concatenate((x0[:,None],y0[:,None]),axis=1)
    for i,opt in enumerate(['Qt Qbb Qc Qz','Qt Qc QbB','QJ QbB']):
        try:
            triangle = scipy.spatial.Delaunay(points, qhull_options=opt)
        except scipy.spatial.qhull.QhullError:
            pass
        else:
            return (i+1,triangle.vertices)

    #-- if still errors: set vertices as an empty list
    delaunay_vertices = []
    return (None,delaunay_vertices)

#-- PURPOSE: calculates the maximum angle within a triangle given the
#-- coordinates of the triangles vertices A(x,y), B(x,y), C(x,y)
def triangle_maximum_angle(Ax, Ay, Bx, By, Cx, Cy):
    """
    Calculates the maximum angle within a triangle with
    vertices A, B and C

    Parameters
    ----------
    Ax: float
        x-coordinate of A vertice
    Ay: float
        y-coordinate of A vertice
    Bx: float
        x-coordinate of B vertice
    By: float
        y-coordinate of B vertice
    Cx: float
        x-coordinate of C vertice
    Cy: float
        y-coordinate of C vertice
    """
    #-- calculate sides of triangle (opposite interior angle at vertex)
    a = np.sqrt((Cx - Bx)**2 + (Cy - By)**2)
    b = np.sqrt((Cx - Ax)**2 + (Cy - Ay)**2)
    c = np.sqrt((Ax - Bx)**2 + (Ay - By)**2)
    #-- calculate interior angles and convert to degrees
    alpha = np.arccos((b**2 + c**2 - a**2)/(2.0*b*c))*180.0/np.pi
    beta = np.arccos((a**2 + c**2 - b**2)/(2.0*a*c))*180.0/np.pi
    gamma = np.arccos((a**2 + b**2 - c**2)/(2.0*a*b))*180.0/np.pi
    #-- return the largest angle within the triangle
    return np.max([alpha, beta, gamma])

#-- PURPOSE: outputting the interpolated data to HDF5
def HDF5_triangulated_data(output_data, MISSION=None, INPUT=None, FILENAME='',
    FILL_VALUE=None, CLOBBER=True):

    #-- setting HDF5 clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'w-'

    #-- open output HDF5 file
    fileID = h5py.File(os.path.expanduser(FILENAME), clobber)

    #-- number of valid points
    n_valid = np.count_nonzero(output_data['data'] != FILL_VALUE)

    #-- HDF5 file attributes
    attrib = {}
    #-- latitude
    attrib['lat'] = {}
    attrib['lat']['long_name'] = 'Latitude_of_measurement'
    attrib['lat']['description'] = ('Corresponding_to_the_measurement_'
        'position_at_the_acquisition_time')
    attrib['lat']['units'] = 'Degrees_North'
    attrib['lat']['coordinates'] = "lon lat time"
    #-- longitude
    attrib['lon'] = {}
    attrib['lon']['long_name'] = 'Longitude_of_measurement'
    attrib['lon']['description'] = ('Corresponding_to_the_measurement_'
        'position_at_the_acquisition_time')
    attrib['lon']['units'] = 'Degrees_East'
    attrib['lon']['coordinates'] = "lon lat time"
    #-- elevation
    attrib['data'] = {}
    attrib['data']['long_name'] = 'Surface_Elevation'
    attrib['data']['description'] = 'Height_above_reference_ellipsoid'
    attrib['data']['units'] = 'meters'
    attrib['data']['_FillValue'] = FILL_VALUE
    attrib['data']['valid_count'] = n_valid
    attrib['data']['coordinates'] = "lon lat time"
    #-- elevation error
    attrib['error'] = {}
    attrib['error']['long_name'] = 'Surface_Elevation_Error'
    attrib['error']['description'] = ('Elevation_error_calculated_'
        'using_triangulated_irregular_networks_(TINs)')
    attrib['error']['units'] = 'meters'
    attrib['error']['_FillValue'] = FILL_VALUE
    attrib['error']['valid_count'] = n_valid
    attrib['error']['coordinates'] = "lon lat time"
    #-- time
    attrib['time'] = {}
    attrib['time']['long_name'] = 'Time'
    attrib['time']['description'] = ('Acquisition_time_measured_as_'
        'seconds_elapsed_since_Jan_1_2000_12:00:00_UTC.')
    attrib['time']['units'] = 'seconds_since_2000-01-01_12:00:00_UTC'
    attrib['time']['calendar'] = "standard"
    attrib['time']['_FillValue'] = FILL_VALUE
    attrib['time']['valid_count'] = n_valid
    attrib['time']['coordinates'] = "lon lat time"
    #-- slope (x-direction)
    attrib['x_slope'] = {}
    attrib['x_slope']['long_name'] = 'X_component_of_surface_slope'
    attrib['x_slope']['description'] = ('Easting_direction_surface_slope_'
        'calculated_using_triangulated_irregular_networks_(TINs)')
    attrib['x_slope']['units'] = 'unitless'
    attrib['x_slope']['_FillValue'] = FILL_VALUE
    attrib['x_slope']['valid_count'] = n_valid
    attrib['x_slope']['coordinates'] = "lon lat time"
    #-- slope (y-direction)
    attrib['y_slope'] = {}
    attrib['y_slope']['long_name'] = 'Y_component_of_surface_slope'
    attrib['y_slope']['description'] = ('Northing_direction_surface_slope_'
        'calculated_using_triangulated_irregular_networks_(TINs)')
    attrib['y_slope']['units'] = 'unitless'
    attrib['y_slope']['_FillValue'] = FILL_VALUE
    attrib['y_slope']['valid_count'] = n_valid
    attrib['y_slope']['coordinates'] = "lon lat time"

    #-- create HDF5 records
    h5 = {}
    for key,val in output_data.items():
        #-- Defining the HDF5 dataset variables
        h5[key] = fileID.create_dataset(key, val.shape, data=val,
            fillvalue=FILL_VALUE, dtype=val.dtype, compression='gzip')
        #-- add HDF5 variable attributes
        for att_name,att_val in attrib[key].items():
            h5[key].attrs[att_name] = att_val
        #-- attach dimensions
        h5[key].dims[0].label = 'RECORD_SIZE'

    #-- HDF5 file title
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'Triangulated_Surface_Elevation_and_Slope_Product'
    fileID.attrs['summary'] = ('Surface_elevation_measurements_over_areas_'
        'including_Greenland_and_Antarctica.')
    fileID.attrs['date_created'] = time.strftime('%Y-%m-%d',time.localtime())
    fileID.attrs['project'] = 'NASA_Operation_IceBridge'
    #-- add attribute for elevation instrument and designated processing level
    instrument = {}
    instrument['BLATM2'] = 'Pre-Icebridge_Airborne_Topographic_Mapper_(icessn)'
    instrument['ILATM2'] = 'IceBridge_Airborne_Topographic_Mapper_(icessn)'
    instrument['BLATM1b'] = 'Pre-Icebridge_Airborne_Topographic_Mapper_(QFIT)'
    instrument['ILATM1b'] = 'IceBridge_Airborne_Topographic_Mapper_(QFIT)'
    instrument['BLVIS2'] = 'Pre-Icebridge_Land,_Vegetation,_and_Ice_Sensor'
    instrument['BVLIS2'] = 'Pre-Icebridge_Land,_Vegetation,_and_Ice_Sensor'
    instrument['ILVIS2'] = 'IceBridge_Land,_Vegetation,_and_Ice_Sensor_(LVIS)'
    instrument['ILVGH2'] = 'Global_Hawk_Land,_Vegetation,_and_Ice_Sensor_(LVIS)'
    fileID.attrs['instrument'] = instrument[MISSION]
    fileID.attrs['processing_level'] = '4'
    #-- add attributes for input elevation file and files triangulated
    fileID.attrs['elevation_file'] = os.path.basename(INPUT[0])
    input_elevation_files = ','.join([os.path.basename(f) for f in INPUT[1:]])
    fileID.attrs['triangulated_files'] = input_elevation_files
    #-- add geospatial and temporal attributes
    fileID.attrs['geospatial_lat_min'] = output_data['lat'].min()
    fileID.attrs['geospatial_lat_max'] = output_data['lat'].max()
    fileID.attrs['geospatial_lon_min'] = output_data['lon'].min()
    fileID.attrs['geospatial_lon_max'] = output_data['lon'].max()
    fileID.attrs['geospatial_lat_units'] = "degrees_north"
    fileID.attrs['geospatial_lon_units'] = "degrees_east"
    fileID.attrs['geospatial_ellipsoid'] = "WGS84"
    fileID.attrs['time_type'] = 'UTC'
    #-- convert start and end time from J2000 seconds into Julian days
    ind, = np.nonzero(output_data['time'] != FILL_VALUE)
    time_start = np.min(output_data['time'][ind])
    time_end = np.max(output_data['time'][ind])
    JD_start = time_start/86400. + 2451545.
    JD_end = time_end/86400. + 2451545.
    #-- convert from julian to calendar dates
    cal_date = icesat2_toolkit.time.convert_julian(np.array([JD_start,JD_end]),
        ASTYPE=np.int64)
    #-- add attributes with measurement date start, end and duration
    args = (cal_date['hour'][0],cal_date['minute'][0],cal_date['second'][0])
    fileID.attrs['RangeBeginningTime'] = '{0:02d}:{1:02d}:{2:02d}'.format(*args)
    args = (cal_date['hour'][-1],cal_date['minute'][-1],cal_date['second'][-1])
    fileID.attrs['RangeEndingTime'] = '{0:02d}:{1:02d}:{2:02d}'.format(*args)
    args = (cal_date['year'][0],cal_date['month'][0],cal_date['day'][0])
    fileID.attrs['RangeBeginningDate'] = '{0:4d}-{1:02d}-{2:02d}'.format(*args)
    args = (cal_date['year'][-1],cal_date['month'][-1],cal_date['day'][-1])
    fileID.attrs['RangeEndingDate'] = '{0:4d}-{1:02d}-{2:02d}'.format(*args)
    duration = np.round(time_end - time_start).astype(np.int64)
    fileID.attrs['DurationTimeSeconds'] = f'{duration:d}'

    #-- Closing the HDF5 file
    fileID.close()

#-- run main program
if __name__ == '__main__':
    main()
