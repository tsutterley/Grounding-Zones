#!/usr/bin/env python
u"""
icebridge.py
Written by Tyler Sutterley (05/2024)
Read altimetry data files from NASA Operation IceBridge (OIB)

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/
f
PROGRAM DEPENDENCIES:
    read_ATM1b_QFIT_binary.py: read ATM1b QFIT binary files (NSIDC version 1)

UPDATE HISTORY:
    Updated 05/2024: added reader for LVIS2 ascii files
        added output writer for LVIS HDF5 files
        include functions for converting ITRF
        use wrapper to importlib for optional dependencies
    Updated 10/2023: add reader for ATM ITRF convention lookup table
    Updated 08/2023: use time functions from timescale.time
    Updated 07/2023: add function docstrings in numpydoc format
    Written 05/2023: moved icebridge data inputs to a separate module
"""

from __future__ import print_function, annotations

import re
import io
import copy
import time
import pathlib
import numpy as np
import grounding_zones as gz

# attempt imports
ATM1b_QFIT = gz.utilities.import_dependency('ATM1b_QFIT')
h5py = gz.utilities.import_dependency('h5py')
timescale = gz.utilities.import_dependency('timescale')

# PURPOSE: read Operation IceBridge data
def from_file(input_file, subset=None, format=None):
    """
    Wrapper function for reading Operation IceBridge data files

    Parameters
    ----------
    input_file: str
        Full path of input file to be read
    subset: np.ndarray or None
        Subsetting array of indices to be read from input file
    format: str or None
        Format of input file

        - ``'ATM'``: Airborne Topographic Mapper Level-2 icessn
        - ``'ATM1b'``: Airborne Topographic Mapper Level-1b QFIT
        - ``'LVIS'``: Land, Vegetation and Ice Sensor Level-2
        - ``'LVGH'``: Land, Vegetation and Ice Sensor Global Hawk Level-2

    Returns
    -------
    dinput: dict
        variables from input file

        - ``lat``: latitude (degrees)
        - ``lon``: longitude (degrees)
        - ``data``: elevation (meters)
        - ``time``: seconds since J2000 epoch
    file_lines: int
        Number of lines within the input file
    HEM: str
        Hemisphere of the input file (``'N'`` or ``'S'``)
    """
    # read data from input_file
    if (format == 'ATM'):
        # load IceBridge ATM data from input_file
        return read_ATM_icessn_file(input_file, subset)
    elif (format == 'ATM1b'):
        # load IceBridge Level-1b ATM data from input_file
        return read_ATM_qfit_file(input_file, subset)
    elif format in ('LVIS','LVGH'):
        # load IceBridge LVIS data from input_file
        return read_LVIS_HDF5_file(input_file, subset)

# PURPOSE: reading the number of file lines removing commented lines
def file_length(input_file, input_subsetter, HDF5=False, QFIT=False):
    """
    Retrieves the number of data points in a file

    Parameters
    ----------
    input_file: str
        Full path of input file to be read
    input_subsetter: np.ndarray or None
        Subsetting array of indices to be read from input file
    HDF5: bool, default False
        Input file is HDF5 format
    QFIT: bool, default False
        Input file is QFIT binary format

    Returns
    -------
    file_lines: int
        Number of lines within the input file
    """
    # verify input file is path
    input_file = pathlib.Path(input_file).expanduser().absolute()
    # subset the data to indices if specified
    if input_subsetter is not None:
        file_lines = len(input_subsetter)
    elif HDF5:
        # read the size of an input variable within a HDF5 file
        with h5py.File(input_file, 'r') as fileID:
            file_lines, = fileID[HDF5].shape
    elif QFIT:
        # read the size of a QFIT binary file
        file_lines = ATM1b_QFIT.ATM1b_QFIT_shape(input_file)
    else:
        # read the input file, split at lines and remove all commented lines
        with input_file.open(mode='r', encoding='utf8') as f:
            i = [i for i in f.readlines() if re.match(r'^(?!\#|\n)',i)]
        file_lines = len(i)
    # return the number of lines
    return file_lines

# PURPOSE: read csv file with the ITRF convention for ATM data
def read_ATM_ITRF_file(header=True, delimiter=','):
    """
    Reads ITRF convention lookup table for ATM campaigns

    Parameters
    ----------
    header: bool, default True
        Input file has a header line
    delimiter: str, default ','
        Column delimiter of input file

    Returns
    -------
    data: dict
        ITRF conventions
    """
    # read ITRF file
    ITRF_file = gz.utilities.get_data_path(['data','ATM1B-ITRF.csv'])
    with ITRF_file.open(mode='r', encoding='utf-8') as f:
        file_contents = f.read().splitlines()
    # get header text and row to start reading data
    if header:
        header_text = file_contents[0].split(delimiter)
        start = 1
    else:
        ncols = len(file_contents[0].split(delimiter))
        header_text = [f'col{i:d}' for i in range(ncols)]
        start = 0
    # allocate dictionary for ITRF data
    data = {col:[] for col in header_text}
    for i,row in enumerate(file_contents[start:]):
        row = row.split(delimiter)
        for j,col in enumerate(header_text):
            data[col].append(row[j])
    # convert data to numpy arrays
    for col in header_text:
        data[col] = np.asarray(data[col])
    # return the parsed data
    return data

# PURPOSE: get the ITRF realization for an OIB dataset
def get_ITRF(
        short_name: str,
        year: int,
        month: int = 0,
        HEM: str = 'N',
    ):
    """
    Get the ITRF realization for an Operation IceBridge dataset

    Parameters
    ----------
    short_name: str
        Name of Operation IceBridge dataset
    year: int
        Year of acquisition of dataset
    month: int, default 0
        Month of acquisition of dataset
    HEM: str, default 'N'
        Region of dataset

        - ``'N'``: Northern Hemisphere
        - ``'S'``: Southern Hemisphere
    """
    if short_name in ('ATM','ATM1b'):
        # get the ITRF of the ATM data
        ITRF_table = gz.io.icebridge.read_ATM_ITRF_file()
        region = dict(N='GR', S='AN')[HEM]
        if (region == 'GR') and (int(month) < 7):
            season = 'SP'
        else:
            season = 'FA'
        # get the row of data from the table
        row, = np.flatnonzero(
            (ITRF_table['year'].astype(int) == int(year)) &
            (ITRF_table['region'] == region) &
            (ITRF_table['season'] == season)
        )
        # find the ITRF for the ATM data
        ITRF = ITRF_table['ITRF'][row]
    elif short_name in ('LVIS','LVGH') and (int(year) <= 2016):
        ITRF = 'ITRF2000'
    elif short_name in ('LVIS','LVGH') and (int(year) >= 2017):
        ITRF = 'ITRF2008'
    # return the reference frame for the OIB dataset
    return ITRF

# PURPOSE: convert the input data to the ITRF reference frame
def convert_ITRF(data, ITRF):
    """
    Convert an Operation IceBridge dataset to a ITRF realization 

    Parameters
    ----------
    data: dict
        Operation IceBridge dataset
    ITRF: str
        ITRF Realization of input dataset
    """
    # get the transform for converting to the latest ITRF
    transform = gz.crs.get_itrf_transform(ITRF)
    # convert time to decimal years
    ts = timescale.time.Timescale().from_deltatime(data['time'],
        epoch=timescale.time._j2000_epoch, standard='UTC')
    # transform the data to a common ITRF
    lon, lat, dat, tdec = transform.transform(
        data['lon'], data['lat'], data['data'], ts.year
    )
    data.update(lon=lon, lat=lat, data=dat)
    # return the updated data dictionary
    return data

## PURPOSE: read the ATM Level-1b data file for variables of interest
def read_ATM_qfit_file(input_file, input_subsetter):
    """
    Reads ATM Level-1b QFIT data files

    Parameters
    ----------
    input_file: str
        Full path of input QFIT file to be read
    input_subsetter: np.ndarray or None
        Subsetting array of indices to be read from input file

    Returns
    -------
    ATM_L1b_input: dict
        Level-1b variables from input file

        - ``lat``: latitude of each shot
        - ``lon``: longitude of each shot
        - ``data``: elevation of each shot
        - ``time``: seconds since J2000 epoch of each shot
    file_lines: int
        Number of lines within the input file
    HEM: str
        Hemisphere of the input file (``'N'`` or ``'S'``)
    """
    # verify input file is path
    input_file = pathlib.Path(input_file).expanduser().absolute()
    # regular expression pattern for extracting parameters
    mission_flag = r'(BLATM1B|ILATM1B|ILNSA1B)'
    regex_pattern = rf'{mission_flag}_(\d+)_(\d+)(.*?).(qi|TXT|h5)'
    regex = re.compile(regex_pattern, re.VERBOSE)
    # extract mission and other parameters from filename
    MISSION,YYMMDD,HHMMSS,AUX,SFX = regex.findall(input_file.name).pop()
    # early date strings omitted century and millennia (e.g. 93 for 1993)
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
        with input_file.open(mode='r', encoding='utf8') as f:
            file_contents = [i for i in f.read().splitlines() if
                re.match(r'^(?!\#|\n)',i)]
        # number of lines of data within file
        file_lines = file_length(input_file, input_subsetter)
        # create output variables with length equal to the number of lines
        ATM_L1b_input['lat'] = np.zeros_like(file_contents, dtype=np.float64)
        ATM_L1b_input['lon'] = np.zeros_like(file_contents, dtype=np.float64)
        ATM_L1b_input['data'] = np.zeros_like(file_contents, dtype=np.float64)
        hour = np.zeros_like(file_contents, dtype=np.float64)
        minute = np.zeros_like(file_contents, dtype=np.float64)
        second = np.zeros_like(file_contents, dtype=np.float64)
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
        fid, h = ATM1b_QFIT.read_ATM1b_QFIT_binary(input_file)
        # number of lines of data within file
        file_lines = file_length(input_file, input_subsetter, QFIT=True)
        ATM_L1b_input['lat'] = fid['latitude'][:]
        ATM_L1b_input['lon'] = fid['longitude'][:]
        ATM_L1b_input['data'] = fid['elevation'][:]
        time_hhmmss = fid['time_hhmmss'][:]
        # extract hour, minute and second from time_hhmmss
        hour = np.zeros_like(time_hhmmss, dtype=np.float64)
        minute = np.zeros_like(time_hhmmss, dtype=np.float64)
        second = np.zeros_like(time_hhmmss, dtype=np.float64)
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
        # number of lines of data within file
        file_lines = file_length(input_file, input_subsetter,
            HDF5='elevation')
        # create output variables with length equal to input elevation
        ATM_L1b_input['lat'] = fileID['latitude'][:]
        ATM_L1b_input['lon'] = fileID['longitude'][:]
        ATM_L1b_input['data'] = fileID['elevation'][:]
        time_hhmmss = fileID['instrument_parameters']['time_hhmmss'][:]
        # extract hour, minute and second from time_hhmmss
        hour = np.zeros_like(time_hhmmss, dtype=np.float64)
        minute = np.zeros_like(time_hhmmss, dtype=np.float64)
        second = np.zeros_like(time_hhmmss, dtype=np.float64)
        # for each line within the file
        for i,packed_time in enumerate(time_hhmmss):
            # convert to zero-padded string with 3 decimal points
            line_contents = f'{packed_time:010.3f}'
            hour[i] = np.float64(line_contents[:2])
            minute[i] = np.float64(line_contents[2:4])
            second[i] = np.float64(line_contents[4:])
        # close the input HDF5 file
        fileID.close()
    # calculate GPS time (seconds since Jan 6, 1980 00:00:00)
    gps_seconds = timescale.time.convert_calendar_dates(
        year, month, day,
        hour=hour, minute=minute, second=second,
        epoch=timescale.time._gps_epoch,
        scale=86400.0)
    # converting to J2000 seconds
    ts = timescale.time.Timescale().from_deltatime(gps_seconds,
        epoch=timescale.time._gps_epoch, standard='GPS')
    ATM_L1b_input['time'] = ts.to_deltatime(
        epoch=timescale.time._j2000_epoch, scale=86400.0
    )
    # subset the data to indices if specified
    if input_subsetter is not None:
        for key,val in ATM_L1b_input.items():
            ATM_L1b_input[key] = val[input_subsetter]
    # hemispheric shot count
    count = {}
    count['N'] = np.count_nonzero(ATM_L1b_input['lat'] >= 0.0)
    count['S'] = np.count_nonzero(ATM_L1b_input['lat'] < 0.0)
    # determine hemisphere with containing shots in file
    HEM, = [key for key, val in count.items() if val]
    # return the output variables
    return ATM_L1b_input, file_lines, HEM

# PURPOSE: read the ATM Level-2 data file for variables of interest
def read_ATM_icessn_file(input_file, input_subsetter):
    """
    Reads ATM Level-2 icessn data files

    Parameters
    ----------
    input_file: str
        Full path of input icessn file to be read
    input_subsetter: np.ndarray or None
        Subsetting array of indices to be read from input file

    Returns
    -------
    ATM_L2_input: dict
        Level-2 variables from input file

        - ``lat``: latitude of each segment
        - ``lon``: longitude of each segment
        - ``data``: elevation of each segment
        - ``error``: estimated elevation uncertainty of each segment
        - ``time``: seconds since J2000 epoch of each segment
        - ``track``: track number of each segment
    file_lines: int
        Number of lines within the input file
    HEM: str
        Hemisphere of the input file (``'N'`` or ``'S'``)
    """
    # verify input file is path
    input_file = pathlib.Path(input_file).expanduser().absolute()
    # regular expression pattern for extracting parameters
    mission_flag = r'(BLATM2|ILATM2)'
    regex_pattern = rf'{mission_flag}_(\d+)_(\d+)_smooth_nadir(.*?)(csv|seg|pt)$'
    regex = re.compile(regex_pattern, re.VERBOSE)
    # extract mission and other parameters from filename
    MISSION,YYMMDD,HHMMSS,AUX,SFX = regex.findall(input_file.name).pop()
    # early date strings omitted century and millennia (e.g. 93 for 1993)
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
    file_lines = file_length(input_file, input_subsetter)
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
    # calculate GPS time (seconds since Jan 6, 1980 00:00:00)
    gps_seconds = timescale.time.convert_calendar_dates(
        year, month, day,
        hour=hour, minute=minute, second=second,
        epoch=timescale.time._gps_epoch,
        scale=86400.0)
    if (MISSION == 'BLATM2') or (SFX != 'csv'):
        # converting to J2000 seconds from GPS seconds
        ts = timescale.time.Timescale().from_deltatime(gps_seconds,
            epoch=timescale.time._gps_epoch, standard='GPS')
    else:
        # converting to J2000 seconds from UTC seconds
        ts = timescale.time.Timescale().from_deltatime(gps_seconds,
            epoch=timescale.time._gps_epoch, standard='UTC')
        leap_seconds = 0.0
    # converting to J2000 seconds
    ATM_L2_input['time'] = ts.to_deltatime(
        epoch=timescale.time._j2000_epoch, scale=86400.0
    )
    # convert RMS from centimeters to meters
    ATM_L2_input['error'] = ATM_L2_input['RMS']/100.0
    # subset the data to indices if specified
    if input_subsetter is not None:
        for key,val in ATM_L2_input.items():
            ATM_L2_input[key] = val[input_subsetter]
    # hemispheric shot count
    count = {}
    count['N'] = np.count_nonzero(ATM_L2_input['lat'] >= 0.0)
    count['S'] = np.count_nonzero(ATM_L2_input['lat'] < 0.0)
    # determine hemisphere with containing shots in file
    HEM, = [key for key, val in count.items() if val]
    # return the output variables
    return ATM_L2_input, file_lines, HEM

# PURPOSE: read LVIS Level-2 data files from NSIDC
def read_LVIS_ascii_file(input_file: str | pathlib.Path | io.BytesIO):
    """
    Reads LVIS Level-2 ascii data files

    Parameters
    ----------
    input_file: str
        Full path of input LVIS file to be read

    Returns
    -------
    ILVIS2_MDS: dict
        Complete set of Level-2 variables from ascii file
    """
    # verify input file is an absolute path
    if not isinstance(input_file, io.BytesIO):
        input_file = pathlib.Path(input_file).expanduser().absolute()
    # LVIS region flags: GL for Greenland and AQ for Antarctica
    lvis_flag = {'GL':'N','AQ':'S'}
    # regular expression pattern for extracting parameters from new format of
    # LVIS2 files (format for LDS 1.04 and 2.0+)
    mission_flag = r'(BLVIS2|BVLIS2|ILVIS2|ILVGH2)'
    regex = rf'{mission_flag}_(GL|AQ)(\d+)_(\d{{2}})(\d{{2}})_(R\d+)_(\d+).TXT'
    rx1 = re.compile(regex, re.IGNORECASE)
    # extract mission, region and other parameters from filename
    MISSION, REGION, YY, MM, DD, RLD, SS = rx1.findall(input_file.name).pop()
    LDS_VERSION = '2.0.2' if (int(RLD[1:3]) >= 18) else '1.04'
    # input file column types for ascii format LVIS files
    # https://lvis.gsfc.nasa.gov/Data/Data_Structure/DataStructure_LDS104.html
    # https://lvis.gsfc.nasa.gov/Data/Data_Structure/DataStructure_LDS202.html
    if (LDS_VERSION == '1.04'):
        file_dtype = {}
        file_dtype['names'] = ('LVIS_LFID','Shot_Number','Time',
            'Longitude_Centroid','Latitude_Centroid','Elevation_Centroid',
            'Longitude_Low','Latitude_Low','Elevation_Low',
            'Longitude_High','Latitude_High','Elevation_High')
        file_dtype['formats']=('i','i','f','f','f','f','f','f','f','f','f','f')
    elif (LDS_VERSION == '2.0.2'):
        file_dtype = {}
        file_dtype['names'] = ('LVIS_LFID','Shot_Number','Time',
            'Longitude_Low','Latitude_Low','Elevation_Low',
            'Longitude_Top','Latitude_Top','Elevation_Top',
            'Longitude_High','Latitude_High','Elevation_High',
            'RH10','RH15','RH20','RH25','RH30','RH35','RH40','RH45','RH50',
            'RH55','RH60','RH65','RH70','RH75','RH80','RH85','RH90','RH95',
            'RH96','RH97','RH98','RH99','RH100','Azimuth','Incident_Angle',
            'Range','Complexity','Flag1','Flag2','Flag3')
        file_dtype['formats'] = ('i','i','f','f','f','f','f','f','f','f','f',
            'f','f','f','f','f','f','f','f','f','f','f','f','f','f','f','f','f',
            'f','f','f','f','f','f','f','f','f','f','f','i','i','i')
    # read icebridge LVIS dataset
    if isinstance(input_file, io.BytesIO):
        file_contents = [i.decode('utf-8') for i in input_file if 
            re.match(rb'^(?!\#|\n)',i)]
    else:
        with input_file.open(mode='r', encoding='utf-8') as f:
            file_contents = [i for i in f.readlines() if 
                re.match(r'^(?!\#|\n)',i)]
    # compile regular expression operator for reading lines (extracts numbers)
    rx2 = re.compile(r'[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?')
    # output python dictionary with variables
    ILVIS2_MDS = {}
    # create output variables with length equal to the number of file lines
    for key,val in zip(file_dtype['names'],file_dtype['formats']):
        ILVIS2_MDS[key] = np.zeros_like(file_contents, dtype=val)
    # for each line within the file
    for line_number,line_entries in enumerate(file_contents):
        # find numerical instances within the line
        line_contents = rx2.findall(line_entries)
        # for each variable of interest: save to dinput as float
        for i,key in enumerate(file_dtype['names']):
            ILVIS2_MDS[key][line_number] = line_contents[i]
    # calculation of julian day (not including hours, minutes and seconds)
    ts = timescale.time.Timescale().from_calendar(
        float(YY), float(MM), float(DD),
        second=ILVIS2_MDS['Time']
    )
    # converting to J2000 seconds and adding seconds since start of day
    ILVIS2_MDS['J2000'] = ts.to_deltatime(
        epoch=timescale.time._j2000_epoch, scale=86400.0
    )
    # save LVIS version
    ILVIS2_MDS['LDS_VERSION'] = copy.copy(LDS_VERSION)
    ILVIS2_MDS['region'] = lvis_flag[REGION]
    # return the output variables
    return ILVIS2_MDS

# PURPOSE: read the LVIS Level-2 data file for variables of interest
def read_LVIS_HDF5_file(input_file, input_subsetter):
    """
    Reads LVIS Level-2 HDF5 data files

    Parameters
    ----------
    input_file: str
        Full path of input LVIS file to be read
    input_subsetter: np.ndarray or None
        Subsetting array of indices to be read from input file

    Returns
    -------
    LVIS_L2_input: dict
        Level-2 variables from input file

        - ``lat``: latitude of each waveform
        - ``lon``: longitude of each waveform
        - ``data``: average elevation of each waveform
        - ``error``: estimated elevation uncertainty of each waveform
        - ``time``: seconds since J2000 epoch of each waveform
    file_lines: int
        Number of lines within the input file
    HEM: str
        Hemisphere of the input file (``'N'`` or ``'S'``)
    """
    # verify input file is path
    input_file = pathlib.Path(input_file).expanduser().absolute()
    # LVIS region flags: GL for Greenland and AQ for Antarctica
    lvis_flag = {'GL':'N','AQ':'S'}
    # regular expression pattern for extracting parameters from new format of
    # LVIS2 files (format for LDS 1.04 and 2.0+)
    mission_flag = r'(BLVIS2|BVLIS2|ILVIS2|ILVGH2)'
    regex_pattern = rf'{mission_flag}_(GL|AQ)(\d+)_(\d+)_(R\d+)_(\d+).H5'
    regex = re.compile(regex_pattern, re.VERBOSE)
    # extract mission, region and other parameters from filename
    MISSION,REGION,YY,MMDD,RLD,SS = regex.findall(input_file.name).pop()
    LDS_VERSION = '2.0.2' if (int(RLD[1:3]) >= 18) else '1.04'
    # input and output python dictionaries with variables
    file_input = {}
    LVIS_L2_input = {}
    fileID = h5py.File(input_file, 'r')
    # create output variables with length equal to input shot number
    file_lines = file_length(input_file, input_subsetter, HDF5='Shot_Number')
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
    LVIS_L2_input['data'] = np.zeros_like(file_input['elev'], dtype=np.float64)
    LVIS_L2_input['lon'] = np.zeros_like(file_input['elev'], dtype=np.float64)
    LVIS_L2_input['lat'] = np.zeros_like(file_input['elev'], dtype=np.float64)
    LVIS_L2_input['error'] = np.zeros_like(file_input['elev'], dtype=np.float64)
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
    if input_subsetter is not None:
        for key,val in LVIS_L2_input.items():
            LVIS_L2_input[key] = val[input_subsetter]
    # return the output variables
    return LVIS_L2_input, file_lines, lvis_flag[REGION]

# PURPOSE: output HDF5 file with geolocated elevation surfaces
# calculated from LVIS Level-1b waveform products
def write_LVIS_HDF5_file(
        ILVIS2_MDS: dict,
        LDS_VERSION: str,
        filename: str | pathlib.Path | None = None,
        lineage: str | pathlib.Path | None = None
    ):
    """
    Writes LVIS Level-2 data to HDF5 files

    Parameters
    ----------
    ILVIS2_MDS: dict
        Complete set of LVIS Level-2 variables
    LDS_VERSION: str
        Version of the LVIS Data Structure (1.04 or 2.0.2)
    filename: str or pathlib.Path or None
        Output HDF5 filename
    lineage: str or pathlib.Path or None
        Original LVIS filename or lineage information
    """
    # open output HDF5 file
    fileID = h5py.File(filename, 'w')

    # create sub-groups within HDF5 file
    fileID.create_group('Time')
    fileID.create_group('Geolocation')
    fileID.create_group('Elevation_Surfaces')
    # sub-groups specific to the LDS version 2.0.2
    if (LDS_VERSION == '2.0.2'):
        fileID.create_group('Waveform')
        fileID.create_group('Instrument_Parameters')

    # Dimensions of parameters
    n_records, = ILVIS2_MDS['Shot_Number'].shape

    # Defining output HDF5 variable attributes
    attributes = {}
    # LVIS_LFID
    attributes['LVIS_LFID'] = {}
    attributes['LVIS_LFID']['long_name'] = 'LVIS Record Index'
    attributes['LVIS_LFID']['description'] = ('LVIS file identification, '
        'including date and time of collection and file number. The third '
        'through seventh values in first field represent the Modified Julian '
        'Date of data collection.')
    # Shot Number
    attributes['Shot_Number'] = {}
    attributes['Shot_Number']['long_name'] = ('Shot Number')
    attributes['Shot_Number']['description'] = ('Laser shot assigned during '
        'collection')
    # Time
    attributes['Time'] = {}
    attributes['Time']['long_name'] = 'Transmit time of each shot'
    attributes['Time']['units'] = 'Seconds'
    attributes['Time']['description'] = 'UTC decimal seconds of the day'
    # J2000
    attributes['J2000'] = {}
    attributes['J2000']['long_name'] = ('Transmit time of each shot in J2000 '
        'seconds')
    attributes['J2000']['units'] = 'seconds since 2000-01-01 12:00:00 UTC'
    attributes['J2000']['description'] = ('The transmit time of each shot in '
        'the 1 second frame measured as UTC seconds elapsed since Jan 1 '
        '2000 12:00:00 UTC.')
    # Centroid
    attributes['Longitude_Centroid'] = {}
    attributes['Longitude_Centroid']['long_name'] = 'Longitude_Centroid'
    attributes['Longitude_Centroid']['units'] = 'Degrees East'
    attributes['Longitude_Centroid']['description'] = ('Corresponding longitude '
        'of the LVIS Level-1B waveform centroid')
    attributes['Latitude_Centroid'] = {}
    attributes['Latitude_Centroid']['long_name'] = 'Latitude_Centroid'
    attributes['Latitude_Centroid']['units'] = 'Degrees North'
    attributes['Latitude_Centroid']['description'] = ('Corresponding latitude of '
        'the LVIS Level-1B waveform centroid')
    attributes['Elevation_Centroid'] = {}
    attributes['Elevation_Centroid']['long_name'] = 'Elevation_Centroid'
    attributes['Elevation_Centroid']['units'] = 'Meters'
    attributes['Elevation_Centroid']['description'] = ('Elevation surface of the '
        'LVIS Level-1B waveform centroid')
    # Lowest mode
    attributes['Longitude_Low'] = {}
    attributes['Longitude_Low']['long_name'] = 'Longitude_Low'
    attributes['Longitude_Low']['units'] = 'Degrees East'
    attributes['Longitude_Low']['description'] = ('Longitude of the '
        'lowest detected mode within the LVIS Level-1B waveform')
    attributes['Latitude_Low'] = {}
    attributes['Latitude_Low']['long_name'] = 'Latitude_Low'
    attributes['Latitude_Low']['units'] = 'Degrees North'
    attributes['Latitude_Low']['description'] = ('Latitude of the '
        'lowest detected mode within the LVIS Level-1B waveform')
    attributes['Elevation_Low'] = {}
    attributes['Elevation_Low']['long_name'] = 'Elevation_Low'
    attributes['Elevation_Low']['units'] = 'Meters'
    attributes['Elevation_Low']['description'] = ('Mean Elevation of the '
        'lowest detected mode within the LVIS Level-1B waveform')
    # Highest mode
    attributes['Longitude_High'] = {}
    attributes['Longitude_High']['long_name'] = 'Longitude_High'
    attributes['Longitude_High']['units'] = 'Degrees East'
    attributes['Longitude_High']['description'] = ('Longitude of the '
        'highest detected mode within the LVIS Level-1B waveform')
    attributes['Latitude_High'] = {}
    attributes['Latitude_High']['long_name'] = 'Latitude_High'
    attributes['Latitude_High']['units'] = 'Degrees North'
    attributes['Latitude_High']['description'] = ('Latitude of the '
        'highest detected mode within the LVIS Level-1B waveform')
    attributes['Elevation_High'] = {}
    attributes['Elevation_High']['long_name'] = 'Elevation_High'
    attributes['Elevation_High']['units'] = 'Meters'
    attributes['Elevation_High']['description'] = ('Mean Elevation of the '
        'highest detected mode within the LVIS Level-1B waveform')
    # Highest detected signal
    attributes['Longitude_Top'] = {}
    attributes['Longitude_Top']['long_name'] = 'Longitude_Top'
    attributes['Longitude_Top']['units'] = 'Degrees East'
    attributes['Longitude_Top']['description'] = ('Longitude of the '
        'highest detected signal within the LVIS Level-1B waveform')
    attributes['Latitude_Top'] = {}
    attributes['Latitude_Top']['long_name'] = 'Latitude_Top'
    attributes['Latitude_Top']['units'] = 'Degrees North'
    attributes['Latitude_Top']['description'] = ('Latitude of the '
        'highest detected signal within the LVIS Level-1B waveform')
    attributes['Elevation_Top'] = {}
    attributes['Elevation_Top']['long_name'] = 'Elevation_Top'
    attributes['Elevation_Top']['units'] = 'Meters'
    attributes['Elevation_Top']['description'] = ('Mean Elevation of the '
        'highest detected signal within the LVIS Level-1B waveform')
    # heights at which a percentage of the waveform energy occurs
    pv = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
        80, 85, 90, 95, 96, 97, 98, 99, 100]
    for RH in pv:
        attributes[f'RH{RH:d}'] = {}
        attributes[f'RH{RH:d}']['long_name'] = f'RH{RH:d}'
        attributes[f'RH{RH:d}']['units'] = 'Meters'
        attributes[f'RH{RH:d}']['description'] = ('Height relative to the '
            f'lowest detected mode at which {RH:d}% of the waveform '
            'energy occurs')
    # Laser parmeters
    # Azimuth
    attributes['Azimuth'] = {}
    attributes['Azimuth']['long_name'] = 'Azimuth'
    attributes['Azimuth']['units'] = 'degrees'
    attributes['Azimuth']['description'] = 'Azimuth angle of the laser beam.'
    attributes['Azimuth']['valid_min'] = 0.0
    attributes['Azimuth']['valid_max'] = 360.0
    # Incident Angle
    attributes['Incident_Angle'] = {}
    attributes['Incident_Angle']['long_name'] = 'Incident_Angle'
    attributes['Incident_Angle']['units'] = 'degrees'
    attributes['Incident_Angle']['description'] = ('Off-nadir incident angle '
        'of the laser beam.')
    attributes['Incident_Angle']['valid_min'] = 0.0
    attributes['Incident_Angle']['valid_max'] = 360.0
    # Range
    attributes['Range'] = {}
    attributes['Range']['long_name'] = 'Range'
    attributes['Range']['units'] = 'meters'
    attributes['Range']['description'] = ('Distance between the instrument and '
        'the ground.')
    # Complexity
    attributes['Complexity'] = {}
    attributes['Complexity']['long_name'] = 'Complexity'
    attributes['Complexity']['description'] = ('Complexity metric for the '
        'return waveform.')
    # Flags
    attributes['Flag1'] = {}
    attributes['Flag1']['long_name'] = 'Flag1'
    attributes['Flag1']['description'] = ('Flag indicating LVIS channel used '
        'to locate lowest detected mode.')
    attributes['Flag2'] = {}
    attributes['Flag2']['long_name'] = 'Flag1'
    attributes['Flag2']['description'] = ('Flag indicating LVIS channel used '
        'to calculate RH metrics.')
    attributes['Flag3'] = {}
    attributes['Flag3']['long_name'] = 'Flag1'
    attributes['Flag3']['description'] = ('Flag indicating LVIS channel '
        'waveform contained in Level-1B file.')

    # Defining the HDF5 dataset variables
    h5 = {}

    # Defining Shot_Number dimension variable
    dim = 'Shot_Number'
    h5[dim] = fileID.create_dataset(dim,
        (n_records,), data=ILVIS2_MDS[dim], dtype=ILVIS2_MDS[dim].dtype,
        compression='gzip')
    h5[dim].make_scale(dim)
    # add HDF5 variable attributes
    for att_name,att_val in attributes[dim].items():
        h5[dim].attrs[att_name] = att_val

    # Time Variables
    for k in ['LVIS_LFID','Time','J2000']:
        v = ILVIS2_MDS[k]
        h5[k] = fileID.create_dataset(f'Time/{k}',
            (n_records,), data=v, dtype=v.dtype,
            compression='gzip')
        # attach dimensions
        h5[k].dims[0].label = dim
        h5[k].dims[0].attach_scale(h5[dim])
        # add HDF5 variable attributes
        for att_name,att_val in attributes[k].items():
            h5[k].attrs[att_name] = att_val

    # Geolocation Variables
    if (LDS_VERSION == '1.04'):
        geolocation_keys = ['Longitude_Centroid','Longitude_Low',
            'Longitude_High','Latitude_Centroid','Latitude_Low','Latitude_High']
    elif (LDS_VERSION == '2.0.2'):
        geolocation_keys = ['Longitude_Low','Longitude_High','Longitude_Top',
            'Latitude_Low','Latitude_High','Latitude_Top']
    for k in geolocation_keys:
        v = ILVIS2_MDS[k]
        h5[k] = fileID.create_dataset(f'Geolocation/{k}',
            (n_records,), data=v, dtype=v.dtype,
            compression='gzip')
        # attach dimensions
        h5[k].dims[0].label = dim
        h5[k].dims[0].attach_scale(h5[dim])
        # add HDF5 variable attributes
        for att_name,att_val in attributes[k].items():
            h5[k].attrs[att_name] = att_val

    # Elevation Surface Variables
    if (LDS_VERSION == '1.04'):
        elevation_keys = ['Elevation_Centroid','Elevation_Low','Elevation_High']
    elif (LDS_VERSION == '2.0.2'):
        elevation_keys = ['Elevation_Low','Elevation_High','Elevation_Top']
    for k in elevation_keys:
        v = ILVIS2_MDS[k]
        h5[k] = fileID.create_dataset(f'Elevation_Surfaces/{k}',
            (n_records,), data=v, dtype=v.dtype,
            compression='gzip')
        # attach dimensions
        h5[k].dims[0].label = dim
        h5[k].dims[0].attach_scale(h5[dim])
        # add HDF5 variable attributes
        for att_name,att_val in attributes[k].items():
            h5[k].attrs[att_name] = att_val

    # variables specific to the LDS version 2.0.2
    if (LDS_VERSION == '2.0.2'):
        # Waveform Variables
        height_keys = ['RH10','RH15','RH20','RH25','RH30','RH35','RH40',
            'RH45','RH50','RH55','RH60','RH65','RH70','RH75','RH80','RH85',
            'RH90','RH95','RH96','RH97','RH98','RH99','RH100','Complexity']
        for k in height_keys:
            v = ILVIS2_MDS[k]
            h5[k] = fileID.create_dataset(f'Waveform/{k}',
                (n_records,), data=v, dtype=v.dtype,
                compression='gzip')
            # attach dimensions
            h5[k].dims[0].label = dim
            h5[k].dims[0].attach_scale(h5[dim])
            # add HDF5 variable attributes
            for att_name,att_val in attributes[k].items():
                h5[k].attrs[att_name] = att_val

        # instrument parameter variables
        instrument_parameter_keys = ['Azimuth','Incident_Angle','Range',
            'Flag1','Flag2','Flag3']
        for k in instrument_parameter_keys:
            v = ILVIS2_MDS[k]
            h5[k]=fileID.create_dataset(f'Instrument_Parameters/{k}',
                (n_records,), data=v, dtype=v.dtype,
                compression='gzip')
            # attach dimensions
            h5[k].dims[0].label = dim
            h5[k].dims[0].attach_scale(h5[dim])
            # add HDF5 variable attributes
            for att_name,att_val in attributes[k].items():
                h5[k].attrs[att_name] = att_val

    # Defining global attributes for output HDF5 file
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'IceBridge LVIS L2 Geolocated Surface Elevation'
    fileID.attrs['comment'] = ('Operation IceBridge products may include test '
        'flight data that are not useful for research and scientific analysis. '
        'Test flights usually occur at the beginning of campaigns. Users '
        'should read flight reports for the flights that collected any of the '
        'data they intend to use')
    fileID.attrs['summary'] = ("Surface elevation measurements over areas "
        "including Greenland and Antarctica. The data were collected as part "
        "of NASA Operation IceBridge funded campaigns.")
    fileID.attrs['references'] = '{0}, {1}'.format('http://lvis.gsfc.nasa.gov/',
        'http://nsidc.org/data/docs/daac/icebridge/ilvis2')
    fileID.attrs['date_created'] = time.strftime('%Y-%m-%d',time.localtime())
    fileID.attrs['project'] = 'NASA Operation IceBridge'
    fileID.attrs['instrument'] = 'Land, Vegetation, and Ice Sensor (LVIS)'
    fileID.attrs['processing_level'] = '2'
    fileID.attrs['lineage'] = pathlib.Path(lineage).name
    # LVIS Data Structure (LDS) version
    # https://lvis.gsfc.nasa.gov/Data/Data_Structure/DataStructure_LDS104.html
    # https://lvis.gsfc.nasa.gov/Data/Data_Structure/DataStructure_LDS202.html
    fileID.attrs['version'] = f'LDSv{LDS_VERSION}'
    # Geospatial and temporal parameters
    fileID.attrs['geospatial_lat_min'] = ILVIS2_MDS['Latitude_Low'].min()
    fileID.attrs['geospatial_lat_max'] = ILVIS2_MDS['Latitude_Low'].max()
    fileID.attrs['geospatial_lon_min'] = ILVIS2_MDS['Longitude_Low'].min()
    fileID.attrs['geospatial_lon_max'] = ILVIS2_MDS['Longitude_Low'].max()
    fileID.attrs['geospatial_lat_units'] = "degrees_north"
    fileID.attrs['geospatial_lon_units'] = "degrees_east"
    fileID.attrs['geospatial_ellipsoid'] = "WGS84"
    fileID.attrs['time_type'] = 'UTC'
    fileID.attrs['date_type'] = 'J2000'
    # create timescale from J2000: seconds since 2000-01-01 12:00:00 UTC
    ts = timescale.time.Timescale().from_deltatime(ILVIS2_MDS['J2000'],
        epoch=timescale.time._j2000_epoch, standard='UTC')
    # add attributes with measurement date start, end and duration
    dt = np.datetime_as_string(ts.to_datetime(), unit='s')
    duration = ts.day*(np.max(ts.MJD) - np.min(ts.MJD))
    fileID.attrs['time_coverage_start'] = str(dt[0])
    fileID.attrs['time_coverage_end'] = str(dt[-1])
    fileID.attrs['time_coverage_duration'] = f'{duration:0.0f}'
    # Closing the HDF5 file
    fileID.close()
