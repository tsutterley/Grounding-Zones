#!/usr/bin/env python
u"""
interp_DAC_icebridge_data.py
Written by Tyler Sutterley (06/2024)
Interpolates AVISO dynamic atmospheric corrections (DAC) to
    Operation IceBridge elevation data

Data will be interpolated for all valid points
    (masking land values will be needed for accurate assessments)

https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/
    atmospheric-corrections.html

Note that the AVISO DAC data can be bz2 compressed netCDF4 files

INPUTS:
    ATM1B, ATM icessn or LVIS file

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -O X, --output-directory X: input/output data directory
    -V, --verbose: Output information about each created file
    -M X, --mode X: Permission mode of directories and files created

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://h5py.org
    netCDF4: Python interface to the netCDF C library
        https://unidata.github.io/netcdf4-python/netCDF4/index.html
    pyTMD: Python-based tidal prediction software
        https://pypi.org/project/pyTMD/
        https://pytmd.readthedocs.io/en/latest/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

PROGRAM DEPENDENCIES:
    spatial.py: utilities for reading and writing spatial data
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Written 06/2024
"""
from __future__ import print_function

import re
import bz2
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
netCDF4 = gz.utilities.import_dependency('netCDF4')
pyproj = gz.utilities.import_dependency('pyproj')
pyTMD = gz.utilities.import_dependency('pyTMD')
timescale = gz.utilities.import_dependency('timescale')

# PURPOSE: read Operation IceBridge data
# calculate and interpolate the dynamic atmospheric correction
def interp_DAC_icebridge_data(base_dir, arg,
    VERBOSE=False, MODE=0o775):

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
    # time
    attrib['time'] = {}
    attrib['time']['long_name'] = 'Time'
    attrib['time']['units'] = 'days since 1992-01-01T00:00:00'
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
    # dynamic atmospheric correction (DAC)
    attrib['dac'] = {}
    attrib['dac']['long_name'] = 'Dynamic_Atmosphere_Correction'
    attrib['dac']['description'] = ("Dynamic_atmospheric_correction_"
        "(DAC)_which_includes_inverse_barometer_(IB)_effects")
    attrib['dac']['reference'] = ('https://www.aviso.altimetry.fr/'
        'en/data/products/auxiliary-products/atmospheric-corrections.html')
    attrib['dac']['source'] = 'Mog2D-G_High_Resolution_barotropic_model'
    attrib['dac']['units'] = 'meters'

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
    logging.info(f'{str(input_file)} -->')
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

    # create timescale from J2000: seconds since 2000-01-01 12:00:00 UTC
    ts = timescale.time.Timescale().from_deltatime(dinput['time'],
        epoch=timescale.time._j2000_epoch, standard='UTC')
    # convert time to days relative to 1950-01-01 (MJD:33282)
    t = ts.to_deltatime(epoch=(1950,1,1,0,0,0))
    YY = np.datetime_as_string(ts.to_datetime(), unit='Y')
    # days and hours to read
    unique_hours, unique_indices = np.unique(
        [np.floor(t*24.0/6.0)*6.0, np.ceil(t*24.0/6.0)*6.0],
        return_index=True)
    days,hours = (unique_hours // 24, unique_hours % 24)
    unique_indices = unique_indices % len(t)

    # pyproj transformer for converting from input coordinates (EPSG)
    # to model coordinates
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_string('+proj=longlat +ellps=WGS84 +datum=WGS84 '
        '+no_defs lon_wrap=180')
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    # calculate projected coordinates of input coordinates
    ix,iy = transformer.transform(dinput['lon'], dinput['lat'])

    # shape of DAC field
    ny,nx = (721, 1440)
    # allocate for DAC fields
    idac = np.ma.zeros((len(days), ny, nx))
    icjd = np.zeros((len(days)))
    for i,CJD in enumerate(days):
        # input file for 6-hour period
        f = f'dac_dif_{CJD:0.0f}_{hours[i]:02.0f}.nc'
        input_file = base_dir.joinpath(YY[unique_indices[i]], f)
        # check if the file exists as a compressed file
        if input_file.with_suffix('.nc.bz2').exists():
            # read bytes from compressed file
            input_file = input_file.with_suffix('.nc.bz2')
            fd = bz2.BZ2File(input_file, 'rb')
        elif input_file.exists():
            # read bytes from uncompressed file
            fd = open(input_file, 'rb')
        else:
            raise FileNotFoundError(f'File not found: {input_file}')
        # read netCDF file for time
        with netCDF4.Dataset('dac', mode='r', memory=fd.read()) as fid:
            ilon = fid['longitude'][:]
            ilat = fid['latitude'][:]
            idac[i,:,:] = fid['dac'][:]
            icjd[i] = fid['dac'].getncattr('Date_CNES_JD')
        # close the compressed file objects
        fd.close()

    # create an interpolator for dynamic atmospheric correction
    RGI = scipy.interpolate.RegularGridInterpolator((icjd,ilat,ilon), idac,
        bounds_error=False)
    # interpolate dynamic atmospheric correction to points
    fill_value = -9999.0
    dinput['dac'] = np.ma.zeros((file_lines), fill_value=fill_value)
    dinput['dac'].data[:] = RGI.__call__(np.c_[t, iy, ix])
    dinput['dac'].mask = np.isnan(dinput['dac'].data)
    dinput['dac'].data[dinput['dac'].mask] = dinput['dac'].fill_value

    # output DAC HDF5 file
    # form: rg_NASA_DAC_WGS84_fl1yyyymmddjjjjj.H5
    # where rg is the hemisphere flag (GR or AN) for the region
    # fl1 and fl2 are the data flags (ATM, LVIS, GLAS)
    # yymmddjjjjj is the year, month, day and second of the input file
    # output region flags: GR for Greenland and AN for Antarctica
    hem_flag = {'N':'GR','S':'AN'}
    # use starting second to distinguish between files for the day
    JJ1 = np.min(dinput['time']) % 86400
    # output file format
    file_format = '{0}_NASA_DAC_WGS84_{1}{2}{3}{4}{5:05.0f}.H5'
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
    fid.attrs['title'] = 'Dynamic atmospheric_correction'
    fid.attrs['summary'] = ('Dynamic atmospheric_correction_interpolated_'
        'to_elevation_measurements.')
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

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates and interpolates dynamic atmospheric
            corrections to Operation IceBridge elevation data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('infile',
        type=str, nargs='+',
        help='Input Operation IceBridge file to run')
    # directory with reanalysis data
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=pathlib.Path.cwd(),
        help='Working data directory')
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

    # run for each input file
    for arg in args.infile:
        interp_DAC_icebridge_data(args.directory, arg,
            VERBOSE=args.verbose, MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
