#!/usr/bin/env python
u"""
calculate_inverse_barometer.py
Written by Tyler Sutterley (05/2024)
Reads hourly mean sea level pressure fields from reanalysis and
    calculates the inverse-barometer response

INPUTS:
    Reanalysis model to run
    ERA-Interim: http://apps.ecmwf.int/datasets/data/interim-full-moda
    ERA5: http://apps.ecmwf.int/data-catalogues/era5/?class=ea
    MERRA-2: https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -Y X, --year X: years to run
    -m X, --mean X: Start and end year range for mean
    -d X, --density X: Density of seawater in kg/m^3
    -V, --verbose: Output information about each created file
    -M X, --mode X: Permission mode of directories and files created

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    netCDF4: Python interface to the netCDF C library
        https://unidata.github.io/netcdf4-python/netCDF4/index.html

PROGRAM DEPENDENCIES:
    utilities.py: download and management utilities for files

REFERENCES:
    Wunsch and Stammer. Atmospheric loading and the oceanic "inverted
        barometer" effect. Reviews of Geophysics, 35(1), 79-107, (1997).
        https://doi.org/10.1029/96RG03037

    Hofmann-Wellenhof and Moritz. Physical Geodesy, (2005).
        https://doi.org/10.1007/978-3-211-33545-1

UPDATE HISTORY:
    Updated 05/2024: use wrapper to importlib for optional dependencies
    Updated 04/2023: using pathlib to define and expand paths
    Updated 12/2022: single implicit import of grounding zone tools
        use constants class from pyTMD for ellipsoidal parameters
    Updated 05/2022: use argparse descriptions within sphinx documentation
    Updated 12/2021: can use variable loglevels for verbose output
    Updated 10/2021: using python logging for handling verbose output
    Updated 07/2021: can use input files to define command line arguments
    Written 03/2021
"""
from __future__ import print_function

import sys
import re
import logging
import pathlib
import argparse
import datetime
import numpy as np
import grounding_zones as gz

# attempt imports
netCDF4 = gz.utilities.import_dependency('netCDF4')
pyTMD = gz.utilities.import_dependency('pyTMD')

# PURPOSE: read land sea mask to get indices of oceanic values
def ncdf_landmask(FILENAME,MASKNAME,OCEAN):
    with netCDF4.Dataset(FILENAME,'r') as fileID:
        landsea = np.squeeze(fileID.variables[MASKNAME][:].copy())
    return (landsea == OCEAN)

# PURPOSE: read reanalysis mean sea level pressure
def ncdf_mean_pressure(FILENAME,VARNAME,LONNAME,LATNAME):
    with netCDF4.Dataset(FILENAME,'r') as fileID:
        # extract pressure and remove singleton dimensions
        mean_pressure = np.array(fileID.variables[VARNAME][:].squeeze())
        longitude = fileID.variables[LONNAME][:].squeeze()
        latitude = fileID.variables[LATNAME][:].squeeze()
    return (mean_pressure,longitude,latitude)

# PURPOSE: read sea level pressure fields and calculate anomalies
# spatially subset pressure maps to latitudinal range of ATL11 points
def ncdf_pressure(fileID, VARNAME, TIMENAME, MEAN, OCEAN, AREA):
    # shape of subsetted pressure field
    ny,nx = np.shape(MEAN)
    # invalid value
    fill_value = fileID.variables[VARNAME]._FillValue
    # allocate for pressure fields
    SLP = np.ma.zeros((24,ny,nx), dtype=np.float32, fill_value=fill_value)
    TPX = np.ma.zeros((24,ny,nx), dtype=np.float32, fill_value=fill_value)
    # calculate total area of reanalysis ocean
    # ocean pressure points will be based on reanalysis mask
    ii,jj = np.nonzero(OCEAN)
    ocean_area = np.sum(AREA[ii,jj])
    # parameters for conventional TOPEX/POSEIDON IB correction
    rho0 = 1025.0
    g0 = -9.80665
    p0 = 101325.0
    # counter for filling arrays
    c = 0
    # convert time to Modified Julian Days
    delta_time = np.copy(fileID.variables[TIMENAME][:])
    for t,dt in enumerate(delta_time):
        # check dimensions for expver slice
        if (fileID.variables[VARNAME].ndim == 4):
            _,nexp,_,_ = fileID.variables[VARNAME].shape
            # sea level pressure for time
            pressure = fileID.variables[VARNAME][t,:,:,:].copy()
            # iterate over expver slices to find valid outputs
            for j in range(nexp):
                # check if any are valid for expver
                if np.any(pressure[j,:,:]):
                    # remove average with respect to time
                    AveRmvd = pressure[j,:,:] - MEAN
                    # conventional TOPEX/POSEIDON IB correction
                    TPX[c,:,:] = (pressure[j,:,:]-p0)/(rho0*g0)
                    break
        else:
            # sea level pressure for time
            pressure = fileID.variables[VARNAME][t,:,:].copy()
            # remove average with respect to time
            AveRmvd = pressure - MEAN
            # conventional TOPEX/POSEIDON IB correction
            TPX[c,:,:] = (pressure[:,:] - p0)/(rho0*g0)
        # calculate average oceanic pressure values
        AVERAGE = np.sum(AveRmvd[ii,jj]*AREA[ii,jj])/ocean_area
        # calculate sea level pressure anomalies and
        # reduce to latitudinal range of ATL11 points
        SLP[c,:,:] = AveRmvd[:,:] - AVERAGE
        # clear temp variables for iteration to free up memory
        pressure,AveRmvd = (None,None)
        # add to counter
        c += 1
    # return sea level pressure anomalies and TOPX/POSEIDON IB correction
    return (SLP, TPX)

# PURPOSE:  calculate the instantaneous inverse barometer response
def calculate_inverse_barometer(base_dir, MODEL, YEAR=None, RANGE=None,
    DENSITY=None, MODE=0o775):

    # directory setup for reanalysis model
    base_dir = pathlib.Path(base_dir).expanduser().absolute()
    ddir = base_dir.joinpath(MODEL)
    # set model specific parameters
    if (MODEL == 'ERA-Interim'):
        # regular expression pattern for finding files
        regex_pattern = (r'ERA\-Interim\-Hourly\-MSL\-'
            r'({0})-(\d{{2}})-(\d{{2}})\.nc$')
        # mean sea level pressure file
        input_mean_file = 'ERA-Interim-Mean-MSL-{0:4d}-{1:4d}.nc'
        # input land-sea mask for ocean redistribution
        input_mask_file = 'ERA-Interim-Invariant-Parameters.nc'
        # output file format
        output_file_format = 'ERA-Interim-Hourly-IB-{0}-{1}-{2}.nc'
        VARNAME = 'msl'
        LONNAME = 'longitude'
        LATNAME = 'latitude'
        TIMENAME = 'time'
        IBNAME = 'ib'
        TPXNAME = 'tpx'
        UNITS = 'm'
        # hours since 1900-01-01 00:00:0.0
        TIME_LONGNAME = 'Time'
        # land-sea mask variable name and value of oceanic points
        MASKNAME = 'lsm'
        OCEAN = 0
    elif (MODEL == 'ERA5'):
        # regular expression pattern for finding files
        regex_pattern = r'ERA5\-Hourly\-MSL\-({0})-(\d{{2}})-(\d{{2}})\.nc$'
        # mean sea level pressure file
        input_mean_file = 'ERA5-Mean-MSL-{0:4d}-{1:4d}.nc'
        # input land-sea mask for ocean redistribution
        input_mask_file = 'ERA5-Invariant-Parameters.nc'
        # output file format
        output_file_format = 'ERA5-Hourly-IB-{0}-{1}-{2}.nc'
        VARNAME = 'msl'
        LONNAME = 'longitude'
        LATNAME = 'latitude'
        TIMENAME = 'time'
        IBNAME = 'ib'
        TPXNAME = 'tpx'
        UNITS = 'm'
        # hours since 1900-01-01 00:00:0.0
        TIME_LONGNAME = 'Time'
        # land-sea mask variable name and value of oceanic points
        MASKNAME = 'lsm'
        OCEAN = 0
    elif (MODEL == 'MERRA-2'):
        # regular expression pattern for finding files
        regex_pattern = (r'MERRA2_(\d{{3}}).tavg1_2d_slv_Nx.'
            r'({0})(\d{{2}})(\d{{2}}).(.*?).nc$')
        # mean sea level pressure file
        input_mean_file = 'MERRA2.Mean_SLP.{0:4d}-{1:4d}.nc'
        # input land-sea mask for ocean redistribution
        input_mask_file = 'MERRA2_101.const_2d_asm_Nx.00000000.nc4'
        # output file format
        output_file_format = 'MERRA2_{0}.tavg1_2d_IB.{1}{2}{3}.{4}.nc'
        VARNAME = 'SLP'
        LONNAME = 'lon'
        LATNAME = 'lat'
        TIMENAME = 'time'
        IBNAME = 'IB'
        TPXNAME = 'TPX'
        UNITS = 'm'
        # minutes since start of file
        TIME_LONGNAME = 'Time'
        # land-sea mask variable name and value of oceanic points
        MASKNAME = 'FROCEAN'
        OCEAN = 1

    # read mean pressure field
    mean_file = ddir.joinpath(input_mean_file.format(RANGE[0], RANGE[1]))
    mean_pressure, lon, lat = ncdf_mean_pressure(mean_file,
        VARNAME, LONNAME, LATNAME)

    # grid step size in radians
    dphi = np.pi*np.abs(lon[1] - lon[0])/180.0
    dth = np.pi*np.abs(lat[1] - lat[0])/180.0
    # calculate meshgrid from latitude and longitude
    gridlon,gridlat = np.meshgrid(lon,lat)
    gridphi = gridlon*np.pi/180.0
    # calculate colatitude
    gridtheta = (90.0 - gridlat)*np.pi/180.0

    # ellipsoidal parameters of WGS84 ellipsoid
    wgs84 = pyTMD.spatial.datum(ellipsoid='WGS84', units='MKS')
    # semimajor and semiminor axes of the ellipsoid [m]
    a_axis = wgs84.a_axis
    b_axis = wgs84.b_axis
    # calculate grid areas globally
    cell_areas = dphi*dth*np.sin(gridtheta)*np.sqrt((a_axis**2)*(b_axis**2) *
        ((np.sin(gridtheta)**2)*(np.cos(gridphi)**2) +
        (np.sin(gridtheta)**2)*(np.sin(gridphi)**2)) +
        (a_axis**4)*(np.cos(gridtheta)**2))
    # read land-sea mask to find ocean values
    # ocean pressure points will be based on reanalysis mask
    ocean_function = ncdf_landmask(ddir.joinpath(input_mask_file),
        MASKNAME, OCEAN)

    # gravitational acceleration at mean sea level at the equator
    ge = 9.780356
    # gravitational acceleration at mean sea level over colatitudes
    # from Heiskanen and Moritz, Physical Geodesy, (1967)
    gs = ge*(1.0 + 5.2885e-3*np.cos(gridtheta)**2 - 5.9e-6*np.cos(2.0*gridtheta)**2)

    # read each reanalysis pressure field for each year
    regex_years = r'\d{4}' if (YEAR is None) else r'|'.join(map(str,YEAR))
    rx = re.compile(regex_pattern.format(regex_years), re.VERBOSE)
    input_files = [fi for fi in ddir.iterdir() if rx.match(fi.name)]
    # for each reanalysis file
    for INPUT_FILE in sorted(input_files):
        # extract parameters from filename
        if MODEL in ('MERRA-2',):
            # extract date from hourly files
            MOD,YEAR,MONTH,DAY,AUX = rx.findall(INPUT_FILE.name).pop()
            # output inverse barometer filename
            FILENAME = output_file_format.format(MOD,YEAR,MONTH,DAY,AUX)
        elif MODEL in ('ERA-Interim','ERA5'):
            # extract date from hourly files
            YEAR,MONTH,DAY = rx.findall(INPUT_FILE.name).pop()
            # output inverse barometer filename
            FILENAME = output_file_format.format(YEAR,MONTH,DAY)
        # full path to output file
        OUTPUT_FILE = ddir.joinpath(FILENAME)

        # read netCDF4 mean sea level file
        with netCDF4.Dataset(INPUT_FILE, 'r') as fileID:
            # number of time points in file
            nt, = fileID.variables[TIMENAME].shape
            # extract time and time units
            dinput = {}
            dinput[TIMENAME] = np.copy(fileID.variables[TIMENAME][:])
            TIME_UNITS = fileID.variables[TIMENAME].units
            # copy latitude and longitude
            dinput[LONNAME] = lon.copy()
            dinput[LATNAME] = lat.copy()
            # calculate sea level pressure anomalies and
            # traditional TOPEX/POSEIDON IB correction
            fill_value = fileID.variables[VARNAME]._FillValue
            SLP, TPX = ncdf_pressure(fileID, VARNAME, TIMENAME,
                mean_pressure, ocean_function, cell_areas)
        # calculate inverse barometer response
        dinput[IBNAME] = -SLP*(DENSITY*gs)**-1
        dinput[TPXNAME] = TPX
        # output to file
        ncdf_IB_write(dinput, fill_value,
            FILENAME=OUTPUT_FILE, IBNAME=IBNAME, TPXNAME=TPXNAME,
            LONNAME=LONNAME, LATNAME=LATNAME, TIMENAME=TIMENAME,
            TIME_UNITS=TIME_UNITS, TIME_LONGNAME=TIME_LONGNAME,
            UNITS=UNITS, DENSITY=DENSITY)
        # change permissions mode
        OUTPUT_FILE.chmod(mode=MODE)

# PURPOSE: write output inverse barometer fields data to file
def ncdf_IB_write(dinput, fill_value, FILENAME=None, IBNAME=None,
    TPXNAME=None, LONNAME=None, LATNAME=None, TIMENAME=None,
    TIME_UNITS=None, TIME_LONGNAME=None, UNITS=None, DENSITY=None):
    # validate input path
    FILENAME = pathlib.Path(FILENAME).expanduser().absolute()
    # opening NetCDF file for writing
    fileID = netCDF4.Dataset(FILENAME, 'w', format="NETCDF4")

    # Defining the NetCDF dimensions
    for key in [LONNAME,LATNAME,TIMENAME]:
        fileID.createDimension(key, len(dinput[key]))

    # defining the NetCDF variables
    nc = {}
    nc[LATNAME]=fileID.createVariable(LATNAME,dinput[LATNAME].dtype,(LATNAME,))
    nc[LONNAME]=fileID.createVariable(LONNAME,dinput[LONNAME].dtype,(LONNAME,))
    nc[TIMENAME]=fileID.createVariable(TIMENAME,dinput[TIMENAME].dtype,(TIMENAME,))
    nc[IBNAME] = fileID.createVariable(IBNAME, dinput[IBNAME].dtype,
        (TIMENAME,LATNAME,LONNAME,), fill_value=fill_value, zlib=True)
    nc[TPXNAME] = fileID.createVariable(TPXNAME, dinput[TPXNAME].dtype,
        (TIMENAME,LATNAME,LONNAME,), fill_value=fill_value, zlib=True)
    # filling NetCDF variables
    for key,val in dinput.items():
        nc[key][:] = val.copy()

    # Defining attributes for longitude and latitude
    nc[LONNAME].long_name = 'Longitude'
    nc[LONNAME].units = 'degrees_east'
    nc[LATNAME].long_name = 'Latitude'
    nc[LATNAME].units = 'degrees_north'
    # Defining attributes for time
    nc[TIMENAME].units = TIME_UNITS
    nc[TIMENAME].long_name = TIME_LONGNAME
    # Defining attributes for inverse barometer effect
    nc[IBNAME].long_name = 'Instantaneous_inverse_barometer_(IB)_response'
    nc[IBNAME].units = UNITS
    nc[IBNAME].density = DENSITY
    # Defining attributes for conventional inverse barometer correction
    nc[TPXNAME].long_name = 'Conventional_(TOPEX/POSEIDON)_IB_response'
    nc[TPXNAME].units = UNITS
    nc[TPXNAME].density = 1025.0
    nc[TPXNAME].gravity = -9.80665
    nc[TPXNAME].pressure = 101325.0

    # add software information
    fileID.software_reference = gz.version.project_name
    fileID.software_version = gz.version.full_version
    fileID.reference = f'Output from {pathlib.Path(sys.argv[0]).name}'

    # Output NetCDF structure information
    logging.info(FILENAME)
    logging.info(list(fileID.variables.keys()))

    # Closing the NetCDF file
    fileID.close()
    # clear nc dictionary variable
    nc = None

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Reads hourly mean sea level pressure
            fields from reanalysis and calculates the
            inverse-barometer response
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    choices = ['ERA-Interim','ERA5','MERRA-2']
    parser.add_argument('model',
        type=str, nargs='+',
        default=['ERA5','MERRA-2'], choices=choices,
        help='Reanalysis Model')
    # directory with reanalysis data
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=pathlib.Path.cwd(),
        help='Working data directory')
    # years to run
    now = datetime.datetime.now()
    parser.add_argument('--year','-Y',
        type=int, nargs='+', default=range(2000,now.year+1),
        help='Years of model outputs to run')
    # start and end years to run for mean
    parser.add_argument('--mean','-m',
        metavar=('START','END'), type=int, nargs=2,
        default=[2000,2020],
        help='Start and end year range for mean')
    # ocean fluidic density [kg/m^3]
    parser.add_argument('--density','-d',
        metavar='RHO', type=float, default=1030.0,
        help='Density of seawater in kg/m^3')
    # verbosity settings
    # verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        action='count', default=0,
        help='Verbose output of processing run')
    # permissions mode of the local directories and files (number in octal)
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

    # create logger
    loglevels = [logging.CRITICAL, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=loglevels[args.verbose])

    # for each reanalysis model
    for MODEL in args.model:
        # run program
        calculate_inverse_barometer(args.directory, MODEL, YEAR=args.year,
            RANGE=args.mean, DENSITY=args.density, MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
