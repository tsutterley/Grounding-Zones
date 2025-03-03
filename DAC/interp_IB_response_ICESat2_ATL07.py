#!/usr/bin/env python
u"""
interp_IB_ICESat2_ATL07.py
Written by Tyler Sutterley (05/2024)
Calculates and interpolates inverse-barometer responses to times and
    locations of ICESat-2 ATL07 sea ice height data

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -O X, --output-directory X: input/output data directory
    -R X, --reanalysis X: Reanalysis model to run
        ERA-Interim: http://apps.ecmwf.int/datasets/data/interim-full-moda
        ERA5: http://apps.ecmwf.int/data-catalogues/era5/?class=ea
        MERRA-2: https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/
    -m X, --mean X: Start and end year range for mean
    -d X, --density X: Density of seawater in kg/m^3
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
    io/ATL07.py: reads ICESat-2 sea ice height data files
    utilities.py: download and management utilities for syncing files

REFERENCES:
    C Wunsch and D Stammer, Atmospheric loading and the oceanic "inverted
        barometer" effect, Reviews of Geophysics, 35(1), 79--107, (1997).
        https://doi.org/10.1029/96RG03037
    P S Callahan, TOPEX/POSEIDON Project GDR Users Handbook, JPL Doc. D-8944,
        Rev. A, 84 pp., (1994)

UPDATE HISTORY:
    Updated 05/2024: use wrapper to importlib for optional dependencies
    Updated 04/2024: use timescale for temporal operations
    Updated 08/2023: create s3 filesystem when using s3 urls as input
        use time functions from pyTMD.time
    Updated 05/2023: use timescale class for time conversion operations
        using pathlib to define and operate on paths
    Updated 12/2022: single implicit import of grounding zone tools
        refactored ICESat-2 data product read programs under io
        use constants class from pyTMD for ellipsoidal parameters
    Updated 05/2022: use argparse descriptions within sphinx documentation
    Updated 10/2021: using python logging for handling verbose output
        added parsing for converting file lines to arguments
    Updated 05/2021: print full path of output filename
    Written 03/2021
"""
from __future__ import print_function

import re
import logging
import pathlib
import argparse
import datetime
import numpy as np
import scipy.interpolate
import grounding_zones as gz

# attempt imports
h5py = gz.utilities.import_dependency('h5py')
is2tk = gz.utilities.import_dependency('icesat2_toolkit')
netCDF4 = gz.utilities.import_dependency('netCDF4')
pyproj = gz.utilities.import_dependency('pyproj')
pyTMD = gz.utilities.import_dependency('pyTMD')
timescale = gz.utilities.import_dependency('timescale')

# PURPOSE: read land sea mask to get indices of oceanic values
def ncdf_landmask(FILENAME, MASKNAME, OCEAN):
    FILENAME = pathlib.Path(FILENAME).expanduser().absolute()
    with netCDF4.Dataset(FILENAME, mode='r') as fileID:
        landsea = np.squeeze(fileID.variables[MASKNAME][:].copy())
    return (landsea == OCEAN)

# PURPOSE: read reanalysis mean sea level pressure
def ncdf_mean_pressure(FILENAME, VARNAME, LONNAME, LATNAME):
    FILENAME = pathlib.Path(FILENAME).expanduser().absolute()
    with netCDF4.Dataset(FILENAME, mode='r') as fileID:
        # extract pressure and remove singleton dimensions
        mean_pressure = np.array(fileID.variables[VARNAME][:].squeeze())
        longitude = fileID.variables[LONNAME][:].squeeze()
        latitude = fileID.variables[LATNAME][:].squeeze()
    return (mean_pressure,longitude,latitude)

# PURPOSE: find pressure files in a directory
def find_pressure_files(ddir, MODEL, MJD):
    # verify input directory exists
    ddir = pathlib.Path(ddir).expanduser().absolute()
    # regular expression pattern for finding files
    if (MODEL == 'ERA-Interim'):
        regex_pattern = r'ERA\-Interim\-Hourly\-MSL\-({0})\.nc$'
        joiner = r'\-'
    elif (MODEL == 'ERA5'):
        regex_pattern = r'ERA5\-Hourly\-MSL\-({0})\.nc$'
        joiner = r'\-'
    elif (MODEL == 'MERRA-2'):
        regex_pattern = r'MERRA2_\d{{3}}.tavg1_2d_slv_Nx.({0}).(.*?).nc$'
        joiner = r''
    # list of dates to read
    dates = []
    # for each unique Modified Julian Day (MJD)
    for mjd in np.unique(np.floor(MJD)):
        # append day prior, day of and day after
        JD = mjd + np.arange(-1,2) + 2400000.5
        # convert from Julian Days to calendar dates
        Y,M,D,_,_,_ = timescale.time.convert_julian(JD,
            ASTYPE=int, FORMAT='tuple')
        # append day as formatted strings
        for y,m,d in zip(Y,M,D):
            dates.append(joiner.join([str(y),str(m).zfill(2),str(d).zfill(2)]))
    # compile regular expression pattern for finding dates
    rx = re.compile(regex_pattern.format('|'.join(dates)))
    flist = [f for f in ddir.iterdir() if rx.match(f.name)]
    # return the sorted list of unique files
    return sorted(set(flist))

# PURPOSE: read sea level pressure fields and calculate anomalies
def ncdf_pressure(FILENAMES,VARNAME,TIMENAME,LATNAME,MEAN,OCEAN,AREA):
    # shape of pressure field
    ny,nx = np.shape(MEAN)
    nfiles = len(FILENAMES)
    # allocate for pressure fields
    SLP = np.ma.zeros((24*nfiles,ny,nx))
    TPX = np.ma.zeros((24*nfiles,ny,nx))
    MJD = np.zeros((24*nfiles))
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
    # for each file
    for FILENAME in FILENAMES:
        FILENAME = pathlib.Path(FILENAME).expanduser().absolute()
        with netCDF4.Dataset(FILENAME, mode='r') as fileID:
            # extract coordinates
            latitude = fileID.variables[LATNAME][:].squeeze()
            # convert time to Modified Julian Days
            delta_time = np.copy(fileID.variables[TIMENAME][:])
            units = fileID.variables[TIMENAME].units
            epoch,to_secs = timescale.time.parse_date_string(units)
            for t,dt in enumerate(delta_time):
                MJD[c] = timescale.time.convert_delta_time(dt*to_secs,
                    epoch1=epoch, epoch2=(1858,11,17,0,0,0), scale=1.0/86400.0)
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
                            TPX[c,:,:] = (pressure[j,:,:] - p0)/(rho0*g0)
                            break
                else:
                    # sea level pressure for time
                    pressure = fileID.variables[VARNAME][t,:,:].copy()
                    # remove average with respect to time
                    AveRmvd = pressure - MEAN
                    # conventional TOPEX/POSEIDON IB correction
                    TPX[c,:,:] = (pressure - p0)/(rho0*g0)
                # calculate average oceanic pressure values
                AVERAGE = np.sum(AveRmvd[ii,jj]*AREA[ii,jj])/ocean_area
                # calculate sea level pressure anomalies
                SLP[c,:,:] = AveRmvd - AVERAGE
                # clear temp variables for iteration to free up memory
                pressure,AveRmvd = (None,None)
                # add to counter
                c += 1
    # verify latitudes are sorted in ascending order
    ilat = np.argsort(latitude)
    SLP = SLP[:,ilat,:]
    TPX = TPX[:,ilat,:]
    latitude = latitude[ilat]
    # verify time is sorted in ascending order
    itime = np.argsort(MJD)
    SLP = SLP[itime,:,:]
    TPX = TPX[itime,:,:]
    MJD = MJD[itime]
    # return the sea level pressure anomalies and times
    return (SLP, TPX, latitude, MJD)

# PURPOSE: read ICESat-2 sea ice height (ATL07)
# calculate and interpolate the instantaneous inverse barometer response
def interp_IB_response_ICESat2(base_dir, INPUT_FILE, MODEL,
    OUTPUT_DIRECTORY=None,
    RANGE=None,
    DENSITY=None,
    VERBOSE=False,
    MODE=0o775):

    # create logger
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # log input file
    logging.info(f'{str(INPUT_FILE)} -->')
    # input granule basename
    GRANULE = INPUT_FILE.name

    # directory setup for reanalysis model
    base_dir = pathlib.Path(base_dir).expanduser().absolute()
    ddir = base_dir.joinpath(MODEL)
    # set model specific parameters
    if (MODEL == 'ERA-Interim'):
        # mean sea level pressure file
        input_mean_file = 'ERA-Interim-Mean-MSL-{0:4d}-{1:4d}.nc'
        # input land-sea mask for ocean redistribution
        input_mask_file = 'ERA-Interim-Invariant-Parameters.nc'
        VARNAME = 'msl'
        LONNAME = 'longitude'
        LATNAME = 'latitude'
        TIMENAME = 'time'
        # land-sea mask variable name and value of oceanic points
        MASKNAME = 'lsm'
        OCEAN = 0
        # projection string
        proj4_params = ('+proj=longlat +ellps=WGS84 +datum=WGS84 '
            '+no_defs lon_wrap=180')
    elif (MODEL == 'ERA5'):
        # mean sea level pressure file
        input_mean_file = 'ERA5-Mean-MSL-{0:4d}-{1:4d}.nc'
        # input land-sea mask for ocean redistribution
        input_mask_file = 'ERA5-Invariant-Parameters.nc'
        VARNAME = 'msl'
        LONNAME = 'longitude'
        LATNAME = 'latitude'
        TIMENAME = 'time'
        # land-sea mask variable name and value of oceanic points
        MASKNAME = 'lsm'
        OCEAN = 0
        # projection string
        proj4_params = ('+proj=longlat +ellps=WGS84 +datum=WGS84 '
            '+no_defs lon_wrap=180')
    elif (MODEL == 'MERRA-2'):
        # mean sea level pressure file
        input_mean_file = 'MERRA2.Mean_SLP.{0:4d}-{1:4d}.nc'
        # input land-sea mask for ocean redistribution
        input_mask_file = 'MERRA2_101.const_2d_asm_Nx.00000000.nc4'
        VARNAME = 'SLP'
        LONNAME = 'lon'
        LATNAME = 'lat'
        TIMENAME = 'time'
        # land-sea mask variable name and value of oceanic points
        MASKNAME = 'FROCEAN'
        OCEAN = 1
        # projection string
        proj4_params = 'epsg:4326'

    # extract parameters from ICESat-2 ATLAS HDF5 sea ice file name
    rx = re.compile(r'(processed_)?(ATL\d{2})-(\d{2})_(\d{4})(\d{2})(\d{2})'
        r'(\d{2})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    try:
        SUB,PRD,HEM,YY,MM,DD,HH,MN,SS,TRK,CYCL,SN,RL,VERS,AUX = \
            rx.findall(GRANULE).pop()
    except:
        # output inverse barometer response HDF5 file (generic)
        args = (INPUT_FILE.stem,MODEL,INPUT_FILE.suffix)
        FILENAME = '{0}_{1}_IB{2}'.format(*args)
    else:
        # output inverse barometer response HDF5 file for ASAS/NSIDC granules
        args = (PRD,HEM,MODEL,YY,MM,DD,HH,MN,SS,TRK,CYCL,SN,RL,VERS,AUX)
        ff = '{0}-{1}_{2}_IB_{3}{4}{5}{6}{7}{8}_{9}{10}{11}_{12}_{13}{14}.h5'
        FILENAME = ff.format(*args)
    # get output directory from input file
    if OUTPUT_DIRECTORY is None:
        OUTPUT_DIRECTORY = INPUT_FILE.parent
    # full path to output file
    OUTPUT_FILE = OUTPUT_DIRECTORY.joinpath(FILENAME)

    # check if data is an s3 presigned url
    if str(INPUT_FILE).startswith('s3:'):
        client = gz.utilities.attempt_login('urs.earthdata.nasa.gov',
            authorization_header=True)
        session = gz.utilities.s3_filesystem()
        INPUT_FILE = session.open(INPUT_FILE, mode='rb')
    else:
        INPUT_FILE = pathlib.Path(INPUT_FILE).expanduser().absolute()

    # read data from input file
    IS2_atl07_mds,IS2_atl07_attrs,IS2_atl07_beams = \
        is2tk.io.ATL07.read_granule(INPUT_FILE, ATTRIBUTES=True)

    # read mean pressure field
    mean_file = ddir.joinpath(input_mean_file.format(RANGE[0], RANGE[1]))
    mean_pressure,lon,lat=ncdf_mean_pressure(mean_file,VARNAME,LONNAME,LATNAME)

    # pyproj transformer for converting from input coordinates (EPSG)
    # to model coordinates
    crs1 = pyproj.CRS.from_string('epsg:{0:d}'.format(4326))
    crs2 = pyproj.CRS.from_string(proj4_params)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

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
    AREA = dphi*dth*np.sin(gridtheta)*np.sqrt((a_axis**2)*(b_axis**2) *
        ((np.sin(gridtheta)**2)*(np.cos(gridphi)**2) +
        (np.sin(gridtheta)**2)*(np.sin(gridphi)**2)) +
        (a_axis**4)*(np.cos(gridtheta)**2))
    # read land-sea mask to find ocean values
    # ocean pressure points will be based on reanalysis mask
    MASK = ncdf_landmask(ddir.joinpath(input_mask_file), MASKNAME, OCEAN)

    # find and read each reanalysis pressure field
    MJD = timescale.time.convert_calendar_dates(int(YY),int(MM),int(DD),
        epoch=(1858,11,17,0,0,0), scale=1.0)
    FILENAMES = find_pressure_files(ddir, MODEL, MJD)
    # read sea level pressure and calculate anomalies
    islp,itpx,ilat,imjd = ncdf_pressure(FILENAMES, VARNAME, TIMENAME,
        LATNAME, mean_pressure, MASK, AREA)
    # create an interpolator for sea level pressure anomalies
    R1 = scipy.interpolate.RegularGridInterpolator((imjd,ilat,lon), islp,
        bounds_error=False)
    R2 = scipy.interpolate.RegularGridInterpolator((imjd,ilat,lon), itpx,
        bounds_error=False)

    # copy variables for outputting to HDF5 file
    IS2_atl07_corr = {}
    IS2_atl07_fill = {}
    IS2_atl07_dims = {}
    IS2_atl07_corr_attrs = {}
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl07_corr['ancillary_data'] = {}
    IS2_atl07_corr_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl07_corr['ancillary_data'][key] = IS2_atl07_mds['ancillary_data'][key]
        # Getting attributes of group and included variables
        IS2_atl07_corr_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl07_attrs['ancillary_data'][key].items():
            IS2_atl07_corr_attrs['ancillary_data'][key][att_name] = att_val
    # for each input beam within the file
    for gtx in sorted(IS2_atl07_beams):
        # output data dictionaries for beam
        IS2_atl07_corr[gtx] = dict(sea_ice_segments={})
        IS2_atl07_fill[gtx] = dict(sea_ice_segments={})
        IS2_atl07_dims[gtx] = dict(sea_ice_segments={})
        IS2_atl07_corr_attrs[gtx] = dict(sea_ice_segments={})

        # number of segments
        val = IS2_atl07_mds[gtx]['sea_ice_segments']
        n_seg = len(val['height_segment_id'])

        # create timescale from ATLAS Standard Epoch time
        # GPS seconds since 2018-01-01 00:00:00 UTC
        ts = timescale.time.Timescale().from_deltatime(val['delta_time'],
            epoch=timescale.time._atlas_sdp_epoch, standard='GPS')

        # calculate projected coordinates of input coordinates
        ix,iy = transformer.transform(val['longitude'], val['latitude'])

        # colatitudes of the ATL07 measurements
        th = (90.0 - val['latitude'])*np.pi/180.0
        # gravitational acceleration at mean sea level at the equator
        ge = 9.780356
        # gravitational acceleration at mean sea level over colatitudes
        # from Heiskanen and Moritz, Physical Geodesy, (1967)
        gs = ge*(1.0 + 5.2885e-3*np.cos(th)**2 - 5.9e-6*np.cos(2.0*th)**2)

        # interpolate sea level pressure anomalies to points
        SLP = R1.__call__(np.c_[ts.MJD,iy,ix])
        # calculate inverse barometer response
        IB = np.ma.zeros((n_seg))
        IB.data[:] = -SLP*(DENSITY*gs)**-1
        # interpolate conventional inverse barometer response to points
        TPX = np.ma.zeros((n_seg))
        TPX.data[:] = R2.__call__(np.c_[ts.MJD,iy,ix])
        # replace any nan values with fill value
        IB.mask = np.isnan(IB.data)
        TPX.mask = np.isnan(TPX.data)
        IB.data[IB.mask] = IB.fill_value
        TPX.data[TPX.mask] = TPX.fill_value

        # group attributes for beam
        IS2_atl07_corr_attrs[gtx]['Description'] = IS2_atl07_attrs[gtx]['Description']
        IS2_atl07_corr_attrs[gtx]['atlas_pce'] = IS2_atl07_attrs[gtx]['atlas_pce']
        IS2_atl07_corr_attrs[gtx]['atlas_beam_type'] = IS2_atl07_attrs[gtx]['atlas_beam_type']
        IS2_atl07_corr_attrs[gtx]['groundtrack_id'] = IS2_atl07_attrs[gtx]['groundtrack_id']
        IS2_atl07_corr_attrs[gtx]['atmosphere_profile'] = IS2_atl07_attrs[gtx]['atmosphere_profile']
        IS2_atl07_corr_attrs[gtx]['atlas_spot_number'] = IS2_atl07_attrs[gtx]['atlas_spot_number']
        IS2_atl07_corr_attrs[gtx]['sc_orientation'] = IS2_atl07_attrs[gtx]['sc_orientation']
        # group attributes for sea_ice_segments
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['Description'] = ("Top group for sea "
            "ice segments as computed by the ATBD algorithm.")
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['data_rate'] = ("Data within this "
            "group are stored at the variable segment rate.")

        # geolocation, time and segment ID
        # delta time
        IS2_atl07_corr[gtx]['sea_ice_segments']['delta_time'] = val['delta_time'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['delta_time'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['delta_time'] = None
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['delta_time'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['delta_time']['standard_name'] = "time"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['delta_time']['source'] = "telemetry"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['delta_time']['calendar'] = "standard"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['delta_time']['description'] = ("Number of "
            "GPS seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch "
            "offset is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS "
            "seconds between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP "
            "epoch. By adding the offset contained within atlas_sdp_gps_epoch to delta time "
            "parameters, the time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['delta_time']['coordinates'] = \
            "height_segment_id latitude longitude"
        # latitude
        IS2_atl07_corr[gtx]['sea_ice_segments']['latitude'] = val['latitude'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['latitude'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['latitude'] = ['delta_time']
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['latitude'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['latitude']['units'] = "degrees_north"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['latitude']['long_name'] = "Latitude"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['latitude']['standard_name'] = "latitude"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['latitude']['description'] = ("Latitude of "
            "segment center")
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['latitude']['valid_min'] = -90.0
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['latitude']['valid_max'] = 90.0
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['latitude']['coordinates'] = \
            "height_segment_id delta_time longitude"
        # longitude
        IS2_atl07_corr[gtx]['sea_ice_segments']['longitude'] = val['longitude'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['longitude'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['longitude'] = ['delta_time']
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['longitude'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['longitude']['units'] = "degrees_east"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['longitude']['long_name'] = "Longitude"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['longitude']['standard_name'] = "longitude"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['longitude']['description'] = ("Longitude of "
            "segment center")
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['longitude']['valid_min'] = -180.0
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['longitude']['valid_max'] = 180.0
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['longitude']['coordinates'] = \
            "height_segment_id delta_time latitude"
        # segment ID
        IS2_atl07_corr[gtx]['sea_ice_segments']['height_segment_id'] = val['height_segment_id']
        IS2_atl07_fill[gtx]['sea_ice_segments']['height_segment_id'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['height_segment_id'] = ['delta_time']
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['height_segment_id'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['height_segment_id']['units'] = "1"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['height_segment_id']['contentType'] = "referenceInformation"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['height_segment_id']['long_name'] = \
            "Identifier of each height segment"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['height_segment_id']['description'] = \
            "Identifier of each height segment"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['height_segment_id']['coordinates'] = \
            "delta_time latitude longitude"
        # geolocation segment beginning
        IS2_atl07_corr[gtx]['sea_ice_segments']['geoseg_beg'] = val['geoseg_beg'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['geoseg_beg'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['geoseg_beg'] = ['delta_time']
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_beg'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_beg']['units'] = "1"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_beg']['contentType'] = "referenceInformation"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_beg']['long_name'] = "Beginning GEOSEG"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_beg']['description'] = \
            "Geolocation segment (geoseg) ID associated with the first photon used in this sea ice segment"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_beg']['coordinates'] = \
            "height_segment_id delta_time latitude longitude"
        # geolocation segment ending
        IS2_atl07_corr[gtx]['sea_ice_segments']['geoseg_end'] = val['geoseg_end'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['geoseg_end'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['geoseg_end'] = ['delta_time']
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_end'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_end']['units'] = "1"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_end']['contentType'] = "referenceInformation"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_end']['long_name'] = "Ending GEOSEG"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_end']['description'] = \
            "Geolocation segment (geoseg) ID associated with the last photon used in this sea ice segment"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geoseg_end']['coordinates'] = \
            "height_segment_id delta_time latitude longitude"
        # along track distance
        IS2_atl07_corr[gtx]['sea_ice_segments']['seg_dist_x'] = val['seg_dist_x'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['seg_dist_x'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['seg_dist_x'] = ['delta_time']
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['seg_dist_x'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['seg_dist_x']['units'] = "meters"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['seg_dist_x']['contentType'] = "referenceInformation"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['seg_dist_x']['long_name'] = "Along track distance"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['seg_dist_x']['description'] = \
            "Along-track distance from the equator crossing to the segment center."
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['seg_dist_x']['coordinates'] = \
            "height_segment_id delta_time latitude longitude"

        # geophysical variables
        IS2_atl07_corr[gtx]['sea_ice_segments']['geophysical'] = {}
        IS2_atl07_fill[gtx]['sea_ice_segments']['geophysical'] = {}
        IS2_atl07_dims[gtx]['sea_ice_segments']['geophysical'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['Description'] = ("Contains geophysical "
            "parameters and corrections used to correct photon heights for geophysical effects, such as tides.")
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['data_rate'] = ("Data within this group "
            "are stored at the sea_ice_height segment rate.")

        # inverse barometer response
        IS2_atl07_corr[gtx]['sea_ice_segments']['geophysical']['height_segment_ib'] = IB.copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['geophysical']['height_segment_ib'] = IB.fill_value
        IS2_atl07_dims[gtx]['sea_ice_segments']['geophysical']['height_segment_ib'] = ['delta_time']
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_ib'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_ib']['units'] = "meters"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_ib']['contentType'] = "referenceInformation"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_ib']['long_name'] = "inverse barometer"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_ib']['description'] = ("Instantaneous inverse "
            "barometer effect due to atmospheric loading")
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_ib']['source'] = MODEL
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_ib']['reference'] = \
            'https://doi.org/10.1029/96RG03037'
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_ib']['coordinates'] = \
            "../segment_id ../delta_time ../latitude ../longitude"
        # conventional (TOPEX/POSEIDON) inverse barometer response
        IS2_atl07_corr[gtx]['sea_ice_segments']['geophysical']['height_segment_ib_tpx'] = TPX.copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['geophysical']['height_segment_ib_tpx'] = TPX.fill_value
        IS2_atl07_dims[gtx]['sea_ice_segments']['geophysical']['height_segment_ib_tpx'] = ['delta_time']
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_ib_tpx'] = {}
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_ib_tpx']['units'] = "meters"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_ib_tpx']['contentType'] = "referenceInformation"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_ib_tpx']['long_name'] = "inverse barometer"
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_ib_tpx']['description'] = ("Conventional "
            "(TOPEX/POSEIDON) instantaneous inverse barometer effect due to atmospheric loading")
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_ib_tpx']['source'] = MODEL
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_ib_tpx']['reference'] = \
            'TOPEX/POSEIDON Project GDR Users Handbook'
        IS2_atl07_corr_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_ib_tpx']['coordinates'] = \
            "../height_segment_id ../delta_time ../latitude ../longitude"

    # print file information
    logging.info(f'\t{str(OUTPUT_FILE)}')
    HDF5_ATL07_corr_write(IS2_atl07_corr, IS2_atl07_corr_attrs,
        CLOBBER=True, INPUT=GRANULE,
        FILL_VALUE=IS2_atl07_fill, DIMENSIONS=IS2_atl07_dims,
        FILENAME=OUTPUT_FILE)
    # change the permissions mode
    OUTPUT_FILE.chmod(mode=MODE)

# PURPOSE: outputting the correction values for ICESat-2 data to HDF5
def HDF5_ATL07_corr_write(IS2_atl07_corr, IS2_atl07_attrs, INPUT=None,
    FILENAME='', FILL_VALUE=None, DIMENSIONS=None, CLOBBER=False):
    # setting HDF5 clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'w-'

    # open output HDF5 file
    FILENAME = pathlib.Path(FILENAME).expanduser().absolute()
    fileID = h5py.File(FILENAME, clobber)

    # create HDF5 records
    h5 = {}

    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    h5['ancillary_data'] = {}
    for k,v in IS2_atl07_corr['ancillary_data'].items():
        # Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS2_atl07_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    # write each output beam
    beams = [k for k in IS2_atl07_corr.keys() if bool(re.match(r'gt\d[lr]',k))]
    for gtx in beams:
        fileID.create_group(gtx)
        # add HDF5 group attributes for beam
        for att_name in ['Description','atlas_pce','atlas_beam_type',
            'groundtrack_id','atmosphere_profile','atlas_spot_number',
            'sc_orientation']:
            fileID[gtx].attrs[att_name] = IS2_atl07_attrs[gtx][att_name]
        # create sea_ice_segments group
        fileID[gtx].create_group('sea_ice_segments')
        h5[gtx] = dict(sea_ice_segments={})
        for att_name in ['Description','data_rate']:
            att_val = IS2_atl07_attrs[gtx]['sea_ice_segments'][att_name]
            fileID[gtx]['sea_ice_segments'].attrs[att_name] = att_val

        # delta_time, geolocation and segment identification variables
        for k in ['delta_time','latitude','longitude','height_segment_id',
            'geoseg_beg','geoseg_end','seg_dist_x']:
            # values and attributes
            v = IS2_atl07_corr[gtx]['sea_ice_segments'][k]
            attrs = IS2_atl07_attrs[gtx]['sea_ice_segments'][k]
            fillvalue = FILL_VALUE[gtx]['sea_ice_segments'][k]
            # Defining the HDF5 dataset variables
            val = '{0}/{1}/{2}'.format(gtx,'sea_ice_segments',k)
            if fillvalue:
                h5[gtx]['sea_ice_segments'][k] = fileID.create_dataset(val,
                    np.shape(v), data=v, dtype=v.dtype, fillvalue=fillvalue,
                    compression='gzip')
            else:
                h5[gtx]['sea_ice_segments'][k] = fileID.create_dataset(val,
                    np.shape(v), data=v, dtype=v.dtype, compression='gzip')
            # create or attach dimensions for HDF5 variable
            if DIMENSIONS[gtx]['sea_ice_segments'][k]:
                # attach dimensions
                for i,dim in enumerate(DIMENSIONS[gtx]['sea_ice_segments'][k]):
                    h5[gtx]['sea_ice_segments'][k].dims[i].attach_scale(
                        h5[gtx]['sea_ice_segments'][dim])
            else:
                # make dimension
                h5[gtx]['sea_ice_segments'][k].make_scale(k)
            # add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5[gtx]['sea_ice_segments'][k].attrs[att_name] = att_val

        # add to geophysical corrections
        key = 'geophysical'
        fileID[gtx]['sea_ice_segments'].create_group(key)
        h5[gtx]['sea_ice_segments'][key] = {}
        for att_name in ['Description','data_rate']:
            att_val=IS2_atl07_attrs[gtx]['sea_ice_segments'][key][att_name]
            fileID[gtx]['sea_ice_segments'][key].attrs[att_name] = att_val
        for k,v in IS2_atl07_corr[gtx]['sea_ice_segments'][key].items():
            # attributes
            attrs = IS2_atl07_attrs[gtx]['sea_ice_segments'][key][k]
            fillvalue = FILL_VALUE[gtx]['sea_ice_segments'][key][k]
            # Defining the HDF5 dataset variables
            val = '{0}/{1}/{2}/{3}'.format(gtx,'sea_ice_segments',key,k)
            if fillvalue:
                h5[gtx]['sea_ice_segments'][key][k] = \
                    fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, fillvalue=fillvalue, compression='gzip')
            else:
                h5[gtx]['sea_ice_segments'][key][k] = \
                    fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, compression='gzip')
            # attach dimensions
            for i,dim in enumerate(DIMENSIONS[gtx]['sea_ice_segments'][key][k]):
                h5[gtx]['sea_ice_segments'][key][k].dims[i].attach_scale(
                    h5[gtx]['sea_ice_segments'][dim])
            # add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5[gtx]['sea_ice_segments'][key][k].attrs[att_name] = att_val

    # HDF5 file title
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'ATLAS/ICESat-2 L3A Sea Ice Height'
    fileID.attrs['summary'] = ('Estimates of the sea ice correction parameters '
        'needed to interpret and assess the quality of the height estimates.')
    fileID.attrs['description'] = ('The data set (ATL07) contains along-track '
        'heights for sea ice and open water leads (at varying length scales) '
        'relative to the WGS84 ellipsoid (ITRF2014 reference frame) after '
        'adjustment for geoidal and tidal variations, and inverted barometer '
        'effects.')
    date_created = datetime.datetime.today()
    fileID.attrs['date_created'] = date_created.isoformat()
    project = 'ICESat-2 > Ice, Cloud, and land Elevation Satellite-2'
    fileID.attrs['project'] = project
    platform = 'ICESat-2 > Ice, Cloud, and land Elevation Satellite-2'
    fileID.attrs['project'] = platform
    # add attribute for elevation instrument and designated processing level
    instrument = 'ATLAS > Advanced Topographic Laser Altimeter System'
    fileID.attrs['instrument'] = instrument
    fileID.attrs['source'] = 'Spacecraft'
    fileID.attrs['references'] = 'https://nsidc.org/data/icesat-2'
    fileID.attrs['processing_level'] = '4'
    # add attributes for input ATL07 file
    fileID.attrs['lineage'] = pathlib.Path(INPUT).name
    # find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for gtx in beams:
        lon = IS2_atl07_corr[gtx]['sea_ice_segments']['longitude']
        lat = IS2_atl07_corr[gtx]['sea_ice_segments']['latitude']
        delta_time = IS2_atl07_corr[gtx]['sea_ice_segments']['delta_time']
        # setting the geospatial and temporal ranges
        lnmn = lon.min() if (lon.min() < lnmn) else lnmn
        lnmx = lon.max() if (lon.max() > lnmx) else lnmx
        ltmn = lat.min() if (lat.min() < ltmn) else ltmn
        ltmx = lat.max() if (lat.max() > ltmx) else ltmx
        tmn = delta_time.min() if (delta_time.min() < tmn) else tmn
        tmx = delta_time.max() if (delta_time.max() > tmx) else tmx
    # add geospatial and temporal attributes
    fileID.attrs['geospatial_lat_min'] = ltmn
    fileID.attrs['geospatial_lat_max'] = ltmx
    fileID.attrs['geospatial_lon_min'] = lnmn
    fileID.attrs['geospatial_lon_max'] = lnmx
    fileID.attrs['geospatial_lat_units'] = "degrees_north"
    fileID.attrs['geospatial_lon_units'] = "degrees_east"
    fileID.attrs['geospatial_ellipsoid'] = "WGS84"
    fileID.attrs['date_type'] = 'UTC'
    fileID.attrs['time_type'] = 'CCSDS UTC-A'
    # convert start and end time from ATLAS SDP seconds into timescale
    ts = timescale.time.Timescale().from_deltatime(np.array([tmn,tmx]),
        epoch=timescale.time._atlas_sdp_epoch, standard='GPS')
    dt = np.datetime_as_string(ts.to_datetime(), unit='s')
    # add attributes with measurement date start, end and duration
    fileID.attrs['time_coverage_start'] = str(dt[0])
    fileID.attrs['time_coverage_end'] = str(dt[1])
    fileID.attrs['time_coverage_duration'] = f'{tmx-tmn:0.0f}'
    # add software information
    fileID.attrs['software_reference'] = gz.version.project_name
    fileID.attrs['software_version'] = gz.version.full_version
    # Closing the HDF5 file
    fileID.close()

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates and interpolates inverse-barometer
            responses to times and locations of ICESat-2 ATL07 sea
            ice height data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='ICESat-2 ATL07 file to run')
    # directory with reanalysis data
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=pathlib.Path.cwd(),
        help='Working data directory')
    # directory with input/output data
    parser.add_argument('--output-directory','-O',
        type=pathlib.Path,
        help='Output data directory')
    # reanalysis model
    choices = ['ERA-Interim','ERA5','MERRA-2']
    parser.add_argument('--reanalysis','-R',
        metavar='REANALYSIS', type=str,
        default='ERA5', choices=choices,
        help='Reanalysis Model')
    # start and end years to run for mean
    parser.add_argument('--mean','-m',
        metavar=('START','END'), type=int, nargs=2,
        default=[2000,2020],
        help='Start and end year range for mean')
    # ocean fluidic density [kg/m^3]
    parser.add_argument('--density','-d',
        metavar='RHO', type=float, default=1030.0,
        help='Density of seawater in kg/m^3')
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

    # run for each input ATL07 file
    for FILE in args.infile:
        interp_IB_response_ICESat2(args.directory, FILE, args.reanalysis,
            OUTPUT_DIRECTORY=args.output_directory,
            RANGE=args.mean,
            DENSITY=args.density,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
