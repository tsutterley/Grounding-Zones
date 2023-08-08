#!/usr/bin/env python
u"""
interp_IB_ICESat_GLA12.py
Written by Tyler Sutterley (08/2023)
Calculates and interpolates inverse-barometer responses for ICESat/GLAS
    L2 GLA12 Antarctic and Greenland Ice Sheet elevation data

Data will be interpolated for all valid points
    (masking land values will be needed for accurate assessments)

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

PROGRAM DEPENDENCIES:
    time.py: utilities for calculating time operations
    spatial.py: utilities for reading and writing spatial data
    utilities.py: download and management utilities for syncing files

REFERENCES:
    C Wunsch and D Stammer, Atmospheric loading and the oceanic "inverted
        barometer" effect, Reviews of Geophysics, 35(1), 79--107, (1997).
        https://doi.org/10.1029/96RG03037
    P S Callahan, TOPEX/POSEIDON Project GDR Users Handbook, JPL Doc. D-8944,
        Rev. A, 84 pp., (1994)

UPDATE HISTORY:
    Updated 08/2023: create s3 filesystem when using s3 urls as input
    Updated 12/2022: single implicit import of grounding zone tools
        use constants class from pyTMD for ellipsoidal parameters
    Updated 11/2022: use f-strings for formatting verbose or ascii output
    Updated 05/2022: use argparse descriptions within sphinx documentation
    Updated 10/2021: using python logging for handling verbose output
        added parsing for converting file lines to arguments
    Updated 05/2021: print full path of output filename
    Updated 03/2021: simplify read pressure values routine
        additionally calculate conventional IB response using an average MSLP
        replaced numpy bool/int to prevent deprecation warnings
    Written 02/2021
"""
from __future__ import print_function

import re
import pyproj
import logging
import pathlib
import argparse
import warnings
import numpy as np
import scipy.interpolate
import grounding_zones as gz

# attempt imports
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
try:
    import netCDF4
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("netCDF4 not available", ImportWarning)
try:
    import pyTMD
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("pyTMD not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

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
    if not ddir.exists():
        raise FileNotFoundError(f'Directory {str(ddir)} not found')
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
        Y,M,D,_,_,_ = is2tk.time.convert_julian(JD,
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
            epoch,to_secs = is2tk.time.parse_date_string(units)
            for t,dt in enumerate(delta_time):
                MJD[c] = is2tk.time.convert_delta_time(dt*to_secs,
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

# PURPOSE: read ICESat ice sheet HDF5 elevation data (GLAH12) from NSIDC
# calculate and interpolate the instantaneous inverse barometer response
def interp_IB_response_ICESat(base_dir, INPUT_FILE, MODEL,
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

    # compile regular expression operator for extracting information from file
    rx = re.compile((r'GLAH(\d{2})_(\d{3})_(\d{1})(\d{1})(\d{2})_(\d{3})_'
        r'(\d{4})_(\d{1})_(\d{2})_(\d{4})\.H5'), re.VERBOSE)
    # extract parameters from ICESat/GLAS HDF5 file name
    # PRD:  Product number (01, 05, 06, 12, 13, 14, or 15)
    # RL:  Release number for process that created the product = 634
    # RGTP:  Repeat ground-track phase (1=8-day, 2=91-day, 3=transfer orbit)
    # ORB:   Reference orbit number (starts at 1 and increments each time a
    #           new reference orbit ground track file is obtained.)
    # INST:  Instance number (increments every time the satellite enters a
    #           different reference orbit)
    # CYCL:   Cycle of reference orbit for this phase
    # TRK: Track within reference orbit
    # SEG:   Segment of orbit
    # GRAN:  Granule version number
    # TYPE:  File type
    try:
        PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE = rx.findall(GRANULE).pop()
    except:
        # output inverse-barometer response HDF5 file (generic)
        FILENAME = f'{INPUT_FILE.stem}_{MODEL}_IB{INPUT_FILE.suffix}'
    else:
        # output inverse-barometer response HDF5 file for NSIDC granules
        args = (PRD,RL,MODEL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        file_format = 'GLAH{0}_{1}_{2}_IB_{3}{4}{5}_{6}_{7}_{8}_{9}_{10}.h5'
        FILENAME = file_format.format(*args)
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

    # read GLAH12 HDF5 file
    fileID = h5py.File(INPUT_FILE, mode='r')
    # get variables and attributes
    rec_ndx_40HZ = fileID['Data_40HZ']['Time']['i_rec_ndx'][:].copy()
    # seconds since 2000-01-01 12:00:00 UTC (J2000)
    DS_UTCTime_40HZ = fileID['Data_40HZ']['DS_UTCTime_40'][:].copy()
    # Latitude (degrees North)
    lat_TPX = fileID['Data_40HZ']['Geolocation']['d_lat'][:].copy()
    # Longitude (degrees East)
    lon_40HZ = fileID['Data_40HZ']['Geolocation']['d_lon'][:].copy()
    # Elevation (height above TOPEX/Poseidon ellipsoid in meters)
    elev_TPX = fileID['Data_40HZ']['Elevation_Surfaces']['d_elev'][:].copy()
    fv = fileID['Data_40HZ']['Elevation_Surfaces']['d_elev'].attrs['_FillValue']

    # create timescale from J2000: seconds since 2000-01-01 12:00:00 UTC
    timescale = pyTMD.time.timescale().from_deltatime(DS_UTCTime_40HZ[:],
        epoch=pyTMD.time._j2000_epoch, standard='UTC')

    # parameters for Topex/Poseidon and WGS84 ellipsoids
    topex = pyTMD.constants('TOPEX')
    wgs84 = pyTMD.constants('WGS84')
    # convert from Topex/Poseidon to WGS84 Ellipsoids
    lat_40HZ,elev_40HZ = is2tk.spatial.convert_ellipsoid(lat_TPX, elev_TPX,
        topex.a_axis, topex.flat, wgs84.a_axis, wgs84.flat, eps=1e-12, itmax=10)
    # colatitude in radians
    theta_40HZ = (90.0 - lat_40HZ)*np.pi/180.0

    # pyproj transformer for converting from input coordinates (EPSG)
    # to model coordinates
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_string(proj4_params)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    # calculate projected coordinates of input coordinates
    ix,iy = transformer.transform(lon_40HZ, lat_40HZ)

    # read mean pressure field
    mean_file = ddir.joinpath(input_mean_file.format(RANGE[0],RANGE[1]))
    mean_pressure,lon,lat = ncdf_mean_pressure(mean_file,VARNAME,LONNAME,LATNAME)

    # grid step size in radians
    dphi = np.pi*np.abs(lon[1] - lon[0])/180.0
    dth = np.pi*np.abs(lat[1] - lat[0])/180.0
    # calculate meshgrid from latitude and longitude
    gridlon,gridlat = np.meshgrid(lon,lat)
    gridphi = gridlon*np.pi/180.0
    # calculate colatitude
    gridtheta = (90.0 - gridlat)*np.pi/180.0

    # semiminor axis of the ellipsoid [m]
    b_axis = (1.0 - wgs84.flat)*wgs84.a_axis
    # calculate grid areas globally
    AREA = dphi*dth*np.sin(gridtheta)*np.sqrt((wgs84.a_axis**2)*(b_axis**2) *
        ((np.sin(gridtheta)**2)*(np.cos(gridphi)**2) +
        (np.sin(gridtheta)**2)*(np.sin(gridphi)**2)) +
        (wgs84.a_axis**4)*(np.cos(gridtheta)**2))
    # read land-sea mask to find ocean values
    # ocean pressure points will be based on reanalysis mask
    MASK = ncdf_landmask(ddir.joinpath(input_mask_file),MASKNAME,OCEAN)

    # find and read each reanalysis pressure field
    FILENAMES = find_pressure_files(ddir, MODEL, timescale.MJD)
    # read sea level pressure and calculate anomalies
    islp,itpx,ilat,imjd = ncdf_pressure(FILENAMES, VARNAME, TIMENAME,
        LATNAME, mean_pressure, MASK, AREA)
    # create an interpolator for sea level pressure anomalies
    R1 = scipy.interpolate.RegularGridInterpolator((imjd,ilat,lon), islp,
        bounds_error=False)
    R2 = scipy.interpolate.RegularGridInterpolator((imjd,ilat,lon), itpx,
        bounds_error=False)

    # gravitational acceleration at mean sea level at the equator
    ge = 9.780356
    # gravitational acceleration at mean sea level over colatitudes
    # from Heiskanen and Moritz, Physical Geodesy, (1967)
    gs = ge*(1.0 + 5.2885e-3*np.cos(theta_40HZ)**2 - 5.9e-6*np.cos(2.0*theta_40HZ)**2)

    # interpolate sea level pressure anomalies to points
    SLP = R1.__call__(np.c_[timescale.MJD, iy, ix])
    # calculate inverse barometer response
    IB = np.ma.zeros((rec_ndx_40HZ), fill_value=fv)
    IB.data = -SLP*(DENSITY*gs)**-1
    # interpolate conventional inverse barometer response to points
    TPX = np.ma.zeros((rec_ndx_40HZ), fill_value=fv)
    TPX.data = R2.__call__(np.c_[timescale.MJD, iy, ix])
    # replace any nan values with fill value
    IB.mask = np.isnan(IB.data)
    TPX.mask = np.isnan(TPX.data)
    IB.data[IB.mask] = IB.fill_value
    TPX.data[TPX.mask] = TPX.fill_value

    # copy variables for outputting to HDF5 file
    IS_gla12_corr = dict(Data_40HZ={})
    IS_gla12_fill = dict(Data_40HZ={})
    IS_gla12_corr_attrs = dict(Data_40HZ={})

    # copy global file attributes
    global_attribute_list = ['featureType','title','comment','summary','license',
        'references','AccessConstraints','CitationforExternalPublication',
        'contributor_role','contributor_name','creator_name','creator_email',
        'publisher_name','publisher_email','publisher_url','platform','instrument',
        'processing_level','date_created','spatial_coverage_type','history',
        'keywords','keywords_vocabulary','naming_authority','project','time_type',
        'date_type','time_coverage_start','time_coverage_end',
        'time_coverage_duration','source','HDFVersion','identifier_product_type',
        'identifier_product_format_version','Conventions','institution',
        'ReprocessingPlanned','ReprocessingActual','LocalGranuleID',
        'ProductionDateTime','LocalVersionID','PGEVersion','OrbitNumber',
        'StartOrbitNumber','StopOrbitNumber','EquatorCrossingLongitude',
        'EquatorCrossingTime','EquatorCrossingDate','ShortName','VersionID',
        'InputPointer','RangeBeginningTime','RangeEndingTime','RangeBeginningDate',
        'RangeEndingDate','PercentGroundHit','OrbitQuality','Cycle','Track',
        'Instrument_State','Timing_Bias','ReferenceOrbit','SP_ICE_PATH_NO',
        'SP_ICE_GLAS_StartBlock','SP_ICE_GLAS_EndBlock','Instance','Range_Bias',
        'Instrument_State_Date','Instrument_State_Time','Range_Bias_Date',
        'Range_Bias_Time','Timing_Bias_Date','Timing_Bias_Time',
        'identifier_product_doi','identifier_file_uuid',
        'identifier_product_doi_authority']
    for att in global_attribute_list:
        IS_gla12_corr_attrs[att] = fileID.attrs[att]
    # copy ICESat campaign name from ancillary data
    IS_gla12_corr_attrs['Campaign'] = fileID['ANCILLARY_DATA'].attrs['Campaign']

    # add attributes for input GLA12 file
    IS_gla12_corr_attrs['lineage'] = GRANULE
    # update geospatial ranges for ellipsoid
    IS_gla12_corr_attrs['geospatial_lat_min'] = np.min(lat_40HZ)
    IS_gla12_corr_attrs['geospatial_lat_max'] = np.max(lat_40HZ)
    IS_gla12_corr_attrs['geospatial_lon_min'] = np.min(lon_40HZ)
    IS_gla12_corr_attrs['geospatial_lon_max'] = np.max(lon_40HZ)
    IS_gla12_corr_attrs['geospatial_lat_units'] = "degrees_north"
    IS_gla12_corr_attrs['geospatial_lon_units'] = "degrees_east"
    IS_gla12_corr_attrs['geospatial_ellipsoid'] = "WGS84"

    # copy 40Hz group attributes
    for att_name,att_val in fileID['Data_40HZ'].attrs.items():
        IS_gla12_corr_attrs['Data_40HZ'][att_name] = att_val
    # copy attributes for time, geolocation and geophysical groups
    for var in ['Time','Geolocation','Geophysical']:
        IS_gla12_corr['Data_40HZ'][var] = {}
        IS_gla12_fill['Data_40HZ'][var] = {}
        IS_gla12_corr_attrs['Data_40HZ'][var] = {}
        for att_name,att_val in fileID['Data_40HZ'][var].attrs.items():
            IS_gla12_corr_attrs['Data_40HZ'][var][att_name] = att_val

    # J2000 time
    IS_gla12_corr['Data_40HZ']['DS_UTCTime_40'] = DS_UTCTime_40HZ
    IS_gla12_fill['Data_40HZ']['DS_UTCTime_40'] = None
    IS_gla12_corr_attrs['Data_40HZ']['DS_UTCTime_40'] = {}
    for att_name,att_val in fileID['Data_40HZ']['DS_UTCTime_40'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_corr_attrs['Data_40HZ']['DS_UTCTime_40'][att_name] = att_val
    # record
    IS_gla12_corr['Data_40HZ']['Time']['i_rec_ndx'] = rec_ndx_40HZ
    IS_gla12_fill['Data_40HZ']['Time']['i_rec_ndx'] = None
    IS_gla12_corr_attrs['Data_40HZ']['Time']['i_rec_ndx'] = {}
    for att_name,att_val in fileID['Data_40HZ']['Time']['i_rec_ndx'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_corr_attrs['Data_40HZ']['Time']['i_rec_ndx'][att_name] = att_val
    # latitude
    IS_gla12_corr['Data_40HZ']['Geolocation']['d_lat'] = lat_40HZ
    IS_gla12_fill['Data_40HZ']['Geolocation']['d_lat'] = None
    IS_gla12_corr_attrs['Data_40HZ']['Geolocation']['d_lat'] = {}
    for att_name,att_val in fileID['Data_40HZ']['Geolocation']['d_lat'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_corr_attrs['Data_40HZ']['Geolocation']['d_lat'][att_name] = att_val
    # longitude
    IS_gla12_corr['Data_40HZ']['Geolocation']['d_lon'] = lon_40HZ
    IS_gla12_fill['Data_40HZ']['Geolocation']['d_lon'] = None
    IS_gla12_corr_attrs['Data_40HZ']['Geolocation']['d_lon'] = {}
    for att_name,att_val in fileID['Data_40HZ']['Geolocation']['d_lon'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_corr_attrs['Data_40HZ']['Geolocation']['d_lon'][att_name] = att_val

    # inverse barometer response
    IS_gla12_corr['Data_40HZ']['Geophysical']['d_ibElv'] = IB.copy()
    IS_gla12_fill['Data_40HZ']['Geophysical']['d_ibElv'] = IB.fill_value
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_ibElv'] = {}
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_ibElv']['units'] = "meters"
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_ibElv']['long_name'] = "inverse barometer"
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_ibElv']['description'] = ("Instantaneous inverse "
        "barometer effect due to atmospheric loading")
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_ibElv']['hertz'] = '40'
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_ibElv']['source'] = MODEL
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_ibElv']['reference'] = \
        'https://doi.org/10.1029/96RG03037'
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_ibElv']['coordinates'] = \
        "../DS_UTCTime_40"

    # conventional (TOPEX/POSEIDON) inverse barometer response
    IS_gla12_corr['Data_40HZ']['Geophysical']['d_tpxElv'] = TPX.copy()
    IS_gla12_fill['Data_40HZ']['Geophysical']['d_tpxElv'] = TPX.fill_value
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_tpxElv'] = {}
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_tpxElv']['units'] = "meters"
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_tpxElv']['long_name'] = "inverse barometer"
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_tpxElv']['description'] = ("Conventional "
        "(TOPEX/POSEIDON) instantaneous inverse barometer effect due to atmospheric loading")
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_tpxElv']['hertz'] = '40'
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_tpxElv']['source'] = MODEL
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_tpxElv']['reference'] = \
        'TOPEX/POSEIDON Project GDR Users Handbook'
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_tpxElv']['coordinates'] = \
        "../DS_UTCTime_40"

    # close the input HDF5 file
    fileID.close()

    # print file information
    logging.info(f'\t{str(OUTPUT_FILE)}')
    HDF5_GLA12_corr_write(IS_gla12_corr, IS_gla12_corr_attrs,
        FILENAME=OUTPUT_FILE,
        FILL_VALUE=IS_gla12_fill,
        CLOBBER=True)
    # change the permissions mode
    OUTPUT_FILE.chmod(mode=MODE)

# PURPOSE: outputting the correction values for ICESat data to HDF5
def HDF5_GLA12_corr_write(IS_gla12_tide, IS_gla12_attrs,
    FILENAME='', FILL_VALUE=None, CLOBBER=False):
    # setting HDF5 clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'w-'

    # open output HDF5 file
    FILENAME = pathlib.Path(FILENAME).expanduser().absolute()
    fileID = h5py.File(FILENAME, clobber)
    # create 40HZ HDF5 records
    h5 = dict(Data_40HZ={})

    # add HDF5 file attributes
    attrs = {a:v for a,v in IS_gla12_attrs.items() if not isinstance(v,dict)}
    for att_name,att_val in attrs.items():
       fileID.attrs[att_name] = att_val

    # add software information
    fileID.attrs['software_reference'] = gz.version.project_name
    fileID.attrs['software_version'] = gz.version.full_version

    # create Data_40HZ group
    fileID.create_group('Data_40HZ')
    # add HDF5 40HZ group attributes
    for att_name,att_val in IS_gla12_attrs['Data_40HZ'].items():
        if att_name not in ('DS_UTCTime_40',) and not isinstance(att_val,dict):
            fileID['Data_40HZ'].attrs[att_name] = att_val

    # add 40HZ time variable
    val = IS_gla12_tide['Data_40HZ']['DS_UTCTime_40']
    attrs = IS_gla12_attrs['Data_40HZ']['DS_UTCTime_40']
    # Defining the HDF5 dataset variables
    h5['Data_40HZ']['DS_UTCTime_40'] = fileID.create_dataset(
        'Data_40HZ/DS_UTCTime_40', np.shape(val),
        data=val, dtype=val.dtype, compression='gzip')
    # make dimension
    h5['Data_40HZ']['DS_UTCTime_40'].make_scale('DS_UTCTime_40')
    # add HDF5 variable attributes
    for att_name,att_val in attrs.items():
        h5['Data_40HZ']['DS_UTCTime_40'].attrs[att_name] = att_val

    # for each variable group
    for group in ['Time','Geolocation','Geophysical']:
        # add group to dict
        h5['Data_40HZ'][group] = {}
        # create Data_40HZ group
        fileID.create_group(f'Data_40HZ/{group}')
        # add HDF5 group attributes
        for att_name,att_val in IS_gla12_attrs['Data_40HZ'][group].items():
            if not isinstance(att_val,dict):
                fileID['Data_40HZ'][group].attrs[att_name] = att_val
        # for each variable in the group
        for key,val in IS_gla12_tide['Data_40HZ'][group].items():
            fillvalue = FILL_VALUE['Data_40HZ'][group][key]
            attrs = IS_gla12_attrs['Data_40HZ'][group][key]
            # Defining the HDF5 dataset variables
            var = f'Data_40HZ/{group}/{key}'
            # use variable compression if containing fill values
            if fillvalue:
                h5['Data_40HZ'][group][key] = fileID.create_dataset(var,
                    np.shape(val), data=val, dtype=val.dtype,
                    fillvalue=fillvalue, compression='gzip')
            else:
                h5['Data_40HZ'][group][key] = fileID.create_dataset(var,
                    np.shape(val), data=val, dtype=val.dtype,
                    compression='gzip')
            # attach dimensions
            for i,dim in enumerate(['DS_UTCTime_40']):
                h5['Data_40HZ'][group][key].dims[i].attach_scale(
                    h5['Data_40HZ'][dim])
            # add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5['Data_40HZ'][group][key].attrs[att_name] = att_val

    # Closing the HDF5 file
    fileID.close()

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates and interpolates inverse-barometer
            responses to ICESat/GLAS L2 GLA12 Antarctic and Greenland
            Ice Sheet elevation data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='ICESat GLA12 file to run')
    # directory with reanalysis data
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=pathlib.Path.cwd(),
        help='Working data directory')
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
        interp_IB_response_ICESat(args.directory, FILE, args.reanalysis,
            RANGE=args.mean, DENSITY=args.density, VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
