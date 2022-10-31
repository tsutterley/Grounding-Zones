#!/usr/bin/env python
u"""
interp_ATL14_DEM_ICESat2_ATL11.py
Written by Tyler Sutterley (10/2022)
Interpolates ATL14 elevations to ICESat-2 ATL11 segment locations

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

PROGRAM DEPENDENCIES:
    convert_delta_time.py: converts from delta time into Julian and year-decimal
    time.py: Utilities for calculating time operations
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Written 10/2022
"""
from __future__ import print_function

import sys
import os
import re
import h5py
import pyproj
import logging
import netCDF4
import datetime
import argparse
import collections
import numpy as np
import scipy.interpolate
from icesat2_toolkit.convert_delta_time import convert_delta_time
import icesat2_toolkit.time
import icesat2_toolkit.utilities
from icesat2_toolkit.read_ICESat2_ATL11 import read_HDF5_ATL11

#-- PURPOSE: read ICESat-2 annual land ice height data (ATL11)
#-- interpolate DEM data to x and y coordinates
def interp_ATL14_DEM_ICESat2(FILE, DEM_MODEL=None, MODE=None):

    #-- read data from FILE
    IS2_atl11_mds,IS2_atl11_attrs,IS2_atl11_pairs = read_HDF5_ATL11(FILE,
        ATTRIBUTES=True)
    DIRECTORY = os.path.dirname(FILE)
    #-- extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})_(\d{2})(\d{2})_'
        r'(\d{3})_(\d{2})(.*?).h5$')
    SUB,PRD,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX = rx.findall(FILE).pop()

    #-- open ATL14 DEM file for reading
    fileID = netCDF4.Dataset(os.path.expanduser(DEM_MODEL), mode='r')
    #-- get coordinate reference system attributes
    crs = {}
    grid_mapping = fileID.variables['h'].getncattr('grid_mapping')
    for att_name in fileID[grid_mapping].ncattrs():
        crs[att_name] = fileID.variables[grid_mapping].getncattr(att_name)
    #-- get original grid coordinates
    x = fileID.variables['x'][:].copy()
    y = fileID.variables['y'][:].copy()
    #-- get grid spacing
    dx = np.abs(x[1] - x[0])
    dy = np.abs(y[1] - y[0])

    #-- pyproj transformer for converting from latitude/longitude
    #-- into DEM file coordinates
    crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(4326))
    crs2 = pyproj.CRS.from_wkt(crs['crs_wkt'])
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    #-- copy variables for outputting to HDF5 file
    IS2_atl11_dem = {}
    IS2_atl11_fill = {}
    IS2_atl11_dims = {}
    IS2_atl11_dem_attrs = {}
    #-- number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    #-- and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    #-- Add this value to delta time parameters to compute full gps_seconds
    IS2_atl11_dem['ancillary_data'] = {}
    IS2_atl11_dem_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        #-- get each HDF5 variable
        IS2_atl11_dem['ancillary_data'][key] = IS2_atl11_mds['ancillary_data'][key]
        #-- Getting attributes of group and included variables
        IS2_atl11_dem_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl11_attrs['ancillary_data'][key].items():
            IS2_atl11_dem_attrs['ancillary_data'][key][att_name] = att_val

    #-- for each input beam pair within the file
    for ptx in sorted(IS2_atl11_pairs):
        #-- output data dictionaries for beam pair
        IS2_atl11_dem[ptx] = dict(ref_surf=collections.OrderedDict())
        IS2_atl11_fill[ptx] = dict(ref_surf={})
        IS2_atl11_dims[ptx] = dict(ref_surf={})
        IS2_atl11_dem_attrs[ptx] = dict(ref_surf={})

        #-- along-track (AT) reference point, latitude, longitude and time
        ref_pt = IS2_atl11_mds[ptx]['ref_pt'].copy()
        latitude = np.ma.array(IS2_atl11_mds[ptx]['latitude'],
            fill_value=IS2_atl11_attrs[ptx]['latitude']['_FillValue'])
        longitude = np.ma.array(IS2_atl11_mds[ptx]['longitude'],
            fill_value=IS2_atl11_attrs[ptx]['longitude']['_FillValue'])
        delta_time = np.ma.array(IS2_atl11_mds[ptx]['delta_time'],
            fill_value=IS2_atl11_attrs[ptx]['delta_time']['_FillValue'])

        #-- convert projection from latitude/longitude to DEM EPSG
        X,Y = transformer.transform(longitude, latitude)

        #-- number of average segments and number of included cycles
        #-- fill_value for invalid heights and corrections
        fv = IS2_atl11_attrs[ptx]['h_corr']['_FillValue']
        #-- shape of along-track and across-track data
        n_points,n_cycles = delta_time.shape

        #-- output interpolated digital elevation model
        dem_h = np.ma.zeros((n_points),fill_value=fv,dtype=np.float32)
        dem_h.mask = np.ones((n_points),dtype=bool)
        dem_h_sigma = np.ma.zeros((n_points),fill_value=fv,dtype=np.float32)
        dem_h_sigma.mask = np.ones((n_points),dtype=bool)

        #-- bounding box of ATL14 file to read for beam pair
        bounds = [[X.min()-5*dx,X.max()+5*dx],[Y.min()-5*dy,Y.max()+5*dy]]
        #-- indices to read
        xind, = np.nonzero((x >= bounds[0][0]) & (x <= bounds[0][1]))
        cols = slice(xind[0],xind[-1],1)
        yind, = np.nonzero((y >= bounds[1][0]) & (y <= bounds[1][1]))
        rows = slice(yind[0],yind[-1],1)
        #-- subset ATL14 elevation field to bounds
        h = fileID.variables['h'][rows,cols]
        h_sigma2 = fileID.variables['h_sigma'][rows,cols]**2
        ice_area = fileID.variables['ice_area'][rows,cols]

        #-- group attributes for beam
        IS2_atl11_dem_attrs[ptx]['description'] = ('Contains the primary science parameters '
            'for this data set')
        IS2_atl11_dem_attrs[ptx]['beam_pair'] = IS2_atl11_attrs[ptx]['beam_pair']
        IS2_atl11_dem_attrs[ptx]['ReferenceGroundTrack'] = IS2_atl11_attrs[ptx]['ReferenceGroundTrack']
        IS2_atl11_dem_attrs[ptx]['first_cycle'] = IS2_atl11_attrs[ptx]['first_cycle']
        IS2_atl11_dem_attrs[ptx]['last_cycle'] = IS2_atl11_attrs[ptx]['last_cycle']
        IS2_atl11_dem_attrs[ptx]['equatorial_radius'] = IS2_atl11_attrs[ptx]['equatorial_radius']
        IS2_atl11_dem_attrs[ptx]['polar_radius'] = IS2_atl11_attrs[ptx]['polar_radius']

        #-- geolocation, time and reference point
        #-- reference point
        IS2_atl11_dem[ptx]['ref_pt'] = ref_pt.copy()
        IS2_atl11_fill[ptx]['ref_pt'] = None
        IS2_atl11_dims[ptx]['ref_pt'] = None
        IS2_atl11_dem_attrs[ptx]['ref_pt'] = collections.OrderedDict()
        IS2_atl11_dem_attrs[ptx]['ref_pt']['units'] = "1"
        IS2_atl11_dem_attrs[ptx]['ref_pt']['contentType'] = "referenceInformation"
        IS2_atl11_dem_attrs[ptx]['ref_pt']['long_name'] = "Reference point number"
        IS2_atl11_dem_attrs[ptx]['ref_pt']['source'] = "ATL06"
        IS2_atl11_dem_attrs[ptx]['ref_pt']['description'] = ("The reference point is the 7 "
            "digit segment_id number corresponding to the center of the ATL06 data used for "
            "each ATL11 point.  These are sequential, starting with 1 for the first segment "
            "after an ascending equatorial crossing node.")
        IS2_atl11_dem_attrs[ptx]['ref_pt']['coordinates'] = \
            "delta_time latitude longitude"
        #-- cycle_number
        IS2_atl11_dem[ptx]['cycle_number'] = IS2_atl11_mds[ptx]['cycle_number'].copy()
        IS2_atl11_fill[ptx]['cycle_number'] = None
        IS2_atl11_dims[ptx]['cycle_number'] = None
        IS2_atl11_dem_attrs[ptx]['cycle_number'] = collections.OrderedDict()
        IS2_atl11_dem_attrs[ptx]['cycle_number']['units'] = "1"
        IS2_atl11_dem_attrs[ptx]['cycle_number']['long_name'] = "Orbital cycle number"
        IS2_atl11_dem_attrs[ptx]['cycle_number']['source'] = "ATL06"
        IS2_atl11_dem_attrs[ptx]['cycle_number']['description'] = ("Number of 91-day periods "
            "that have elapsed since ICESat-2 entered the science orbit. Each of the 1,387 "
            "reference ground track (RGTs) is targeted in the polar regions once "
            "every 91 days.")
        #-- delta time
        IS2_atl11_dem[ptx]['delta_time'] = delta_time.copy()
        IS2_atl11_fill[ptx]['delta_time'] = delta_time.fill_value
        IS2_atl11_dims[ptx]['delta_time'] = ['ref_pt','cycle_number']
        IS2_atl11_dem_attrs[ptx]['delta_time'] = collections.OrderedDict()
        IS2_atl11_dem_attrs[ptx]['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl11_dem_attrs[ptx]['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl11_dem_attrs[ptx]['delta_time']['standard_name'] = "time"
        IS2_atl11_dem_attrs[ptx]['delta_time']['calendar'] = "standard"
        IS2_atl11_dem_attrs[ptx]['delta_time']['source'] = "ATL06"
        IS2_atl11_dem_attrs[ptx]['delta_time']['description'] = ("Number of GPS "
            "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
            "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
            "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
            "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
            "time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl11_dem_attrs[ptx]['delta_time']['coordinates'] = \
            "ref_pt cycle_number latitude longitude"
        #-- latitude
        IS2_atl11_dem[ptx]['latitude'] = latitude.copy()
        IS2_atl11_fill[ptx]['latitude'] = latitude.fill_value
        IS2_atl11_dims[ptx]['latitude'] = ['ref_pt']
        IS2_atl11_dem_attrs[ptx]['latitude'] = collections.OrderedDict()
        IS2_atl11_dem_attrs[ptx]['latitude']['units'] = "degrees_north"
        IS2_atl11_dem_attrs[ptx]['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_dem_attrs[ptx]['latitude']['long_name'] = "Latitude"
        IS2_atl11_dem_attrs[ptx]['latitude']['standard_name'] = "latitude"
        IS2_atl11_dem_attrs[ptx]['latitude']['source'] = "ATL06"
        IS2_atl11_dem_attrs[ptx]['latitude']['description'] = ("Center latitude of "
            "selected segments")
        IS2_atl11_dem_attrs[ptx]['latitude']['valid_min'] = -90.0
        IS2_atl11_dem_attrs[ptx]['latitude']['valid_max'] = 90.0
        IS2_atl11_dem_attrs[ptx]['latitude']['coordinates'] = \
            "ref_pt delta_time longitude"
        #-- longitude
        IS2_atl11_dem[ptx]['longitude'] = longitude.copy()
        IS2_atl11_fill[ptx]['longitude'] = longitude.fill_value
        IS2_atl11_dims[ptx]['longitude'] = ['ref_pt']
        IS2_atl11_dem_attrs[ptx]['longitude'] = collections.OrderedDict()
        IS2_atl11_dem_attrs[ptx]['longitude']['units'] = "degrees_east"
        IS2_atl11_dem_attrs[ptx]['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_dem_attrs[ptx]['longitude']['long_name'] = "Longitude"
        IS2_atl11_dem_attrs[ptx]['longitude']['standard_name'] = "longitude"
        IS2_atl11_dem_attrs[ptx]['longitude']['source'] = "ATL06"
        IS2_atl11_dem_attrs[ptx]['longitude']['description'] = ("Center longitude of "
            "selected segments")
        IS2_atl11_dem_attrs[ptx]['longitude']['valid_min'] = -180.0
        IS2_atl11_dem_attrs[ptx]['longitude']['valid_max'] = 180.0
        IS2_atl11_dem_attrs[ptx]['longitude']['coordinates'] = \
            "ref_pt delta_time latitude"

        #-- reference surface variables
        IS2_atl11_dem_attrs[ptx]['ref_surf']['Description'] = ("The ref_surf subgroup contains "
            "parameters that describe the reference surface fit at each reference point, "
            "including slope information from ATL06, the polynomial coefficients used for the "
            "fit, and misfit statistics.")
        IS2_atl11_dem_attrs[ptx]['ref_surf']['data_rate'] = ("Data within this group "
            "are stored at the average segment rate.")

        #-- use spline interpolation to calculate DEM values at coordinates
        f1 = scipy.interpolate.RectBivariateSpline(x[cols],y[rows],h.T,kx=1,ky=1)
        f2 = scipy.interpolate.RectBivariateSpline(x[cols],y[rows],h_sigma2.T,kx=1,ky=1)
        f3 = scipy.interpolate.RectBivariateSpline(x[cols],y[rows],ice_area.T,kx=1,ky=1)
        dem_h.data[:] = f1.ev(X,Y)
        dem_h_sigma.data[:] = np.sqrt(f2.ev(X,Y))
        dem_ice_area = f3.ev(X,Y)
        #-- update masks and replace fill values
        dem_h.mask[:] = (dem_ice_area <= 0.0) | (np.abs(dem_h.data) >= 1e4)
        dem_h_sigma.mask[:] = (dem_ice_area <= 0.0) | (np.abs(dem_h.data) >= 1e4)
        dem_h.data[dem_h.mask] = dem_h.fill_value
        dem_h_sigma.data[dem_h_sigma.mask] = dem_h_sigma.fill_value

        #-- save ATL14 DEM elevation for pair track
        IS2_atl11_dem[ptx]['ref_surf']['dem_h'] = dem_h
        IS2_atl11_fill[ptx]['ref_surf']['dem_h'] = dem_h.fill_value
        IS2_atl11_dims[ptx]['ref_surf']['dem_h'] = ['ref_pt']
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h'] = collections.OrderedDict()
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h']['units'] = "meters"
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h']['contentType'] = "referenceInformation"
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h']['long_name'] = "DEM Height"
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h']['description'] = ("Height of the DEM, "
            "interpolated by bivariate-spline interpolation in the DEM coordinate system "
            "to the segment location.")
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h']['source'] = 'ATL14'
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h']['coordinates'] = \
            "../ref_pt ../delta_time ../latitude ../longitude"

        #-- save ATl14 DEM elevation uncertainty for pair track
        IS2_atl11_dem[ptx]['ref_surf']['dem_h_sigma'] = dem_h_sigma
        IS2_atl11_fill[ptx]['ref_surf']['dem_h_sigma'] = dem_h_sigma.fill_value
        IS2_atl11_dims[ptx]['ref_surf']['dem_h_sigma'] = ['ref_pt']
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h_sigma'] = collections.OrderedDict()
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h_sigma']['units'] = "meters"
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h_sigma']['contentType'] = "referenceInformation"
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h_sigma']['long_name'] = "DEM Uncertainty"
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h_sigma']['description'] = ("Uncertainty in the "
            "DEM surface height, interpolated by bivariate-spline interpolation in the DEM "
            "coordinate system to the segment location.")
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h_sigma']['source'] = 'ATL14'
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h_sigma']['coordinates'] = \
            "../ref_pt ../delta_time ../latitude ../longitude"

    #-- close the ATL14 elevation file
    fileID.close()

    #-- output HDF5 files with output masks
    fargs = ('ATL14',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
    file_format = '{0}_{1}{2}_{3}{4}_{5}_{6}{7}.h5'
    output_file = os.path.join(DIRECTORY,file_format.format(*fargs))
    #-- print file information
    logging.info(f'\t{output_file}')
    #-- write to output HDF5 file
    HDF5_ATL11_dem_write(IS2_atl11_dem, IS2_atl11_dem_attrs,
        CLOBBER=True, INPUT=os.path.basename(FILE),
        FILL_VALUE=IS2_atl11_fill, DIMENSIONS=IS2_atl11_dims,
        FILENAME=output_file)
    #-- change the permissions mode
    os.chmod(output_file, MODE)

#-- PURPOSE: outputting the interpolated DEM data for ICESat-2 data to HDF5
def HDF5_ATL11_dem_write(IS2_atl11_dem, IS2_atl11_attrs, INPUT=None,
    FILENAME='', FILL_VALUE=None, DIMENSIONS=None, CLOBBER=True):
    #-- setting HDF5 clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'w-'

    #-- open output HDF5 file
    fileID = h5py.File(os.path.expanduser(FILENAME), clobber)

    #-- create HDF5 records
    h5 = {}

    #-- number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    #-- and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    h5['ancillary_data'] = {}
    for k,v in IS2_atl11_dem['ancillary_data'].items():
        #-- Defining the HDF5 dataset variables
        val = f'ancillary_data/{k}'
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        #-- add HDF5 variable attributes
        for att_name,att_val in IS2_atl11_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    #-- write each output beam pair
    pairs = [k for k in IS2_atl11_dem.keys() if bool(re.match(r'pt\d',k))]
    for ptx in pairs:
        fileID.create_group(ptx)
        h5[ptx] = {}
        #-- add HDF5 group attributes for beam pair
        for att_name in ['description','beam_pair','ReferenceGroundTrack',
            'first_cycle','last_cycle','equatorial_radius','polar_radius']:
            fileID[ptx].attrs[att_name] = IS2_atl11_attrs[ptx][att_name]

        #-- ref_pt, cycle number, geolocation and delta_time variables
        for k in ['ref_pt','cycle_number','delta_time','latitude','longitude']:
            #-- values and attributes
            v = IS2_atl11_dem[ptx][k]
            attrs = IS2_atl11_attrs[ptx][k]
            fillvalue = FILL_VALUE[ptx][k]
            #-- Defining the HDF5 dataset variables
            val = f'{ptx}/{k}'
            if fillvalue:
                h5[ptx][k] = fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, fillvalue=fillvalue, compression='gzip')
            else:
                h5[ptx][k] = fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, compression='gzip')
            #-- create or attach dimensions for HDF5 variable
            if DIMENSIONS[ptx][k]:
                #-- attach dimensions
                for i,dim in enumerate(DIMENSIONS[ptx][k]):
                    h5[ptx][k].dims[i].attach_scale(h5[ptx][dim])
            else:
                #-- make dimension
                h5[ptx][k].make_scale(k)
            #-- add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5[ptx][k].attrs[att_name] = att_val

        #-- add to output variables
        for key in ['ref_surf',]:
            fileID[ptx].create_group(key)
            h5[ptx][key] = {}
            for att_name in ['Description','data_rate']:
                att_val=IS2_atl11_attrs[ptx][key][att_name]
                fileID[ptx][key].attrs[att_name] = att_val
            for k,v in IS2_atl11_dem[ptx][key].items():
                #-- attributes
                attrs = IS2_atl11_attrs[ptx][key][k]
                fillvalue = FILL_VALUE[ptx][key][k]
                #-- Defining the HDF5 dataset variables
                val = f'{ptx}/{key}/{k}'
                if fillvalue:
                    h5[ptx][key][k] = fileID.create_dataset(val, np.shape(v),
                        data=v, dtype=v.dtype, fillvalue=fillvalue,
                        compression='gzip')
                else:
                    h5[ptx][key][k] = fileID.create_dataset(val, np.shape(v),
                        data=v, dtype=v.dtype, compression='gzip')
                #-- attach dimensions
                for i,dim in enumerate(DIMENSIONS[ptx][key][k]):
                    h5[ptx][key][k].dims[i].attach_scale(h5[ptx][dim])
                #-- add HDF5 variable attributes
                for att_name,att_val in attrs.items():
                    h5[ptx][key][k].attrs[att_name] = att_val

    #-- HDF5 file title
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'ATLAS/ICESat-2 Land Ice Height'
    fileID.attrs['summary'] = ('Geophysical parameters for land ice segments '
        'needed to interpret and assess the quality of the height estimates.')
    fileID.attrs['description'] = ('Land ice parameters for each beam pair. '
        'All parameters are calculated for the same along-track increments '
        'for each beam pair and repeat.')
    date_created = datetime.datetime.today()
    fileID.attrs['date_created'] = date_created.isoformat()
    project = 'ICESat-2 > Ice, Cloud, and land Elevation Satellite-2'
    fileID.attrs['project'] = project
    platform = 'ICESat-2 > Ice, Cloud, and land Elevation Satellite-2'
    fileID.attrs['project'] = platform
    #-- add attribute for elevation instrument and designated processing level
    instrument = 'ATLAS > Advanced Topographic Laser Altimeter System'
    fileID.attrs['instrument'] = instrument
    fileID.attrs['source'] = 'Spacecraft'
    fileID.attrs['references'] = 'https://nsidc.org/data/icesat-2'
    fileID.attrs['processing_level'] = '4'
    #-- add attributes for input ATL11 files
    fileID.attrs['input_files'] = ','.join([os.path.basename(i) for i in INPUT])
    #-- find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for ptx in pairs:
        lon = IS2_atl11_dem[ptx]['longitude']
        lat = IS2_atl11_dem[ptx]['latitude']
        delta_time = IS2_atl11_dem[ptx]['delta_time']
        valid = np.nonzero(delta_time != FILL_VALUE[ptx]['delta_time'])
        #-- setting the geospatial and temporal ranges
        lnmn = lon.min() if (lon.min() < lnmn) else lnmn
        lnmx = lon.max() if (lon.max() > lnmx) else lnmx
        ltmn = lat.min() if (lat.min() < ltmn) else ltmn
        ltmx = lat.max() if (lat.max() > ltmx) else ltmx
        tmn = delta_time[valid].min() if (delta_time[valid].min() < tmn) else tmn
        tmx = delta_time[valid].max() if (delta_time[valid].max() > tmx) else tmx
    #-- add geospatial and temporal attributes
    fileID.attrs['geospatial_lat_min'] = ltmn
    fileID.attrs['geospatial_lat_max'] = ltmx
    fileID.attrs['geospatial_lon_min'] = lnmn
    fileID.attrs['geospatial_lon_max'] = lnmx
    fileID.attrs['geospatial_lat_units'] = "degrees_north"
    fileID.attrs['geospatial_lon_units'] = "degrees_east"
    fileID.attrs['geospatial_ellipsoid'] = "WGS84"
    fileID.attrs['date_type'] = 'UTC'
    fileID.attrs['time_type'] = 'CCSDS UTC-A'
    #-- convert start and end time from ATLAS SDP seconds into UTC time
    time_utc = convert_delta_time(np.array([tmn,tmx]))
    #-- convert to calendar date
    YY,MM,DD,HH,MN,SS = icesat2_toolkit.time.convert_julian(time_utc['julian'],
        format='tuple')
    #-- add attributes with measurement date start, end and duration
    tcs = datetime.datetime(int(YY[0]), int(MM[0]), int(DD[0]),
        int(HH[0]), int(MN[0]), int(SS[0]), int(1e6*(SS[0] % 1)))
    fileID.attrs['time_coverage_start'] = tcs.isoformat()
    tce = datetime.datetime(int(YY[1]), int(MM[1]), int(DD[1]),
        int(HH[1]), int(MN[1]), int(SS[1]), int(1e6*(SS[1] % 1)))
    fileID.attrs['time_coverage_end'] = tce.isoformat()
    fileID.attrs['time_coverage_duration'] = '{0:0.0f}'.format(tmx-tmn)
    #-- Closing the HDF5 file
    fileID.close()

#-- PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Interpolate DEMs to ICESat-2 ATL11 annual land
            ice height locations
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = \
        icesat2_toolkit.utilities.convert_arg_line_to_args
    #-- command line parameters
    parser.add_argument('file',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        help='ICESat-2 ATL11 file to run')
    #-- full path to ATL14 digital elevation file
    parser.add_argument('--dem-model','-,',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='ICESat-2 ATL14 DEM file to run')
    #-- verbosity settings
    #-- verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of run')
    #-- permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permissions mode of output files')
    # return the parser
    return parser

#-- This is the main part of the program that calls the individual functions
def main():
    #-- Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    #-- create logger
    loglevel = logging.INFO if args.verbose else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    #-- run program with parameters
    interp_ATL14_DEM_ICESat2(args.file,
        DEM_MODEL=args.dem_model,
        MODE=args.mode)

#-- run main program
if __name__ == '__main__':
    main()
