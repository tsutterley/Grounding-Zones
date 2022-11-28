#!/usr/bin/env python
u"""
interp_ATL14_DEM_ICESat2_ATL06.py
Written by Tyler Sutterley (11/2022)
Interpolates ATL14 elevations to locations of ICESat-2 ATL06 segments

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
    read_ICESat2_ATL06.py: reads ICESat-2 land ice along-track height data files
    convert_delta_time.py: converts from delta time into Julian and year-decimal
    time.py: Utilities for calculating time operations
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 11/2022: check that granule intersects ATL14 DEM
    Written 11/2022
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
import numpy as np
import scipy.interpolate
from icesat2_toolkit.convert_delta_time import convert_delta_time
import icesat2_toolkit.time
import icesat2_toolkit.utilities
from icesat2_toolkit.read_ICESat2_ATL06 import read_HDF5_ATL06

#-- PURPOSE: read ICESat-2 land ice height data (ATL06)
#-- interpolate DEM data to x and y coordinates
def interp_ATL14_DEM_ICESat2(FILE, DEM_MODEL=None, MODE=None):

    #-- read data from FILE
    IS2_atl06_mds,IS2_atl06_attrs,IS2_atl06_beams = read_HDF5_ATL06(FILE,
        ATTRIBUTES=True)
    DIRECTORY = os.path.dirname(FILE)
    #-- extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})'
        r'(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    SUB,PRD,YY,MM,DD,HH,MN,SS,TRK,CYC,GRN,RL,VRS,AUX=rx.findall(FILE).pop()

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
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_wkt(crs['crs_wkt'])
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    #-- copy variables for outputting to HDF5 file
    IS2_atl06_dem = {}
    IS2_atl06_fill = {}
    IS2_atl06_dims = {}
    IS2_atl06_dem_attrs = {}
    #-- number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    #-- and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    #-- Add this value to delta time parameters to compute full gps_seconds
    IS2_atl06_dem['ancillary_data'] = {}
    IS2_atl06_dem_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        #-- get each HDF5 variable
        IS2_atl06_dem['ancillary_data'][key] = IS2_atl06_mds['ancillary_data'][key]
        #-- Getting attributes of group and included variables
        IS2_atl06_dem_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl06_attrs['ancillary_data'][key].items():
            IS2_atl06_dem_attrs['ancillary_data'][key][att_name] = att_val

    #-- for each input beam within the file
    for gtx in sorted(IS2_atl06_beams):
        #-- variables for beam
        val = IS2_atl06_mds[gtx]['land_ice_segments']
        #-- number of segments
        n_seg, = val['segment_id'].shape
        #-- fill_value for invalid heights and corrections
        fv = IS2_atl06_attrs[gtx]['land_ice_segments']['h_li']['_FillValue']

        #-- convert projection from latitude/longitude to DEM EPSG
        X,Y = transformer.transform(val['longitude'], val['latitude'])

        #-- check that beam coordinates intersect ATL14
        valid = (X >= x.min()) & (X <= x.max()) & (Y >= y.min()) & (Y <= y.max())
        if not np.any(valid):
            continue

        #-- output data dictionaries for beam pair
        IS2_atl06_dem[gtx] = dict(land_ice_segments={})
        IS2_atl06_fill[gtx] = dict(land_ice_segments={})
        IS2_atl06_dims[gtx] = dict(land_ice_segments={})
        IS2_atl06_dem_attrs[gtx] = dict(land_ice_segments={})

        #-- output interpolated digital elevation model
        dem_h = np.ma.zeros((n_seg),fill_value=fv,dtype=np.float32)
        dem_h.mask = np.ones((n_seg),dtype=bool)
        dem_h_sigma = np.ma.zeros((n_seg),fill_value=fv,dtype=np.float32)
        dem_h_sigma.mask = np.ones((n_seg),dtype=bool)

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
        IS2_atl06_dem_attrs[gtx]['Description'] = IS2_atl06_attrs[gtx]['Description']
        IS2_atl06_dem_attrs[gtx]['atlas_pce'] = IS2_atl06_attrs[gtx]['atlas_pce']
        IS2_atl06_dem_attrs[gtx]['atlas_beam_type'] = IS2_atl06_attrs[gtx]['atlas_beam_type']
        IS2_atl06_dem_attrs[gtx]['groundtrack_id'] = IS2_atl06_attrs[gtx]['groundtrack_id']
        IS2_atl06_dem_attrs[gtx]['atmosphere_profile'] = IS2_atl06_attrs[gtx]['atmosphere_profile']
        IS2_atl06_dem_attrs[gtx]['atlas_spot_number'] = IS2_atl06_attrs[gtx]['atlas_spot_number']
        IS2_atl06_dem_attrs[gtx]['sc_orientation'] = IS2_atl06_attrs[gtx]['sc_orientation']
        #-- group attributes for land_ice_segments
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['Description'] = ("The land_ice_segments group "
            "contains the primary set of derived products. This includes geolocation, height, and "
            "standard error and quality measures for each segment. This group is sparse, meaning "
            "that parameters are provided only for pairs of segments for which at least one beam "
            "has a valid surface-height measurement.")
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['data_rate'] = ("Data within this group are "
            "sparse.  Data values are provided only for those ICESat-2 20m segments where at "
            "least one beam has a valid land ice height measurement.")

        #-- geolocation, time and segment ID
        #-- delta time
        IS2_atl06_dem[gtx]['land_ice_segments']['delta_time'] = val['delta_time'].copy()
        IS2_atl06_fill[gtx]['land_ice_segments']['delta_time'] = None
        IS2_atl06_dims[gtx]['land_ice_segments']['delta_time'] = None
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['delta_time'] = {}
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['delta_time']['standard_name'] = "time"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['delta_time']['calendar'] = "standard"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['delta_time']['description'] = ("Number of GPS "
            "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
            "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
            "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
            "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
            "time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['delta_time']['coordinates'] = \
            "segment_id latitude longitude"
        #-- latitude
        IS2_atl06_dem[gtx]['land_ice_segments']['latitude'] = val['latitude'].copy()
        IS2_atl06_fill[gtx]['land_ice_segments']['latitude'] = None
        IS2_atl06_dims[gtx]['land_ice_segments']['latitude'] = ['delta_time']
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['latitude'] = {}
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['latitude']['units'] = "degrees_north"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['latitude']['long_name'] = "Latitude"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['latitude']['standard_name'] = "latitude"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['latitude']['description'] = ("Latitude of "
            "segment center")
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['latitude']['valid_min'] = -90.0
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['latitude']['valid_max'] = 90.0
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['latitude']['coordinates'] = \
            "segment_id delta_time longitude"
        #-- longitude
        IS2_atl06_dem[gtx]['land_ice_segments']['longitude'] = val['longitude'].copy()
        IS2_atl06_fill[gtx]['land_ice_segments']['longitude'] = None
        IS2_atl06_dims[gtx]['land_ice_segments']['longitude'] = ['delta_time']
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['longitude'] = {}
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['longitude']['units'] = "degrees_east"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['longitude']['long_name'] = "Longitude"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['longitude']['standard_name'] = "longitude"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['longitude']['description'] = ("Longitude of "
            "segment center")
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['longitude']['valid_min'] = -180.0
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['longitude']['valid_max'] = 180.0
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['longitude']['coordinates'] = \
            "segment_id delta_time latitude"
        #-- segment ID
        IS2_atl06_dem[gtx]['land_ice_segments']['segment_id'] = val['segment_id'].copy()
        IS2_atl06_fill[gtx]['land_ice_segments']['segment_id'] = None
        IS2_atl06_dims[gtx]['land_ice_segments']['segment_id'] = ['delta_time']
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['segment_id'] = {}
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['segment_id']['units'] = "1"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['segment_id']['contentType'] = "referenceInformation"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['segment_id']['long_name'] = "Along-track segment ID number"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['segment_id']['description'] = ("A 7 digit number "
            "identifying the along-track geolocation segment number.  These are sequential, starting with "
            "1 for the first segment after an ascending equatorial crossing node. Equal to the segment_id for "
            "the second of the two 20m ATL03 segments included in the 40m ATL06 segment")
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['segment_id']['coordinates'] = \
            "delta_time latitude longitude"

        #-- dem variables
        IS2_atl06_dem[gtx]['land_ice_segments']['dem'] = {}
        IS2_atl06_fill[gtx]['land_ice_segments']['dem'] = {}
        IS2_atl06_dims[gtx]['land_ice_segments']['dem'] = {}
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem'] = {}
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['Description'] = ("The dem group "
            "contains the reference digital elevation model and geoid heights.")
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['data_rate'] = ("Data within this group "
            "are stored at the land_ice_segments segment rate.")

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
        IS2_atl06_dem[gtx]['land_ice_segments']['dem']['dem_h'] = dem_h
        IS2_atl06_fill[gtx]['land_ice_segments']['dem']['dem_h'] = dem_h.fill_value
        IS2_atl06_dims[gtx]['land_ice_segments']['dem']['dem_h'] = ['delta_time']
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h'] = {}
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h']['units'] = "meters"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h']['contentType'] = "referenceInformation"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h']['long_name'] = "DEM Height"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h']['description'] = ("Height of the DEM, "
            "interpolated by bivariate-spline interpolation in the DEM coordinate system "
            "to the segment location.")
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h']['source'] = 'ATL14'
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h']['coordinates'] = \
            "../segment_id ../delta_time ../latitude ../longitude"

        #-- save ATl14 DEM elevation uncertainty for pair track
        IS2_atl06_dem[gtx]['land_ice_segments']['dem']['dem_h_sigma'] = dem_h_sigma
        IS2_atl06_fill[gtx]['land_ice_segments']['dem']['dem_h_sigma'] = dem_h_sigma.fill_value
        IS2_atl06_dims[gtx]['land_ice_segments']['dem']['dem_h_sigma'] = ['delta_time']
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h_sigma'] = {}
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h_sigma']['units'] = "meters"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h_sigma']['contentType'] = "referenceInformation"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h_sigma']['long_name'] = "DEM Uncertainty"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h_sigma']['description'] = ("Uncertainty in the "
            "DEM surface height, interpolated by bivariate-spline interpolation in the DEM "
            "coordinate system to the segment location.")
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h_sigma']['source'] = 'ATL14'
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h_sigma']['coordinates'] = \
            "../segment_id ../delta_time ../latitude ../longitude"

    #-- close the ATL14 elevation file
    fileID.close()

    #-- check that there are any valid beams in the dataset
    if bool([k for k in IS2_atl06_dem.keys() if bool(re.match(r'gt\d[lr]',k))]):
        #-- output HDF5 files with output masks
        fargs = ('ATL14',YY,MM,DD,HH,MN,SS,TRK,CYC,GRN,RL,VRS,AUX)
        file_format = '{0}_{1}{2}{3}{4}{5}{6}_{7}{8}{9}_{10}_{11}{12}.h5'
        output_file = os.path.join(DIRECTORY,file_format.format(*fargs))
        #-- print file information
        logging.info(f'\t{output_file}')
        #-- write to output HDF5 file
        HDF5_ATL06_dem_write(IS2_atl06_dem, IS2_atl06_dem_attrs,
            CLOBBER=True, INPUT=os.path.basename(FILE),
            FILL_VALUE=IS2_atl06_fill, DIMENSIONS=IS2_atl06_dims,
            FILENAME=output_file)
        #-- change the permissions mode
        os.chmod(output_file, MODE)

#-- PURPOSE: outputting the interpolated DEM data for ICESat-2 data to HDF5
def HDF5_ATL06_dem_write(IS2_atl06_dem, IS2_atl06_attrs, INPUT=None,
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
    for k,v in IS2_atl06_dem['ancillary_data'].items():
        #-- Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        #-- add HDF5 variable attributes
        for att_name,att_val in IS2_atl06_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    #-- write each output beam
    beams = [k for k in IS2_atl06_dem.keys() if bool(re.match(r'gt\d[lr]',k))]
    for gtx in beams:
        fileID.create_group(gtx)
        #-- add HDF5 group attributes for beam
        for att_name in ['Description','atlas_pce','atlas_beam_type',
            'groundtrack_id','atmosphere_profile','atlas_spot_number',
            'sc_orientation']:
            fileID[gtx].attrs[att_name] = IS2_atl06_attrs[gtx][att_name]
        #-- create land_ice_segments group
        fileID[gtx].create_group('land_ice_segments')
        h5[gtx] = dict(land_ice_segments={})
        for att_name in ['Description','data_rate']:
            att_val = IS2_atl06_attrs[gtx]['land_ice_segments'][att_name]
            fileID[gtx]['land_ice_segments'].attrs[att_name] = att_val

        #-- segment_id, geolocation, time and height variables
        for k in ['delta_time','latitude','longitude','segment_id']:
            #-- values and attributes
            v = IS2_atl06_dem[gtx]['land_ice_segments'][k]
            attrs = IS2_atl06_attrs[gtx]['land_ice_segments'][k]
            fillvalue = FILL_VALUE[gtx]['land_ice_segments'][k]
            #-- Defining the HDF5 dataset variables
            val = '{0}/{1}/{2}'.format(gtx,'land_ice_segments',k)
            if fillvalue:
                h5[gtx]['land_ice_segments'][k] = fileID.create_dataset(val,
                    np.shape(v), data=v, dtype=v.dtype, fillvalue=fillvalue,
                    compression='gzip')
            else:
                h5[gtx]['land_ice_segments'][k] = fileID.create_dataset(val,
                    np.shape(v), data=v, dtype=v.dtype, compression='gzip')
            #-- create or attach dimensions for HDF5 variable
            if DIMENSIONS[gtx]['land_ice_segments'][k]:
                #-- attach dimensions
                for i,dim in enumerate(DIMENSIONS[gtx]['land_ice_segments'][k]):
                    h5[gtx]['land_ice_segments'][k].dims[i].attach_scale(
                        h5[gtx]['land_ice_segments'][dim])
            else:
                #-- make dimension
                h5[gtx]['land_ice_segments'][k].make_scale(k)
            #-- add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5[gtx]['land_ice_segments'][k].attrs[att_name] = att_val

        #-- add to output variables
        for key in ['dem',]:
            fileID[gtx]['land_ice_segments'].create_group(key)
            h5[gtx]['land_ice_segments'][key] = {}
            for att_name in ['Description','data_rate']:
                att_val=IS2_atl06_attrs[gtx]['land_ice_segments'][key][att_name]
                fileID[gtx]['land_ice_segments'][key].attrs[att_name] = att_val
            for k,v in IS2_atl06_dem[gtx]['land_ice_segments'][key].items():
                #-- attributes
                attrs = IS2_atl06_attrs[gtx]['land_ice_segments'][key][k]
                fillvalue = FILL_VALUE[gtx]['land_ice_segments'][key][k]
                #-- Defining the HDF5 dataset variables
                val = '{0}/{1}/{2}/{3}'.format(gtx,'land_ice_segments',key,k)
                if fillvalue:
                    h5[gtx]['land_ice_segments'][key][k] = \
                        fileID.create_dataset(val, np.shape(v), data=v,
                        dtype=v.dtype, fillvalue=fillvalue, compression='gzip')
                else:
                    h5[gtx]['land_ice_segments'][key][k] = \
                        fileID.create_dataset(val, np.shape(v), data=v,
                        dtype=v.dtype, compression='gzip')
                #-- attach dimensions
                for i,dim in enumerate(DIMENSIONS[gtx]['land_ice_segments'][key][k]):
                    h5[gtx]['land_ice_segments'][key][k].dims[i].attach_scale(
                        h5[gtx]['land_ice_segments'][dim])
                #-- add HDF5 variable attributes
                for att_name,att_val in attrs.items():
                    h5[gtx]['land_ice_segments'][key][k].attrs[att_name] = att_val

    #-- HDF5 file title
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'ATLAS/ICESat-2 Land Ice Height'
    fileID.attrs['summary'] = ('Geophysical parameters for land ice segments '
        'needed to interpret and assess the quality of land height estimates.')
    fileID.attrs['description'] = ('Land ice parameters for each beam.  All '
        'parameters are calculated for the same along-track increments for '
        'each beam and repeat.')
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
    #-- add attributes for input ATL06 file
    fileID.attrs['input_files'] = os.path.basename(INPUT)
    #-- find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for gtx in beams:
        lon = IS2_atl06_dem[gtx]['land_ice_segments']['longitude']
        lat = IS2_atl06_dem[gtx]['land_ice_segments']['latitude']
        delta_time = IS2_atl06_dem[gtx]['land_ice_segments']['delta_time']
        #-- setting the geospatial and temporal ranges
        lnmn = lon.min() if (lon.min() < lnmn) else lnmn
        lnmx = lon.max() if (lon.max() > lnmx) else lnmx
        ltmn = lat.min() if (lat.min() < ltmn) else ltmn
        ltmx = lat.max() if (lat.max() > ltmx) else ltmx
        tmn = delta_time.min() if (delta_time.min() < tmn) else tmn
        tmx = delta_time.max() if (delta_time.max() > tmx) else tmx
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
    fileID.attrs['time_coverage_duration'] = f'{tmx-tmn:0.0f}'
    #-- Closing the HDF5 file
    fileID.close()

#-- PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Interpolate DEMs to ICESat-2 ATL06 land
            ice height locations
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = \
        icesat2_toolkit.utilities.convert_arg_line_to_args
    #-- command line parameters
    parser.add_argument('file',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        help='ICESat-2 ATL06 file to run')
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
