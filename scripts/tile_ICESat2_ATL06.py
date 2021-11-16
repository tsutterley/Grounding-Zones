#!/usr/bin/env python
u"""
tile_ICESat2_ATL06.py
Written by Tyler Sutterley (11/2021)
Creates tile index files of ICESat-2 land ice elevation data

COMMAND LINE OPTIONS:
    --help: list the command line options
    -H X, --hemisphere X: Region of interest to run
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
    read_ICESat2_ATL06.py: reads ICESat-2 land ice along-track height data files

UPDATE HISTORY:
    Updated 11/2021: adjust tiling to index by center coordinates
    Written 10/2021
"""
import sys
import os
import re
import h5py
import pyproj
import logging
import argparse
import datetime
import collections
import numpy as np
from icesat2_toolkit.read_ICESat2_ATL06 import read_HDF5_ATL06

#-- PURPOSE: set the hemisphere of interest based on the granule
def set_hemisphere(GRANULE):
    if GRANULE in ('10','11','12'):
        return 'S'
    elif GRANULE in ('03','04','05'):
        return 'N'
    else:
        raise Exception('Non-polar granule')

#-- PURPOSE: create tile index files of ICESat-2 elevation data
def tile_ICESat2_ATL06(FILE,
    SPACING=None,
    VERBOSE=False,
    MODE=0o775):

    #-- create logger
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    #-- read data from input file
    logging.info(FILE)
    IS2_atl06_mds,IS2_atl06_attrs,IS2_atl06_beams = read_HDF5_ATL06(FILE,
        ATTRIBUTES=True)
    #-- extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_'
        r'(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$',re.VERBOSE)
    PRD,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX = rx.findall(FILE).pop()
    #-- set the hemisphere flag based on ICESat-2 granule
    HEM = set_hemisphere(GRAN)

    #-- index directory for hemisphere
    index_directory = 'north' if (HEM == 'N') else 'south'
    #-- output directory and index file
    DIRECTORY = os.path.dirname(FILE)
    output_file = os.path.join(DIRECTORY, index_directory,
        os.path.basename(FILE))

    #-- pyproj transformer for converting to polar stereographic
    EPSG = dict(N=3413,S=3031)
    crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(4326))
    crs2 = pyproj.CRS.from_string("epsg:{0:d}".format(EPSG[HEM]))
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    #-- dictionary of coordinate reference system variables
    cs_to_cf = crs2.cs_to_cf()

    #-- attributes for each output item
    attributes = dict(x={},y={},index={})
    #-- x and y
    for att_name in ['long_name','standard_name','units']:
        attributes['x'][att_name] = cs_to_cf[0][att_name]
        attributes['y'][att_name] = cs_to_cf[1][att_name]
    #-- index
    attributes['index']['long_name'] = 'Index'
    attributes['index']['grid_mapping'] = 'Polar_Stereographic'
    attributes['index']['units'] = '1'
    attributes['index']['coordinates'] = 'x y'
    #-- beam group attribute keys
    attributes['beam'] = ['Description','groundtrack_id',
        'atmosphere_profile','atlas_spot_number',
        'sc_orientation','atlas_beam_type','atlas_pce']

    #-- create index directory for hemisphere
    if not os.access(os.path.join(DIRECTORY,index_directory),os.F_OK):
        os.makedirs(os.path.join(DIRECTORY,index_directory),
            mode=MODE, exist_ok=True)

    #-- open output index file
    f2 = h5py.File(output_file,'w')
    f2.attrs['featureType'] = 'trajectory'
    f2.attrs['GDAL_AREA_OR_POINT'] = 'Point'
    f2.attrs['time_type'] = 'GPS'
    today = datetime.datetime.now().isoformat()
    f2.attrs['date_created'] = today
    #-- create projection variable
    h5 = f2.create_dataset('Polar_Stereographic',(),dtype=np.byte)
    #-- add projection attributes
    h5.attrs['standard_name'] = 'Polar_Stereographic'
    h5.attrs['spatial_epsg'] = crs2.to_epsg()
    h5.attrs['spatial_ref'] = crs2.to_wkt()
    for att_name,att_val in crs2.to_cf().items():
        h5.attrs[att_name] = att_val

    #-- for each input beam within the file
    for gtx in sorted(IS2_atl06_beams):
        #-- extract latitude, longitude, delta_time and segment_id
        latitude = IS2_atl06_mds[gtx]['land_ice_segments']['latitude']
        longitude = IS2_atl06_mds[gtx]['land_ice_segments']['longitude']
        delta_time = IS2_atl06_mds[gtx]['land_ice_segments']['delta_time']
        segment_id = IS2_atl06_mds[gtx]['land_ice_segments']['segment_id']
        #-- convert latitude and longitude to regional projection
        x,y = transformer.transform(longitude,latitude)
        #-- large-scale tiles
        xtile = (x-0.5*SPACING)//SPACING
        ytile = (y-0.5*SPACING)//SPACING
        #-- add delta time and segment id attributes
        for key in ['delta_time','segment_id']:
            for att_name in ('DIMENSION_LIST','CLASS','NAME'):
                IS2_atl06_attrs[gtx]['land_ice_segments'][key].pop(att_name,None)
            attributes[key] =  IS2_atl06_attrs[gtx]['land_ice_segments'][key]
        #-- for each valid tile pair
        for xp,yp in set(zip(xtile,ytile)):
            #-- center of each tile (adjust due to integer truncation)
            xc = (xp+1)*SPACING
            yc = (yp+1)*SPACING
            #-- create group
            tile_group = 'E{0:0.0f}_N{1:0.0f}'.format(xc/1e3,yc/1e3)
            if tile_group not in f2:
                g1 = f2.create_group(tile_group)
            else:
                g1 = f2[tile_group]
            #-- add group attributes
            g1.attrs['x_center'] = xc
            g1.attrs['y_center'] = yc
            g1.attrs['spacing'] = SPACING

            #-- create merged tile file if not existing
            tile_file = os.path.join(DIRECTORY,index_directory,
                '{0}.h5'.format(tile_group))
            clobber = 'a' if os.access(tile_file,os.F_OK) else 'w'
            #-- open output merged tile file
            f3 = h5py.File(tile_file,clobber)
            #-- create group for file
            if os.path.basename(FILE) not in f3:
                g3 = f3.create_group(os.path.basename(FILE))
            else:
                g3 = f3[os.path.basename(FILE)]
            #-- add file-level variables and attributes
            if (clobber == 'w'):
                #-- create projection variable
                h5 = f3.create_dataset('Polar_Stereographic',(),
                    dtype=np.byte)
                #-- add projection attributes
                h5.attrs['standard_name'] = 'Polar_Stereographic'
                h5.attrs['spatial_epsg'] = crs2.to_epsg()
                h5.attrs['spatial_ref'] = crs2.to_wkt()
                for att_name,att_val in crs2.to_cf().items():
                    h5.attrs[att_name] = att_val
                #-- add file attributes
                f3.attrs['featureType'] = 'trajectory'
                f3.attrs['x_center'] = xc
                f3.attrs['y_center'] = yc
                f3.attrs['spacing'] = SPACING
                f3.attrs['GDAL_AREA_OR_POINT'] = 'Point'
                f3.attrs['time_type'] = 'UTC'
                f3.attrs['date_created'] = today

            #-- indices of points within tile
            indices, = np.nonzero((xtile == xp) & (ytile == yp))
            #-- output variables for index file
            output = collections.OrderedDict()
            output['delta_time'] = delta_time[indices].copy()
            output['segment_id'] = segment_id[indices].copy()
            output['x'] = x[indices].copy()
            output['y'] = y[indices].copy()
            output['index'] = indices.copy()

            #-- create group for beam
            g2 = f2.create_group('{0}/{1}'.format(tile_group,gtx))
            g4 = f3.create_group('{0}/{1}'.format(os.path.basename(FILE),gtx))
            #-- for each group
            for g in [g2,g4]:
                #-- add attributes for ATL06 beam
                for att_name in attributes['beam']:
                    g.attrs[att_name] = IS2_atl06_attrs[gtx][att_name]
                #-- for each output variable
                h5 = {}
                for key,val in output.items():
                    #-- create HDF5 variables
                    h5[key] = g.create_dataset(key, val.shape, data=val,
                        dtype=val.dtype, compression='gzip')
                    #-- add variable attributes
                    for att_name,att_val in attributes[key].items():
                        h5[key].attrs[att_name] = att_val
                    #-- create or attach dimensions
                    if key not in ('delta_time',):
                        for i,dim in enumerate(['delta_time']):
                            h5[key].dims[i].attach_scale(h5[dim])
                    else:
                        h5[key].make_scale(key)

    #-- Output HDF5 structure information
    logging.info(list(f2.keys()))
    #-- close the output file
    f2.close()
    #-- change the permissions mode of the output file
    os.chmod(output_file, mode=MODE)

#-- Main program that calls tile_ICESat2_ATL06()
def main():
   #-- Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Creates tile index files of ICESat-2 ATL06
            land ice elevation data
            """
    )
    #-- command line parameters
    #-- input ICESat-2 land ice height files
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat-2 ATL06 file to run')
    #-- output grid spacing
    parser.add_argument('--spacing','-S',
        type=float, default=10e3,
        help='Output grid spacing')
    #-- verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of run')
    #-- permissions mode of the directories and files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of directories and files')
    args,_ = parser.parse_known_args()

    #-- run program for each product
    for FILE in args.infile:
        tile_ICESat2_ATL06(FILE,
            SPACING=args.spacing,
            VERBOSE=args.verbose,
            MODE=args.mode)

#-- run main program
if __name__ == '__main__':
    main()
