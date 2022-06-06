#!/usr/bin/env python
u"""
tile_ICESat2_ATL11.py
Written by Tyler Sutterley (06/2022)
Creates tile index files of ICESat-2 annual land ice elevation data

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
    read_ICESat2_ATL11.py: reads ICESat-2 annual land ice height data files

UPDATE HISTORY:
    Updated 06/2022: add checks if variables and groups already exist
    Updated 05/2022: use argparse descriptions within documentation
    Updated 11/2021: adjust tiling to index by center coordinates
        wait if merged HDF5 tile file is unavailable
    Written 10/2021
"""
import sys
import os
import re
import h5py
import time
import pyproj
import logging
import argparse
import collections
import numpy as np
from icesat2_toolkit.read_ICESat2_ATL11 import read_HDF5_ATL11

#-- PURPOSE: set the hemisphere of interest based on the granule
def set_hemisphere(GRANULE):
    if GRANULE in ('10','11','12'):
        return 'S'
    elif GRANULE in ('03','04','05'):
        return 'N'
    else:
        raise Exception('Non-polar granule')

#-- PURPOSE: attempt to open an HDF5 file and wait if already open
def multiprocess_h5py(filename, *args, **kwargs):
    while True:
        try:
            fileID = h5py.File(filename, *args, **kwargs)
            break
        except (IOError, OSError, PermissionError) as e:
            time.sleep(1)
    return fileID

#-- PURPOSE: create tile index files of ICESat-2 elevation data
def tile_ICESat2_ATL11(FILE,
    SPACING=None,
    VERBOSE=False,
    MODE=0o775):

    #-- create logger
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    #-- read data from input file
    logging.info(FILE)
    IS2_atl11_mds,IS2_atl11_attrs,IS2_atl11_pairs = read_HDF5_ATL11(FILE,
        ATTRIBUTES=True)
    #-- extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})_(\d{2})(\d{2})_'
        r'(\d{3})_(\d{2})(.*?).h5$')
    SUB,PRD,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX = rx.findall(FILE).pop()
    #-- set the hemisphere flag based on ICESat-2 granule
    HEM = set_hemisphere(GRAN)

    #-- index directory for hemisphere
    index_directory = 'north' if (HEM == 'N') else 'south'
    #-- output directory and index file
    DIRECTORY = os.path.dirname(FILE)
    BASENAME = os.path.basename(FILE)
    output_file = os.path.join(DIRECTORY, index_directory, BASENAME)

    #-- pyproj transformer for converting to polar stereographic
    EPSG = dict(N=3413,S=3031)
    crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(4326))
    crs2 = pyproj.CRS.from_string("epsg:{0:d}".format(EPSG[HEM]))
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    #-- dictionary of coordinate reference system variables
    cs_to_cf = crs2.cs_to_cf()
    crs_to_dict = crs2.to_dict()

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
    #-- beam pair group attribute keys
    attributes['pair'] = ['beam_pair','beam_spacing','seg_atc_spacing',
        'ReferenceGroundTrack','first_cycle','last_cycle',
        'L_search_AT','L_search_XT','seg_sigma_threshold_min','N_search',
        'poly_max_degree_AT','poly_max_degree_XT','xy_scale','t_scale',
        'max_fit_iterations','equatorial_radius','polar_radius',
        'seg_number_skip','pair_yatc_ctr_tol','N_poly_coeffs','N_coeffs']

    #-- create index directory for hemisphere
    if not os.access(os.path.join(DIRECTORY,index_directory),os.F_OK):
        os.makedirs(os.path.join(DIRECTORY,index_directory),
            mode=MODE, exist_ok=True)

    #-- open output index file
    f2 = h5py.File(output_file,'w')
    f2.attrs['featureType'] = 'trajectory'
    f2.attrs['GDAL_AREA_OR_POINT'] = 'Point'
    today = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    f2.attrs['date_created'] = today
    #-- create projection variable
    h5 = f2.create_dataset('Polar_Stereographic', (), dtype=np.byte)
    #-- add projection attributes
    h5.attrs['standard_name'] = 'Polar_Stereographic'
    h5.attrs['spatial_epsg'] = crs2.to_epsg()
    h5.attrs['spatial_ref'] = crs2.to_wkt()
    h5.attrs['proj4_params'] = crs2.to_proj4()
    h5.attrs['latitude_of_projection_origin'] = crs_to_dict['lat_0']
    for att_name,att_val in crs2.to_cf().items():
        h5.attrs[att_name] = att_val

    #-- for each input beam pair within the file
    for ptx in sorted(IS2_atl11_pairs):
        #-- along-track (AT) reference point, latitude and longitude
        ref_pt = IS2_atl11_mds[ptx]['ref_pt'].copy()
        latitude = np.ma.array(IS2_atl11_mds[ptx]['latitude'],
            fill_value=IS2_atl11_attrs[ptx]['latitude']['_FillValue'])
        longitude = np.ma.array(IS2_atl11_mds[ptx]['longitude'],
            fill_value=IS2_atl11_attrs[ptx]['longitude']['_FillValue'])
        #-- convert latitude and longitude to regional projection
        x,y = transformer.transform(longitude,latitude)
        #-- large-scale tiles
        xtile = (x-0.5*SPACING)//SPACING
        ytile = (y-0.5*SPACING)//SPACING
        #-- find valid latitudes
        valid, = np.nonzero(latitude.data != latitude.fill_value)
        #-- add ref_pt attributes
        for key in ['ref_pt']:
            for att_name in ('DIMENSION_LIST','CLASS','NAME'):
                IS2_atl11_attrs[ptx][key].pop(att_name,None)
            attributes[key] = IS2_atl11_attrs[ptx][key]
        #-- for each valid tile pair
        for xp,yp in set(zip(xtile[valid],ytile[valid])):
            #-- center of each tile (adjust due to integer truncation)
            xc = (xp+1)*SPACING
            yc = (yp+1)*SPACING
            #-- create group
            tile_group = 'E{0:0.0f}_N{1:0.0f}'.format(xc/1e3, yc/1e3)
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
            f3 = multiprocess_h5py(tile_file,clobber)
            #-- create group for file
            if BASENAME not in f3:
                g3 = f3.create_group(BASENAME)
            else:
                g3 = f3[BASENAME]
            #-- add file-level variables and attributes
            if (clobber == 'w'):
                #-- create projection variable
                h5 = f3.create_dataset('Polar_Stereographic', (),
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
                f3.attrs['date_created'] = today

            #-- indices of points within tile
            indices, = np.nonzero((xtile == xp) & (ytile == yp))
            #-- output variables for index file
            output = collections.OrderedDict()
            output['ref_pt'] = ref_pt[indices].copy()
            output['x'] = x[indices].copy()
            output['y'] = y[indices].copy()
            output['index'] = indices.copy()

            #-- groups for beam pair
            tile_pair_group = '{0}/{1}'.format(tile_group,ptx)
            pair_group = '{0}/{1}'.format(BASENAME,ptx)
            #-- try to create groups for each beam pair
            if tile_pair_group not in f2:
                g2 = f2.create_group(tile_pair_group)
            else:
                g2 = f2[tile_pair_group]
            if pair_group not in f3:
                g4 = f3.create_group(pair_group)
            else:
                g4 = f3[pair_group]
            #-- for each group
            for g in [g2,g4]:
                #-- add attributes for ATL11 beam pair
                for att_name in attributes['pair']:
                    g.attrs[att_name] = IS2_atl11_attrs[ptx][att_name]
                #-- for each output variable
                h5 = {}
                for key,val in output.items():
                    #-- check if HDF5 variable exists
                    if key not in g:
                        #-- create HDF5 variable
                        h5[key] = g.create_dataset(key, val.shape, data=val,
                            dtype=val.dtype, compression='gzip')
                    else:
                        #-- overwrite HDF5 variable
                        h5[key] = g[key]
                        h5[key][...] = val
                    #-- add variable attributes
                    for att_name,att_val in attributes[key].items():
                        h5[key].attrs[att_name] = att_val
                    #-- create or attach dimensions
                    if key not in ('ref_pt',):
                        for i,dim in enumerate(['ref_pt']):
                            h5[key].dims[i].attach_scale(h5[dim])
                    else:
                        h5[key].make_scale(key)

    #-- Output HDF5 structure information
    logging.info(list(f2.keys()))
    #-- close the output file
    f2.close()
    #-- change the permissions mode of the output file
    os.chmod(output_file, mode=MODE)

#-- PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Creates tile index files of ICESat-2 ATL11
            annual land ice elevation data
            """
    )
    #-- command line parameters
    #-- input ICESat-2 annual land ice height files
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat-2 ATL11 file to run')
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
    #-- return the parser
    return parser

#-- This is the main part of the program that calls the individual functions
def main():
    #-- Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    #-- run program for each file
    for FILE in args.infile:
        tile_ICESat2_ATL11(FILE,
            SPACING=args.spacing,
            VERBOSE=args.verbose,
            MODE=args.mode)

#-- run main program
if __name__ == '__main__':
    main()
