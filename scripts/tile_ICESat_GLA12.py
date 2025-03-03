#!/usr/bin/env python
u"""
tile_ICESat_GLA12.py
Written by Tyler Sutterley (05/2024)
Creates tile index files of ICESat/GLAS L2 GLA12 Antarctic and
    Greenland Ice Sheet elevation data

INPUTS:
    input_file: ICESat GLA12 data file

COMMAND LINE OPTIONS:
    --help: list the command line options
    -H X, --hemisphere X: Region of interest to run
    -S X, --spacing X: Output grid spacing
    -V, --verbose: Verbose output of run
    -M X, --mode X: Permissions mode of the directories and files

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/

PROGRAM DEPENDENCIES:
    spatial: utilities for reading, writing and operating on spatial data

UPDATE HISTORY:
    Updated 05/2024: adjust default spacing of tiles to 80 km
        return if no valid points in hemisphere
        use wrapper to importlib for optional dependencies
        change permissions mode of the output tile files
        moved multiprocess h5py reader to io utilities module
        write 40HZ reference track number in output files
    Updated 05/2023: using pathlib to define and operate on paths
    Updated 12/2022: check that file exists within multiprocess HDF5 function
        use constants class from pyTMD for ellipsoidal parameters
        single implicit import of grounding zone tools
    Updated 07/2022: place some imports within try/except statements
    Updated 06/2022: add checks if variables and groups already exist
    Updated 05/2022: use argparse descriptions within documentation
    Written 02/2022
"""
import sys
import re
import copy
import time
import logging
import pathlib
import argparse
import collections
import numpy as np
import grounding_zones as gz

# attempt imports
h5py = gz.utilities.import_dependency('h5py')
pyproj = gz.utilities.import_dependency('pyproj')
pyTMD = gz.utilities.import_dependency('pyTMD')

# PURPOSE: create tile index files of ICESat ice sheet
# HDF5 elevation data (GLAH12)
def tile_ICESat_GLA12(input_file,
    SPACING=None,
    HEM=None,
    VERBOSE=False,
    MODE=0o775):

    # create logger
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # index directory for hemisphere
    index_directory = 'north' if (HEM == 'N') else 'south'
    # output directory and index file
    input_file = pathlib.Path(input_file).expanduser().absolute()
    DIRECTORY = input_file.with_name(index_directory)
    output_file = DIRECTORY.joinpath(input_file.name)
    # create index directory for hemisphere
    DIRECTORY.mkdir(mode=MODE, parents=True, exist_ok=True)

    # compile regular expression operator for extracting information from file
    rx = re.compile((r'GLAH(\d{2})_(\d{3})_(\d{1})(\d{1})(\d{2})_(\d{3})_'
        r'(\d{4})_(\d{1})_(\d{2})_(\d{4})\.H5$'), re.VERBOSE)
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
    PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE = \
        rx.findall(input_file.name).pop()

    # pyproj transformer for converting to polar stereographic
    EPSG = dict(N=3413, S=3031)
    SIGN = dict(N=1.0, S=-1.0)
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_epsg(EPSG[HEM])
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    # dictionary of coordinate reference system variables
    cs_to_cf = crs2.cs_to_cf()
    crs_to_dict = crs2.to_dict()

    # attributes for each output item
    attributes = dict(x={},y={},index={},i_rec_ndx={},DS_UTCTime_40={})
    # x and y
    attributes['x'],attributes['y'] = ({},{})
    for att_name in ['long_name','standard_name','units']:
        attributes['x'][att_name] = cs_to_cf[0][att_name]
        attributes['y'][att_name] = cs_to_cf[1][att_name]
    # index
    attributes['index'] = {}
    attributes['index']['long_name'] = 'Index'
    attributes['index']['grid_mapping'] = 'Polar_Stereographic'
    attributes['index']['units'] = '1'
    attributes['index']['coordinates'] = 'x y'
    # track
    attributes['i_track'] = {}
    attributes['i_track']['long_name'] = 'Track'
    attributes['i_track']['description'] = 'Reference track number'
    attributes['i_track']['grid_mapping'] = 'Polar_Stereographic'
    attributes['i_track']['units'] = '1'
    attributes['i_track']['coordinates'] = 'x y'

    # track file progress
    logging.info(str(input_file))

    # read GLAH12 HDF5 file
    fileID = h5py.File(input_file, mode='r')
    n_40HZ, = fileID['Data_40HZ']['Time']['i_rec_ndx'].shape
    # get variables and attributes
    # copy ICESat campaign name from ancillary data
    campaign = copy.copy(fileID['ANCILLARY_DATA'].attrs['Campaign'])
    # ICESat record
    key = 'i_rec_ndx'
    rec_ndx_1HZ = fileID['Data_1HZ']['Time'][key][:].copy()
    rec_ndx_40HZ = fileID['Data_40HZ']['Time'][key][:].copy()
    for att_name,att_val in fileID['Data_40HZ']['Time'][key].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            attributes['i_rec_ndx'][att_name] = copy.copy(att_val)
    # seconds since 2000-01-01 12:00:00 UTC (J2000)
    key = 'DS_UTCTime_40'
    DS_UTCTime_40HZ = fileID['Data_40HZ'][key][:].copy()
    for att_name,att_val in fileID['Data_40HZ'][key].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            attributes['DS_UTCTime_40'][att_name] = copy.copy(att_val)
    # ICESat track number
    i_track_1HZ = fileID['Data_1HZ']['Geolocation']['i_track'][:].copy()
    i_track_40HZ = np.zeros((n_40HZ), dtype=i_track_1HZ.dtype)
    # Latitude (TOPEX/Poseidon ellipsoid degrees North)
    lat_TPX = fileID['Data_40HZ']['Geolocation']['d_lat'][:].copy()
    # Longitude (degrees East)
    lon_40HZ = fileID['Data_40HZ']['Geolocation']['d_lon'][:].copy()
    # Elevation (height above TOPEX/Poseidon ellipsoid in meters)
    elev_TPX = fileID['Data_40HZ']['Elevation_Surfaces']['d_elev'][:].copy()
    fv = fileID['Data_40HZ']['Elevation_Surfaces']['d_elev'].attrs['_FillValue']
    # map 1HZ data to 40HZ data
    for k,record in enumerate(rec_ndx_1HZ):
        # indice mapping the 40HZ data to the 1HZ data
        map_1HZ_40HZ, = np.nonzero(rec_ndx_40HZ == record)
        i_track_40HZ[map_1HZ_40HZ] = i_track_1HZ[k]

    # parameters for Topex/Poseidon and WGS84 ellipsoids
    topex = pyTMD.spatial.datum(ellipsoid='TOPEX', units='MKS')
    wgs84 = pyTMD.spatial.datum(ellipsoid='WGS84', units='MKS')
    # convert from Topex/Poseidon to WGS84 Ellipsoids
    lat_40HZ, elev_40HZ = pyTMD.spatial.convert_ellipsoid(
        lat_TPX, elev_TPX,
        topex.a_axis, topex.flat,
        wgs84.a_axis, wgs84.flat,
        eps=1e-12, itmax=10)

    # indices of points in hemisphere
    valid, = np.nonzero((np.sign(lat_40HZ) == SIGN[HEM]) & (elev_TPX != fv))
    if not valid.any():
        logging.error('No valid points in hemisphere')
        return
    # convert latitude and longitude to regional projection
    x,y = transformer.transform(lon_40HZ,lat_40HZ)
    # large-scale tiles
    xtile = (x-0.5*SPACING)//SPACING
    ytile = (y-0.5*SPACING)//SPACING

    # open output index file
    f2 = h5py.File(output_file, mode='w')
    f2.attrs['featureType'] = 'trajectory'
    f2.attrs['GDAL_AREA_OR_POINT'] = 'Point'
    f2.attrs['time_type'] = 'UTC'
    today = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    f2.attrs['date_created'] = today
    f2.attrs['campaign'] = campaign
    # add software information
    f2.attrs['software_reference'] = gz.version.project_name
    f2.attrs['software_version'] = gz.version.full_version
    # create projection variable
    h5 = f2.create_dataset('Polar_Stereographic', (), dtype=np.byte)
    # add projection attributes
    h5.attrs['standard_name'] = 'Polar_Stereographic'
    h5.attrs['spatial_epsg'] = crs2.to_epsg()
    h5.attrs['spatial_ref'] = crs2.to_wkt()
    h5.attrs['proj4_params'] = crs2.to_proj4()
    h5.attrs['latitude_of_projection_origin'] = crs_to_dict['lat_0']
    for att_name,att_val in crs2.to_cf().items():
        h5.attrs[att_name] = att_val
    # for each valid tile pair
    for xp,yp in set(zip(xtile[valid],ytile[valid])):
        # center of each tile (adjust due to integer truncation)
        xc = (xp+1)*SPACING
        yc = (yp+1)*SPACING
        # create group
        tile_group = f'E{xc/1e3:0.0f}_N{yc/1e3:0.0f}'
        if tile_group not in f2:
            g2 = f2.create_group(tile_group)
        else:
            g2 = f2[tile_group]
        # add group attributes
        g2.attrs['x_center'] = xc
        g2.attrs['y_center'] = yc
        g2.attrs['spacing'] = SPACING

        # create merged tile file if not existing
        tile_file = DIRECTORY.joinpath(f'{tile_group}.h5')
        clobber = 'a' if tile_file.exists() else 'w'
        # open output merged tile file
        f3 = gz.io.multiprocess_h5py(tile_file, mode=clobber)
        # create file group
        if input_file.name not in f3:
            g3 = f3.create_group(input_file.name)
        else:
            g3 = f3[input_file.name]
        # add file-level variables and attributes
        if (clobber == 'w'):
            # create projection variable
            h5 = f3.create_dataset('Polar_Stereographic', (),
                dtype=np.byte)
            # add projection attributes
            h5.attrs['standard_name'] = 'Polar_Stereographic'
            h5.attrs['spatial_epsg'] = crs2.to_epsg()
            h5.attrs['spatial_ref'] = crs2.to_wkt()
            for att_name,att_val in crs2.to_cf().items():
                h5.attrs[att_name] = att_val
            # add file attributes
            f3.attrs['featureType'] = 'trajectory'
            f3.attrs['x_center'] = xc
            f3.attrs['y_center'] = yc
            f3.attrs['spacing'] = SPACING
            f3.attrs['GDAL_AREA_OR_POINT'] = 'Point'
            f3.attrs['time_type'] = 'UTC'
            f3.attrs['date_created'] = today
            # add software information
            f3.attrs['software_reference'] = gz.version.project_name
            f3.attrs['software_version'] = gz.version.full_version

        # indices of points within tile
        indices, = np.nonzero((xtile == xp) & (ytile == yp))
        # output variables for index file
        output = collections.OrderedDict()
        output['DS_UTCTime_40'] = DS_UTCTime_40HZ[indices].copy()
        output['i_rec_ndx'] = rec_ndx_40HZ[indices].copy()
        output['i_track'] = i_track_40HZ[indices].copy()
        output['x'] = x[indices].copy()
        output['y'] = y[indices].copy()
        output['index'] = indices.copy()
        # for each output group
        for g in [g2,g3]:
            # for each output variable
            h5 = {}
            for key,val in output.items():
                # check if HDF5 variable exists
                if key not in g:
                    # create HDF5 variable
                    h5[key] = g.create_dataset(key, val.shape, data=val,
                        dtype=val.dtype, compression='gzip')
                else:
                    # overwrite HDF5 variable
                    h5[key] = g[key]
                    h5[key][...] = val
                # add variable attributes
                for att_name,att_val in attributes[key].items():
                    h5[key].attrs[att_name] = att_val
                # create or attach dimensions
                if key not in ('DS_UTCTime_40',):
                    for i,dim in enumerate(['DS_UTCTime_40']):
                        h5[key].dims[i].attach_scale(h5[dim])
                else:
                    h5[key].make_scale(key)
        # close the merged tile file
        f3.close()
        # change the permissions mode of the merged tile file
        tile_file.chmod(mode=MODE)

    # Output HDF5 structure information
    logging.info(list(f2.keys()))
    # close the output file
    f2.close()
    # change the permissions mode of the output file
    output_file.chmod(mode=MODE)

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Creates tile index files of ICESat/GLAS L2 GLA12
            Antarctic and Greenland Ice Sheet elevation data
            """
    )
    # command line parameters
    # input ICESat GLAS files
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='ICESat GLA12 file to run')
    # region of interest to run
    parser.add_argument('--hemisphere','-H',
        type=str, default='N', choices=('N','S'),
        help='Hemisphere')
    # output grid spacing
    parser.add_argument('--spacing','-S',
        type=float, default=80e3,
        help='Output grid spacing')
    # verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of run')
    # permissions mode of the directories and files (number in octal)
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

    # run for each input GLAH12 file
    for FILE in args.infile:
        tile_ICESat_GLA12(FILE,
            SPACING=args.spacing,
            HEM=args.hemisphere,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
