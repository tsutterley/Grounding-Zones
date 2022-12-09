#!/usr/bin/env python
u"""
tile_ICESat_GLA12.py
Written by Tyler Sutterley (12/2022)
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
    time.py: utilities for calculating time operations
    spatial: utilities for reading, writing and operating on spatial data

UPDATE HISTORY:
    Updated 12/2022: check that file exists within multiprocess HDF5 function
    Updated 07/2022: place some imports within try/except statements
    Updated 06/2022: add checks if variables and groups already exist
    Updated 05/2022: use argparse descriptions within documentation
    Written 02/2022
"""
import sys
import os
import re
import copy
import time
import pyproj
import logging
import argparse
import warnings
import collections
import numpy as np
import grounding_zones as gz

# attempt imports
try:
    import h5py
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("h5py not available")
try:
    import pyTMD
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("pyTMD not available")
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: attempt to open an HDF5 file and wait if already open
def multiprocess_h5py(filename, *args, **kwargs):
    # check that file exists if entering with read mode
    if kwargs['mode'] in ('r','r+') and not os.access(filename, os.F_OK):
        raise FileNotFoundError(filename)
    # attempt to open HDF5 file
    while True:
        try:
            fileID = h5py.File(filename, *args, **kwargs)
            break
        except (IOError, BlockingIOError, PermissionError) as e:
            time.sleep(1)
    # return the file access object
    return fileID

# PURPOSE: create tile index files of ICESat ice sheet HDF5 elevation
# data (GLAH12) from NSIDC
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
    DIRECTORY = os.path.dirname(input_file)
    BASENAME = os.path.basename(input_file)
    output_file = os.path.join(DIRECTORY, index_directory, BASENAME)
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
    PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE = rx.findall(input_file).pop()

    # pyproj transformer for converting to polar stereographic
    EPSG = dict(N=3413,S=3031)
    SIGN = dict(N=1.0,S=-1.0)
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

    # create index directory for hemisphere
    if not os.access(os.path.join(DIRECTORY,index_directory),os.F_OK):
        os.makedirs(os.path.join(DIRECTORY,index_directory),
            mode=MODE, exist_ok=True)

    # track file progress
    logging.info(input_file)

    # read GLAH12 HDF5 file
    fileID = h5py.File(input_file,'r')
    n_40HZ, = fileID['Data_40HZ']['Time']['i_rec_ndx'].shape
    # get variables and attributes
    # copy ICESat campaign name from ancillary data
    campaign = copy.copy(fileID['ANCILLARY_DATA'].attrs['Campaign'])
    # ICESat record
    key = 'i_rec_ndx'
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
    # Latitude (TOPEX/Poseidon ellipsoid degrees North)
    lat_TPX = fileID['Data_40HZ']['Geolocation']['d_lat'][:].copy()
    # Longitude (degrees East)
    lon_40HZ = fileID['Data_40HZ']['Geolocation']['d_lon'][:].copy()
    # Elevation (height above TOPEX/Poseidon ellipsoid in meters)
    elev_TPX = fileID['Data_40HZ']['Elevation_Surfaces']['d_elev'][:].copy()
    fv = fileID['Data_40HZ']['Elevation_Surfaces']['d_elev'].attrs['_FillValue']

    # semimajor axis (a) and flattening (f) for TP and WGS84 ellipsoids
    atop,ftop = (6378136.3,1.0/298.257)
    awgs,fwgs = (6378137.0,1.0/298.257223563)
    # convert from Topex/Poseidon to WGS84 Ellipsoids
    lat_40HZ,elev_40HZ = pyTMD.spatial.convert_ellipsoid(lat_TPX, elev_TPX,
        atop, ftop, awgs, fwgs, eps=1e-12, itmax=10)

    # create index directory for hemisphere
    if not os.access(os.path.join(DIRECTORY,index_directory),os.F_OK):
        os.makedirs(os.path.join(DIRECTORY,index_directory),
            mode=MODE, exist_ok=True)

    # indices of points in hemisphere
    valid, = np.nonzero((np.sign(lat_40HZ) == SIGN[HEM]) & (elev_TPX != fv))
    # convert latitude and longitude to regional projection
    x,y = transformer.transform(lon_40HZ,lat_40HZ)
    # large-scale tiles
    xtile = (x-0.5*SPACING)//SPACING
    ytile = (y-0.5*SPACING)//SPACING

    # open output index file
    f2 = h5py.File(output_file, 'w')
    f2.attrs['featureType'] = 'trajectory'
    f2.attrs['GDAL_AREA_OR_POINT'] = 'Point'
    f2.attrs['time_type'] = 'UTC'
    today = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    f2.attrs['date_created'] = today
    f2.attrs['campaign'] = campaign
    # add software information
    git_revision_hash =  gz.utilities.get_git_revision_hash()
    f2.attrs['software_reference'] = gz.version.project_name
    f2.attrs['software_version'] = gz.version.full_version
    f2.attrs['software_revision'] = git_revision_hash
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
        tile_file = os.path.join(DIRECTORY, index_directory,
            f'{tile_group}.h5')
        clobber = 'a' if os.access(tile_file, os.F_OK) else 'w'
        # open output merged tile file
        f3 = multiprocess_h5py(tile_file, mode=clobber)
        # create file group
        if BASENAME not in f3:
            g3 = f3.create_group(BASENAME)
        else:
            g3 = f3[BASENAME]
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
            f3.attrs['software_revision'] = git_revision_hash

        # indices of points within tile
        indices, = np.nonzero((xtile == xp) & (ytile == yp))
        # output variables for index file
        output = collections.OrderedDict()
        output['DS_UTCTime_40'] = DS_UTCTime_40HZ[indices].copy()
        output['i_rec_ndx'] = rec_ndx_40HZ[indices].copy()
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

    # Output HDF5 structure information
    logging.info(list(f2.keys()))
    # close the output file
    f2.close()
    # change the permissions mode of the output file
    os.chmod(output_file, mode=MODE)

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
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat GLA12 file to run')
    # region of interest to run
    parser.add_argument('--hemisphere','-H',
        type=str, default='N', choices=('N','S'),
        help='Hemisphere')
    # output grid spacing
    parser.add_argument('--spacing','-S',
        type=float, default=10e3,
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

    # run program for each file
    for FILE in args.infile:
        tile_ICESat_GLA12(FILE,
            SPACING=args.spacing,
            HEM=args.hemisphere,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
