#!/usr/bin/env python
u"""
track_ICESat_GLA12.py
Written by Tyler Sutterley (05/2024)
Creates index files of the ICESat/GLAS L2 GLA12 Antarctic and
    Greenland Ice Sheet tracks

INPUTS:
    input_file: ICESat GLA12 data file

COMMAND LINE OPTIONS:
    --help: list the command line options
    -V, --verbose: Verbose output of run
    -M X, --mode X: Permissions mode of the directories and files

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/

UPDATE HISTORY:
    Written 06/2024
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

# PURPOSE: create index files of ICESat ice sheet HDF5 (GLAH12) tracks
def track_ICESat_GLA12(input_file, VERBOSE=False, MODE=0o775):

    # create logger
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # index directory for tracks
    index_directory = 'track'
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

    # attributes for each output item
    attributes = dict(i_rec_ndx={},DS_UTCTime_40={},d_lon={},d_lat={})
    # index
    attributes['index'] = {}
    attributes['index']['long_name'] = 'Index'
    attributes['index']['units'] = '1'
    attributes['index']['coordinates'] = 'd_lon d_lat'
    # track
    attributes['i_track'] = {}
    attributes['i_track']['long_name'] = 'Track'
    attributes['i_track']['description'] = 'Reference track number'
    attributes['i_track']['units'] = '1'
    attributes['i_track']['coordinates'] = 'd_lon d_lat'
    # repeat ground-track phase
    attributes['i_rgtp'] = {}
    attributes['i_rgtp']['long_name'] = 'Repeat Ground-Track Phase'
    attributes['i_rgtp']['description'] = ('Repeat ground-track phase '
        '(1=8-day, 2=91-day, 3=transfer orbit)')
    attributes['i_rgtp']['units'] = '1'
    attributes['i_rgtp']['valid_min'] = 1
    attributes['i_rgtp']['valid_max'] = 3
    attributes['i_rgtp']['flag_meanings'] = '8_day 91_day transfer_orbit'
    attributes['i_rgtp']['flag_values'] = [1,2,3]
    attributes['i_rgtp']['coordinates'] = 'd_lon d_lat'

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
        if att_name not in ('DIMENSION_LIST','CLASS','NAME','coordinates'):
            attributes[key][att_name] = copy.copy(att_val)
    attributes[key]['coordinates'] = 'd_lon d_lat'
    # repeat ground-track phase
    i_rgtp_40HZ = np.zeros((n_40HZ), dtype=int)
    i_rgtp_40HZ[:] = int(RGTP)
    # seconds since 2000-01-01 12:00:00 UTC (J2000)
    key = 'DS_UTCTime_40'
    DS_UTCTime_40HZ = fileID['Data_40HZ'][key][:].copy()
    for att_name,att_val in fileID['Data_40HZ'][key].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME','coordinates'):
            attributes[key][att_name] = copy.copy(att_val)
    attributes[key]['coordinates'] = 'd_lon d_lat'
    # Latitude (TOPEX/Poseidon ellipsoid degrees North)
    key = 'd_lat'
    lat_TPX = fileID['Data_40HZ']['Geolocation'][key][:].copy()
    for att_name,att_val in fileID['Data_40HZ']['Geolocation'][key].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME','coordinates'):
            attributes[key][att_name] = copy.copy(att_val)
    # Longitude (degrees East)
    key = 'd_lon'
    lon_TPX = fileID['Data_40HZ']['Geolocation'][key][:].copy()
    for att_name,att_val in fileID['Data_40HZ']['Geolocation'][key].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME','coordinates'):
            attributes[key][att_name] = copy.copy(att_val)
    # ICESat track number
    i_track_1HZ = fileID['Data_1HZ']['Geolocation']['i_track'][:].copy()
    i_track_40HZ = np.zeros((n_40HZ), dtype=i_track_1HZ.dtype)
    # map 1HZ data to 40HZ data
    for k,record in enumerate(rec_ndx_1HZ):
        # indice mapping the 40HZ data to the 1HZ data
        map_1HZ_40HZ, = np.nonzero(rec_ndx_40HZ == record)
        i_track_40HZ[map_1HZ_40HZ] = i_track_1HZ[k]
    # unique tracks
    tracks = np.unique(i_track_1HZ)

    # open output index file
    f2 = h5py.File(output_file, mode='w')
    f2.attrs['featureType'] = 'trajectory'
    f2.attrs['time_type'] = 'UTC'
    today = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    f2.attrs['date_created'] = today
    f2.attrs['campaign'] = campaign
    # add software information
    f2.attrs['software_reference'] = gz.version.project_name
    f2.attrs['software_version'] = gz.version.full_version

    # for each unique ground track
    for i, rgt in enumerate(tracks):
        # create group
        track_group = f'{rgt:04.0f}'
        if track_group not in f2:
            g2 = f2.create_group(track_group)
        else:
            g2 = f2[track_group]

        # create merged track file if not existing
        track_file = DIRECTORY.joinpath(f'{track_group}.h5')
        clobber = 'a' if track_file.exists() else 'w'
        # open output merged track file
        f3 = gz.io.multiprocess_h5py(track_file, mode=clobber)
        # create file group
        if input_file.name not in f3:
            g3 = f3.create_group(input_file.name)
        else:
            g3 = f3[input_file.name]
        # add file-level variables and attributes
        if (clobber == 'w'):
            # add file attributes
            f3.attrs['featureType'] = 'trajectory'
            f3.attrs['time_type'] = 'UTC'
            f3.attrs['date_created'] = today
            # add software information
            f3.attrs['software_reference'] = gz.version.project_name
            f3.attrs['software_version'] = gz.version.full_version

        # indices of points in the track
        indices, = np.nonzero(i_track_40HZ == rgt)
        # output variables for index file
        output = collections.OrderedDict()
        output['DS_UTCTime_40'] = DS_UTCTime_40HZ[indices].copy()
        output['i_rec_ndx'] = rec_ndx_40HZ[indices].copy()
        output['i_track'] = i_track_40HZ[indices].copy()
        output['i_rgtp'] = i_rgtp_40HZ[indices].copy()
        output['index'] = indices.copy()
        output['d_lon'] = lon_TPX[indices].copy()
        output['d_lat'] = lat_TPX[indices].copy()
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
        # close the merged track file
        f3.close()
        # change the permissions mode of the merged track file
        track_file.chmod(mode=MODE)

    # Output HDF5 structure information
    logging.info(list(f2.keys()))
    # close the output file
    f2.close()
    # change the permissions mode of the output file
    output_file.chmod(mode=MODE)

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Creates index files of ICESat/GLAS L2 GLA12
            Antarctic and Greenland Ice Sheet tracks
            """
    )
    # command line parameters
    # input ICESat GLAS files
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='ICESat GLA12 file to run')
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
        track_ICESat_GLA12(FILE,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
