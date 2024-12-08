#!/usr/bin/env python
u"""
tile_data_ICESat2_ATL11.py
Written by Tyler Sutterley (12/2024)
Extracts a subset of data for tiles of ICESat-2 ATL11 annual
land ice elevation data

COMMAND LINE OPTIONS:
    -O X, --output-directory X: input/output data directory
    -T X, --tide X: Tide model used in correction
    -R X, --reanalysis X: Reanalysis model for inverse-barometer correction
    -M X, --mode X: Permission mode of directories and files created
    -F X, --format X: output data format
        HDF5 (default)
        parquet
    -V, --verbose: Output information about each created file

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://h5py.org
    pandas: Python Data Analysis Library
        https://pandas.pydata.org/
    PyArrow: Apache Arrow Python bindings
        https://arrow.apache.org/docs/python/

PROGRAM DEPENDENCIES:
    io/ATL11.py: reads ICESat-2 annual land ice height data files

UPDATE HISTORY:
    Written 12/2024
"""
from __future__ import print_function

import sys
import os
import re
import logging
import pathlib
import argparse
import traceback
import numpy as np
import grounding_zones as gz

# attempt imports
h5py = gz.utilities.import_dependency('h5py')
is2tk = gz.utilities.import_dependency('icesat2_toolkit')
pyTMD = gz.utilities.import_dependency('pyTMD')

# PURPOSE: keep track of threads
def info(args):
    logging.debug(pathlib.Path(sys.argv[0]).name)
    logging.debug(args)
    logging.debug(f'module name: {__name__}')
    if hasattr(os, 'getppid'):
        logging.debug(f'parent process: {os.getppid():d}')
    logging.debug(f'process id: {os.getpid():d}')

# PURPOSE: extract ICESat-2 ATL11 data for tiles
def tile_data_ICESat2_ATL11(tile_file,
        OUTPUT_DIRECTORY=None,
        TIDE_MODEL=None,
        REANALYSIS=None,
        THRESHOLD=(None,None),
        MAX_ERROR=None,
        FORMAT='HDF5',
        MODE=0o775
    ):

    # input tile data file
    tile_file = pathlib.Path(tile_file).expanduser().absolute()
    # regular expression pattern for tile files
    R1 = re.compile(r'E([-+]?\d+)_N([-+]?\d+)', re.VERBOSE)
    # regular expression pattern for ICESat-2 ATL11 files
    R2 = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})_(\d{2})(\d{2})_'
        r'(\d{3})_(\d{2})(.*?).h5$', re.VERBOSE)
    # directory with ATL11 data
    if OUTPUT_DIRECTORY is None:
        OUTPUT_DIRECTORY = tile_file.parent
    # create output directory if non-existent
    if not OUTPUT_DIRECTORY.exists():
        OUTPUT_DIRECTORY.mkdir(mode=MODE, parents=True, exist_ok=True)
    # file format for mask and tide prediction files
    file_format = '{0}_{1}{2}_{3}{4}_{5}{6}_{7}_{8}{9}.h5'
    # extract tile centers from filename
    tile_centers = R1.findall(tile_file.name).pop()
    xc, yc = 1000.0*np.array(tile_centers, dtype=np.float64)
    logging.info(f'Tile File: {str(tile_file)}')
    suffix = dict(HDF5='h5', parquet='parquet')[FORMAT]
    tile_file_formatted = f'E{xc/1e3:0.0f}_N{yc/1e3:0.0f}.{suffix}'
    logging.info(f'Tile Center: {xc:0.1f} {yc:0.1f}')

    # total height count for all cycles
    npts = 0
    # read the HDF5 file
    f1 = gz.io.multiprocess_h5py(tile_file, mode='r')
    d1 = tile_file.parents[1]
    # find ATL11 files within tile
    ATL11_files = [f for f in f1.keys() if R2.match(f)]
    # read each ATL11 group
    for ATL11 in ATL11_files:
        # for each ATL11 beam pairs within the tile
        for ptx, subgroup in f1[ATL11].items():
            indices = subgroup['index'][:].copy()
            ncycles = len(subgroup['cycle_number'])
            npts += ncycles*len(indices)
    # close the tile file
    f1.close()
    # log total number of points
    logging.info(f'Total Points: {npts:d}')

    # root group attributes
    attributes = dict(ROOT={})
    attributes['ROOT']['x_center'] = xc
    attributes['ROOT']['y_center'] = xc
    if TIDE_MODEL:
        attributes['ROOT']['tide_model'] = TIDE_MODEL
    if REANALYSIS:
        attributes['ROOT']['ib_model'] = REANALYSIS
    attributes['threshold'] = THRESHOLD
    attributes['max_error'] = MAX_ERROR

    # allocate for combined variables
    d = {}
    d['longitude'] = np.zeros((npts), dtype=np.float64)
    d['latitude'] = np.zeros((npts), dtype=np.float64)
    d['x_atc'] = np.zeros((npts), dtype=np.float64)
    d['ref_pt'] = np.zeros((npts), dtype=np.int64)
    d['delta_time'] = np.zeros((npts), dtype=np.float64)
    # save heights relative to geoid
    d['h_corr'] = np.zeros((npts), dtype=np.float64)
    d['h_sigma'] = np.zeros((npts), dtype=np.float64)
    d['tide_ocean'] = np.zeros((npts), dtype=np.float64)
    d['ib'] = np.zeros((npts), dtype=np.float64)
    d['geoid_h'] = np.zeros((npts), dtype=np.float64)
    # static variables
    d['cycle'] = np.zeros((npts), dtype=np.int64)
    d['pair'] = np.zeros((npts), dtype=np.int64)
    d['rgt'] = np.zeros((npts), dtype=np.int64)
    # mask for reducing files
    mask = np.zeros((npts), dtype=bool)

    # counter for filling arrays
    c = 0
    # read the HDF5 tile file
    logging.info(f'Reading File: {str(tile_file)}')
    f1 = gz.io.multiprocess_h5py(tile_file, mode='r')
    # read each ATL11 group
    for ATL11 in ATL11_files:
        # full path to data file
        FILE2 = d1.joinpath(ATL11)
        f2 = gz.io.multiprocess_h5py(FILE2, mode='r')
        SUB,PRD,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX = R2.findall(ATL11).pop()
        reference_ground_track = int(TRK)
        # read tide model corrections
        if TIDE_MODEL:
            # read ATL11 tide correction HDF5 file
            a3 = (PRD,TIDE_MODEL,'_TIDES',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
            FILE3 = d1.joinpath(file_format.format(*a3))
            f3 = gz.io.multiprocess_h5py(FILE3, mode='r')
        # read inverse barometer correction
        if REANALYSIS:
            # read ATL11 inverse barometer HDF5 file
            a4 = (PRD,REANALYSIS,'IB',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
            FILE4 = d1.joinpath(file_format.format(*a4))
            f4 = gz.io.multiprocess_h5py(FILE4, mode='r')

        # for each beam pair within the tile
        # extract number of points from subgroup
        # assuming that all data is valid
        for ptx, subgroup in f1[ATL11].items():
            # indices within tile
            indices = subgroup['index'][:].copy()
            file_length = len(indices)
            # read dataset
            mds, attrs = is2tk.io.ATL11.read_pair(f2, ptx,
                ATTRIBUTES=True, REFERENCE=True, KEEP=True)
            # invalid value for heights
            invalid = attrs[ptx]['h_corr']['_FillValue']
            # pair number
            beam_pair = int(attrs[ptx]['beam_pair'])
            # reduce variables to indices
            delta_time = mds[ptx]['delta_time'][indices,:].copy()
            # combine errors (ignore overflow at invalid points)
            with np.errstate(over='ignore'):
                error = np.sqrt(
                    mds[ptx]['h_corr_sigma'][indices,:]**2 +
                    mds[ptx]['h_corr_sigma_systematic'][indices,:]**2
                )
            # read tide model corrections
            if TIDE_MODEL:
                # read tide data and reduce to indices
                temp = f3[ptx]['cycle_stats']['tide_ocean'][:].copy()
                tide_ocean = temp[indices,:]
            else:
                tide_ocean = np.zeros((file_length,ncycles))
            # read inverse barometer correction
            if REANALYSIS:
                # read IB data and reduce to indices
                temp = f4[ptx]['cycle_stats']['ib'][:].copy()
                IB = temp[indices,:]
            else:
                # reduce DAC to indices
                IB = mds[ptx]['cycle_stats']['dac'][indices,:]

            # copy annual land ice height variables
            for k,cycle_number in enumerate(mds[ptx]['cycle_number']):
                # height variables for cycle
                h = np.ma.array(mds[ptx]['h_corr'][indices,k].copy(),
                    fill_value=invalid)
                # create masks for height variables
                h.mask = (h.data == h.fill_value)
                # quality summary for height variables
                qs1 = mds[ptx]['quality_summary'][indices,k]
                # quasi-freeboard: WGS84 elevation - geoid height
                # reference heights to geoid
                h -= mds[ptx]['ref_surf']['geoid_h'][indices]
                # correct heights for DAC/IB
                h -= IB[:,k]
                # correct heights for ocean tides
                h -= tide_ocean[:,k]
                # save to output dictionary
                d['h_corr'][c:c+file_length] = h.copy()
                d['h_sigma'][c:c+file_length] = error[:,k].copy()
                d['tide_ocean'][c:c+file_length] = tide_ocean[:,k].copy()
                d['ib'][c:c+file_length] = IB[:,k].copy()
                d['delta_time'][c:c+file_length] = delta_time[:,k].copy()
                d['cycle'][c:c+file_length] = cycle_number
                d['pair'][c:c+file_length] = beam_pair
                d['rgt'][c:c+file_length] = reference_ground_track
                # add 1d variables to output dictionary
                for i in ['longitude','latitude','ref_pt']:
                    d[i][c:c+file_length] = mds[ptx][i][indices].copy()
                # add 1d surface variables to output dictionary
                group = 'ref_surf'
                for i in ['x_atc','geoid_h']:
                    d[i][c:c+file_length] = mds[ptx][group][i][indices].copy()
                # mask for reducing to valid values
                mask[c:c+file_length] = \
                    np.logical_not(h.mask) & \
                    (tide_ocean[:,k] != invalid) & \
                    (h > THRESHOLD[0]) & (h <= THRESHOLD[1]) & \
                    (error[:,k] <= MAX_ERROR) & \
                    (qs1 == 0)
                # add to counter
                c += file_length
        # close the input dataset(s)
        f2.close()
        if TIDE_MODEL:
            f3.close()
        if REANALYSIS:
            f4.close()
    # close the tile file
    f1.close()

    # reduce variables to valid values
    valid, = np.nonzero(mask)
    for key,val in d.items():
        d[key] = val[valid].copy()  
    
    # output file
    output_file = OUTPUT_DIRECTORY.joinpath(tile_file_formatted)
    logging.info(output_file)
    if (FORMAT == 'HDF5'):
        # open output HDF5 file in append mode
        fileID = gz.io.multiprocess_h5py(output_file, mode='a')
        # create data group if non-existent
        group = 'data'
        if group not in fileID:
            g1 = fileID.create_group(group)
        else:
            g1 = fileID[group]
        # add root attributes
        for att_name, att_val in attributes['ROOT'].items():
            g1.attrs[att_name] = att_val
        # for each output variable
        h5 = {}
        for key,val in d.items():
            # create or overwrite HDF5 variables
            if key not in fileID[group]:
                # create HDF5 variables
                if val.shape:
                    h5[key] = g1.create_dataset(key, val.shape, data=val,
                        dtype=val.dtype, compression='gzip')
                else:
                    h5[key] = g1.create_dataset(key, val.shape,
                        dtype=val.dtype)
            else:
                # overwrite HDF5 variables
                g1[key][...] = val.copy()
        # close the output file
        logging.info(fileID[group].keys())
        fileID.close()
    elif (FORMAT == 'parquet'):
        # write to parquet
        pyTMD.spatial.to_parquet(d, attributes, output_file,
            geoparquet=False, crs=4326)

    # change the permissions mode
    output_file.chmod(mode=MODE)

# PURPOSE: create arguments parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Extracts a subset of data for tiles of
            ICESat-2 ATL11 annual land ice elevation data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    group = parser.add_mutually_exclusive_group(required=False)
    # command line parameters
    parser.add_argument('infile',
        type=pathlib.Path,
        help='Tile file to run')
    # directory with input/output data
    parser.add_argument('--output-directory','-O',
        type=pathlib.Path,
        help='Output data directory')
    # tide model to use
    group.add_argument('--tide','-T',
        metavar='TIDE', type=str,
        help='Tide model to use in correction')
    # inverse barometer response to use
    parser.add_argument('--reanalysis','-R',
        metavar='REANALYSIS', type=str,
        help='Reanalysis model to use in inverse-barometer correction')
    # height threshold (filter points to range)
    parser.add_argument('--threshold','-t',
        nargs=2, type=float, default=(0, 150),
        help='Min/max height thresholds for reducing points')
    # maximum valid error for height change
    parser.add_argument('--max-error','-e',
        type=float, default=1.0,
        help='Maximum error for reducing points')    
    # output data format
    parser.add_argument('--format','-F',
        type=str, default='HDF5',
        choices=('HDF5','parquet'),
        help='Output data format')
    # verbose output of processing run
    # print information about processing run
    parser.add_argument('--verbose','-V',
        action='count', default=0,
        help='Verbose output of processing run')
    # permissions mode of the directories and files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Local permissions mode of the output file')
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

    # run program
    try:
        info(args)
        tile_data_ICESat2_ATL11(args.infile,
            OUTPUT_DIRECTORY=args.output_directory,
            TIDE_MODEL=args.tide,
            REANALYSIS=args.reanalysis,
            THRESHOLD=args.threshold,
            MAX_ERROR=args.max_error,
            FORMAT=args.format,
            MODE=args.mode)
    except Exception as exc:
        # if there has been an error exception
        # print the type, value, and stack trace of the
        # current exception being handled
        logging.critical(f'process id {os.getpid():d} failed')
        logging.error(traceback.format_exc())

# run main program
if __name__ == '__main__':
    main()
