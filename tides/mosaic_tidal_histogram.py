#!/usr/bin/env python
"""
mosaic_tidal_histogram.py
Written by Tyler Sutterley (10/2025)

Creates a mosaic of tidal histograms

COMMAND LINE OPTIONS:
    --help: list the command line options
    -d X, --directory X: directory to run
    -H X, --hemisphere X: Region of interest to run
    -r X, --range X: valid range of tiles to read [xmin,xmax,ymin,ymax]
    -c X, --crop X: crop mosaic to bounds [xmin,xmax,ymin,ymax]
    -T X, --tide X: Tide model used in correction
    -O X, --output-file X: output filename
    -V, --verbose: verbose output of run
    -M X, --mode X: Local permissions mode of the output mosaic

UPDATE HISTORY:
    Written 10/2025
"""
import sys
import os
import re
import logging
import pathlib
import argparse
import pyTMD.io
import traceback
import numpy as np
import grounding_zones as gz

# attempt imports
h5py = gz.utilities.import_dependency('h5py')
pyproj = gz.utilities.import_dependency('pyproj')

# PURPOSE: keep track of threads
def info(args):
    logging.debug(pathlib.Path(sys.argv[0]).name)
    logging.debug(args)
    logging.debug(f'module name: {__name__}')
    if hasattr(os, 'getppid'):
        logging.debug(f'parent process: {os.getppid():d}')
    logging.debug(f'process id: {os.getpid():d}')

# PURPOSE: mosaic interpolated tiles to a complete grid
def mosaic_tidal_histogram(base_dir, output_file,
        HEM=None,
        RANGE=None,
        CROP=None,
        TIDE_MODEL=None,
        DEFINITION_FILE=None,
        MODE=0o775
    ):

    # directory setup
    base_dir = pathlib.Path(base_dir).expanduser().absolute()
    # index directory for hemisphere
    index_directory = 'north' if (HEM == 'N') else 'south'
    tile_directory = base_dir.joinpath(index_directory)
    # regular expression pattern for tile files
    R1 = re.compile(r'E([-+]?\d+)_N([-+]?\d+)', re.VERBOSE)

    # find list of valid files
    initial_file_list = [f for f in tile_directory.iterdir() if R1.match(f.name)]
    logging.info(f"Found {len(initial_file_list)} files")

    # valid range of tiles
    xmin, xmax, ymin, ymax = np.array(RANGE)
    # reduce file list to those within the valid range
    valid_file_list = []
    for tile in initial_file_list:
        try:
            xc,yc = [int(item)*1e3 for item in R1.search(tile.name).groups()]
        except Exception as exc:
            continue
        # check that tile center is within range
        if ((xc >= xmin) and (xc <= xmax) & (yc >= ymin) and (yc <= ymax)):
            valid_file_list.append(tile)
    logging.info(f"Found {len(valid_file_list)} files within range")

    # get tide model parameters from definition file or model name
    if DEFINITION_FILE is not None:
        model = pyTMD.io.model(None, verify=False).from_file(
            DEFINITION_FILE)
    elif TIDE_MODEL is not None:
        model = pyTMD.io.model(None, verify=False).from_database(TIDE_MODEL)
    else:
        # default for uncorrected heights
        model = type('model', (), dict(name=None, corrections='GOT'))
    # tide model group for different corrections
    group = model.name if model.name else 'uncorrected'

    # get bounds, grid spacing and dimensions of output mosaic
    mosaic = gz.mosaic()
    for tile in sorted(valid_file_list):
        # read tile grid from HDF5
        try:
            with h5py.File(tile) as fileID:
                x = fileID[group]['x'][:]
                y = fileID[group]['y'][:]
                bins = fileID[group]['bins'][:]
                invalid = fileID[group]['dh_hist'].fillvalue
        except (KeyError, ValueError) as exc:
            # drop invalid files
            valid_file_list.remove(tile)
        else:
            # update the mosaic attributes
            mosaic.update_spacing(x, y)
            mosaic.update_bounds(x, y)
    # grid dimensions
    ny, nx = mosaic.dimensions
    nbins = len(bins)
    logging.info(f'Grid Dimensions {ny:d} {nx:d} {nbins:d}')
    logging.info(f'Grid Spacing {mosaic.spacing[0]} {mosaic.spacing[1]}')

    # pyproj transformer for converting to polar stereographic
    EPSG = dict(N=3413, S=3031)[HEM]
    crs = pyproj.CRS.from_epsg(EPSG)
    # dictionary of coordinate reference system variables
    cs_to_cf = crs.cs_to_cf()
    crs_to_dict = crs.to_dict()
    # flattening and standard parallel of datum and projection
    crs_to_cf = crs.to_cf()

    # allocate for output variables
    output = {}
    # projection variable
    output['crs'] = np.empty((), dtype=np.byte)
    # use centered coordinates
    output['x'] = mosaic.x
    output['y'] = mosaic.y
    output['bins'] = bins.copy()
    # cell area (accounting for polar stereographic distortion)
    output['cell_area'] = np.zeros((ny, nx))
    # histogram of height differences
    output['dh_hist'] = np.ma.zeros((ny, nx, nbins), fill_value=invalid)
    # data count
    output['count'] = np.zeros((ny, nx), dtype=np.int64)

    # attributes for each output item
    attributes = dict(ROOT={}, x={}, y={})
    fill_value = {}
    # root group attributes
    attributes['ROOT']['x_center'] = xc
    attributes['ROOT']['y_center'] = yc
    attributes['ROOT']['spacing'] = mosaic.spacing
    # projection attributes
    attributes['crs'] = {}
    fill_value['crs'] = None
    # add projection attributes
    attributes['crs']['standard_name'] = \
        crs_to_cf['grid_mapping_name'].title()
    attributes['crs']['spatial_epsg'] = crs.to_epsg()
    attributes['crs']['spatial_ref'] = crs.to_wkt()
    attributes['crs']['proj4_params'] = crs.to_proj4()
    attributes['crs']['latitude_of_projection_origin'] = \
        crs_to_dict['lat_0']
    for att_name,att_val in crs_to_cf.items():
        attributes['crs'][att_name] = att_val
    # x and y
    attributes['x'],attributes['y'] = ({},{})
    fill_value['x'],fill_value['y'] = (None,None)
    for att_name in ['long_name', 'standard_name', 'units']:
        attributes['x'][att_name] = cs_to_cf[0][att_name]
        attributes['y'][att_name] = cs_to_cf[1][att_name]
    # histogram bin
    attributes['bins'] = {}
    attributes['bins']['long_name'] = 'Histogram bins'
    attributes['bins']['units'] = '1'
    attributes['bins']['description'] = \
        'Center of each height difference histogram bin'
    fill_value['bins'] = None
    # ice area
    attributes['cell_area'] = {}
    attributes['cell_area']['long_name'] = 'Cell area'
    attributes['cell_area']['description'] = ('Area of each grid cell, '
        'accounting for polar stereographic distortion')
    attributes['cell_area']['units'] = 'm^2'
    attributes['cell_area']['coordinates'] = 'y x'
    attributes['cell_area']['grid_mapping'] = 'crs'
    fill_value['cell_area'] = 0
    # height difference histogram
    attributes['dh_hist'] = {}
    attributes['dh_hist']['long_name'] = 'Height difference histogram'
    attributes['dh_hist']['description'] = 'Histogram of height differences'
    attributes['dh_hist']['units'] = 'meters'
    attributes['dh_hist']['coordinates'] = 'y x bins'
    attributes['dh_hist']['grid_mapping'] = 'crs'
    fill_value['dh_hist'] = invalid
    # mean height difference
    attributes['dh_mean'] = {}
    attributes['dh_mean']['long_name'] = 'Mean height difference'
    attributes['dh_mean']['description'] = \
        'Mean of height difference histogram'
    attributes['dh_mean']['units'] = 'meters'
    attributes['dh_mean']['coordinates'] = 'y x'
    attributes['dh_mean']['grid_mapping'] = 'crs'
    fill_value['dh_mean'] = invalid
    # standard deviation of height differences
    attributes['dh_stdev'] = {}
    attributes['dh_stdev']['long_name'] = \
        'Standard deviation of height differences'
    attributes['dh_stdev']['description'] = \
        'Standard deviation of height difference histogram'
    attributes['dh_stdev']['units'] = 'meters'
    attributes['dh_stdev']['coordinates'] = 'y x'
    attributes['dh_stdev']['grid_mapping'] = 'crs'
    fill_value['dh_stdev'] = invalid
    # data count
    attributes['count'] = {}
    attributes['count']['long_name'] = 'Number of data points'
    attributes['count']['units'] = '1'
    attributes['count']['coordinates'] = 'y x'
    attributes['count']['grid_mapping'] = 'crs'
    fill_value['count'] = 0

    # build the output mosaic
    for tile in sorted(valid_file_list):
        # read tile grid from HDF5
        fileID = h5py.File(tile)
        x = fileID[group]['x'][:]
        y = fileID[group]['y'][:]
        cell_area = fileID[group]['cell_area'][:]
        dh_hist = fileID[group]['dh_hist'][:]
        count = fileID[group]['count'][:]
        # get image coordinates of tile
        iy, ix = mosaic.image_coordinates(x, y)
        # add tile to output mosaics
        output['cell_area'][iy, ix] = cell_area[:]
        output['dh_hist'][iy, ix, :] = dh_hist[:]
        output['count'][iy, ix] = count[:]
        # close the input HDF5 file
        fileID.close()

    # replace masked values with fill value
    output['dh_hist'].mask = (output['dh_hist'].data == invalid)
    # find valid points
    valid = (output['count'] > 0)
    # compute mean and standard deviation of height differences
    ii, jj = np.nonzero(valid)
    # histogram mean
    b2 = np.broadcast_to(bins, (ny, nx, nbins))
    output['dh_mean'] = np.ma.zeros((ny, nx), fill_value=invalid)
    output['dh_mean'][ii,jj] = np.average(b2[ii,jj,:], axis=1,
        weights=output['dh_hist'][ii,jj,:])
    output['dh_mean'].mask = np.logical_not(valid)
    # standard deviation of histogram    
    hmean = np.broadcast_to(output['dh_mean'][:,:,None], (ny, nx, nbins))
    hvariance = np.average((b2[ii,jj,:] - hmean[ii,jj,:])**2, axis=1,
        weights=output['dh_hist'][ii,jj,:])
    output['dh_stdev'] = np.ma.zeros((ny, nx), fill_value=invalid)
    output['dh_stdev'][ii,jj] = np.sqrt(hvariance)
    output['dh_stdev'].mask = np.logical_not(valid)

    # crop mosaic to bounds
    if np.any(CROP):
        # column and row indices
        xind, = np.nonzero((mosaic.x >= CROP[0]) & (mosaic.x <= CROP[1]))
        xslice = slice(xind[0], xind[-1], 1)
        yind, = np.nonzero((mosaic.y >= CROP[2]) & (mosaic.y <= CROP[3]))
        yslice = slice(yind[0], yind[-1], 1)
        # crop the output variables to range
        output['x'] = np.copy(output['x'][xslice])
        output['y'] = np.copy(output['y'][yslice])
        # crop the 2D and 3D variables
        for key in ['cell_area', 'count', 'dh_mean', 'dh_stdev']:
            output[key] = np.copy(output[key][yslice, xslice])
        for key in ['dh_hist']:
            output[key] = np.copy(output[key][yslice, xslice, :])

    # open output HDF5 file
    fileID = h5py.File(output_file, mode='a')
    # create tide model group if non-existent
    if group not in fileID:
        g1 = fileID.create_group(group)
    else:
        g1 = fileID[group]
    # add root attributes
    for att_name, att_val in attributes['ROOT'].items():
        g1.attrs[att_name] = att_val
    # for each output variable
    h5 = {}
    for key,val in output.items():
        # create or overwrite HDF5 variables
        logging.info(f'{key}')
        # create HDF5 variables
        if fill_value[key]:
            h5[key] = g1.create_dataset(key, val.shape, data=val,
                dtype=val.dtype, fillvalue=fill_value[key],
                compression='gzip')
        elif val.shape:
            h5[key] = g1.create_dataset(key, val.shape, data=val,
                dtype=val.dtype, compression='gzip')
        else:
            h5[key] = g1.create_dataset(key, val.shape,
                dtype=val.dtype)
        # add variable attributes
        for att_name,att_val in attributes[key].items():
            h5[key].attrs[att_name] = att_val
    # close the output file
    fileID.close()
    # change the permissions mode
    output_file.chmod(mode=MODE)

# PURPOSE: create arguments parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Creates a mosaic of tidal histograms
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    parser.add_argument('--directory','-d',
        type=pathlib.Path,
        help='directory to run')
    # region of interest to run
    parser.add_argument('--hemisphere','-H',
        type=str, default='S', choices=('N','S'),
        help='Region of interest to run')
    # input range of tiles to read
    parser.add_argument('--range','-r', type=float,
        nargs=4, default=[-np.inf,np.inf,-np.inf,np.inf],
        metavar=('xmin','xmax','ymin','ymax'),
        help='valid range of tiles to read')
    # bounds of output mosaic
    parser.add_argument('--crop','-c', type=float,
        nargs=4, default=[None, None, None, None],
        metavar=('xmin','xmax','ymin','ymax'),
        help='Crop mosaic to bounds')
    # tide model to use
    parser.add_argument('--tide','-T',
        metavar='TIDE', type=str,
        help='Tide model used in correction')
    # output filename
    parser.add_argument('--output-file','-O',
        type=pathlib.Path,
        help='Output filename')
    # verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of run')
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
    loglevel = logging.INFO if args.verbose else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # run tide mosaic program
    try:
        info(args)
        mosaic_tidal_histogram(args.directory, args.output_file,
            HEM=args.hemisphere,
            RANGE=args.range,
            CROP=args.crop,
            TIDE_MODEL=args.tide,
            MODE=args.mode
        )
    except Exception as exc:
        # if there has been an error exception
        # print the type, value, and stack trace of the
        # current exception being handled
        logging.critical(f'process id {os.getpid():d} failed')
        logging.error(traceback.format_exc())

# run main program
if __name__ == '__main__':
    main()
