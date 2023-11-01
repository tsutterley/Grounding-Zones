#!/usr/bin/env python
"""
mosaic_tide_adjustment.py
Written by Tyler Sutterley (11/2023)

Creates a mosaic of interpolated tidal adjustment scale factors

COMMAND LINE OPTIONS:
    --help: list the command line options
    -d X, --directory X: directory to run
    -H X, --hemisphere X: Region of interest to run
    -r X, --range X: valid range of tiles to read [xmin,xmax,ymin,ymax]
    -c X, --crop X: crop mosaic to bounds [xmin,xmax,ymin,ymax]
    -m X, --mask X: geotiff mask file for valid points
    -T X, --tide X: Tide model used in correction
    -O X, --output-file X: output filename
    -V, --verbose: verbose output of run
    -M X, --mode X: Local permissions mode of the output mosaic

UPDATE HISTORY:
    Updated 11/2023: mask individual tiles before building mosaic
    Updated 10/2023: use grounding zone mosaic and raster utilities
    Updated 05/2023: using pathlib to define and operate on paths
    Updated 12/2022: single implicit import of grounding zone tools
    Updated 07/2022: place some imports within try/except statements
    Updated 06/2022: use argparse descriptions within documentation
    Updated 07/2021: added option for cropping output mosaic
    Updated 03/2020: made output filename a command line option
    Written 03/2020
"""
import re
import logging
import pathlib
import argparse
import warnings
import numpy as np
import grounding_zones as gz

# attempt imports
try:
    import h5py
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.warn("h5py not available", ImportWarning)
try:
    import pyproj
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.warn("pyproj not available", ImportWarning)
# filter warnings
warnings.filterwarnings("ignore")

# PURPOSE: mosaic interpolated tiles to a complete grid
def mosaic_tide_adjustment(base_dir, output_file,
    HEM=None,
    RANGE=None,
    CROP=None,
    MASK=None,
    TIDE_MODEL=None,
    MODE=0o775):

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

    # get bounds, grid spacing and dimensions of output mosaic
    mosaic = gz.mosaic()
    for tile in sorted(valid_file_list):
        # read tile grid from HDF5
        try:
            with h5py.File(tile) as fileID:
                x = fileID['geophysical']['x'][:]
                y = fileID['geophysical']['y'][:]
        except (KeyError, ValueError) as exc:
            # drop invalid files
            valid_file_list.remove(tile)
        else:
            # update the mosaic attributes
            mosaic.update_spacing(x, y)
            mosaic.update_bounds(x, y)
    # grid dimensions
    ny, nx = mosaic.dimensions
    logging.info(f'Grid Dimensions {ny:d} {nx:d}')

    # update mask for grounded ice values
    if MASK is not None:
        # read mask from geotiff file
        # flip to be monotonically increasing in y dimension
        MASK = pathlib.Path(MASK).expanduser().absolute()
        raster = gz.io.raster().from_file(MASK, format='geotiff').flip()

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
    output['Polar_Stereographic'] = np.empty((), dtype=np.byte)
    # use centered coordinates
    output['x'] = mosaic.x
    output['y'] = mosaic.y
    output['tide_adj_scale'] = np.zeros(((ny,nx)))
    output['weight'] = np.zeros(((ny,nx)))

    # attributes for each output item
    attributes = dict(x={}, y={}, tide_adj_scale={}, weight={})
    fill_value = {}
    # projection attributes
    attributes['Polar_Stereographic'] = {}
    fill_value['Polar_Stereographic'] = None
    # add projection attributes
    attributes['Polar_Stereographic']['standard_name'] = 'Polar_Stereographic'
    attributes['Polar_Stereographic']['spatial_epsg'] = crs.to_epsg()
    attributes['Polar_Stereographic']['spatial_ref'] = crs.to_wkt()
    attributes['Polar_Stereographic']['proj4_params'] = crs.to_proj4()
    attributes['Polar_Stereographic']['latitude_of_projection_origin'] = \
        crs_to_dict['lat_0']
    for att_name,att_val in crs_to_cf.items():
        attributes['Polar_Stereographic'][att_name] = att_val
    # x and y
    attributes['x'],attributes['y'] = ({},{})
    fill_value['x'],fill_value['y'] = (None,None)
    for att_name in ['long_name', 'standard_name', 'units']:
        attributes['x'][att_name] = cs_to_cf[0][att_name]
        attributes['y'][att_name] = cs_to_cf[1][att_name]
    # tide_adj_scale
    attributes['tide_adj_scale']['description'] = ('Scale factor for adjusting '
        'tidal amplitudes to account for ice flexure')
    attributes['tide_adj_scale']['long_name'] = 'Tide Scale Factor'
    attributes['tide_adj_scale']['units'] = '1'
    attributes['tide_adj_scale']['coordinates'] = 'y x'
    attributes['tide_adj_scale']['source'] = 'ATL11'
    attributes['tide_adj_scale']['model'] = TIDE_MODEL
    attributes['tide_adj_scale']['grid_mapping'] = 'Polar_Stereographic'
    fill_value['tide_adj_scale'] = 0
    # weight
    attributes['weight']['long_name'] = 'Tile weight'
    attributes['weight']['units'] = '1'
    attributes['weight']['coordinates'] = 'y x'
    attributes['weight']['grid_mapping'] = 'Polar_Stereographic'
    fill_value['weight'] = 0

    # build the output mosaic
    for tile in sorted(valid_file_list):
        # read tile grid from HDF5
        fileID = h5py.File(tile)
        x = fileID['geophysical']['x'][:]
        y = fileID['geophysical']['y'][:]
        tide_adj_scale = fileID['geophysical']['tide_adj_scale'][:]
        weight = fileID['geophysical']['weight'][:]
        # mask tide adjustment scale factor
        if MASK is not None:
            # warp to output grid and mask tide adjustment grid
            gridx, gridy = np.meshgrid(x, y)
            mask = raster.warp(gridx, gridy, order=1)
            ii, jj = np.nonzero(mask.data <= np.finfo(np.float32).eps)
            tide_adj_scale[ii, jj] = 0.0
        # get image coordinates of tile
        iy, ix = mosaic.image_coordinates(x, y)
        # add tile to output mosaic
        output['tide_adj_scale'][iy, ix] = tide_adj_scale[:]
        output['weight'][iy, ix] = weight[:]
        # close the input HDF5 file
        fileID.close()

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
        for key in ['tide_adj_scale', 'weight']:
            output[key] = np.copy(output[key][yslice, xslice])

    # open output HDF5 file
    fileID = h5py.File(output_file, mode='w')
    # for each output variable
    h5 = {}
    for key,val in output.items():
        # create or overwrite HDF5 variables
        logging.info(f'{key}')
        # create HDF5 variables
        if fill_value[key]:
            h5[key] = fileID.create_dataset(key, val.shape, data=val,
                dtype=val.dtype, fillvalue=fill_value[key],
                compression='gzip')
        elif val.shape:
            h5[key] = fileID.create_dataset(key, val.shape, data=val,
                dtype=val.dtype, compression='gzip')
        else:
            h5[key] = fileID.create_dataset(key, val.shape,
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
        description="""Creates a mosaic of interpolated tidal
            adjustment scale factors
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
    # use a mask for valid points
    parser.add_argument('--mask','-m',
        type=pathlib.Path,
        help='geotiff mask file for valid points')
    # tide model to use
    parser.add_argument('--tide','-T',
        metavar='TIDE', type=str, default='CATS2008-v2023',
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
    mosaic_tide_adjustment(args.directory, args.output_file,
        HEM=args.hemisphere,
        RANGE=args.range,
        CROP=args.crop,
        MASK=args.mask,
        TIDE_MODEL=args.tide,
        MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
