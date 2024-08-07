#!/usr/bin/env python
u"""
interpolate_tide_adjustment.py
Written by Tyler Sutterley (05/2024)
Interpolates tidal adjustment scale factors to output grids

COMMAND LINE OPTIONS:
    --help: list the command line options
    -O X, --output-directory X: input/output data directory
    -H X, --hemisphere X: Region of interest to run
    -W X, --width: Width of tile grid
    -s X, --subset: Width of interpolation subset
    -S X, --spacing X: Output grid spacing
    -P X, --pad X: Tile pad for creating mosaics
    -T X, --tide X: Tide model used in correction
    -I X, --interpolate X: Interpolation method
    -t X, --tension X: Biharmonic spline tension
    -w X, --smooth X: Radial basis function smoothing weight
    -e X, --epsilon X: Radial basis function adjustable constant
    -p X, --polynomial X: Polynomial order for radial basis functions
    -V, --verbose: Verbose output of run
    -m X, --mode X: Local permissions mode of the output mosaic

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/

UPDATE HISTORY:
    Updated 05/2024: use wrapper to importlib for optional dependencies
        moved multiprocess h5py reader to io utilities module
    Updated 12/2023: don't have a default tide model in arguments
    Updated 11/2023: only mask out invalid points within fit domain
    Updated 10/2023: mask out invalid tide adjustment points before fit
    Updated 08/2023: can set the output directory to be separate
    Updated 05/2023: using pathlib to define and operate on paths
    Updated 12/2022: check that file exists within multiprocess HDF5 function
        single implicit import of grounding zone tools
    Updated 07/2022: place some imports within try/except statements
    Updated 06/2022: use argparse descriptions within documentation
    Updated 01/2022: added options for using radial basis functions
        wait if HDF5 tile file is unavailable for read or write
    Written 12/2021
"""

import re
import time
import logging
import pathlib
import argparse
import numpy as np
import grounding_zones as gz

# attempt imports
h5py = gz.utilities.import_dependency('h5py')
pyproj = gz.utilities.import_dependency('pyproj')
spi = gz.utilities.import_dependency('spatial_interpolators')

# PURPOSE: reduce a matrix using a selected function
def reduce(val, method=np.min, axis=1):
    return method(val, axis=axis)

def interpolate_tide_adjustment(tile_file,
        OUTPUT_DIRECTORY=None,
        HEM='S',
        W=80e3,
        SUBSET=10e3,
        SPACING=None,
        PAD=0,
        TIDE_MODEL=None,
        METHOD=None,
        TENSION=0,
        SMOOTH=0,
        EPSILON=0,
        POLYNOMIAL=0,
        MODE=0o775
    ):

    # input tile data file
    tile_file = pathlib.Path(tile_file).expanduser().absolute()
    tile_file_format = 'E{0:0.0f}_N{1:0.0f}.h5'
    # regular expression pattern for tile files
    R1 = re.compile(r'E([-+]?\d+)_N([-+]?\d+)', re.VERBOSE)
    # regular expression pattern for ICESat-2 ATL11 files
    R2 = re.compile(r'(ATL\d{2})_(\d{4})(\d{2})_(\d{2})(\d{2})_'
        r'(\d{3})_(\d{2})(.*?).h5$', re.VERBOSE)
    # directory with ATL11 data
    if OUTPUT_DIRECTORY is None:
        OUTPUT_DIRECTORY = tile_file.parents[1]
    # file format for mask and tide fit files
    file_format = '{0}_{1}{2}_{3}{4}_{5}{6}_{7}_{8}{9}.h5'
    # extract tile centers from filename
    tile_centers = R1.findall(tile_file.name).pop()
    xc, yc = 1000.0*np.array(tile_centers, dtype=np.float64)
    logging.info(f'Tile File: {str(tile_file)}')

    # grid dimensions
    xmin,xmax = xc + np.array([-0.5,0.5])*W
    ymin,ymax = yc + np.array([-0.5,0.5])*W
    dx,dy = np.broadcast_to(np.atleast_1d(SPACING),(2,))
    nx = np.int64(W//dx) + 1
    ny = np.int64(W//dy) + 1
    # minimum number of points to run interpolation for a tile
    point_threshold = 3

    # pyproj transformer for converting to polar stereographic
    EPSG = dict(N=3413, S=3031)[HEM]
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_epsg(EPSG)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    # dictionary of coordinate reference system variables
    cs_to_cf = crs2.cs_to_cf()
    crs_to_dict = crs2.to_dict()
    # flattening and standard parallel of datum and projection
    crs_to_cf = crs2.to_cf()

    # total point count
    npts = 0
    # indices of buffers
    d_i,d_j = np.meshgrid([-1., 0, 1.], [-1., 0., 1.])
    # iterate over buffer indices
    for ii,jj in zip(d_i.ravel(), d_j.ravel()):
        # tile file for buffer
        tile = tile_file_format.format((xc+W*ii)/1e3, (yc+W*jj)/1e3)
        buffer_file = tile_file.with_name(tile)
        if not buffer_file.exists():
            continue
        # read the HDF5 file
        logging.info(f'Reading Buffer File: {str(buffer_file)}')
        f1 = gz.io.multiprocess_h5py(buffer_file, mode='r')
        # find ATL11 files within tile
        ATL11_files = [f for f in f1.keys() if R2.match(f)]
        # read each ATL11 group
        for ATL11 in ATL11_files:
            # for each ATL11 beam pairs within the tile
            for ptx in f1[ATL11].keys():
                indices = f1[ATL11][ptx]['index'][:].copy()
                npts += len(indices)
        # close the tile file
        f1.close()
    # log total number of points
    logging.info(f'Total Points: {npts:d}')

    # allocate for combined variables
    d = {}
    d['ref_pt'] = np.zeros((npts), dtype=np.int64)
    d['rgt'] = np.zeros((npts), dtype=np.int16)
    d['pair'] = np.zeros((npts), dtype=np.int8)
    d['longitude'] = np.zeros((npts), dtype=np.float64)
    d['latitude'] = np.zeros((npts), dtype=np.float64)
    d['tide_adj'] = np.ma.zeros((npts), dtype=np.float64)
    d['tide_adj_sigma'] = np.ma.zeros((npts), dtype=np.float64)
    d['mask'] = np.zeros((npts), dtype=bool)
    d['ice_gz'] = np.ones((npts), dtype=bool)
    Reducer = dict(tide_adj=np.min, tide_adj_sigma=np.max)
    # indices for each pair track
    pair = dict(pt1=1, pt2=2, pt3=3)
    # counter for filling arrays
    c = 0
    # iterate over buffer indices
    for ii,jj in zip(d_i.ravel(), d_j.ravel()):
        # tile file for buffer
        tile = tile_file_format.format((xc+W*ii)/1e3, (yc+W*jj)/1e3)
        buffer_file = tile_file.with_name(tile)
        if not buffer_file.exists():
            continue
        # read the HDF5 file
        logging.info(f'Reading Buffer File: {str(buffer_file)}')
        f1 = gz.io.multiprocess_h5py(buffer_file, mode='r')
        # find ATL11 files within tile
        ATL11_files = [f for f in f1.keys() if R2.match(f)]
        # read each ATL11 file and estimate errors
        for ATL11 in ATL11_files:
            # extract parameters from ATL11 filename
            PRD,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX = R2.findall(ATL11).pop()
            # ATL11 flexure correction HDF5 file
            FILE2 = OUTPUT_DIRECTORY.joinpath(file_format.format(
                PRD,TIDE_MODEL,'_FIT_TIDES',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX))
            # ATL11 raster mask HDF5 file
            FILE3 = OUTPUT_DIRECTORY.joinpath(file_format.format(
                PRD,'MASK','',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX))
            # skip file if not currently accessible
            if not FILE2.exists():
                continue
            # open ATL11 flexure correction HDF5 file
            f2 = gz.io.multiprocess_h5py(FILE2, mode='r')
            # open ATL11 grounded mask HDF5 file
            f3 = gz.io.multiprocess_h5py(FILE3, mode='r')
            # for each ATL11 beam pairs within the tile
            for ptx in f1[ATL11].keys():
                # reference points and indices within tile
                ref_pt = f1[ATL11][ptx]['ref_pt'][:].copy()
                indices = f1[ATL11][ptx]['index'][:].copy()
                file_length = len(indices)
                # reference ground track and pair track
                d['rgt'][c:c+file_length] = int(TRK)
                d['pair'][c:c+file_length] = pair[ptx]
                # extract ATL11 fields for pair
                for k in ['latitude','longitude','ref_pt']:
                    temp = f2[ptx][k][:].copy()
                    # reduce to indices
                    d[k][c:c+file_length] = temp[indices]
                # verify reference points
                assert set(ref_pt) == set(d['ref_pt'][c:c+file_length])
                # try to extract geophysical variables
                for k in ['tide_adj','tide_adj_sigma']:
                    try:
                        temp = f2[ptx]['cycle_stats'][k][:].copy()
                        fv = f2[ptx]['cycle_stats'][k].fillvalue
                    except Exception as exc:
                        pass
                    else:
                        # reduce matrix to indices
                        d[k][c:c+file_length] = reduce(temp[indices,:],
                            method=Reducer[k], axis=1)
                        d[k].fill_value = fv
                # try to extract subsetting variables
                for k in ['ice_gz']:
                    try:
                        temp = f2[ptx]['subsetting'][k][:].copy()
                    except Exception as exc:
                        pass
                    else:
                        # reduce to indices
                        d[k][c:c+file_length] = temp[indices]
                for k in ['mask']:
                    try:
                        temp = f3[ptx]['subsetting'][k][:].copy()
                    except Exception as exc:
                        pass
                    else:
                        # reduce to indices
                        d[k][c:c+file_length] = temp[indices]
                # add to counter
                c += file_length
            # close the ATL11 tidal flexure and mask file
            f2.close()
            f3.close()
        # close the tile file
        f1.close()

    # replace fill values
    for k in ['tide_adj','tide_adj_sigma']:
        d[k].mask = (d[k].data == d[k].fill_value) | np.isnan(d[k].data)
        d[k].data[d[k].mask] = d[k].fill_value

    # make a global reference point number
    # combining ref_pt, rgt and pair
    global_ref_pt = 3*1387*d['ref_pt'] + 3*(d['rgt']-1) + (d['pair']-1)
    _, indices = np.unique(global_ref_pt, return_index=True)
    logging.info(f'Unique Points: {len(indices):d}')
    # reduce to unique indices
    for key,val in d.items():
        d[key] = val[indices]
    # calculate coordinates in polar stereographic
    d['x'], d['y'] = transformer.transform(d['longitude'], d['latitude'])

    # allocate for output variables
    output = {}
    # projection variable
    output['Polar_Stereographic'] = np.empty((), dtype=np.byte)
    # coordinates
    output['x'] = np.arange(xmin+dx/2.0, xmax+dx, dx)
    output['y'] = np.arange(ymin+dx/2.0, ymax+dy, dy)
    output['tide_adj_scale'] = np.zeros((ny,nx))
    output['weight'] = np.zeros((ny,nx))
    logging.info(f'Grid Dimensions {ny:d} {nx:d}')
    # attributes for each output item
    attributes = dict(x={}, y={}, tide_adj_scale={}, weight={})
    fill_value = {}
    # projection attributes
    attributes['Polar_Stereographic'] = {}
    fill_value['Polar_Stereographic'] = None
    # add projection attributes
    attributes['Polar_Stereographic']['standard_name'] = 'Polar_Stereographic'
    attributes['Polar_Stereographic']['spatial_epsg'] = crs2.to_epsg()
    attributes['Polar_Stereographic']['spatial_ref'] = crs2.to_wkt()
    attributes['Polar_Stereographic']['proj4_params'] = crs2.to_proj4()
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

    # calculate mosaic over centers, edges and corners
    mosaic = np.zeros((ny,nx))
    weight = np.zeros((ny,nx))
    # for each subset coordinate
    for xi in np.arange(xmin, xmin + W, SUBSET):
        for yi in np.arange(ymin, ymin + W, SUBSET):
            # iterate over centers, edges and corners
            for ii,jj in zip(d_i.ravel(), d_j.ravel()):
                # minimum x and y for iteration
                xm = xi + SUBSET*ii/2
                ym = yi + SUBSET*jj/2
                # reduce to valid fit points
                ice_gz, = np.nonzero(d['ice_gz'])
                valid_ice_gz = np.ones_like(d['ice_gz'])
                valid_ice_gz[ice_gz] ^= d['tide_adj'].mask[ice_gz]
                # clip unique points to coordinates
                # buffer to improve determination at edges
                clipped = np.nonzero((d['x'] >= xm-0.1*SUBSET) &
                    (d['x'] <= xm+1.1*SUBSET) &
                    (d['y'] >= ym-0.1*SUBSET) &
                    (d['y'] <= ym+1.1*SUBSET) &
                    valid_ice_gz)
                # skip iteration if there are no points
                if not np.any(clipped):
                    continue
                # create clipped data
                u = {}
                for key,val in d.items():
                    u[key] = val[clipped]
                # mask out grounded, rock and lake points
                u['tide_adj'].mask |= np.logical_not(u['mask'])
                tide_adj_scale = u['tide_adj'].filled(np.nan)
                # output coordinates for grid subset
                X = np.arange(xm, xm+SUBSET + dx, dx)
                Y = np.arange(ym, ym+SUBSET + dy, dy)
                # reduce to points within complete tile
                X = X[(X >= xmin) & ((X <= xmin+W))]
                Y = Y[(Y >= ymin) & ((Y <= ymin+W))]
                # grid indices
                iy = np.array((Y[:,None]-ymin)//dy, dtype='i')
                ix = np.array((X[None,:]-xmin)//dx, dtype='i')
                # normalize x and y coordinates
                xnorm = (u['x'] - (xm - 0.1*SUBSET))/(1.2*SUBSET)
                ynorm = (u['y'] - (ym - 0.1*SUBSET))/(1.2*SUBSET)
                # convert from edges to centers and then normalize
                gridx,gridy = np.meshgrid(X + 0.5*dx, Y + 0.5*dy)
                XN = (gridx - (xm - 0.1*SUBSET))/(1.2*SUBSET)
                YN = (gridy - (ym - 0.1*SUBSET))/(1.2*SUBSET)
                # create output grids for interpolation and weights
                interp = np.ones((len(Y),len(X)))
                count = np.ones((len(Y),len(X)))
                # pad the interpolated matrix to remove edges
                if (PAD > 0):
                    xpad = np.array([xm + PAD, xm + SUBSET - PAD])
                    ypad = np.array([ym + PAD, ym + SUBSET - PAD])
                    indy,indx = np.nonzero((gridx < xpad[0]) |
                        (gridx > xpad[1]) |
                        (gridy < ypad[0]) |
                        (gridy > ypad[1]))
                    interp[indy,indx] = 0.0
                    count[indy,indx] = 0.0
                # check if adjustment exists or is uniform
                if np.all(tide_adj_scale == 1):
                    mosaic[iy,ix] += interp.copy()
                    weight[iy,ix] += count.copy()
                    continue
                elif np.all(tide_adj_scale == 0):
                    weight[iy,ix] += count.copy()
                    continue
                elif np.all(np.isnan(tide_adj_scale)):
                    weight[iy,ix] += count.copy()
                    continue
                elif np.any(np.isnan(tide_adj_scale)):
                    # replace invalid points
                    tide_adj_scale = np.nan_to_num(tide_adj_scale, nan=0.0)
                elif (len(np.atleast_1d(tide_adj_scale)) <= point_threshold):
                    weight[iy,ix] += count.copy()
                    continue
                # interpolate sparse points to grid
                if METHOD in ('spline',):
                    # interpolate with biharmonic splines in tension
                    INTERP = spi.biharmonic_spline(xnorm, ynorm,
                        u['tide_adj'], XN.flatten(), YN.flatten(),
                        metric='euclidean', tension=TENSION, eps=1e-7)
                elif METHOD in ('radial',):
                    # interpolate with radial basis functions
                    INTERP = spi.radial_basis(xnorm, ynorm,
                        u['tide_adj'], XN.flatten(), YN.flatten(),
                        metric='euclidean', smooth=SMOOTH,
                        epsilon=EPSILON, polynomial=POLYNOMIAL)
                # clip to valid values and add to output mosaic
                np.clip(INTERP, 0.0, 1.0, out=INTERP)
                interp[:,:] = INTERP.reshape(len(Y),len(X))
                # pad the interpolated matrix to remove edges
                if (PAD > 0):
                    # reset interpolation grids for padded
                    interp[indy,indx] = 0.0
                    count[indy,indx] = 0.0
                # add to output mosaic
                mosaic[iy,ix] += interp.copy()
                weight[iy,ix] += count.copy()

    # calculate average tide adjustment scale
    ii,jj = np.nonzero(weight > 0)
    output['tide_adj_scale'][ii,jj] = mosaic[ii,jj]/weight[ii,jj]
    output['weight'][ii,jj] = weight[ii,jj]

    # open original HDF5 file in append mode
    fileID = gz.io.multiprocess_h5py(tile_file, mode='a')
    # create geophysical group if non-existent
    group = 'geophysical'
    if group not in fileID:
        g1 = fileID.create_group(group)
    else:
        g1 = fileID[group]
    # add group attributes
    g1.attrs['x_center'] = xc
    g1.attrs['y_center'] = yc
    g1.attrs['spacing'] = SPACING
    # for each output variable
    h5 = {}
    for key,val in output.items():
        # create or overwrite HDF5 variables
        logging.info(f'{group}/{key}')
        if key not in fileID[group]:
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
        else:
            # overwrite HDF5 variables
            g1[key][...] = val.copy()
    # close the output file
    fileID.close()
    # change the permissions mode
    tile_file.chmod(mode=MODE)

# PURPOSE: create arguments parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Interpolates tidal adjustment scale
            factors to output grids
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='ICESat-2 ATL11 tile file to run')
    # directory with input/output data
    parser.add_argument('--output-directory','-O',
        type=pathlib.Path,
        help='Output data directory')
    # region of interest to run
    parser.add_argument('--hemisphere','-H',
        type=str, default='S', choices=('N','S'),
        help='Region of interest to run')
    # tile grid width
    parser.add_argument('--width', '-W',
        type=float, default=80e3,
        help='Width of tile grid')
    # interpolation subset width
    parser.add_argument('--subset', '-s',
        type=float, default=10e3,
        help='Width of interpolaton subset')
    # output grid spacing
    parser.add_argument('--spacing','-S',
        type=float, nargs='+', default=200,
        help='Output grid spacing')
    # pad tiles in mosaic to remove less determined edges
    parser.add_argument('--pad','-P',
        type=float, default=1e3,
        help='Tile pad for creating mosaics')
    # tide model to use
    parser.add_argument('--tide','-T',
        metavar='TIDE', type=str,
        help='Tide model used in correction')
    # interpolation method
    parser.add_argument('--interpolate','-I',
        type=str, default='radial', choices=('spline','radial'),
        help='Interpolation Method')
    # biharmonic spline tension
    parser.add_argument('--tension','-t',
        type=float, default=0.5,
        help='Biharmonic spline tension')
    # radial basis function smoothing
    parser.add_argument('--smooth','-w',
        type=float, default=1.0,
        help='Radial basis function smoothing weights')
    # radial basis function epsilon
    parser.add_argument('--epsilon','-e',
        type=float, default=1.0,
        help='Radial basis function adjustable constant')
    # radial basis function polynomial
    parser.add_argument('--polynomial','-p',
        type=int, default=0,
        help='Polynomial order for augmenting radial basis functions')
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

    # run program for each file
    for FILE in args.infile:
        interpolate_tide_adjustment(FILE,
            OUTPUT_DIRECTORY=args.output_directory,
            HEM=args.hemisphere,
            W=args.width,
            SUBSET=args.subset,
            SPACING=args.spacing,
            PAD=args.pad,
            TIDE_MODEL=args.tide,
            METHOD=args.interpolate,
            TENSION=args.tension,
            SMOOTH=args.smooth,
            EPSILON=args.epsilon,
            POLYNOMIAL=args.polynomial,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
