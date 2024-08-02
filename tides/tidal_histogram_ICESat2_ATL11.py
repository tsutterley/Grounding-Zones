#!/usr/bin/env python
u"""
tidal_histogram_ICESat2_ATL11.py
Written by Tyler Sutterley (07/2024)
Calculates histograms of variances between ICESat-2 ATL11
annual land ice height data and tide predictions

COMMAND LINE OPTIONS:
    -O X, --output-directory X: input/output data directory
    -H X, --hemisphere X: Region of interest to run
    -W X, --width: Width of tile grid
    -S X, --spacing X: Output grid spacing
    -T X, --tide X: Tide model used in correction
    -B X, --histogram X: lower end, upper end and spacing of histogram
    -M X, --mode X: Permission mode of directories and files created
    -V, --verbose: Output information about each created file

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://h5py.org
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/

PROGRAM DEPENDENCIES:
    io/ATL11.py: reads ICESat-2 annual land ice height data files

UPDATE HISTORY:
    Updated 07/2024: use wrapper to importlib for optional dependencies
        use multiprocess h5py reader from io utilities module
        use updated ATL11 readers within icesat2_toolkit
        use pathlib to define and operate on paths
        use raster file to define valid grid cells in mask 
        include cell_area in output HDF5 file
    Written 03/2021
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
import scipy.stats
import scipy.interpolate
import grounding_zones as gz

# attempt imports
h5py = gz.utilities.import_dependency('h5py')
is2tk = gz.utilities.import_dependency('icesat2_toolkit')
pyproj = gz.utilities.import_dependency('pyproj')
pyTMD = gz.utilities.import_dependency('pyTMD')

# PURPOSE: keep track of threads
def info(args):
    logging.debug(pathlib.Path(sys.argv[0]).name)
    logging.debug(args)
    logging.debug(f'module name: {__name__}')
    if hasattr(os, 'getppid'):
        logging.debug(f'parent process: {os.getppid():d}')
    logging.debug(f'process id: {os.getpid():d}')

# PURPOSE: read a raster file and return the data
def read_raster_file(raster_file, **kwargs):
    """
    Read a geotiff file and prepare for use in interpolation
    """
    # read raster image for spatial coordinates and data
    dinput = is2tk.spatial.from_geotiff(raster_file, **kwargs)
    spacing = np.array(dinput['attributes']['spacing'])
    # check that x and y are strictly increasing
    if (np.sign(spacing[0]) == -1):
        dinput['x'] = dinput['x'][::-1]
        dinput['data'] = dinput['data'][:,::-1]
        spacing[0] *= -1.0
    if (np.sign(spacing[1]) == -1):
        dinput['y'] = dinput['y'][::-1]
        dinput['data'] = dinput['data'][::-1,:]
        spacing[1] *= -1.0
    # update the spacing attribute
    dinput['attributes']['spacing'] = spacing
    # return the mask object
    return dinput

# PURPOSE: calculates histograms of tide differences
def tidal_histogram(tile_file,
        OUTPUT_DIRECTORY=None,
        HEM='S',
        W=80e3,
        SPACING=None,
        MASK_FILE=None,
        TIDE_MODEL=None,
        DEFINITION_FILE=None,
        DEFINITION_FORMAT='ascii',
        REANALYSIS=None,
        HISTOGRAM=(-8,8,0.01),
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
    OUTPUT_DIRECTORY.mkdir(mode=MODE, parents=True, exist_ok=True)
    # file format for mask and tide prediction files
    file_format = '{0}_{1}{2}_{3}{4}_{5}{6}_{7}_{8}{9}.h5'
    # extract tile centers from filename
    tile_centers = R1.findall(tile_file.name).pop()
    xc, yc = 1000.0*np.array(tile_centers, dtype=np.float64)
    logging.info(f'Tile File: {str(tile_file)}')
    tile_file_formatted = f'E{xc/1e3:0.0f}_N{yc/1e3:0.0f}.h5'
    logging.info(f'Tile Center: {xc:0.1f} {yc:0.1f}')

    # grid dimensions
    xmin,xmax = xc + np.array([-0.5,0.5])*W
    ymin,ymax = yc + np.array([-0.5,0.5])*W
    dx,dy = np.broadcast_to(np.atleast_1d(SPACING),(2,))
    nx = np.int64(W//dx) + 1
    ny = np.int64(W//dy) + 1

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
    flat = 1.0/crs_to_cf['inverse_flattening']
    reference_latitude = crs_to_cf['standard_parallel']

    # get tide model parameters from definition file
    if DEFINITION_FILE is not None:
        model = pyTMD.io.model(None, verify=False).from_file(
            DEFINITION_FILE, format=DEFINITION_FORMAT)
        TIDE_MODEL = model.name

    # read the input file
    if MASK_FILE is not None:
        bounds = [xmin-dx, xmax+dx, ymin-dy, ymax+dy]
        m = read_raster_file(MASK_FILE, bounds=bounds)
        # calculate polar stereographic distortion
        # interpolate raster to output grid
        DX, DY = m['attributes']['spacing']
        GRIDx, GRIDy = np.meshgrid(m['x'], m['y'])
        IMx = np.clip((GRIDx.flatten() - xmin)//dx, 0, nx-1).astype(int)
        IMy = np.clip((GRIDy.flatten() - ymin)//dy, 0, ny-1).astype(int)
        mask = np.zeros((ny, nx))
        area = np.zeros((ny, nx))
        for i, msk in enumerate(m['data'].flatten()):
            mask[IMy[i], IMx[i]] += DX*DY*msk
            area[IMy[i], IMx[i]] += DX*DY
        # convert to average
        i, j = np.nonzero(area)
        mask[i,j] /= area[i,j]
        # create an interpolator for input raster data
        SPL = scipy.interpolate.RectBivariateSpline(
            m['x'], m['y'], m['data'].T, kx=1, ky=1)
        # tolerance to find valid elevations
        TOLERANCE = 0.5
    else:
        # assume all ice covered
        mask = np.ones((ny, nx))
        SPL = None

    # output coordinates (centered)
    x = np.arange(xmin + dx/2.0, xmax + dx, dx)
    y = np.arange(ymin + dx/2.0, ymax + dy, dy)
    gridx, gridy = np.meshgrid(x, y)
    # convert grid coordinates to latitude and longitude
    _, gridlat = transformer.transform(gridx, gridy, direction='inverse')
    ps_scale = is2tk.spatial.scale_factors(gridlat, flat=flat,
        reference_latitude=reference_latitude, metric='area')
    # calculate cell areas (m^2)
    cell_area = dx*dy*mask*ps_scale

    # total point count
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
            npts += (ncycles-1)*len(indices)
    # close the tile file
    f1.close()
    # log total number of points
    logging.info(f'Total Points: {npts:d}')

    # histogram parameters
    vmin,vmax,w = np.copy(HISTOGRAM)
    b1 = np.arange(vmin,vmax+w,w)
    b2 = (b1[1:] + b1[0:-1])/2.0
    nbins = np.int64((vmax-vmin)/w)

    # allocate for combined variables
    d = {}
    d['longitude'] = np.zeros((npts), dtype=np.float64)
    d['latitude'] = np.zeros((npts), dtype=np.float64)
    d['dh_corr'] = np.ma.zeros((npts), dtype=np.float64)
    d['dh_sigma'] = np.ma.zeros((npts), dtype=np.float64)
    d['mask'] = np.zeros((npts), dtype=bool)

    # height threshold (filter points below 0m elevation)
    THRESHOLD = 0.0
    # maximum valid error for height change
    MAX_ERROR = 1.0
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
        # read tide model corrections
        if TIDE_MODEL:
            # ATL11 tide correction HDF5 file
            a3 = (PRD,TIDE_MODEL,'_TIDES',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
            FILE3 = d1.joinpath(file_format.format(*a3))
            f3 = gz.io.multiprocess_h5py(FILE3, mode='r')
        # read inverse barometer correction
        if REANALYSIS:
            # read inverse barometer HDF5 file
            a4 = (PRD,REANALYSIS,'IB',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
            FILE4 = d1.joinpath(file_format.format(*a4))
            f4 = gz.io.multiprocess_h5py(FILE4, mode='r')

        # for each beam pair within the tile
        # extract number of points from subgroup
        # assuming that all data is valid
        for ptx, subgroup in f1[ATL11].items():
            # indices within tile
            indices = subgroup['index'][:].copy()
            cycle_number = subgroup['cycle_number'][:].copy()
            file_length = len(indices)
            # read dataset
            mds, attrs = is2tk.io.ATL11.read_pair(f2, ptx,
                ATTRIBUTES=True, REFERENCE=True, KEEP=True)
            # invalid value for heights
            invalid = attrs[ptx]['h_corr']['_FillValue']
            # combine errors (ignore overflow at invalid points)
            with np.errstate(over='ignore'):
                error = np.sqrt(
                    mds[ptx]['h_corr_sigma'][indices,:]**2 +
                    mds[ptx]['h_corr_sigma_systematic'][indices,:]**2
                )
            # read tide model corrections
            if TIDE_MODEL:
                # read data and reduce to indices
                temp = f3[ptx]['cycle_stats']['tide_ocean'][:].copy()
                tide_ocean = temp[indices,:]
            else:
                tide_ocean = np.zeros((file_length,ncycles))
            # read inverse barometer correction
            if REANALYSIS:
                # read data and reduce to indices
                temp = f3[ptx]['cycle_stats']['ib'][:].copy()
                IB = temp[indices,:]
            else:
                # reduce DAC to indices
                IB = mds[ptx]['cycle_stats']['dac'][indices,:]
            # for each cycle
            for k in range(ncycles-1):
                # copy annual land ice height variables
                h1 = np.ma.array(mds[ptx]['h_corr'][indices,k],
                    fill_value=invalid)
                h2 = np.ma.array(mds[ptx]['h_corr'][indices,k+1],
                    fill_value=invalid)
                e1, e2 = error[:,k], error[:,k+1]
                # create masks for height variables
                h1.mask = (h1.data == h1.fill_value)
                h2.mask = (h2.data == h2.fill_value)
                # quality summary for height variables
                qs1 = mds[ptx]['quality_summary'][indices,k]
                qs2 = mds[ptx]['quality_summary'][indices,k+1]
                # reference heights to geoid
                h1 -= mds[ptx]['ref_surf']['geoid_h'][indices]
                h2 -= mds[ptx]['ref_surf']['geoid_h'][indices]
                # correct heights for DAC
                h1 -= IB[:,k]
                h2 -= IB[:,k+1]
                # correct heights for ocean tides
                h1 -= tide_ocean[:,k]
                h2 -= tide_ocean[:,k+1]
                # calculate tide-corrected height differences
                d['dh_corr'][c:c+file_length] = (h2 - h1)
                d['dh_sigma'][c:c+file_length] = np.sqrt(e1**2 + e2**2)
                # add longitude, and latitude
                for i in ['longitude','latitude']:
                    d[i][c:c+file_length] = mds[ptx][i][indices].copy()
                # mask for reducing to valid values
                d['mask'][c:c+file_length] = \
                    np.logical_not(h1.mask) & np.logical_not(h2.mask) & \
                    (tide_ocean[:,k] != invalid) & \
                    (tide_ocean[:,k+1] != invalid) & \
                    (h1 > THRESHOLD) & (h2 > THRESHOLD) & \
                    (qs1 == 0) & (qs2 == 0)
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

    # calculate coordinates in coordinate reference system of tiles
    d['x'], d['y'] = transformer.transform(d['longitude'], d['latitude'])
    # reduce to valid surfaces with raster mask
    if SPL is not None:
        d['mask'] &= (SPL.ev(d['x'], d['y']) >= TOLERANCE)

    # check if there are any valid points
    if not np.any(d['mask']):
        raise ValueError('No valid points found for tile')

    # log total number of valid points
    valid, = np.nonzero(d['mask'])
    logging.info(f'Total Valid: {len(valid):d}')

    # output attributes
    attributes = dict(ROOT={}, x={}, y={})
    # invalid values for each variablce
    fill_value = {}
    # root group attributes
    attributes['ROOT']['x_center'] = xc
    attributes['ROOT']['y_center'] = xc
    attributes['ROOT']['tile_width'] = W
    attributes['ROOT']['spacing'] = SPACING
    # projection attributes
    attributes['crs'] = {}
    fill_value['crs'] = None
    # add projection attributes
    attributes['crs']['standard_name'] = \
        crs_to_cf['grid_mapping_name'].title()
    attributes['crs']['spatial_epsg'] = crs2.to_epsg()
    attributes['crs']['spatial_ref'] = crs2.to_wkt()
    attributes['crs']['proj4_params'] = crs2.to_proj4()
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
    attributes['bins']['units'] = 'meters'
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
    attributes['dh_hist']['units'] = 'meters'
    attributes['dh_hist']['coordinates'] = 'y x'
    attributes['dh_hist']['grid_mapping'] = 'crs'
    fill_value['dh_hist'] = invalid
    # data count
    attributes['count'] = {}
    attributes['count']['long_name'] = 'Number of data points'
    attributes['count']['units'] = '1'
    attributes['count']['coordinates'] = 'y x'
    attributes['count']['grid_mapping'] = 'crs'
    fill_value['count'] = 0

    # allocate for output variables
    output = {}
    # projection variable
    output['crs'] = np.empty((), dtype=np.byte)
    # coordinates
    output['x'] = np.copy(x)
    output['y'] = np.copy(y)
    output['bins'] = np.copy(b2)
    # cell area (possibly masked for non-ice areas)
    output['cell_area'] = np.copy(cell_area)
    # allocate for histogram variables
    output['dh_hist'] = np.ma.zeros((ny, nx, nbins), fill_value=invalid)
    output['count'] = np.zeros((ny, nx), dtype=np.int64)
    hist = np.zeros((nbins), dtype=np.float64)

    # calculate histogram over each raster pixel
    for xi in np.arange(xmin, xmin + W, dx):
        for yi in np.arange(ymin, ymin + W, dy):
            # grid indices
            indy = int((yi - ymin)//dy)
            indx = int((xi - xmin)//dx)
            # clip unique points to coordinates
            clipped = np.nonzero((d['x'] >= xi) & (d['x'] < xi+dx) &
                (d['y'] >= yi) & (d['y'] < yi+dy) &
                d['mask'] & (d['dh_sigma'] <= MAX_ERROR))
            # skip iteration if there are no (valid) points
            if not np.any(clipped):
                continue
            # create clipped data
            u = {}
            for key,val in d.items():
                u[key] = val[clipped]
            # calculate histogram
            hh,hb = np.histogram(u['dh_corr'].compressed(), bins=b1)
            # add to histogram
            output['dh_hist'][indy, indx] += hh.astype(np.float64)
            output['count'][indy, indx] += np.sum(hh)
            hist += hh.astype(np.float64)

    # find and replace invalid values
    indy, indx = np.nonzero(output['count'] == 0)
    # update values for invalid points
    output['dh_hist'][indy, indx, :] = invalid

    # calculate histogram statistics
    N = np.sum(hist)
    # histogram mean and standard deviation
    hmean = np.average(b2, weights=hist)
    hvariance = np.average((b2-hmean)**2, weights=hist)
    hstdev = np.sqrt(hvariance)
    # histogram skewness and excess kurtosis
    hskewness = np.average((b2-hmean)**3, weights=hist)/(hstdev**3)
    hkurtosis = np.average((b2-hmean)**4, weights=hist)/(hstdev**4)
    hkurtosis_excess = hkurtosis - 3.0
    # omnibus chi-squared test of normality
    mu1 = np.sqrt(6.0*N*(N-1.0)/(N-2.0)/(N+1.0)/(N+3.0))
    mu2 = 2.0*mu1*np.sqrt((N*N-1.0)/(N-3.0)/(N+5.0))
    chi2 = (hskewness/mu1)**2 + (hkurtosis_excess/mu2)**2
    pvalue = 1.0 - scipy.stats.chi2.cdf(chi2,2)
    # cumulative probability distribution function of histogram
    cpdf = np.cumsum(hist/np.sum(hist))
    # calculate percentiles for IQR and RDE
    # IQR: first and third quartiles (25th and 75th percentiles)
    # RDE: 16th and 84th percentiles
    # median: 50th percentile
    Q1,Q3,P16,P84,hmedian = np.interp([0.25,0.75,0.16,0.84,0.5],cpdf,b2)
    # calculate interquartile range
    hIQR = Q3 - Q1
    # calculate robust dispersion estimator (RDE)
    hRDE = P84 - P16
    # add statistics attributes
    attributes['dh_hist']['mean'] = hmean
    attributes['dh_hist']['median'] = hmedian
    attributes['dh_hist']['stdev'] = hstdev
    attributes['dh_hist']['skewness'] = hskewness
    attributes['dh_hist']['kurtosis'] = hkurtosis_excess
    attributes['dh_hist']['IQR'] = hIQR
    attributes['dh_hist']['RDE'] = hRDE

    # open output HDF5 file in append mode
    output_file = OUTPUT_DIRECTORY.joinpath(tile_file_formatted)
    logging.info(output_file)
    fileID = gz.io.multiprocess_h5py(output_file, mode='a')
    # create fit_statistics group if non-existent
    group = TIDE_MODEL if TIDE_MODEL else 'uncorrected'
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
        if key not in fileID[group]:
            # create HDF5 variables
            if fill_value[key] is not None:
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
    logging.info(fileID[group].keys())
    fileID.close()
    # change the permissions mode
    output_file.chmod(mode=MODE)

# PURPOSE: create a list of available ocean tide models
def get_available_models():
    """Create a list of available tide models
    """
    try:
        return sorted(pyTMD.io.model.ocean_elevation())
    except (NameError, AttributeError):
        return None

# PURPOSE: create arguments parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates histograms of variances between ICESat-2
            ATL11 annual land ice height data and tide predictions
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
    # region of interest to run
    parser.add_argument('--hemisphere','-H',
        type=str, default='S', choices=('N','S'),
        help='Region of interest to run')
    # tile grid width
    parser.add_argument('--width','-W',
        type=float, default=80e3,
        help='Width of tile grid')
    # output grid spacing
    parser.add_argument('--spacing','-S',
        type=float, nargs='+', default=5e3,
        help='Output grid spacing')
    # mask file for non-ice points
    parser.add_argument('--mask-file',
        type=pathlib.Path,
        help='Ice mask file')
    # tide model to use
    group.add_argument('--tide','-T',
        metavar='TIDE', type=str,
        choices=get_available_models(),
        help='Tide model to use in correction')
    # tide model definition file to set an undefined model
    group.add_argument('--definition-file',
        type=pathlib.Path,
        help='Tide model definition file')
    parser.add_argument('--definition-format',
        type=str, default='ascii', choices=('ascii', 'json'),
        help='Format for model definition file')
    # inverse barometer response to use
    parser.add_argument('--reanalysis','-R',
        metavar='REANALYSIS', type=str,
        help='Reanalysis model to use in inverse-barometer correction')
    # histogram parameters
    parser.add_argument('--histogram','-B',
        type=float, nargs=3, metavar=('VMIN','VMAX','W'),
        default=(-8,8,0.01),
        help='Histogram bin lower end, upper end and spacing')
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
        tidal_histogram(args.infile,
            OUTPUT_DIRECTORY=args.output_directory,
            HEM=args.hemisphere,
            W=args.width,
            SPACING=args.spacing,
            MASK_FILE=args.mask_file,
            TIDE_MODEL=args.tide,
            DEFINITION_FILE=args.definition_file,
            DEFINITION_FORMAT=args.definition_format,
            REANALYSIS=args.reanalysis,
            HISTOGRAM=args.histogram,
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
