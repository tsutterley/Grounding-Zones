#!/usr/bin/env python
u"""
tile_icebridge_data.py
Written by Tyler Sutterley (05/2024)
Creates tile index files of Operation IceBridge elevation data

INPUTS:
    ATM1B, ATM icessn or LVIS file from NSIDC

COMMAND LINE OPTIONS:
    --help: list the command line options
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
    read_ATM1b_QFIT_binary.py: read ATM1b QFIT binary files (NSIDC version 1)

UPDATE HISTORY:
    Updated 05/2024: adjust default spacing of tiles to 80 km
        return if no valid points in hemisphere
        save icebridge filename with suffix as groups in tile files
        use wrapper to importlib for optional dependencies
        change permissions mode of the output tile files
        moved multiprocess h5py reader to io utilities module
    Updated 05/2023: using pathlib to define and operate on paths
        move icebridge data inputs to a separate module in io
    Updated 12/2022: check that file exists within multiprocess HDF5 function
        single implicit import of grounding zone tools
    Updated 07/2022: update imports of ATM1b QFIT functions to released version
        place some imports within try/except statements
    Updated 06/2022: add checks if variables and groups already exist
    Updated 05/2022: use argparse descriptions within documentation
    Updated 11/2021: adjust tiling to index by center coordinates
        wait if merged HDF5 tile file is unavailable
    Written 10/2021
"""
import sys
import re
import time
import logging
import pathlib
import argparse
import collections
import numpy as np
import grounding_zones as gz

# attempt imports
h5py = gz.utilities.import_dependency('h5py')
is2tk = gz.utilities.import_dependency('icesat2_toolkit')
pyproj = gz.utilities.import_dependency('pyproj')

# PURPOSE: create tile index files of Operation IceBridge data
def tile_icebridge_data(arg,
    SPACING=None,
    VERBOSE=False,
    MODE=0o775):

    # create logger
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # extract file name and subsetter indices lists
    match_object = re.match(r'(.*?)(\[(.*?)\])?$', str(arg))
    input_file = pathlib.Path(match_object.group(1)).expanduser().absolute()
    # subset input file to indices
    if match_object.group(2):
        # decompress ranges and add to list
        input_subsetter = []
        for i in re.findall(r'((\d+)-(\d+)|(\d+))',match_object.group(3)):
            input_subsetter.append(int(i[3])) if i[3] else \
                input_subsetter.extend(range(int(i[1]),int(i[2])+1))
    else:
        input_subsetter = None

    # calculate if input files are from ATM or LVIS (+GH)
    regex = {}
    regex['ATM'] = r'(BLATM2|ILATM2)_(\d+)_(\d+)_smooth_nadir(.*?)(csv|seg|pt)$'
    regex['ATM1b'] = r'(BLATM1b|ILATM1b)_(\d+)_(\d+)(.*?).(qi|TXT|h5)$'
    regex['LVIS'] = r'(BLVIS2|BVLIS2|ILVIS2)_(.*?)(\d+)_(\d+)_(R\d+)_(\d+).H5$'
    regex['LVGH'] = r'(ILVGH2)_(.*?)(\d+)_(\d+)_(R\d+)_(\d+).H5$'
    for key,val in regex.items():
        if re.match(val, input_file.name):
            OIB = key
            rx = re.compile(val)

    # extract information from first input file
    # acquisition year, month and day
    # number of points
    # instrument (PRE-OIB ATM or LVIS, OIB ATM or LVIS)
    if OIB in ('ATM','ATM1b'):
        M1,YYMMDD1,HHMMSS1,AX1,SF1 = rx.findall(input_file.name).pop()
        # early date strings omitted century and millennia (e.g. 93 for 1993)
        if (len(YYMMDD1) == 6):
            year_two_digit,MM1,DD1 = YYMMDD1[:2],YYMMDD1[2:4],YYMMDD1[4:]
            year_two_digit = float(year_two_digit)
            if (year_two_digit >= 90):
                YY1 = f'{1900.0+year_two_digit:4.0f}'
            else:
                YY1 = f'{2000.0+year_two_digit:4.0f}'
        elif (len(YYMMDD1) == 8):
            YY1,MM1,DD1 = YYMMDD1[:4],YYMMDD1[4:6],YYMMDD1[6:]
    elif OIB in ('LVIS','LVGH'):
        M1,RG1,YY1,MMDD1,RLD1,SS1 = rx.findall(input_file.name).pop()
        MM1,DD1 = MMDD1[:2],MMDD1[2:]

    # track file progress
    logging.info(input_file)
    # read data from input_file
    if (OIB == 'ATM'):
        # load IceBridge ATM data from input_file
        dinput, file_lines, HEM = gz.io.icebridge.read_ATM_icessn_file(
            input_file, input_subsetter)
    elif (OIB == 'ATM1b'):
        # load IceBridge Level-1b ATM data from input_file
        dinput, file_lines, HEM = gz.io.icebridge.read_ATM_qfit_file(
            input_file, input_subsetter)
    elif OIB in ('LVIS','LVGH'):
        # load IceBridge LVIS data from input_file
        dinput, file_lines, HEM = gz.io.icebridge.read_LVIS_HDF5_file(
            input_file, input_subsetter)

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
    attributes = collections.OrderedDict()
    # time
    attributes['time'] = {}
    attributes['time']['long_name'] = 'time'
    attributes['time']['standard_name'] = 'time'
    attributes['time']['description'] = ('The transmit time of each shot in '
        'the 1 second frame measured as UTC seconds elapsed since Jan 1 '
        '2000 12:00:00 UTC.')
    attributes['time']['units'] = 'seconds since 2000-01-01 12:00:00 UTC'
    attributes['time']['calendar'] = 'standard'
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

    # index directory for hemisphere
    index_directory = 'north' if (HEM == 'N') else 'south'
    # output directory and index file
    DIRECTORY = input_file.with_name(index_directory)
    output_file = DIRECTORY.joinpath(f'{input_file.stem}.h5')
    # create index directory for hemisphere
    DIRECTORY.mkdir(mode=MODE, parents=True, exist_ok=True)

    # indices of points in hemisphere
    valid, = np.nonzero(np.sign(dinput['lat']) == SIGN[HEM])
    if not valid.any():
        logging.error('No valid points in hemisphere')
        return
    # convert latitude and longitude to regional projection
    x,y = transformer.transform(dinput['lon'],dinput['lat'])
    # large-scale tiles
    xtile = (x-0.5*SPACING)//SPACING
    ytile = (y-0.5*SPACING)//SPACING

    # open output index file
    f2 = h5py.File(output_file,'w')
    f2.attrs['featureType'] = 'trajectory'
    f2.attrs['GDAL_AREA_OR_POINT'] = 'Point'
    f2.attrs['time_type'] = 'UTC'
    today = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    f2.attrs['date_created'] = today
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
        output['time'] = dinput['time'][indices].copy()
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
                if key not in ('time',):
                    for i,dim in enumerate(['time']):
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
        description="""Creates tile index files of Operation
            IceBridge elevation data
            """
    )
    # command line parameters
    # input operation icebridge files
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='Input Operation IceBridge file')
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

    # run program for each file
    for arg in args.infile:
        tile_icebridge_data(arg,
            SPACING=args.spacing,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
