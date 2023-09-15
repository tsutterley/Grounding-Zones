#!/usr/bin/env python
u"""
symbolic_ICESat_GLA12.py
Written by Tyler Sutterley (05/2023)
Creates symbolic links for ICESat/GLAS L2 GLA12 Antarctic and Greenland
    Ice Sheet elevation files organized by date

CALLING SEQUENCE:
    python symbolic_ICESat_GLA12.py --directory <path_to_directory> \
        --incoming <path_to_incoming> --verbose --mode 0o775

COMMAND LINE OPTIONS:
    -h, --help: list the command line options
    -D X, --directory X: local working directory for creating symbolic links
    --incoming X: directory with ICESat GLA12 data
    -V, --verbose: output information about each symbolic link
    -M X, --mode X: permission mode of directories

UPDATE HISTORY:
    Updated 05/2023: using pathlib to define and operate on paths
    Updated 05/2022: use argparse descriptions within documentation
    Written 02/2022
"""
from __future__ import print_function

import sys
import re
import copy
import logging
import pathlib
import argparse
import warnings

# attempt imports
try:
    import h5py
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.warn("h5py not available", ImportWarning)

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Creates symbolic links for ICESat/GLAS L2
            GLA12 Antarctic and Greenland Ice Sheet elevation files
            organized by date
            """
    )
    # command line parameters
    # working data directory
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=pathlib.Path.cwd(),
        help='Working data directory for symbolic link')
    # incoming directory with ICESat GLA12 data
    parser.add_argument('--incoming',
        type=pathlib.Path,
        help='Directory with ICESat GLA12 data')
    # verbose will output information about each symbolic link
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Output information about each symbolic link')
    # permissions mode of the local directories (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='permissions mode of output directories')
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

    # run program
    symbolic_ICESat_GLA12(args.directory, args.incoming, MODE=args.mode)

# PURPOSE: copy ICESat GLA12 files to data directory with subdirectories
def symbolic_ICESat_GLA12(base_dir, incoming, MODE=0o775):
    # regular expression pattern for finding subdirectories
    rx1 = re.compile(r'(\d+)\.(\d+)\.(\d+)',re.VERBOSE)
    # compile regular expression operator for extracting information from file
    rx2 = re.compile((r'GLAH(\d{2})_(\d{3})_(\d{1})(\d{1})(\d{2})_(\d{3})_'
        r'(\d{4})_(\d{1})_(\d{2})_(\d{4})\.H5$'), re.VERBOSE)

    # find subdirectories within incoming
    subdir = [s for s in incoming.iterdir() if rx1.match(s.name) and s.is_dir()]
    for sd in sorted(subdir):
        # put symlinks in directories similar to NSIDC
        local_dir = base_dir.joinpath(sd.name)
        local_dir.mkdir(mode=MODE, parents=True, exist_ok=True)
        # find each ICESat GLA12 file within the subdirectory
        files = [f for f in sd.iterdir() if rx2.match(f.name)]
        # for each ICESat GLA12 file
        for f in sorted(files):
            # attempt to create the symbolic link else continue
            try:
                # create symbolic link of file from incoming to outgoing
                outgoing = local_dir.joinpath(f.name)
                f.symlink_to(outgoing)
            except FileExistsError:
                continue
            else:
                # print original and symbolic link of file
                logging.info(f'{str(f)} -->\n\t{str(outgoing)}')

    # find files within incoming (if flattened)
    # find each ICESat GLA12 file within the subdirectory
    files = [f for f in incoming.iterdir() if rx2.match(f.name)]
    for f in sorted(files):
        # get the date information from the input file
        year,month,day = parse_GLA12_HDF5_file(incoming.joinpath(f))
        # put symlinks in directories similar to NSIDC
        local_dir = base_dir.joinpath(f'{year:4d}.{month:02d}.{day:02d}')
        local_dir.mkdir(mode=MODE, parents=True, exist_ok=True)
        # attempt to create the symbolic link else continue
        try:
            # create symbolic link of file from scf_outgoing to local
            outgoing = local_dir.joinpath(f.name)
            f.symlink_to(outgoing)
        except FileExistsError:
            continue
        else:
            # print original and symbolic link of file
            logging.info(f'{str(f)} -->\n\t{str(outgoing)}')

# PURPOSE: extract date information from HDF5 ancillary data attributes
def parse_GLA12_HDF5_file(input_file):
    attributes = {}
    # open input HDF5 file for reading
    input_file = pathlib.Path(input_file).expanduser().absolute()
    with h5py.File(input_file, mode='r') as fileID:
        # get inventory metadata
        for key,val in fileID['METADATA']['INVENTORYMETADATA'].attrs.items():
            attributes[key] = copy.copy(val)
    # extract year month and day from attributes
    YY,MM,DD = attributes['RangeBeginningDate'].decode('utf-8').split('-')
    return (int(YY),int(MM),int(DD))

# run main program
if __name__ == '__main__':
    main()
