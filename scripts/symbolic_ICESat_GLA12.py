#!/usr/bin/env python
u"""
symbolic_ICESat_GLA12.py
Written by Tyler Sutterley (05/2022)
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
    Updated 05/2022: use argparse descriptions within documentation
    Written 02/2022
"""
from __future__ import print_function

import sys
import os
import re
import copy
import logging
import argparse
import warnings

# attempt imports
try:
    import h5py
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("module")
    warnings.warn("h5py not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

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
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory for symbolic link')
    # incoming directory with ICESat GLA12 data
    parser.add_argument('--incoming',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
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
    subdirectories = [s for s in os.listdir(incoming) if rx1.match(s) and
        os.path.isdir(os.path.join(incoming,s))]
    for sd in sorted(subdirectories):
        # put symlinks in directories similar to NSIDC
        local_dir = os.path.join(base_dir,sd)
        os.makedirs(local_dir,MODE) if not os.path.exists(local_dir) else None
        # find each  ICESat GLA12 file within the subdirectory
        files = [f for f in os.listdir(os.path.join(incoming,sd)) if rx2.match(f)]
        # for each ICESat GLA12 file
        for f in sorted(files):
            # attempt to create the symbolic link else continue
            try:
                # create symbolic link of file from scf_outgoing to local
                os.symlink(os.path.join(incoming,sd,f), os.path.join(local_dir,f))
            except FileExistsError:
                continue
            else:
                # print original and symbolic link of file
                args = (os.path.join(incoming,sd,f),os.path.join(local_dir,f))
                logging.info('{0} -->\n\t{1}'.format(*args))
    # find files within incoming (if flattened)
    # find each ICESat GLA12 file within the subdirectory
    files = [f for f in os.listdir(incoming) if rx2.match(f)]
    for f in sorted(files):
        # get the date information from the input file
        year,month,day = parse_GLA12_HDF5_file(os.path.join(incoming,f))
        # put symlinks in directories similar to NSIDC
        sd = '{year:4d}.{month:02d}.{day:02d}'.format(year,month,day)
        local_dir = os.path.join(base_dir,sd)
        os.makedirs(local_dir,MODE) if not os.path.exists(local_dir) else None
        # attempt to create the symbolic link else continue
        try:
            # create symbolic link of file from scf_outgoing to local
            os.symlink(os.path.join(incoming,f), os.path.join(local_dir,f))
        except FileExistsError:
            continue
        else:
            # print original and symbolic link of file
            args = (os.path.join(incoming,f),os.path.join(local_dir,f))
            logging.info('{0} -->\n\t{1}'.format(*args))

# PURPOSE: extract date information from HDF5 ancillary data attributes
def parse_GLA12_HDF5_file(input_file):
    attributes = {}
    with h5py.File(os.path.expanduser(input_file),'r') as fileID:
        for key,val in fileID['METADATA']['INVENTORYMETADATA'].attrs.items():
            attributes[key] = copy.copy(val)
    # extract year month and day from attributes
    YY,MM,DD = attributes['RangeBeginningDate'].decode('utf-8').split('-')
    return (int(YY),int(MM),int(DD))

# run main program
if __name__ == '__main__':
    main()
