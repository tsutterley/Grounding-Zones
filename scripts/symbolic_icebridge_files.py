#!/usr/bin/env python
u"""
symbolic_icebridge_files.py
Written by Tyler Sutterley (05/2022)
Creates symbolic links for Operation IceBridge files organized by date

CALLING SEQUENCE:
    python symbolic_icebridge_files.py --product ILATM2 \
        --directory <path_to_directory> --incoming <path_to_incoming> \
        --verbose --mode 0o775

COMMAND LINE OPTIONS:
    -h, --help: list the command line options
    -D X, --directory X: local working directory for creating symbolic links
    --product X: Operation IceBridge data product to create symbolic links
    --incoming X: directory with Operation IceBridge data
    -V, --verbose: output information about each symbolic link
    -M X, --mode X: permission mode of directories

UPDATE HISTORY:
    Updated 05/2022: use argparse descriptions within documentation
    Updated 10/2021: using python logging for handling verbose output
    Updated 10/2020: using argparse to set command line parameters
    Written 07/2019
"""
from __future__ import print_function

import sys
import os
import re
import logging
import argparse
import numpy as np

#-- PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Creates symbolic links for Operation IceBridge
            files to a separate directory organized by date
            """
    )
    #-- command line parameters
    #-- working data directory
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory for symbolic link')
    #-- Operation IceBridge product
    choices = ('BLATM1B','BLATM2','ILATM1B','ILATM2','ILVGH2','ILVIS2')
    parser.add_argument('--product','-p',
        metavar='PRODUCTS', type=str,
        choices=choices, default='ILATM2.002',
        help='OIB data product to create symbolic links')
    #-- incoming directory with Operation IceBridge data
    parser.add_argument('--incoming',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        help='Directory with Operation IceBridge data')
    #-- verbose will output information about each symbolic link
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Output information about each symbolic link')
    #-- permissions mode of the local directories (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='permissions mode of output directories')
    #-- return the parser
    return parser

#-- This is the main part of the program that calls the individual functions
def main():
    #-- Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    #-- create logger
    loglevel = logging.INFO if args.verbose else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    #-- run program
    symbolic_icebridge_files(args.directory, args.incoming, args.product,
        MODE=args.mode)

#-- PURPOSE: copy Operation IceBridge files to data directory with
#-- data subdirectories
def symbolic_icebridge_files(base_dir, incoming, PRODUCT, MODE=0o775):
    #-- regular expression pattern for finding subdirectories
    rx1 = re.compile(r'(\d+)\.(\d+)\.(\d+)',re.VERBOSE)
    #-- regular expression patterns for Operation IceBridge files
    R = {}
    R['BLATM2'] = r'(BLATM2|ILATM2)_(\d+)_(\d+)_smooth_nadir(.*?)(csv|seg|pt)$'
    R['ILATM2'] = r'(BLATM2|ILATM2)_(\d+)_(\d+)_smooth_nadir(.*?)(csv|seg|pt)$'
    R['BLATM1B'] = r'(BLATM1b|ILATM1b)_(\d+)_(\d+)(.*?).(qi|TXT|h5)$'
    R['ILATM1B'] = r'(BLATM1b|ILATM1b)_(\d+)_(\d+)(.*?).(qi|TXT|h5)$'
    R['ILVIS2'] = r'(BLVIS2|BVLIS2|ILVIS2)_(.*?)(\d+)_(\d+)_(R\d+)_(\d+).(H5|TXT)$'
    R['ILVGH2'] = r'(ILVGH2)_(.*?)(\d+)_(\d+)_(R\d+)_(\d+).(H5|TXT)$'
    rx2 = re.compile(R[PRODUCT],re.VERBOSE)
    #-- find subdirectories within incoming
    subdirectories = [s for s in os.listdir(incoming) if rx1.match(s) and
        os.path.isdir(os.path.join(incoming,s))]
    for sd in sorted(subdirectories):
        #-- put symlinks in directories similar to NSIDC
        local_dir = os.path.join(base_dir,sd)
        os.makedirs(local_dir,MODE) if not os.path.exists(local_dir) else None
        #-- find each operation icebridge file within the subdirectory
        files = [f for f in os.listdir(os.path.join(incoming,sd)) if rx2.match(f)]
        #-- for each operation icebridge file
        for f in sorted(files):
            #-- attempt to create the symbolic link else continue
            try:
                #-- create symbolic link of file from scf_outgoing to local
                os.symlink(os.path.join(incoming,sd,f), os.path.join(local_dir,f))
            except FileExistsError:
                continue
            else:
                #-- print original and symbolic link of file
                args = (os.path.join(incoming,sd,f),os.path.join(local_dir,f))
                logging.info('{0} -->\n\t{1}'.format(*args))
    #-- find files within incoming (if flattened)
    #-- find each operation icebridge file within the subdirectory
    files = [f for f in os.listdir(incoming) if rx2.match(f)]
    for f in sorted(files):
        #-- get the date information from the input file
        year,month,day = parse_icebridge_file(f, PRODUCT)
        #-- put symlinks in directories similar to NSIDC
        sd = '{year:4d}.{month:02d}.{day:02d}'.format(year,month,day)
        local_dir = os.path.join(base_dir,sd)
        os.makedirs(local_dir,MODE) if not os.path.exists(local_dir) else None
        #-- attempt to create the symbolic link else continue
        try:
            #-- create symbolic link of file from scf_outgoing to local
            os.symlink(os.path.join(incoming,f), os.path.join(local_dir,f))
        except FileExistsError:
            continue
        else:
            #-- print original and symbolic link of file
            args = (os.path.join(incoming,f),os.path.join(local_dir,f))
            logging.info('{0} -->\n\t{1}'.format(*args))

#-- PURPOSE: wrapper function for parsing files
def parse_icebridge_file(input_file, PRODUCT):
    if PRODUCT in ('BLATM1B','ILATM1B'):
        year,month,day = parse_ATM_qfit_file(input_file)
    elif PRODUCT in ('BLATM2','ILATM2'):
        year,month,day = parse_ATM_icessn_file(input_file)
    elif PRODUCT in ('ILVIS2','ILVGH2'):
        year,month,day = parse_LVIS_elevation_file(input_file)
    # return the elevations
    return (year,month,day)

#-- PURPOSE: extract information from ATM Level-1B QFIT files
def parse_ATM_qfit_file(input_file):
    #-- regular expression pattern for extracting parameters
    regex_pattern = r'(BLATM1B|ILATM1B|ILNSA1B)_(\d+)_(\d+)(.*?).(qi|TXT|h5)$'
    rx = re.compile(regex_pattern, re.VERBOSE)
    #-- extract mission and other parameters from filename
    MISSION,YYMMDD,HHMMSS,AUX,SFX = rx.findall(input_file).pop()
    #-- early date strings omitted century and millenia (e.g. 93 for 1993)
    if (len(YYMMDD) == 6):
        yr2d,month,day = np.array([YYMMDD[:2],YYMMDD[2:4],YYMMDD[4:]],dtype='i')
        year = (yr2d + 1900.0) if (yr2d >= 90) else (yr2d + 2000.0)
    elif (len(YYMMDD) == 8):
        year,month,day = np.array([YYMMDD[:4],YYMMDD[4:6],YYMMDD[6:]],dtype='i')
    #-- return the year, month and day
    return (int(year),month,day)

#-- PURPOSE: extract information from ATM Level-2 icessn files
def parse_ATM_icessn_file(input_file):
    #-- regular expression pattern for extracting parameters
    mission_flag = r'(BLATM2|ILATM2)'
    regex_pattern = rf'{mission_flag}_(\d+)_(\d+)_smooth_nadir(.*?)(csv|seg|pt)$'
    rx = re.compile(regex_pattern, re.VERBOSE)
    #-- extract mission and other parameters from filename
    MISSION,YYMMDD,HHMMSS,AUX,SFX = rx.findall(input_file).pop()
    #-- early date strings omitted century and millenia (e.g. 93 for 1993)
    if (len(YYMMDD) == 6):
        yr2d,month,day = np.array([YYMMDD[:2],YYMMDD[2:4],YYMMDD[4:]],dtype='i')
        year = (yr2d + 1900.0) if (yr2d >= 90) else (yr2d + 2000.0)
    elif (len(YYMMDD) == 8):
        year,month,day = np.array([YYMMDD[:4],YYMMDD[4:6],YYMMDD[6:]],dtype='i')
    #-- return the year, month and day
    return (int(year),month,day)

#-- PURPOSE: extract information from LVIS HDF5 files
def parse_LVIS_elevation_file(input_file):
    #-- regular expression pattern for extracting parameters
    regex_pattern = (r'(BLVIS2|BVLIS2|ILVIS2|ILVGH2)_(.*?)(\d+)_'
        r'(\d{2})(\d{2})_(R\d+)_(\d+).(H5|TXT)$')
    rx = re.compile(regex_pattern, re.VERBOSE)
    #-- extract mission, region and other parameters from filename
    MISSION,REGION,YY,MM,DD,RLD,SS,SFX = rx.findall(input_file).pop()
    LDS_VERSION = '2.0.2' if (int(RLD[1:3]) >= 18) else '1.04'
    #-- return the year, month and day
    return (int(YY),int(MM),int(DD))

#-- run main program
if __name__ == '__main__':
    main()
