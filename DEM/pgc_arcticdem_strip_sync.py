#!/usr/bin/env python
u"""
pgc_arcticdem_strip_sync.py
Written by Tyler Sutterley (11/2022)

Syncs ArcticDEM strip tar files from the Polar Geospatial Center (PGC)
    https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/strips

CALLING SEQUENCE:
    python pgc_arcticdem_strip_sync.py --version s2s041 --resolution 2m

COMMAND LINE OPTIONS:
    --help: list the command line options
    -D X, --directory X: Working data directory
    -v X, --version X: ArcticDEM version
        s2s041 (default)
    -r X, --resolution X: ArcticDEM spatial resolution
        2m (default)
    -s X, --strip X: ArcticDEM strips to sync (default=All)
    -T X, --timeout X: Timeout in seconds for blocking operations
    -R X, --retry X: Connection retry attempts
    -L, --list: print files to be transferred, but do not execute transfer
    -l, --log: output log of files downloaded
    --clobber: Overwrite existing data in transfer
    -M X, --mode X: Local permissions mode of the directories and files synced

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    dateutil: powerful extensions to datetime
        https://dateutil.readthedocs.io/en/stable/
    lxml: Pythonic XML and HTML processing library using libxml2/libxslt
        https://lxml.de/
        https://github.com/lxml/lxml
    future: Compatibility layer between Python 2 and Python 3
        https://python-future.org/

PROGRAM DEPENDENCIES:
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 11/2022: new ArcticDEM strip version and directory structure
    Written 05/2022
"""
from __future__ import print_function

import sys
import os
import re
import ssl
import time
import shutil
import logging
import argparse
import posixpath
import traceback
import lxml.etree
import grounding_zones as gz

# PURPOSE: sync local ArcticDEM strip files with PGC public server
def pgc_arcticdem_strip_sync(base_dir, VERSION, RESOLUTION, STRIPS=None,
    TIMEOUT=None, RETRY=1, LOG=False, LIST=False, CLOBBER=False,
    MODE=None):
    # data directory
    DIRECTORY = os.path.join(base_dir,'ArcticDEM')
    # check if directory exists and recursively create if not
    os.makedirs(DIRECTORY,MODE) if not os.path.exists(DIRECTORY) else None

    # create log file with list of synchronized files (or print to terminal)
    if LOG:
        # format: PGC_ArcticDEM_strip_sync_2002-04-01.log
        today = time.strftime('%Y-%m-%d',time.localtime())
        LOGFILE = f'PGC_ArcticDEM_strip_sync_{today}.log'
        logging.basicConfig(filename=os.path.join(DIRECTORY,LOGFILE),
            level=logging.INFO)
        logging.info(f'PGC ArcticDEM Strip Sync Log ({today})')
        logging.info(f'VERSION={VERSION}')
        logging.info(f'RESOLUTION={RESOLUTION}')
        logging.info('STRIPS={0}'.format(','.join(STRIPS))) if STRIPS else None
    else:
        # standard output (terminal output)
        logging.basicConfig(level=logging.INFO)

    # remote http server for PGC DEM data
    HOST = ['http://data.pgc.umn.edu','elev','dem','setsm']
    # compile regular expression operators for strips
    R1 = re.compile(r'|'.join(STRIPS)) if STRIPS else re.compile(r'[n|s]\d+[e|w]\d+')
    R2 = re.compile((r'(SETSM)_(\w+)_(\d{4})(\d{2})(\d{2})_'
        r'(\w+)_(\w+)_(seg\d+)_(\d+m)_(.*?)\.tar\.gz'))
    # compile regular expression operators for shapefiles
    R3 = re.compile(rf'(.*?)_Strip_Index_({VERSION})_shp\.zip')

    # compile HTML parser for lxml
    parser = lxml.etree.HTMLParser()

    # remote directory for strip version and resolution
    remote_path = [*HOST, 'ArcticDEM', 'strips', VERSION, RESOLUTION]
    # open connection with PGC server at remote directory
    remote_sub,collastmod,_ = gz.utilities.pgc_list(remote_path,
        timeout=TIMEOUT, parser=parser, pattern=R1, sort=True)
    # for each tile subdirectory
    for sd,lmd in zip(remote_sub,collastmod):
        # check if data directory exists and recursively create if not
        local_dir = os.path.join(DIRECTORY,sd)
        if not os.access(local_dir, os.F_OK) and not LIST:
            os.makedirs(local_dir,MODE)
        # open connection with PGC server at remote directory
        remote_path = [*HOST, 'ArcticDEM', 'strips', VERSION, RESOLUTION, sd]
        remote_dir = posixpath.join(*remote_path)
        # read and parse request for files (names and modified dates)
        colnames,collastmod,_ = gz.utilities.pgc_list(remote_path,
            timeout=TIMEOUT, parser=parser, pattern=R2, sort=True)
        # sync each ArcticDEM strip file
        for colname,remote_mtime in zip(colnames,collastmod):
            # remote and local versions of the file
            remote_file = posixpath.join(remote_dir,colname)
            local_file = os.path.join(local_dir,colname)
            # sync ArcticDEM strip tar file
            http_pull_file(remote_file, remote_mtime, local_file,
                TIMEOUT=TIMEOUT, RETRY=RETRY, LIST=LIST,
                CLOBBER=CLOBBER, MODE=MODE)
        # keep remote modification time of directory and local access time
        os.utime(local_dir, (os.stat(local_dir).st_atime, lmd))

    # remote directory for shapefiles of strip version
    remote_path = [*HOST, 'ArcticDEM', 'indexes']
    remote_dir = posixpath.join(*remote_path)
    # read and parse request for files (names and modified dates)
    colnames,collastmod,_ = gz.utilities.pgc_list(remote_path,
        timeout=TIMEOUT, parser=parser, pattern=R3, sort=True)
    # sync each ArcticDEM strip shapefile
    for colname,remote_mtime in zip(colnames,collastmod):
        # remote and local versions of the file
        remote_file = posixpath.join(remote_dir,colname)
        local_file = os.path.join(DIRECTORY,colname)
        # sync ArcticDEM strip shapefile
        http_pull_file(remote_file, remote_mtime, local_file,
            TIMEOUT=TIMEOUT, RETRY=RETRY, LIST=LIST,
            CLOBBER=CLOBBER, MODE=MODE)

    # close log file and set permissions level to MODE
    if LOG:
        os.chmod(os.path.join(DIRECTORY, LOGFILE), MODE)

# PURPOSE: Try downloading a file up to a set number of times
def retry_download(remote_file, local=None, timeout=None,
    retry=1, chunk=0, context=ssl.SSLContext()):
    # attempt to download up to the number of retries
    retry_counter = 0
    while (retry_counter < retry):
        # attempt to retrieve file from https server
        try:
            # Create and submit request.
            # There are a range of exceptions that can be thrown here
            # including HTTPError and URLError.
            request = gz.utilities.urllib2.Request(remote_file)
            response = gz.utilities.urllib2.urlopen(request,
                context=context, timeout=timeout)
            # get the length of the remote file
            remote_length = int(response.headers['content-length'])
            # copy contents to file using chunked transfer encoding
            # transfer should work with ascii and binary data formats
            with open(local, 'wb') as f:
                shutil.copyfileobj(response, f, chunk)
            local_length = os.path.getsize(local)
        except Exception as e:
            logging.error(traceback.format_exc())
            pass
        else:
            # check that downloaded file matches original length
            if (local_length == remote_length):
                break
        # add to retry counter
        retry_counter += 1
    # check if maximum number of retries were reached
    if (retry_counter == retry):
        raise TimeoutError('Maximum number of retries reached')

# PURPOSE: pull file from a remote host checking if file exists locally
# and if the remote file is newer than the local file
def http_pull_file(remote_file, remote_mtime, local_file, TIMEOUT=None,
    RETRY=1, LIST=False, CLOBBER=False, MODE=0o775):
    # chunked transfer encoding size
    CHUNK = 16 * 1024
    # if file exists in file system: check if remote file is newer
    TEST = False
    OVERWRITE = ' (clobber)'
    # check if local version of file exists
    if os.access(local_file, os.F_OK):
        # check last modification time of local file
        local_mtime = os.stat(local_file).st_mtime
        # if remote file is newer: overwrite the local file
        if (remote_mtime > local_mtime):
            TEST = True
            OVERWRITE = ' (overwrite)'
    else:
        TEST = True
        OVERWRITE = ' (new)'
    # if file does not exist locally, is to be overwritten, or CLOBBER is set
    if TEST or CLOBBER:
        # Printing files transferred
        logging.info(f'{remote_file} --> ')
        logging.info(f'\t{local_file}{OVERWRITE}\n')
        # if executing copy command (not only printing the files)
        if not LIST:
            # attempt to retry the download
            retry_download(remote_file, local=local_file,
                timeout=TIMEOUT, retry=RETRY, chunk=CHUNK)
            # keep remote modification time of file and local access time
            os.utime(local_file, (os.stat(local_file).st_atime, remote_mtime))
            os.chmod(local_file, MODE)

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Sync ArcticDEM strip tar files from the Polar
            Geospatial Center (PGC)
            """
    )
    # command line parameters
    # working data directory
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory')
    # ArcticDEM model version
    parser.add_argument('--version','-v',
        type=str, choices=('s2s041',), default='s2s041',
        help='ArcticDEM version')
    # DEM spatial resolution
    parser.add_argument('--resolution','-r',
        type=str, choices=('2m',), default='2m',
        help='ArcticDEM spatial resolution')
    # ArcticDEM strip parameters
    parser.add_argument('--strip','-s',
        type=str, nargs='+',
        help='ArcticDEM strips to sync')
    # connection timeout and number of retry attempts
    parser.add_argument('--timeout','-T',
        type=int, default=120,
        help='Timeout in seconds for blocking operations')
    parser.add_argument('--retry','-R',
        type=int, default=5,
        help='Connection retry attempts')
    # Output log file in form
    # format: PGC_ArcticDEM_strip_sync_2002-04-01.log
    parser.add_argument('--log','-l',
        default=False, action='store_true',
        help='Output log file')
    # sync options
    parser.add_argument('--list','-L',
        default=False, action='store_true',
        help='Only print files that could be transferred')
    parser.add_argument('--clobber','-C',
        default=False, action='store_true',
        help='Overwrite existing data in transfer')
    # permissions mode of the local directories and files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of directories and files synced')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # check internet connection before attempting to run program
    # attempt to connect to public http Polar Geospatial Center host
    HOST = posixpath.join('http://data.pgc.umn.edu','elev','dem')
    if gz.utilities.check_connection(HOST):
        pgc_arcticdem_strip_sync(args.directory, args.version, args.resolution,
            STRIPS=args.strip, TIMEOUT=args.timeout, RETRY=args.retry,
            LIST=args.list, LOG=args.log, CLOBBER=args.clobber,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
