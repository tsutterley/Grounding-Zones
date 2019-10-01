#!/usr/bin/env python
u"""
pgc_rema_sync.py
Written by Tyler Sutterley (01/2021)

Syncs Reference Elevation Map of Antarctica (REMA) DEM tar files
    from the Polar Geospatial Center (PGC)
    http://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic

CALLING SEQUENCE:
    python pgc_rema_sync.py --version v1.1 --resolution 8m

COMMAND LINE OPTIONS:
    --help: list the command line options
    -D X, --directory=X: Working data directory
    -v X, --version=X: REMA DEM version
        v1.0
        v1.1 (default)
    -r X, --resolution X: REMA DEM spatial resolution
        1km
        200m
        100m
        8m (default)
    -t X, --tile X: REMA DEM tiles to sync (default=All)
    -L, --list: print files to be transferred, but do not execute transfer
    -l, --log: output log of files downloaded
    --clobber: Overwrite existing data in transfer
    -M X, --mode=X: Local permissions mode of the directories and files synced

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
    Updated 01/2021: using utilities modules to list from server
        using argparse to set command line options
    Updated 10/2019: added ssl context to urlopen instances
    Updated 09/2019: set last modified time of subdirectories
    Updated 05/2019: sort tile directories of interest.  copy tile shapefiles
    Updated 04/2019: check python version for urllib compatibility with python 2
    Written 04/2019
"""
from __future__ import print_function

import sys
import os
import re
import ssl
import time
import shutil
import argparse
import posixpath
import lxml.etree
import grounding_zones.utilities

#-- PURPOSE: sync local REMA DEM files with PGC public server
def pgc_rema_sync(base_dir, VERSION, RESOLUTION, TILES=None,
    LOG=False, LIST=False, CLOBBER=False, MODE=None):
    #-- data directory
    DIRECTORY = os.path.join(base_dir,'REMA')
    #-- check if directory exists and recursively create if not
    os.makedirs(DIRECTORY,MODE) if not os.path.exists(DIRECTORY) else None

    #-- create log file with list of synchronized files (or print to terminal)
    if LOG:
        #-- format: PGC_REMA_sync_2002-04-01.log
        today = time.strftime('%Y-%m-%d',time.localtime())
        LOGFILE = 'PGC_REMA_sync_{0}.log'.format(today)
        fid = open(os.path.join(DIRECTORY,LOGFILE),'w')
        print('PGC REMA Sync Log ({0})'.format(today), file=fid)
        print('VERSION={0}'.format(VERSION), file=fid)
        print('RESOLUTION={0}'.format(RESOLUTION), file=fid)
        print('TILES={0}'.format(','.join(TILES)), file=fid) if TILES else None
    else:
        #-- standard output (terminal output)
        fid = sys.stdout

    #-- remote http server for PGC DEM data
    HOST = ['http://data.pgc.umn.edu','elev','dem','setsm']
    #-- compile regular expression operators for tiles
    R1 = re.compile(r'|'.join(TILES)) if TILES else re.compile(r'\d+_\d+')
    R2 = re.compile(r'(\d+_\d+)_(.*?)\.tar\.gz')
    #-- compile regular expression operators for shapefiles
    R3 = re.compile(r'(.*?)_Tile_Index_Rel(\d+)\.zip')

    #-- compile HTML parser for lxml
    parser = lxml.etree.HTMLParser()

    #-- remote directory for data version and resolution
    remote_path = HOST + ['REMA','mosaic',VERSION,RESOLUTION]
    #-- open connection with PGC server at remote directory
    remote_sub,collastmod,_ = grounding_zones.utilities.pgc_list(remote_path,
        timeout=20, parser=parser, pattern=R1, sort=True)
    #-- for each tile subdirectory
    for sd,lmd in zip(remote_sub,collastmod):
        #-- check if data directory exists and recursively create if not
        local_dir = os.path.join(DIRECTORY,sd)
        if not os.access(local_dir, os.F_OK) and not LIST:
            os.makedirs(local_dir,MODE)
        #-- open connection with PGC server at remote directory
        remote_path = HOST + ['REMA','mosaic',VERSION,RESOLUTION,sd]
        remote_dir = posixpath.join(*remote_path)
        #-- read and parse request for files (names and modified dates)
        colnames,collastmod,_ = grounding_zones.utilities.pgc_list(remote_path,
            timeout=20, parser=parser, pattern=R2, sort=True)
        #-- sync each REMA DEM data file
        for colname,remote_mtime in zip(colnames,collastmod):
            #-- remote and local versions of the file
            remote_file = posixpath.join(remote_dir,colname)
            local_file = os.path.join(local_dir,colname)
            #-- sync REMA DEM tar file
            http_pull_file(fid, remote_file, remote_mtime, local_file, LIST,
                CLOBBER, MODE)
        #-- keep remote modification time of directory and local access time
        os.utime(local_dir, (os.stat(local_dir).st_atime, lmd))

    #-- remote directory for shapefiles of data version
    remote_path = HOST + ['REMA','indexes']
    remote_dir = posixpath.join(*remote_path)
    #-- read and parse request for files (names and modified dates)
    colnames,collastmod,_ = grounding_zones.utilities.pgc_list(remote_path,
        timeout=20, parser=parser, pattern=R3, sort=True)
    #-- sync each REMA DEM shapefile
    for colname,remote_mtime in zip(colnames,collastmod):
        #-- remote and local versions of the file
        remote_file = posixpath.join(remote_dir,colname)
        local_file = os.path.join(DIRECTORY,colname)
        #-- sync REMA DEM shapefile
        http_pull_file(fid, remote_file, remote_mtime, local_file, LIST,
            CLOBBER, MODE)

    #-- close log file and set permissions level to MODE
    if LOG:
        fid.close()
        os.chmod(os.path.join(DIRECTORY,LOGFILE), MODE)

#-- PURPOSE: pull file from a remote host checking if file exists locally
#-- and if the remote file is newer than the local file
def http_pull_file(fid,remote_file,remote_mtime,local_file,LIST,CLOBBER,MODE):
    #-- if file exists in file system: check if remote file is newer
    TEST = False
    OVERWRITE = ' (clobber)'
    #-- check if local version of file exists
    if os.access(local_file, os.F_OK):
        #-- check last modification time of local file
        local_mtime = os.stat(local_file).st_mtime
        #-- if remote file is newer: overwrite the local file
        if (remote_mtime > local_mtime):
            TEST = True
            OVERWRITE = ' (overwrite)'
    else:
        TEST = True
        OVERWRITE = ' (new)'
    #-- if file does not exist locally, is to be overwritten, or CLOBBER is set
    if TEST or CLOBBER:
        #-- Printing files transferred
        print('{0} --> '.format(remote_file), file=fid)
        print('\t{0}{1}\n'.format(local_file,OVERWRITE), file=fid)
        #-- if executing copy command (not only printing the files)
        if not LIST:
            #-- Create and submit request. There are a wide range of exceptions
            #-- that can be thrown here, including HTTPError and URLError.
            request = grounding_zones.utilities.urllib2.Request(remote_file)
            response = grounding_zones.utilities.urllib2.urlopen(request,
                timeout=20, context=ssl.SSLContext())
            #-- chunked transfer encoding size
            CHUNK = 16 * 1024
            #-- copy contents to local file using chunked transfer encoding
            #-- transfer should work properly with ascii and binary data formats
            with open(local_file, 'wb') as f:
                shutil.copyfileobj(response, f, CHUNK)
            #-- keep remote modification time of file and local access time
            os.utime(local_file, (os.stat(local_file).st_atime, remote_mtime))
            os.chmod(local_file, MODE)

#-- Main program that calls pgc_rema_sync()
def main():
    #-- Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Syncs Reference Elevation Map of Antarctica
            (REMA) DEM tar files from the Polar Geospatial Center (PGC)
            """
    )
    #-- command line parameters
    #-- working data directory
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory')
    #-- REMA DEM model version
    parser.add_argument('--version','-v',
        type=str, choices=('v1.0','v1.1'), default='v1.1',
        help='REMA DEM version')
    #-- DEM spatial resolution
    parser.add_argument('--resolution','-r',
        type=str, choices=('8m','100m','200m','1km'),
        default='8m', help='REMA DEM spatial resolution')
    #-- REMA DEM parameters
    parser.add_argument('--tile','-t',
        type=str, nargs='+',
        help='REMA DEM tiles to sync')
    #-- Output log file in form
    #-- format: PGC_REMA_sync_2002-04-01.log
    parser.add_argument('--log','-l',
        default=False, action='store_true',
        help='Output log file')
    #-- sync options
    parser.add_argument('--list','-L',
        default=False, action='store_true',
        help='Only print files that could be transferred')
    parser.add_argument('--clobber','-C',
        default=False, action='store_true',
        help='Overwrite existing data in transfer')
    #-- permissions mode of the local directories and files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of directories and files synced')
    args = parser.parse_args()

    #-- check internet connection before attempting to run program
    #-- attempt to connect to public http Polar Geospatial Center host
    HOST = posixpath.join('http://data.pgc.umn.edu','elev','dem')
    if grounding_zones.utilities.check_connection(HOST):
        pgc_rema_sync(args.directory, args.version, args.resolution,
            TILES=args.tile, LIST=args.list, LOG=args.log,
            CLOBBER=args.clobber, MODE=args.mode)

#-- run main program
if __name__ == '__main__':
    main()
