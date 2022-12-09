#!/usr/bin/env python
u"""
copernicus_sea_level_sync.py
Written by Tyler Sutterley (12/2022)

Syncs sea surface anomalies calculated from AVISO and distributed by the EU
    ftp://my.cmems-du.eu/Core/SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047/
        dataset-duacs-rep-global-merged-allsat-phy-l4

CALLING SEQUENCE:
    python copernicus_sea_level_sync.py --user <username> --year 2017
    where <username> is your Copernicus data dissemination server username

COMMAND LINE OPTIONS:
    --help: list the command line options
    -U X, --user: username for Copernicus FTP servers
    -P X, --password: password for Copernicus FTP servers
    -N X, --netrc X: path to .netrc file for alternative authentication
    -Y X, --year X: years to sync
    -m X, --month X: specific months to sync
    -D X, --directory X: working data directory
    -t X, --timeout X: Timeout in seconds for blocking operations
    -l, --log: output log of files downloaded
    -L, --list: print files to be transferred, but do not execute transfer
    -C, --clobber: Overwrite existing data in transfer
    -M X, --mode X: Local permissions mode of the directories and files synced

UPDATE HISTORY:
    Updated 12/2022: single implicit import of grounding zone tools
    Updated 06/2022: using argparse to set command line parameters
        use logging for verbose and log output
    Updated 12/2018: using new Copernicus ftp server
    Written 09/2017
"""
from __future__ import print_function

import sys
import re
import os
import netrc
import getpass
import logging
import argparse
import builtins
import posixpath
import calendar, time
import ftplib
import grounding_zones as gz

# PURPOSE: sync local Copernicus Sea Level files with ftp server
def copernicus_sea_level_sync(DIRECTORY,
    USER=None,
    PASSWORD=None,
    YEAR=None,
    MONTHS=None,
    TIMEOUT=None,
    LOG=False,
    LIST=False,
    MODE=None,
    CLOBBER=False):

    # connect and login to Copernicus ftp server
    ftp = ftplib.FTP('my.cmems-du.eu')
    ftp.login(USER, PASSWORD)

    # output of synchronized files
    if LOG:
        # format: Copernicus_Sea_Level_sync_2002-04-01.log
        today = time.strftime('%Y-%m-%d',time.localtime())
        LOGFILE = f'Copernicus_Sea_Level_sync_{today}.log'
        logging.basicConfig(filename=os.path.join(DIRECTORY,LOGFILE),
            level=logging.INFO)
        logging.info(f'Copernicus Sea Level Sync Log ({today})')

    else:
        # standard output (terminal output)
        logging.basicConfig(level=logging.INFO)

    # compile regular expression operator for years to sync
    regex_years = r'|'.join(rf'{y:d}' for y in YEAR) if YEAR else r'\d+'
    R1 = re.compile(rf'({regex_years})', re.VERBOSE)
    # compile regular expression pattern for finding files
    if MONTHS:
        regex_months = r'('+r'|'.join(['{0:02d}'.format(m) for m in MONTHS])+r')'
    else:
        regex_months = r'\d{2}'
    # compile regular expression pattern for finding files
    R2 = re.compile((r'dt_global_allsat_phy_l4_(\d{{4}})({0})(\d{{2}})_'
        r'(\d{{4}})(\d{{2}})(\d{{2}}).nc').format(regex_months), re.VERBOSE)

    # path to remote directory for sea level anomalies
    RD = ['Core','SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047',
        'dataset-duacs-rep-global-merged-allsat-phy-l4']
    # find remote yearly directories for sea level anomalies within YEARS
    YEARS,_ = gz.utilities.ftp_list([ftp.host,RD[0],RD[1],RD[2]],
        username=USER, password=PASSWORD, timeout=TIMEOUT,
        basename=True, pattern=R1, sort=True)
    for Y in YEARS:
        # remote and local directory for data product of year
        local_dir = os.path.join(DIRECTORY,Y)
        # check if local directory exists and recursively create if not
        os.makedirs(local_dir,MODE) if not os.path.exists(local_dir) else None
        # get filenames from remote directory
        remote_files,remote_mtimes = gz.utilities.ftp_list(
            [ftp.host,RD[0],RD[1],RD[2],Y],
            username=USER, password=PASSWORD, timeout=TIMEOUT,
            basename=True, pattern=R2, sort=True)
        for fi,remote_mtime in zip(remote_files,remote_mtimes):
            # extract filename from regex object
            remote_path = [ftp.host,RD[0],RD[1],RD[2],Y,fi]
            local_file = os.path.join(local_dir,fi)
            ftp_mirror_file(ftp, remote_path, remote_mtime, local_file,
                LIST=LIST, CLOBBER=CLOBBER, MODE=MODE)
    # close the ftp connection
    ftp.quit()
    # close log file and set permissions level to MODE
    if LOG:
        os.chmod(os.path.join(DIRECTORY,LOGFILE), MODE)

# PURPOSE: pull file from a remote host checking if file exists locally
# and if the remote file is newer than the local file
def ftp_mirror_file(ftp, remote_path, remote_mtime, local_file,
    LIST=False, CLOBBER=False, MODE=0o775):
    # if file exists in file system: check if remote file is newer
    TEST = False
    OVERWRITE = ' (clobber)'
    # check if local version of file exists
    if os.access(local_file, os.F_OK):
        # check last modification time of local file
        local_mtime = os.stat(local_file).st_mtime
        # if remote file is newer: overwrite the local file
        if (gz.utilities.even(remote_mtime) >
            gz.utilities.even(local_mtime)):
            TEST = True
            OVERWRITE = ' (overwrite)'
    else:
        TEST = True
        OVERWRITE = ' (new)'
    # if file does not exist locally, is to be overwritten, or CLOBBER is set
    if TEST or CLOBBER:
        # Printing files transferred
        remote_ftp_url = posixpath.join('ftp://',*remote_path)
        logging.info(f'{remote_ftp_url} -->')
        logging.info(f'\t{local_file}{OVERWRITE}\n')
        # if executing copy command (not only printing the files)
        if not LIST:
            # path to remote file
            remote_file = posixpath.join(*remote_path[1:])
            # copy remote file contents to local file
            with open(local_file, 'wb') as f:
                ftp.retrbinary('RETR {0}'.format(remote_file), f.write)
            # keep remote modification time of file and local access time
            os.utime(local_file, (os.stat(local_file).st_atime, remote_mtime))
            os.chmod(local_file, MODE)

# PURPOSE: create argument parser
def arguments():
    # Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Syncs sea surface anomalies calculated
            from AVISO and distributed by the EU
            """
    )
    # command line parameters
    # Copernicus ftp credentials
    parser.add_argument('--user','-U',
        type=str, default=os.environ.get('COPERNICUS_USERNAME'),
        help='Username for Copernicus Login')
    parser.add_argument('--password','-W',
        type=str, default=os.environ.get('COPERNICUS_PASSWORD'),
        help='Password for Copernicus Login')
    parser.add_argument('--netrc','-N',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.path.join(os.path.expanduser('~'),'.netrc'),
        help='Path to .netrc file for authentication')
    # working data directory
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory')
    # dates of data to sync
    parser.add_argument('--year','-Y',
        type=int, nargs='+',
        help='Years to sync')
    parser.add_argument('--month','-m',
        type=int, nargs='+',
        help='Months of the year to sync')
    # connection timeout
    parser.add_argument('--timeout','-t',
        type=int, default=360,
        help='Timeout in seconds for blocking operations')
    # Output log file
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
        help='permissions mode of output files')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # Copernicus ftp hostname
    HOST = 'my.cmems-du.eu'
    # get authentication
    if not args.user and not os.access(args.netrc,os.F_OK):
        # check that AVISO credentials were entered
        args.user= builtins.input(f'Username for {HOST}: ')
        # enter password securely from command-line
        args.password = getpass.getpass(f'Password for {args.user}@{HOST}: ')
    elif os.access(args.netrc, os.F_OK):
        args.user,_,args.password=netrc.netrc(args.netrc).authenticators(HOST)
    elif args.user and not args.password:
        # enter password securely from command-line
        args.password = getpass.getpass(f'Password for {args.user}@{HOST}: ')

    # check AVISO credentials before attempting to run program
    if gz.utilities.check_ftp_connection(HOST,
            username=args.user,password=args.password):
        copernicus_sea_level_sync(args.directory,
            USER=args.user,
            PASSWORD=args.password,
            YEAR=args.year,
            MONTHS=args.month,
            TIMEOUT=args.timeout,
            LOG=args.log,
            LIST=args.list,
            MODE=args.mode,
            CLOBBER=args.clobber)

# run main program
if __name__ == '__main__':
    main()
