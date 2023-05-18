#!/usr/bin/env python
u"""
aviso_dac_sync.py
Written by Tyler Sutterley (04/2023)

Syncs the dynamic atmospheric correction (DAC) from AVISO
    https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/
        atmospheric-corrections.html
    https://www.aviso.altimetry.fr/en/data/data-access.html

CALLING SEQUENCE:
    python aviso_dac_sync.py --user <username> --year 2017
    where <username> is your AVISO data dissemination server username

COMMAND LINE OPTIONS:
    --help: list the command line options
    -U X, --user: username for AVISO FTP servers (email)
    -P X, --password: password for AVISO FTP servers
    -N X, --netrc X: path to .netrc file for alternative authentication
    -Y X, --year X: years to sync
    -D X, --directory X: working data directory
    -t X, --timeout X: Timeout in seconds for blocking operations
    -l, --log: output log of files downloaded
    -L, --list: print files to be transferred, but do not execute transfer
    -C, --clobber: Overwrite existing data in transfer
    -M X, --mode X: Local permissions mode of the directories and files synced

UPDATE HISTORY:
    Updated 04/2023: using pathlib to define and expand paths
    Updated 12/2022: single implicit import of grounding zone tools
    Updated 11/2022: use f-strings for formatting verbose or ascii output
    Updated 02/2022: using argparse to set command line parameters
        use logging for verbose and log output
    Updated 05/2019: new authenticated ftp host (changed 2018-05-31)
    Written 09/2017
"""
from __future__ import print_function

import sys
import re
import os
import netrc
import getpass
import logging
import pathlib
import argparse
import builtins
import posixpath
import calendar, time
import ftplib
import grounding_zones as gz

# PURPOSE: sync local AVISO DAC files with ftp server
def aviso_dac_sync(DIRECTORY,
    USER=None,
    PASSWORD=None,
    YEAR=None,
    TIMEOUT=None,
    LOG=False,
    LIST=False,
    MODE=None,
    CLOBBER=False):

    # connect and login to AVISO ftp server
    ftp = ftplib.FTP('ftp-access.aviso.altimetry.fr', timeout=TIMEOUT)
    ftp.login(USER, PASSWORD)

    # output directory
    DIRECTORY = pathlib.Path(DIRECTORY).expanduser().absolute()
    DIRECTORY.mkdir(mode=MODE, parents=True, exist_ok=True)

    # output of synchronized files
    if LOG:
        # format: AVISO_DAC_sync_2002-04-01.log
        today = time.strftime('%Y-%m-%d',time.localtime())
        LOGFILE = DIRECTORY.joinpath(f'AVISO_DAC_sync_{today}.log')
        logging.basicConfig(filename=LOGFILE, level=logging.INFO)
        logging.info(f'AVISO DAC Sync Log ({today})')

    else:
        # standard output (terminal output)
        logging.basicConfig(level=logging.INFO)

    # compile regular expression operator for years to sync
    regex_years = r'|'.join(rf'{y:d}' for y in YEAR) if YEAR else r'\d+'
    R1 = re.compile(rf'({regex_years})', re.VERBOSE)
    # compile regular expression pattern for finding files
    R2 = re.compile(r'dac_dif_(\d+)_(\d+).nc.bz2$', re.VERBOSE)

    # find remote yearly directories for DAC
    YEARS,_ = gz.utilities.ftp_list(
        [ftp.host,'auxiliary','dac','dac_delayed_global'],
        username=USER, password=PASSWORD, timeout=TIMEOUT,
        basename=True, pattern=R1, sort=True)
    for Y in YEARS:
        # remote and local directory for data product of year
        # check if local directory exists and recursively create if not
        DIRECTORY.joinpath(Y).mkdir(mode=MODE, exist_ok=True)
        # get filenames from remote directory
        remote_files,remote_mtimes = gz.utilities.ftp_list(
            [ftp.host,'auxiliary','dac','dac_delayed_global',Y],
            username=USER, password=PASSWORD, timeout=TIMEOUT,
            basename=True, pattern=R2, sort=True)
        for fi,remote_mtime in zip(remote_files,remote_mtimes):
            # extract filename from regex object
            remote_path = [ftp.host,'auxiliary','dac','dac_delayed_global',Y,fi]
            local_file = DIRECTORY.joinpath(Y,fi)
            ftp_mirror_file(ftp, remote_path, remote_mtime, local_file,
                LIST=LIST, CLOBBER=CLOBBER, MODE=MODE)
    # close the ftp connection
    ftp.quit()
    # close log file and set permissions level to MODE
    if LOG:
        LOGFILE.chmod(mode=MODE)

# PURPOSE: pull file from a remote host checking if file exists locally
# and if the remote file is newer than the local file
def ftp_mirror_file(ftp, remote_path, remote_mtime, local_file,
    LIST=False, CLOBBER=False, MODE=0o775):
    # if file exists in file system: check if remote file is newer
    TEST = False
    OVERWRITE = ' (clobber)'
    # check if local version of file exists
    local_file = pathlib.Path(local_file).expanduser
    if local_file.exists():
        # check last modification time of local file
        local_mtime = local_file.stat().st_mtime
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
                ftp.retrbinary(f'RETR {remote_file}', f.write)
            # keep remote modification time of file and local access time
            os.utime(local_file, (local_file.stat().st_atime, remote_mtime))
            local_file.chmod(mode=MODE)

# PURPOSE: create argument parser
def arguments():
    # Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Syncs the dynamic atmospheric correction
            (DAC) from AVISO
            """
    )
    # command line parameters
    # AVISO ftp credentials
    parser.add_argument('--user','-U',
        type=str, default=os.environ.get('AVISO_USERNAME'),
        help='Username for AVISO Login')
    parser.add_argument('--password','-W',
        type=str, default=os.environ.get('AVISO_PASSWORD'),
        help='Password for AVISO Login')
    parser.add_argument('--netrc','-N',
        type=pathlib.Path, default=pathlib.Path().home().joinpath('.netrc'),
        help='Path to .netrc file for authentication')
    # working data directory
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=pathlib.Path.cwd(),
        help='Working data directory')
    # years of data to sync
    parser.add_argument('--year','-Y',
        type=int, nargs='+',
        help='Years to sync')
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

    # AVISO ftp hostname
    HOST = 'ftp-access.aviso.altimetry.fr'
    # get authentication
    if not args.user and not args.netrc.exists():
        # check that AVISO credentials were entered
        args.user = builtins.input(f'Username for {HOST}: ')
        # enter password securely from command-line
        args.password = getpass.getpass(f'Password for {args.user}@{HOST}: ')
    elif args.netrc.exists():
        args.user,_,args.password = netrc.netrc(args.netrc).authenticators(HOST)
    elif args.user and not args.password:
        # enter password securely from command-line
        args.password = getpass.getpass(f'Password for {args.user}@{HOST}: ')

    # check AVISO credentials before attempting to run program
    if gz.utilities.check_ftp_connection(HOST,
            username=args.user, password=args.password):
        # run AVISO sync program
        aviso_dac_sync(args.directory,
            USER=args.user,
            PASSWORD=args.password,
            YEAR=args.year,
            TIMEOUT=args.timeout,
            LOG=args.log,
            LIST=args.list,
            MODE=args.mode,
            CLOBBER=args.clobber)

# run main program
if __name__ == '__main__':
    main()
