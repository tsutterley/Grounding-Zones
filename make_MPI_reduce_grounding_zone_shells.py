#!/usr/bin/env python
u"""
make_MPI_reduce_grounding_zone_shells.py
Written by Tyler Sutterley (07/2019)

Creates a shell script to reduce ICESat-2 data to grounding zone regions

COMMAND LINE OPTIONS:
	-P X, --np=X: Run in parallel with X number of processes
	--directory=X: Working data base directory (default $PYTHONDATA)
	-B X, --buffer=X: Distance in kilometers to buffer from grounding line
	-Y X, --year=X: Years of ICESat-2 data to run
	--product=X: ICESat-2 data product to run
	--release=X: ICESat-2 data release to run
	--version=X: ICESat-2 data version to run
	--granule=X: ICESat-2 granule regions to run
	--track=X: ICESat-2 tracks to run
	-C, --clobber: Overwrite existing tidal elevation files
	-V, --verbose: output MPI rank and size for each job
	-M X, --mode=X: Permission mode of files created
	--shell=X: Output directory for shell script
	-H X, --header=X: Print header at top of shell script
		SLURM: SLURM workload manager
			https://slurm.schedmd.com
		PBS: PBS/Torque resource manager
			http://www.adaptivecomputing.com/products/open-source/torque
		OIB: OIBSERVE module load commands
	--queue=X: submission queue for SLURM and PBS job management software
	--memory=X: memory to allocate for each job (e.g. 36G)
	--walltime=X: time to allocate for total job (e.g. 5-00:00:00 for 5 days)
	--mail-type=X: Notify user via email if event occurs
		valid types: NONE, BEGIN, END, FAIL, REQUEUE, STAGE_OUT, ALL
		Multiple type values may be specified in a comma separated list
	--mail-user=X: email notification of state changes as defined by mail-type

REQUIRES MPI PROGRAM
	MPI: standardized and portable message-passing system
		https://www.open-mpi.org/
		http://mpitutorial.com/

PYTHON DEPENDENCIES:
	numpy: Scientific Computing Tools For Python
		http://www.numpy.org
		http://www.scipy.org/NumPy_for_Matlab_Users
	mpi4py: MPI for Python
		http://pythonhosted.org/mpi4py/
		http://mpi4py.readthedocs.org/en/stable/
	h5py: Python interface for Hierarchal Data Format 5 (HDF5)
		http://h5py.org
	fiona: Python wrapper for vector data access functions from the OGR library
		https://fiona.readthedocs.io/en/latest/manual.html
	shapely: PostGIS-ish operations outside a database context for Python
		http://toblerity.org/shapely/index.html
	pyproj: Python interface to PROJ library
		https://pypi.org/project/pyproj/

UPDATE HISTORY:
	Updated 07/2019: using regex for file versions
	Written 04/2019
"""
from __future__ import print_function

import sys
import os
import re
import h5py
import getopt
import inspect
import datetime
import numpy as np

#-- current file path for the child programs
filename = inspect.getframeinfo(inspect.currentframe()).filename
filepath = os.path.dirname(os.path.abspath(filename))
#-- tilde compress if in HOME directory
if filepath.startswith(os.path.expanduser('~')):
	filepath = filepath.replace(os.path.expanduser('~'),'~')
#-- MPI child program
child_program = {}
child_program['ATL03'] = os.path.join(filepath,'MPI_reduce_ICESat2_ATL03_grounding_zone.py')
child_program['ATL06'] = os.path.join(filepath,'MPI_reduce_ICESat2_ATL06_grounding_zone.py')

#-- compile regular expression operator for finding subdirectories
#-- and extracting date information from the subdirectory
rx = re.compile('(\d+)\.(\d+)\.(\d+)',re.VERBOSE)

#-- PURPOSE: help module to describe the optional input command-line parameters
def usage():
	print('\nHelp: {0}'.format(os.path.basename(sys.argv[0])))
	print(' -P X, --np=X\t\tRun in parallel with X number of processes')
	print(' -D X, --directory=X\tWorking data directory')
	print(' -B X, --buffer=X\tDistance to buffer from grounding line')
	print(' -Y X, --year=X\t\tYears of ICESat-2 data to run')
	print(' --product=X\t\tICESat-2 data product to run')
	print(' --release=X\t\tICESat-2 data release to run')
	print(' --version=X\t\tICESat-2 data version to run')
	print(' --granule=X\t\tICESat-2 granule regions to run')
	print(' --track=X\t\tICESat-2 tracks to run')
	print(' -C, --clobber\t\tOverwrite existing tidal elevation files')
	print(' -V, --verbose\t\tOutput MPI rank and size for each job')
	print(' -M X, --mode=X\t\tPermission mode of files created')
	print(' -S X, --shell=X\tOutput directory for shell script')
	print(' -H X, --header=X\tPrint header at top of shell script')
	print('\tSLURM: SLURM workload manager')
	print('\tPBS: PBS/Torque resource manager')
	print('\tOIB: OIBSERVE module load commands')
	print(' --queue=X\t\tQueue for SLURM and PBS job management software')
	print(' --memory=X\t\tMemory to allocate for each job')
	print(' --walltime=X\t\tTime to allocate for total job')
	print(' --mail-user=X:\t\temail to notify user of state changes as '
		'defined by mail-type')
	print(' --mail-type=X:\t\tNotify user via email if event occurs')
	print('\tValid types: NONE, BEGIN, END, FAIL, REQUEUE, STAGE_OUT, ALL')
	print('\tMultiple type values may be specified in a comma separated list\n')

#-- This is the main part of the program that calls the individual modules
def main():
	#-- Read the system arguments listed after the program
	long_options = ['help','np=','directory=','buffer=','year=','product=',
		'release=','version=','granule=','track=','shell=','clobber','verbose',
		'mode=','header=','memory=','queue=','walltime=','mail-user=','mail-type=']
	optlist,arglist=getopt.getopt(sys.argv[1:],'hP:D:B:Y:S:CVM:H:',long_options)

	#-- command line parameters
	PROCESSES = 1
	#-- working data directory
	base_dir = os.getcwd()
	filename = inspect.getframeinfo(inspect.currentframe()).filename
	filepath = os.path.dirname(os.path.abspath(filename))
	YEARS = np.arange(2018,2019)
	buffer_flag = ''
	PRODUCT = 'ATL06'
	RELEASE = '001'
	VERSIONS = None
	GRANULES = None
	TRACKS = None
	CLOBBER = False
	VERBOSE = False
	verbose_flag = ''
	#-- permissions mode of the output files (number in octal)
	MODE = 0o775
	mode_flag = ''
	print_header = False
	QUEUE = None
	MEMORY = ''
	WALLTIME = ''
	MAIL_USER = None
	MAIL_TYPE = None
	for opt, arg in optlist:
		if opt in ('-h','--help'):
			usage()
			sys.exit()
		elif opt in ("-P","--np"):
			PROCESSES = np.int(arg)
		elif opt in ("--directory"):
			base_dir = os.path.expanduser(arg)
		elif opt in ("-S","--shell"):
			filepath = os.path.expanduser(arg)
		elif opt in ("-Y","--year"):
			YEARS = np.array(arg.split(','),dtype=np.int)
		elif opt in ("-B","--buffer"):
			buffer_flag = ' --buffer={0}'.format(arg)
		elif opt in ("--product"):
			PRODUCT = arg
		elif opt in ("--release"):
			RELEASE = arg
		elif opt in ("--version"):
			VERSIONS = np.array(arg.split(','), dtype=np.int)
		elif opt in ("--granule"):
			GRANULES = np.array(arg.split(','), dtype=np.int)
		elif opt in ("--track"):
			TRACKS = np.sort(arg.split(',')).astype(np.int)
		elif opt in ("-C","--clobber"):
			CLOBBER = True
		elif opt in ("-V","--verbose"):
			VERBOSE = True
			verbose_flag = ' {0}'.format(opt)
		elif opt in ("-M","--mode"):
			mode_flag = ' --mode={0}'.format(arg)
			MODE = int(arg, 8)
		elif opt in ("-H","--header"):
			print_header = arg.upper()
		elif opt in ("--queue"):
			QUEUE = arg
		elif opt in ("--memory"):
			MEMORY = arg.upper()
		elif opt in ("--walltime"):
			WALLTIME = arg
		elif opt in ("--mail-user"):
			MAIL_USER = arg
		elif opt in ("--mail-type"):
			MAIL_TYPE = arg

	#-- shell prefix necessary to run MPI and python
	cloud_flag = ' -mca btl ^openib' if (print_header == 'GCP') else ''
	pyvers = 'python3' if (sys.version_info[0] == 3) else 'python'
	mpi_prefix = 'mpiexec -np {0:d}{1} {2}'.format(PROCESSES,cloud_flag,pyvers)
	#-- create output shell directory if not currently in file system
	if not os.access(filepath, os.F_OK):
		os.mkdir(filepath)

	#-- add date to each output shell file
	date_created = datetime.datetime.today()
	today = date_created.strftime("%Y-%m-%d")

	#-- run all ICESat-2 file subdirectories
	#-- compile regular expression operator for finding subdirectories
	#-- and extracting date information from the subdirectory
	regex_year = '|'.join(['{0:4d}'.format(Y) for Y in sorted(YEARS)])
	rx = re.compile('({0})\.(\d+)\.(\d+)'.format(regex_year),re.VERBOSE)
	#-- lists all items in the ICESat-2 data directory
	#-- checks if each item in list is a directory and a number
	DIRECTORY = os.path.join(base_dir,'icesat2.dir','{0}.dir'.format(PRODUCT))
	subdirectories = [sd for sd in os.listdir(DIRECTORY) if bool(rx.match(sd))
		and os.path.isdir(os.path.join(DIRECTORY, sd))]

	#-- write the output shell script
	args = (PRODUCT,'grounding_zone',today)
	output_shell_script = "reduce_{0}_{1}_{2}.sh".format(*args)
	f = open(os.path.join(filepath,output_shell_script), 'w')
	if (print_header == 'SLURM'):
		#-- add SLURM headers
		print_slurm_header(f, PRODUCT, RELEASE, PROCESSES, MEMORY,
			WALLTIME, QUEUE=QUEUE, MAIL_USER=MAIL_USER, MAIL_TYPE=MAIL_TYPE)
	elif (print_header == 'PBS'):
		#-- add PBS headers
		print_pbs_header(f, PRODUCT, RELEASE, PROCESSES, MEMORY,
			WALLTIME, QUEUE=QUEUE, MAIL_USER=MAIL_USER, MAIL_TYPE=MAIL_TYPE)
	elif (print_header == 'OIB'):
		#-- print module load commands for OIBSERVE
		print_oibserve_header(f)
	#-- for each subdirectory
	for sd in sorted(subdirectories):
		#-- find files to run
		FILES = find_ICESat2_files(base_dir, sd, PRODUCT, RELEASE, VERSIONS,
			GRANULES, TRACKS, VERBOSE=VERBOSE, CLOBBER=CLOBBER)
		#-- print run command for files
		for FILE in FILES:
			#-- tilde compress if in HOME directory
			if FILE.startswith(os.path.expanduser('~')):
				FILE = FILE.replace(os.path.expanduser('~'),'~')
			#-- print command for file
			args = (mpi_prefix, child_program[PRODUCT], buffer_flag,
				verbose_flag, mode_flag, FILE)
			print("{0} {1} {2}{3}{4} {5}".format(*args), file=f)
	#-- close the shell script and change permissions to MODE
	f.close()
	os.chmod(os.path.join(filepath,output_shell_script), MODE)

#-- PURPOSE: find ICESat-2 ATL06 HDF5 files
def find_ICESat2_files(base_dir, SUBDIRECTORY, PRODUCT, RELEASE, VERSIONS,
	GRANULES, TRACKS, VERBOSE=False, CLOBBER=False):
	#-- local directory
	DIRECTORY = os.path.join(base_dir,'icesat2.dir','{0}.dir'.format(PRODUCT),
		SUBDIRECTORY)

	#-- find ICESat-2 HDF5 files in the subdirectory for product and release
	TRACKS = np.arange(1,1388) if not np.any(TRACKS) else TRACKS
	GRANULES = np.arange(1,15) if not np.any(GRANULES) else GRANULES
	regex_track = '|'.join(['{0:04d}'.format(T) for T in TRACKS])
	regex_granule = '|'.join(['{0:02d}'.format(G) for G in GRANULES])
	regex_version = '|'.join(['{0:02d}'.format(V) for V in VERSIONS])
	#-- compile regular expression operator for extracting data from files
	args = (PRODUCT,regex_track,regex_granule,RELEASE,regex_version)
	rx1 = re.compile(('{0}_(\d{{4}})(\d{{2}})(\d{{2}})(\d{{2}})(\d{{2}})'
		'(\d{{2}})_({1})(\d{{2}})({2})_({3})_({4})(.*?).h5$'.format(*args)))
	input_files = [f for f in os.listdir(DIRECTORY) if bool(rx1.match(f))]

	#-- list of ATL06 files
	file_list = []
	output_files = []
	file_format = '{0}_{1}_{2}{3}{4}{5}{6}{7}_{8}{9}{10}_{11}_{12}{13}.h5'
	#-- for each input file
	for f1 in input_files:
		#-- extract base parameters from ICESat-2 HDF5 files
		YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX = rx1.findall(f1).pop()
		#-- check if there is an output file for the input file
		f2 = file_format.format(PRD,'GROUNDING_ZONE_MASK',YY,MM,DD,HH,MN,SS,
			TRK,CYCL,GRAN,RL,VERS,AUX)
		#-- initial status of flags
		TEST = False
		OVERWRITE = 'clobber'
		#-- test if output file exists in file system
		if os.access(os.path.join(DIRECTORY,f2), os.F_OK):
			#-- compare last modified dates of input and output files
			MT1 = os.stat(os.path.join(DIRECTORY,f1)).st_mtime
			MT2 = os.stat(os.path.join(DIRECTORY,f2)).st_mtime
			if np.any(MT1 > MT2):
				TEST = True
				OVERWRITE = 'overwrite'
		else:
			TEST = True
			OVERWRITE = 'new'
		#-- if writing or overwriting
		if TEST or CLOBBER:
			#-- if using verbose output
			print('{0} ({1})'.format(f1, OVERWRITE)) if VERBOSE else None
			file_list.append(os.path.join(DIRECTORY,f1))
	#-- return the list of input files
	return file_list

#-- PURPOSE: add SLURM headers
def print_slurm_header(fid, Y, M, D, PRODUCT, RELEASE, PROCESSES,
	MEMORY, WALLTIME, QUEUE=None, MAIL_USER=None, MAIL_TYPE=None):
	print("#!/bin/bash", file=fid)
	#-- print number of processes
	print("#SBATCH -N{0:d}".format(1), file=fid)
	print("#SBATCH -n{0:d}".format(1), file=fid)
	#-- print required memory
	print("#SBATCH --mem={0}".format(MEMORY), file=fid)
	#-- print wall time
	print("#SBATCH -t{0}".format(WALLTIME), file=fid)
	#-- print queue
	QUEUE = 'c6145' if QUEUE is None else QUEUE
	print("#SBATCH -p {0}".format(QUEUE), file=fid)
	#-- print job name
	args = (PRODUCT,Y,M,D)
	print("#SBATCH --job-name=gz_{0}_{1}-{2}-{3}".format(*args),file=fid)
	#-- print mail-user and mail-type commands
	if MAIL_USER:
		print("#SBATCH --mail-user={0}".format(MAIL_USER), file=fid)
	if MAIL_TYPE:
		print("#SBATCH --mail-type={0}".format(MAIL_TYPE), file=fid)
	#-- print module load commands
	print("module load python/2.7.9", file=fid)
	print("module load compiler/gnu/5.3.0", file=fid)
	print("module load mpi/openmpi/1.10.2/gnu_5.3.0\n", file=fid)

#-- PURPOSE: add PBS headers
def print_pbs_header(f, Y, M, D, PRODUCT, RELEASE, PROCESSES, MEMORY,
	WALLTIME, QUEUE=None, MAIL_USER=None, MAIL_TYPE=None):
	print("#!/bin/bash", file=f)
	#-- print number of processes, required memory and wall time
	args = (MEMORY,WALLTIME)
	print("#PBS -l nodes=1:ppn=1,mem={0},walltime={1}".format(*args),file=f)
	#-- print queue
	QUEUE = 'free64' if QUEUE is None else QUEUE
	print("#PBS -q {0}".format(QUEUE), file=f)
	#-- print job name
	args = (PRODUCT,Y,M,D)
	print("#PBS -N gz_{0}_{1}-{2}-{3}".format(*args),file=f)
	#-- print mail-user and mail-type commands
	if MAIL_USER:
		print("#PBS -M {0}".format(MAIL_USER), file=f)
	if MAIL_TYPE:
		#-- email sent if the job (a) is aborted, (b) begins or (e) ends
		#-- can do any in combination (i.e. abe for all three)
		print("#PBS -m {0}".format(MAIL_TYPE), file=f)
	#-- print module load commands
	print("module load python/2.7.9", file=f)
	print("module load compiler/gnu/5.3.0", file=f)
	print("module load mpi/openmpi/1.10.2/gnu_5.3.0\n", file=f)

#-- PURPOSE: add OIBSERVE headers
def print_oibserve_header(f):
	print("#!/bin/bash", file=f)
	print("module load openmpi-1.10-x86_64", file=f)
	print("module load python/2.7.13\n", file=f)

#-- run main program
if __name__ == '__main__':
	main()
