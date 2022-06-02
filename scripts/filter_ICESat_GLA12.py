#!/usr/bin/env python
u"""
filter_ICESat_GLA12.py
Written by Tyler Sutterley (05/2022)
Calculates quality summary flags for ICESat/GLAS L2 GLA12
    Antarctic and Greenland Ice Sheet elevation data

INPUTS:
    input_file: ICESat GLA12 data file

COMMAND LINE OPTIONS:
    --help: list the command line options
    --IceSVar: criteria for standard deviation of gaussian fit
    --gval_rcv: riteria for unscaled gain value
    --reflctUC: criteria for reflectivity
    --numPk: criteria for number of peaks in waveform
    -V, --verbose: Output information about each created file
    -M X, --mode X: Permission mode of files created

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/

REFERENCES:
I. M. Howat, B. E. Smith, I. R. Joughin, and T. A. Scambos,
    "Rates of southeast Greenland ice volume loss from combined
    ICESat and ASTER observations", Geophysical Research Letters,
    35(17), L17505, (2008). https://doi.org/10.1029/2008GL034496

H. D. Pritchard, S. R. M. Ligtenberg, H. A. Fricker, D. G. Vaughan,
    M. R. van den Broeke, L. Padman, "Antarctic ice-sheet loss
    driven by basal melting of ice shelves", Nature, 484(7395),
    502-505, (2012). https://doi.org/10.1038/nature10968

B. E. Smith, C. R. Bentley, and C. F. Raymond, "Recent elevation
    changes on the ice streams and ridges of the Ross Embayment
    from ICESat crossovers", Geophysical Research Letters,
    32(25), L21S09, (2005). https://doi.org/10.1029/2005GL024365

L. S. Sorensen, S. B. Simonsen, K. Nielsen, P. Lucas-Picher,
    G. Spada, G. Adalgeirsdottir, R. Forsberg, and C. S. Hvidberg,
    "Mass balance of the Greenland ice sheet (2003--2008) from
    ICESat data -- the impact of interpolation, sampling and firn
    density", The Cryosphere, 5(1), 173-186, (2011).
    https://doi.org/10.5194/tc-5-173-2011

UPDATE HISTORY:
    Updated 05/2022: use argparse descriptions within documentation
    Forked 02/2022 from icesat_glas_correct.py
"""
from __future__ import print_function

import sys
import os
import re
import h5py
import logging
import argparse
import numpy as np

#-- PURPOSE: Calculates quality summary flags for ICESat/GLAS L2 GLA12
#-- Antarctic and Greenland Ice Sheet elevation data
def filter_ICESat_GLA12(INPUT_FILE,
    IceSVar=0.03,
    gval_rcv=200,
    reflctUC=0.1,
    numPk=1,
    VERBOSE=False,
    MODE=0o775):

    #-- create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    #-- get directory from INPUT_FILE
    logging.info('{0} -->'.format(INPUT_FILE))
    DIRECTORY = os.path.dirname(INPUT_FILE)

    #-- compile regular expression operator for extracting information from file
    rx = re.compile((r'GLAH(\d{2})_(\d{3})_(\d{1})(\d{1})(\d{2})_(\d{3})_'
        r'(\d{4})_(\d{1})_(\d{2})_(\d{4})\.H5'), re.VERBOSE)
    #-- extract parameters from ICESat/GLAS HDF5 file name
    #-- PRD:  Product number (01, 05, 06, 12, 13, 14, or 15)
    #-- RL:  Release number for process that created the product = 634
    #-- RGTP:  Repeat ground-track phase (1=8-day, 2=91-day, 3=transfer orbit)
    #-- ORB:   Reference orbit number (starts at 1 and increments each time a
    #--           new reference orbit ground track file is obtained.)
    #-- INST:  Instance number (increments every time the satellite enters a
    #--           different reference orbit)
    #-- CYCL:   Cycle of reference orbit for this phase
    #-- TRK: Track within reference orbit
    #-- SEG:   Segment of orbit
    #-- GRAN:  Granule version number
    #-- TYPE:  File type
    try:
        PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE = rx.findall(INPUT_FILE).pop()
    except:
        #-- output quality summary HDF5 file (generic)
        fileBasename,fileExtension = os.path.splitext(INPUT_FILE)
        OUTPUT_FILE = '{0}_{1}{2}'.format(fileBasename,'MASK',fileExtension)
    else:
        #-- output quality summary HDF5 file for NSIDC granules
        args = (PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        file_format = 'GLAH{0}_{1}_MASK_{2}{3}{4}_{5}_{6}_{7}_{8}_{9}.h5'
        OUTPUT_FILE = file_format.format(*args)

    #-- read GLAH12 HDF5 file
    f = h5py.File(os.path.expanduser(INPUT_FILE), 'r')
    #-- copy variables for outputting to HDF5 file
    IS_gla12_mask = dict(Data_40HZ={})
    IS_gla12_attrs = dict(Data_40HZ={})

    #-- copy global file attributes
    global_attribute_list = ['featureType','title','comment','summary','license',
        'references','AccessConstraints','CitationforExternalPublication',
        'contributor_role','contributor_name','creator_name','creator_email',
        'publisher_name','publisher_email','publisher_url','platform','instrument',
        'processing_level','date_created','spatial_coverage_type','history',
        'keywords','keywords_vocabulary','naming_authority','project','time_type',
        'date_type','time_coverage_start','time_coverage_end',
        'time_coverage_duration','source','HDFVersion','identifier_product_type',
        'identifier_product_format_version','Conventions','institution',
        'ReprocessingPlanned','ReprocessingActual','LocalGranuleID',
        'ProductionDateTime','LocalVersionID','PGEVersion','OrbitNumber',
        'StartOrbitNumber','StopOrbitNumber','EquatorCrossingLongitude',
        'EquatorCrossingTime','EquatorCrossingDate','ShortName','VersionID',
        'InputPointer','RangeBeginningTime','RangeEndingTime','RangeBeginningDate',
        'RangeEndingDate','PercentGroundHit','OrbitQuality','Cycle','Track',
        'Instrument_State','Timing_Bias','ReferenceOrbit','SP_ICE_PATH_NO',
        'SP_ICE_GLAS_StartBlock','SP_ICE_GLAS_EndBlock','Instance','Range_Bias',
        'Instrument_State_Date','Instrument_State_Time','Range_Bias_Date',
        'Range_Bias_Time','Timing_Bias_Date','Timing_Bias_Time',
        'identifier_product_doi','identifier_file_uuid',
        'identifier_product_doi_authority']
    for att in global_attribute_list:
        IS_gla12_attrs[att] = f.attrs[att]
    #-- copy ICESat campaign name from ancillary data
    IS_gla12_attrs['Campaign'] = f['ANCILLARY_DATA'].attrs['Campaign']

    #-- get variables and attributes
    fv = f['Data_40HZ']['Elevation_Surfaces']['d_elev'].attrs['_FillValue']
    rec_ndx_40HZ = f['Data_40HZ']['Time']['i_rec_ndx'][:].copy()
    #-- seconds since 2000-01-01 12:00:00 UTC (J2000)
    DS_UTCTime_40HZ = f['Data_40HZ']['DS_UTCTime_40'][:].copy()
    #-- Latitude (degrees North)
    lat_40HZ = f['Data_40HZ']['Geolocation']['d_lat'][:].copy()
    #-- Longitude (degrees East)
    lon_40HZ = f['Data_40HZ']['Geolocation']['d_lon'][:].copy()
    #-- create combined mask
    elev_TPX = f['Data_40HZ']['Elevation_Surfaces']['d_elev'][:]
    quality_mask = (elev_TPX == fv)
    #-- elevation use flag (0=pass)
    elev_use_flag = f['Data_40HZ']['Quality']['elev_use_flg'][:]
    quality_mask |= (elev_use_flag != 0)
    #-- standard deviation of gaussian fit (IceSVar/LandVar)
    #-- Smith and Howat culled > 0.030
    #-- Pritchard (2012) culled > 0.035
    gauss_fit_flag = f['Data_40HZ']['Elevation_Surfaces']['d_IceSVar'][:]
    #-- ice sheet standard deviation
    quality_mask |= (gauss_fit_flag > IceSVar)
    #-- attitude quality flag (0=good, 50=warning, 100=bad)
    sigma_att_flag = f['Data_40HZ']['Quality']['sigma_att_flg'][:]
    quality_mask |= (sigma_att_flag != 0)
    #-- saturation flag
    #-- 0 = Not Saturated
    #-- 1 = Sat. Correction is Inconsequential
    #-- 2 = Sat. Correction is Applicable
    #-- 3 = Sat. Correction is Not Computable
    #-- 4 = Sat. Correction model is Not Applicable
    sat_corr_flag = f['Data_40HZ']['Quality']['sat_corr_flg'][:]
    quality_mask |= (sat_corr_flag > 2)
    #-- unscaled gain value (Pritchard 2012 culled > 200)
    gain_value_flag = f['Data_40HZ']['Waveform']['i_gval_rcv'][:]
    quality_mask |= (gain_value_flag > gval_rcv)
    #-- number of peaks found in the return echo of gaussian fit
    #-- Sorensen uses == 1, Smith and Howat use == 1
    num_peaks_flag = f['Data_40HZ']['Waveform']['i_numPk'][:]
    quality_mask |= (num_peaks_flag > numPk)
    #-- Reflectivity not corrected for Atmospheric effects
    #-- The atmospheric corrected reflectivity may be calculated from
    #-- this uncorrected reflectivity by multiplying it by d_reflCor_atm
    reflective_flag = f['Data_40HZ']['Reflectivity']['d_reflctUC'][:]
    #-- reflectivity Pritchard culls reflect < 0.1
    #-- valid surface reflectivity (received energy/transmit energy)
    quality_mask |= (num_peaks_flag == reflective_flag)
    quality_mask |= (num_peaks_flag < reflctUC)

    #-- copy attributes for time, geolocation and quality groups
    for var in ['Time','Geolocation','Quality']:
        IS_gla12_mask['Data_40HZ'][var] = {}
        IS_gla12_attrs['Data_40HZ'][var] = {}
        for att_name,att_val in f['Data_40HZ'][var].attrs.items():
            IS_gla12_attrs['Data_40HZ'][var][att_name] = att_val

    #-- J2000 time
    IS_gla12_mask['Data_40HZ']['DS_UTCTime_40'] = DS_UTCTime_40HZ
    IS_gla12_attrs['Data_40HZ']['DS_UTCTime_40'] = {}
    for att_name,att_val in f['Data_40HZ']['DS_UTCTime_40'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_attrs['Data_40HZ']['DS_UTCTime_40'][att_name] = att_val
    #-- record
    IS_gla12_mask['Data_40HZ']['Time']['i_rec_ndx'] = rec_ndx_40HZ
    IS_gla12_attrs['Data_40HZ']['Time']['i_rec_ndx'] = {}
    for att_name,att_val in f['Data_40HZ']['Time']['i_rec_ndx'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_attrs['Data_40HZ']['Time']['i_rec_ndx'][att_name] = att_val
    #-- latitude
    IS_gla12_mask['Data_40HZ']['Geolocation']['d_lat'] = lat_40HZ
    IS_gla12_attrs['Data_40HZ']['Geolocation']['d_lat'] = {}
    for att_name,att_val in f['Data_40HZ']['Geolocation']['d_lat'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_attrs['Data_40HZ']['Geolocation']['d_lat'][att_name] = att_val
    #-- longitude
    IS_gla12_mask['Data_40HZ']['Geolocation']['d_lon'] = lon_40HZ
    IS_gla12_attrs['Data_40HZ']['Geolocation']['d_lon'] = {}
    for att_name,att_val in f['Data_40HZ']['Geolocation']['d_lon'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_attrs['Data_40HZ']['Geolocation']['d_lon'][att_name] = att_val

    #-- close the input HDF5 file
    f.close()

    #-- create quality summary with valid points == 0
    IS_gla12_mask['Data_40HZ']['Quality']['quality_summary'] = \
        np.logical_not(quality_mask).astype('b')

    #-- attributes for quality summary
    IS_gla12_attrs['Data_40HZ']['Quality']['quality_summary'] = {}
    IS_gla12_attrs['Data_40HZ']['Quality']['quality_summary']['long_name'] = \
        "GLA12_Quality_Summary"
    IS_gla12_attrs['Data_40HZ']['Quality']['quality_summary']['units'] = "1"
    IS_gla12_attrs['Data_40HZ']['Quality']['quality_summary']['valid_min'] = 0
    IS_gla12_attrs['Data_40HZ']['Quality']['quality_summary']['valid_max'] = 1
    IS_gla12_attrs['Data_40HZ']['Quality']['quality_summary']['contentType'] = \
        "qualityInformation"
    IS_gla12_attrs['Data_40HZ']['Quality']['quality_summary']['description'] = ("The "
        "quality_summary parameter indicates the best-quality subset of all GLA12 "
        "data. A zero in this parameter implies that no data-quality tests have "
        "found a problem with the elevation and that the data can be corrected for "
        "the effects of atmospheric saturation, a one implies that some potential "
        "problem has been found. This flag can be used for obtaining high-quality "
        "data, but will likely miss a significant fraction of usable data, "
        "particularly in cloudy, rough, or low-surface-reflectance conditions.")
    IS_gla12_attrs['Data_40HZ']['Quality']['quality_summary']['coordinates'] = \
        "DS_UTCTime_40"
    IS_gla12_attrs['Data_40HZ']['Quality']['quality_summary']['flag_meanings'] = \
        "best_quality potential_problem"
    IS_gla12_attrs['Data_40HZ']['Quality']['quality_summary']['flag_values'] = [0,1]

    #-- print file information
    logging.info('\t{0}'.format(OUTPUT_FILE))
    HDF5_GLA12_mask_write(IS_gla12_mask, IS_gla12_attrs,
        FILENAME=os.path.join(DIRECTORY,OUTPUT_FILE),
        CLOBBER=True)
    #-- change the permissions mode
    os.chmod(os.path.join(DIRECTORY,OUTPUT_FILE), MODE)

#-- PURPOSE: outputting the mask values for ICESat data to HDF5
def HDF5_GLA12_mask_write(IS_gla12_tide, IS_gla12_attrs,
    FILENAME='', CLOBBER=False):
    #-- setting HDF5 clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'w-'

    #-- open output HDF5 file
    fileID = h5py.File(os.path.expanduser(FILENAME), clobber)
    #-- create 40HZ HDF5 records
    h5 = dict(Data_40HZ={})

    #-- add HDF5 file attributes
    attrs = {a:v for a,v in IS_gla12_attrs.items() if not isinstance(v,dict)}
    for att_name,att_val in attrs.items():
       fileID.attrs[att_name] = att_val

    #-- create Data_40HZ group
    fileID.create_group('Data_40HZ')
    #-- add HDF5 40HZ group attributes
    for att_name,att_val in IS_gla12_attrs['Data_40HZ'].items():
        if att_name not in ('DS_UTCTime_40',) and not isinstance(att_val,dict):
            fileID['Data_40HZ'].attrs[att_name] = att_val

    #-- add 40HZ time variable
    val = IS_gla12_tide['Data_40HZ']['DS_UTCTime_40']
    attrs = IS_gla12_attrs['Data_40HZ']['DS_UTCTime_40']
    #-- Defining the HDF5 dataset variables
    var = '{0}/{1}'.format('Data_40HZ','DS_UTCTime_40')
    h5['Data_40HZ']['DS_UTCTime_40'] = fileID.create_dataset(var,
        np.shape(val), data=val, dtype=val.dtype, compression='gzip')
    #-- make dimension
    h5['Data_40HZ']['DS_UTCTime_40'].make_scale('DS_UTCTime_40')
    #-- add HDF5 variable attributes
    for att_name,att_val in attrs.items():
        h5['Data_40HZ']['DS_UTCTime_40'].attrs[att_name] = att_val

    #-- for each variable group
    for group in ['Time','Geolocation','Quality']:
        #-- add group to dict
        h5['Data_40HZ'][group] = {}
        #-- create Data_40HZ group
        fileID.create_group('Data_40HZ/{0}'.format(group))
        #-- add HDF5 group attributes
        for att_name,att_val in IS_gla12_attrs['Data_40HZ'][group].items():
            if not isinstance(att_val,dict):
                fileID['Data_40HZ'][group].attrs[att_name] = att_val
        #-- for each variable in the group
        for key,val in IS_gla12_tide['Data_40HZ'][group].items():
            attrs = IS_gla12_attrs['Data_40HZ'][group][key]
            #-- Defining the HDF5 dataset variables
            var = '{0}/{1}/{2}'.format('Data_40HZ',group,key)
            h5['Data_40HZ'][group][key] = fileID.create_dataset(var,
                np.shape(val), data=val, dtype=val.dtype,
                compression='gzip')
            #-- attach dimensions
            for i,dim in enumerate(['DS_UTCTime_40']):
                h5['Data_40HZ'][group][key].dims[i].attach_scale(
                    h5['Data_40HZ'][dim])
            #-- add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5['Data_40HZ'][group][key].attrs[att_name] = att_val

    #-- Closing the HDF5 file
    fileID.close()

#-- PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates quality summary flags for
            ICESat/GLAS L2 GLA12 Antarctic and Greenland Ice
            Sheet elevation data
            """
    )
    #-- command line parameters
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat GLA12 file to run')
    #-- filter flag criteria
    parser.add_argument('--IceSVar',
        type=float, default=0.03,
        help='Criteria for standard deviation of gaussian fit')
    parser.add_argument('--gval_rcv',
        type=float, default=200,
        help='Criteria for unscaled gain value')
    parser.add_argument('--reflctUC',
        type=float, default=0.1,
        help='Criteria for reflectivity')
    parser.add_argument('--numPk',
        type=int, default=1,
        help='Criteria for number of peaks in waveform')
    #-- verbosity settings
    #-- verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Output information about each created file')
    #-- permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of directories and files created')
    #-- return the parser
    return parser

#-- This is the main part of the program that calls the individual functions
def main():
    #-- Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    #-- run for each input GLA12 file
    for FILE in args.infile:
        filter_ICESat_GLA12(FILE, IceSVar=args.IceSVar,
            gval_rcv=args.gval_rcv, reflctUC=args.reflctUC,
            numPk=args.numPk, VERBOSE=args.verbose, MODE=args.mode)

#-- run main program
if __name__ == '__main__':
    main()
