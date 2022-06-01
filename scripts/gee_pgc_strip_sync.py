#!/usr/bin/env python
u"""
gee_pgc_strip_sync.py
Written by Tyler Sutterley (05/2022)

Syncs Reference Elevation Map of Antarctica (REMA) DEM or ArcticDEM
    strip tar files from Google Earth Engine

CALLING SEQUENCE:
    python gee_pgc_strip_sync.py --model REMA

COMMAND LINE OPTIONS:
    --help: list the command line options
    -m X, --model: PGC digital elevation model
        ArcticDEM
        REMA
    -v X, --version X: DEM version
        v1.0 (default)
        v3.0
    -r X, --resolution X: DEM spatial resolution
        8m
        2m (default)
    -S X, --scale X: Output spatial resolution (resampled)
        default is the same as the original strip
    -Y X, --year X: Year of DEM strips to sync (default=All)
    -M, --matchtag: Output matchtag raster files
    -I, --index: Output index shapefiles

PYTHON DEPENDENCIES:
    ee: Python bindings for calling the Earth Engine API
        https://github.com/google/earthengine-api

UPDATE HISTORY:
    Written 05/2022
"""
from __future__ import print_function

import ee
import logging
import argparse

# PURPOSE: sync local PGC DEM strip files with Google Earth Engine
def gee_pgc_strip_sync(model, version, resolution,
    YEARS=None,
    SCALE=None,
    MATCHTAG=False,
    INDEX=False):

    # initialize Google Earth Engine API
    ee.Initialize()
    #-- standard logging output
    logging.basicConfig(level=logging.INFO)
    # format version and scale
    VERSION = version[:2].upper()
    if not SCALE:
        SCALE = int(resolution[:-1])
    # image collection with PGC DEM strip data
    collection = ee.ImageCollection(f'UMN/PGC/{model}/{VERSION}/{resolution}')
    # for each year of strip data
    for _, year in enumerate(YEARS):
        # reduce image collection to year
        filtered = collection.filterDate(f'{year}-01-01', f'{year}-12-31')
        n_images = filtered.size().getInfo()
        # convert image collection to list
        collection_list = filtered.toList(n_images)
        # for each image in the list
        for i in range(n_images):
            img = ee.Image(collection_list.get(i))
            granule = img.id().getInfo()
            # log granule
            logging.info(granule)
            properties = img.getInfo()['properties']
            # get elevation geotiff
            elev = img.select('elevation')
            # calculate DEM standard deviation
            maxPixels = (SCALE//int(resolution[:-1]))**2
            stdev = elev.reduceResolution(ee.Reducer.sampleStdDev(),
                maxPixels=maxPixels)
            # combine elevation and standard deviation
            image = ee.Image.cat([elev, stdev.float()])
            # scale and reduce resolution
            # export to google drive
            task = ee.batch.Export.image.toDrive(**{
                'image': image,
                'description': f'{granule}_{SCALE}m_{version}_dem',
                'scale': SCALE,
                'fileFormat': 'GeoTIFF',
                'folder': f'{model}_{SCALE}m',
                'formatOptions': {'cloudOptimized': True}
            })
            task.start()

            # output the DEM matchtag raster file
            if MATCHTAG:
                # get matchtag geotiff
                matchtag = img.select('matchtag')
                # export to google drive
                task = ee.batch.Export.image.toDrive(**{
                    'image': matchtag,
                    'description': f'{granule}_{SCALE}m_{version}_matchtag',
                    'scale': SCALE,
                    'fileFormat': 'GeoTIFF',
                    'folder': f'{model}_{SCALE}m',
                    'formatOptions': {'cloudOptimized': True}
                })
                task.start()

            # output the DEM index shapefiles
            if INDEX:
                # get coordinates of footprint linear ring
                coordinates = ee.Geometry.LinearRing(
                    properties['system:footprint']['coordinates'])
                features = ee.FeatureCollection([coordinates])
                # export to google drive
                task = ee.batch.Export.table.toDrive(**{
                    'collection': features,
                    'description': f'{granule}_{SCALE}m_{version}_index',
                    'fileFormat': 'SHP',
                    'folder': f'{model}_{SCALE}m',
                })
                task.start()

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Syncs Reference Elevation Map of Antarctica
            (REMA) DEM or ArcticDEM strip tar files from Google
            Earth Engine
            """
    )
    # command line parameters
    # DEM model
    parser.add_argument('--model', '-m',
        type=str, choices=('ArcticDEM','REMA'), default='REMA',
        help='PGC digital elevation model (DEM)')
    # DEM model version
    parser.add_argument('--version', '-v',
        type=str, choices=('v1.0','v3.0'), default='v1.0',
        help='PGC DEM version')
    # DEM spatial resolution
    parser.add_argument('--resolution', '-r',
        type=str, choices=('2m','8m'), default='2m',
        help='PGC DEM spatial resolution')
    # output spatial resolution
    # default is the same as the original DEM strip
    parser.add_argument('--scale', '-S',
        type=int, help='Output spatial resolution')
    # PGC DEM strip parameters
    parser.add_argument('--year', '-Y',
        type=int, nargs='+', default=range(2014, 2018),
        help='Years of PGC DEM strips to sync')
    # output matchtag raster files
    parser.add_argument('--matchtag','-M',
        default=False, action='store_true',
        help='Output PGC DEM matchtag raster files')
    # output index shapefiles
    parser.add_argument('--index','-I',
        default=False, action='store_true',
        help='Output PGC DEM index shapefiles')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args, _ = parser.parse_known_args()
    # run Google Earth Engine sync
    gee_pgc_strip_sync(args.model, args.version, args.resolution,
        YEARS=args.year,
        SCALE=args.scale,
        MATCHTAG=args.matchtag,
        INDEX=args.index)

# run main program
if __name__ == '__main__':
    main()
