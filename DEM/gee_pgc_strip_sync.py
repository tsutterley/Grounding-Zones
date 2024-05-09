#!/usr/bin/env python
u"""
gee_pgc_strip_sync.py
Written by Tyler Sutterley (05/2024)

Processes and syncs Reference Elevation Map of Antarctica (REMA) DEM
    or ArcticDEM strip tar files from Google Earth Engine
Can resample the DEM to a specified spatial scale and calculate the
    standard deviation of the image at the resampled pixel size

CALLING SEQUENCE:
    python gee_pgc_strip_sync.py --model REMA --scale 32

COMMAND LINE OPTIONS:
    --help: list the command line options
    -m X, --model: PGC digital elevation model
        ArcticDEM
        REMA (default)
    -v X, --version X: DEM version
        v1.0 (default)
        v3.0
    -r X, --resolution X: DEM spatial resolution
        8m
        2m (default)
    -S X, --scale X: Output spatial resolution (resampled)
    -T X, --time X: Time range for reducing image collection
    -B X, --bbox X: Bounding box for reducing image collection
    -P X, --polygon X: Georeferenced file for reducing image collection
    -R X, --restart X: Indice for restarting PGC DEM sync
    -A X, --active X: Number of currently active tasks allowed
    -M, --matchtag: Output matchtag raster files
    -I, --index: Output index shapefiles
    -D X, --directory X: Output Directory on Google Drive
    -c, --cloud-optimized: Output as cloud-optimized geotiffs (COGs)

PYTHON DEPENDENCIES:
    ee: Python bindings for calling the Earth Engine API
        https://github.com/google/earthengine-api
    dateutil: powerful extensions to datetime
        https://dateutil.readthedocs.io/en/stable/

UPDATE HISTORY:
    Updated 05/2024: use wrapper to importlib for optional dependencies
    Updated 07/2023: using pathlib to define and operate on paths
    Updated 07/2022: made COG output optional and not the default
        place some imports within try/except statements
        added option to use a georeferenced polygon file
        added option to specify the output folder on Google Drive
    Updated 06/2022: added restart and bbox command line options
        add task limiter to prevent maximum active worker stoppages
        changed temporal filter to start time and end time
    Written 05/2022
"""
from __future__ import print_function

import sys
import os
import time
import logging
import pathlib
import argparse
import dateutil.parser
import grounding_zones as gz

# attempt imports
fiona = gz.utilities.import_dependency('fiona')
ee = gz.utilities.import_dependency('ee')

# PURPOSE: keep track of threads
def info(args):
    logging.info(pathlib.Path(sys.argv[0]).name)
    logging.info(args)
    logging.info(f'module name: {__name__}')
    if hasattr(os, 'getppid'):
        logging.info(f'parent process: {os.getppid():d}')
    logging.info(f'process id: {os.getpid():d}')

# PURPOSE: get number of currently pending or running tasks
def current_tasks():
    """
    Get number of currently pending or running tasks in
    Google Earth Engine
    """
    # list of tasks from Earth Engine Task Manager
    tasks = ee.data.listOperations()
    # reduce tasks to pending or running
    queue = [t for t in tasks if t['metadata']['state'] in
        ('PENDING', 'READY', 'RUNNING')]
    return len(queue)

# PURPOSE: limits number of currently active tasks
def task_limiter(tasks, limit=3000, wait=5):
    """
    Limits number of currently active tasks in Google Earth Engine

    Parameters
    ----------
    tasks: list
        Tasks to be submitted to Google Earth Engine
    limit: int, default 3000
        Number of queued tasks allowed by the GEE batch system
    wait: int, default 5
        Number of seconds to wait before retrying submission
    """
    n_submit = len(tasks)
    while True:
        # get the number of current tasks
        n_queue = current_tasks()
        # check if number of tasks is under the limit
        if ((n_queue + n_submit) < limit):
            # start each task
            for task in tasks:
                task.start()
            return
        else:
            # wait while other tasks are currently active
            time.sleep(wait)

# PURPOSE: sync local PGC DEM strip files with Google Earth Engine
def gee_pgc_strip_sync(model, version, resolution,
    TIME=None,
    BOUNDS=None,
    POLYGON=None,
    START=0,
    LIMIT=3000,
    SCALE=None,
    MATCHTAG=False,
    INDEX=False,
    FOLDER=None,
    COG=False):

    # initialize Google Earth Engine API
    ee.Initialize()
    # format model version and scale
    VERSION = version[:2].upper()
    if not SCALE:
        SCALE = int(resolution[:-1])
    # default folder for output on Google Drive
    if FOLDER is None:
        FOLDER = f'{model}_{SCALE}m'

    # image collection with PGC DEM strip data
    collection = ee.ImageCollection(f'UMN/PGC/{model}/{VERSION}/{resolution}')
    # reduce image collection to temporal range
    if TIME is not None:
        # verify that start and end time are in ISO format
        start_time = dateutil.parser.parse(TIME[0]).isoformat()
        end_time = dateutil.parser.parse(TIME[1]).isoformat()
        logging.info(f'Start Time: {start_time}')
        logging.info(f'End Time: {end_time}')
        collection = collection.filterDate(start_time, end_time)
    # reduce image collection to spatial bounding box
    if BOUNDS is not None:
        BBOX = ','.join(map(str,BOUNDS))
        logging.info(f'Bounding Box: {BBOX}')
        collection = collection.filterBounds(ee.Geometry.BBox(*BOUNDS))
    # reduce image collection to spatial geometry
    if POLYGON is not None:
        # read georeferenced file
        POLYGON = pathlib.Path(POLYGON).expanduser().absolute()
        logging.info(f'Georeferenced File: {str(POLYGON)}')
        shape = fiona.open(str(POLYGON))
        # convert input polygons into a list of geometries
        polys = [ee.Geometry(rec['geometry'], shape.crs['init'])
            for i,rec in enumerate(shape)]
        # convert into a single multipolygon and reduce collection
        geometry = ee.Geometry.MultiPolygon(polys)
        collection = collection.filterBounds(geometry)
    # number of images in filtered image collection
    n_images = collection.size().getInfo()
    # create list from filtered image collection
    collection_list = collection.toList(n_images)
    # track the number of images to process
    logging.info(f'Number of Images: {n_images:d}')

    # for each image in the list
    for i in range(START,n_images):
        # get image from collection
        img = ee.Image(collection_list.get(i))
        granule = img.id().getInfo()
        # log granule
        logging.info(granule)
        properties = img.getInfo()['properties']
        # create task list for granule
        tasks = []
        # if reducing resolution: output standard deviation band
        if (SCALE > int(resolution[:-1])):
            # get elevation geotiff
            elev = img.select('elevation')
            # calculate DEM standard deviation
            maxPixels = (SCALE//int(resolution[:-1]))**2
            stdev = elev.reduceResolution(ee.Reducer.sampleStdDev(),
                maxPixels=maxPixels)
            # combine elevation and standard deviation
            image = ee.Image.cat([elev, stdev.float()])
        else:
            # get elevation geotiff
            image = img.select('elevation')
        # scale and reduce resolution
        # export to google drive
        task = ee.batch.Export.image.toDrive(**{
            'image': image,
            'description': f'{granule}_{SCALE}m_{version}_dem',
            'scale': SCALE,
            'fileFormat': 'GeoTIFF',
            'folder': FOLDER,
            'formatOptions': {'cloudOptimized': COG}
        })
        # add to task list
        tasks.append(task)
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
                'folder': FOLDER,
                'formatOptions': {'cloudOptimized': COG}
            })
            # add to task list
            tasks.append(task)
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
                'folder': FOLDER,
            })
            # add to task list
            tasks.append(task)
        # attempt to run tasks
        task_limiter(tasks, limit=LIMIT)

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
    parser.add_argument('--scale', '-S',
        type=int, default=32,
        help='Output spatial resolution')
    # PGC DEM strip parameters
    # temporal range for reducing image collection
    parser.add_argument('--time','-T',
        type=str, nargs=2, metavar=('start_time','end_time'),
        help='Temporal range for reducing image collection')
    # bounding box for reducing image collection
    parser.add_argument('--bbox','-B',
        type=float, nargs=4,
        metavar=('lon_min','lat_min','lon_max','lat_max'),
        help='Bounding box for reducing image collection')
    parser.add_argument('--polygon','-P',
        type=pathlib.Path,
        help='Georeferenced file for reducing image collection')
    # restart sync at a particular image
    parser.add_argument('--restart', '-R',
        type=int, default=0,
        help='Indice for restarting PGC DEM sync')
    # limit number of currently active tasks
    parser.add_argument('--limit', '-L',
        type=int, default=3000,
        help='Number of currently active tasks allowed')
    # output matchtag raster files
    parser.add_argument('--matchtag','-M',
        default=False, action='store_true',
        help='Output PGC DEM matchtag raster files')
    # output index shapefiles
    parser.add_argument('--index','-I',
        default=False, action='store_true',
        help='Output PGC DEM index shapefiles')
    # output directory on Google Drive
    parser.add_argument('--directory','-D',
        type=str, help='Output Directory on Google Drive')
    # output as cloud-optimized geotiffs (COGs)
    parser.add_argument('--cloud-optimized','-c',
        default=False, action='store_true',
        help='Output as cloud-optimized geotiffs (COGs)')
    # print information about processing run
    parser.add_argument('--verbose','-V',
        action='count', default=0,
        help='Verbose output of processing run')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args, _ = parser.parse_known_args()
    # create logger
    loglevels = [logging.CRITICAL, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=loglevels[args.verbose])
    # log parameters
    info(args)
    # run Google Earth Engine sync
    gee_pgc_strip_sync(args.model, args.version, args.resolution,
        TIME=args.time,
        BOUNDS=args.bbox,
        POLYGON=args.polygon,
        START=args.restart,
        LIMIT=args.limit,
        SCALE=args.scale,
        MATCHTAG=args.matchtag,
        INDEX=args.index,
        FOLDER=args.directory,
        COG=args.cloud_optimized)

# run main program
if __name__ == '__main__':
    main()
