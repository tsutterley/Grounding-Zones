#!/usr/bin/env python
u"""
gee_pgc_strip_sync.py
Written by Tyler Sutterley (06/2022)

Processes and syncs Reference Elevation Map of Antarctica (REMA) DEM
    or ArcticDEM strip tar files from Google Earth Engine

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
    -R X, --restart X: Indice for restarting PGC DEM sync
    -A X, --active X: Number of currently active tasks allowed
    -M, --matchtag: Output matchtag raster files
    -I, --index: Output index shapefiles

PYTHON DEPENDENCIES:
    ee: Python bindings for calling the Earth Engine API
        https://github.com/google/earthengine-api
    dateutil: powerful extensions to datetime
        https://dateutil.readthedocs.io/en/stable/

UPDATE HISTORY:
    Updated 06/2022: added restart and bbox command line options
        add task limiter to prevent maximum active worker stoppages
        changed temporal filter to start time and end time
    Written 05/2022
"""
from __future__ import print_function
import argparse
import time
import logging
import dateutil.parser
import ee

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
    START=0,
    LIMIT=3000,
    SCALE=None,
    MATCHTAG=False,
    INDEX=False):

    # initialize Google Earth Engine API
    ee.Initialize()
    # standard logging output
    logging.basicConfig(level=logging.INFO)
    # format model version and scale
    VERSION = version[:2].upper()
    if not SCALE:
        SCALE = int(resolution[:-1])
    # image collection with PGC DEM strip data
    collection = ee.ImageCollection(f'UMN/PGC/{model}/{VERSION}/{resolution}')
    # reduce image collection to temporal range
    if TIME is not None:
        #-- verify that start and end time are in ISO format
        start_time = dateutil.parser.parse(TIME[0]).isoformat()
        end_time = dateutil.parser.parse(TIME[1]).isoformat()
        collection = collection.filterDate(start_time, end_time)
    # reduce image collection to spatial bounding box
    if BOUNDS is not None:
        collection = collection.filterBounds(ee.Geometry.BBox(*BOUNDS))
    # number of images in filtered image collection
    n_images = collection.size().getInfo()
    # create list from filtered image collection
    collection_list = collection.toList(n_images)
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
                'folder': f'{model}_{SCALE}m',
                'formatOptions': {'cloudOptimized': True}
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
                'folder': f'{model}_{SCALE}m',
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
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args, _ = parser.parse_known_args()
    # run Google Earth Engine sync
    gee_pgc_strip_sync(args.model, args.version, args.resolution,
        TIME=args.time,
        BOUNDS=args.bbox,
        START=args.restart,
        LIMIT=args.limit,
        SCALE=args.scale,
        MATCHTAG=args.matchtag,
        INDEX=args.index)

# run main program
if __name__ == '__main__':
    main()
