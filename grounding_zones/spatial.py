#!/usr/bin/env python
u"""
spatial.py
Written by Tyler Sutterley (12/2024)

Utilities for reading, writing and operating on spatial data

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

UPDATE HISTORY:
    Updated 12/2024: add latitude and longitude as potential dimension names
    Updated 11/2024: added function to calculate the altitude and azimuth
    Updated 09/2024: deprecation fix case where an array is output to scalars
    Updated 08/2024: changed from 'geotiff' to 'GTiff' and 'cog' formats
        added functions to convert to and from East-North-Up coordinates
    Updated 07/2024: added functions to convert to and from DMS
    Updated 06/2024: added function to write parquet files with metadata
    Updated 05/2024: added function to read from parquet files
        allowing for decoding of the geometry column from WKB
        deprecation update to use exceptions with osgeo osr
    Updated 04/2024: use timescale for temporal operations
        use wrapper to importlib for optional dependencies
    Updated 03/2024: can calculate polar stereographic distortion for distances
    Updated 02/2024: changed class name for ellipsoid parameters to datum
    Updated 10/2023: can read from netCDF4 or HDF5 variable groups
        apply no formatting to columns in ascii file output
    Updated 09/2023: add function to invert field mapping keys and values
        use datetime64[ns] for parsing dates from ascii files
    Updated 08/2023: remove possible crs variables from output fields list
        place PyYAML behind try/except statement to reduce build size
    Updated 05/2023: use datetime parser within pyTMD.time module
    Updated 04/2023: copy inputs in cartesian to not modify original arrays
        added iterative methods for converting from cartesian to geodetic
        allow netCDF4 and HDF5 outputs to be appended to existing files
        using pathlib to define and expand paths
    Updated 03/2023: add basic variable typing to function inputs
    Updated 02/2023: use outputs from constants class for WGS84 parameters
        include more possible dimension names for gridded and drift outputs
    Updated 01/2023: added default field mapping for reading from netCDF4/HDF5
        split netCDF4 output into separate functions for grid and drift types
    Updated 12/2022: add software information to output HDF5 and netCDF4
    Updated 11/2022: place some imports within try/except statements
        added encoding for writing ascii files
        use f-strings for formatting verbose or ascii output
    Updated 10/2022: added datetime parser for ascii time columns
    Updated 06/2022: added field_mapping options to netCDF4 and HDF5 reads
        added from_file wrapper function to read from particular formats
    Updated 04/2022: add option to reduce input GDAL raster datasets
        updated docstrings to numpy documentation format
        use gzip virtual filesystem for reading compressed geotiffs
        include utf-8 encoding in reads to be windows compliant
    Updated 03/2022: add option to specify output GDAL driver
    Updated 01/2022: use iteration breaks in convert ellipsoid function
        remove fill_value attribute after creating netCDF4 and HDF5 variables
    Updated 11/2021: added empty cases to netCDF4 and HDF5 output for crs
        try to get grid mapping attributes from netCDF4 and HDF5
    Updated 10/2021: add pole case in stereographic area scale calculation
        using python logging for handling verbose output
    Updated 09/2021: can calculate height differences between ellipsoids
    Updated 07/2021: added function for determining input variable type
    Updated 03/2021: added polar stereographic area scale calculation
        add routines for converting to and from cartesian coordinates
        replaced numpy bool/int to prevent deprecation warnings
    Updated 01/2021: add streaming from bytes for ascii, netCDF4, HDF5, geotiff
        set default time for geotiff files to 0
    Updated 12/2020: added module for converting ellipsoids
    Updated 11/2020: output data as masked arrays if containing fill values
        add functions to read from and write to geotiff image formats
    Written 09/2020
"""

from __future__ import annotations

import re
import io
import copy
import gzip
import json
import uuid
import logging
import pathlib
import warnings
import datetime
import collections
import numpy as np
import timescale.time
import grounding_zones.version
from grounding_zones.utilities import import_dependency

# attempt imports
osgeo = import_dependency('osgeo')
osgeo.gdal = import_dependency('osgeo.gdal')
osgeo.osr = import_dependency('osgeo.osr')
osgeo.gdalconst = import_dependency('osgeo.gdalconst')
h5py = import_dependency('h5py')
netCDF4 = import_dependency('netCDF4')
pd = import_dependency('pandas')
pyarrow = import_dependency('pyarrow')
pyarrow.parquet = import_dependency('pyarrow.parquet')
shapely = import_dependency('shapely')
shapely.geometry = import_dependency('shapely.geometry')
yaml = import_dependency('yaml')

__all__ = [
    "case_insensitive_filename",
    "from_file",
    "from_ascii",
    "from_netCDF4",
    "from_HDF5",
    "from_geotiff",
    "from_parquet",
    "to_file",
    "to_ascii",
    "to_netCDF4",
    "_drift_netCDF4",
    "_grid_netCDF4",
    "_time_series_netCDF4",
    "to_HDF5",
    "to_geotiff",
    "to_parquet",
    "expand_dims",
    "default_field_mapping",
    "inverse_mapping",
]

def case_insensitive_filename(filename: str | pathlib.Path):
    """
    Searches a directory for a filename without case dependence

    Parameters
    ----------
    filename: str
        input filename
    """
    # check if file presently exists with input case
    filename = pathlib.Path(filename).expanduser().absolute()
    if not filename.exists():
        # search for filename without case dependence
        f = [f.name for f in filename.parent.iterdir() if
            re.match(filename.name, f.name, re.I)]
        # raise error if no file found
        if not f:
            raise FileNotFoundError(str(filename))
        filename = filename.with_name(f.pop())
    # return the matched filename
    return filename

def from_file(filename: str, format: str, **kwargs):
    """
    Wrapper function for reading data from an input format

    Parameters
    ----------
    filename: str
        full path of input file
    format: str
        format of input file
    **kwargs: dict
        Keyword arguments for file reader
    """
    # read input file to extract spatial coordinates and data
    if (format == 'ascii'):
        dinput = from_ascii(filename, **kwargs)
    elif (format == 'netCDF4'):
        dinput = from_netCDF4(filename, **kwargs)
    elif (format == 'HDF5'):
        dinput = from_HDF5(filename, **kwargs)
    elif format in ('GTiff','cog'):
        dinput = from_geotiff(filename, **kwargs)
    elif (format == 'parquet'):
        dinput = from_parquet(filename, **kwargs)
    else:
        raise ValueError(f'Invalid format {format}')
    return dinput

def from_ascii(filename: str, **kwargs):
    """
    Read data from an ascii file

    Parameters
    ----------
    filename: str
        full path of input ascii file
    compression: str or NoneType, default None
        file compression type
    columns: list, default ['time', 'y', 'x', 'data']
        column names of ascii file
    delimiter: str, default ','
        Delimiter for csv or ascii files
    header: int, default 0
        header lines to skip from start of file
    parse_dates: bool, default False
        Try parsing the time column
    """
    # set default keyword arguments
    kwargs.setdefault('compression', None)
    kwargs.setdefault('columns', ['time', 'y', 'x', 'data'])
    kwargs.setdefault('delimiter', ',')
    kwargs.setdefault('header', 0)
    kwargs.setdefault('parse_dates', False)
    # print filename
    logging.info(str(filename))
    # get column names
    columns = copy.copy(kwargs['columns'])
    # open the ascii file and extract contents
    if (kwargs['compression'] == 'gzip'):
        # read input ascii data from gzip compressed file and split lines
        filename = case_insensitive_filename(filename)
        with gzip.open(filename, 'r') as f:
            file_contents = f.read().decode('ISO-8859-1').splitlines()
    elif (kwargs['compression'] == 'bytes'):
        # read input file object and split lines
        file_contents = filename.read().splitlines()
    else:
        # read input ascii file (.txt, .asc) and split lines
        filename = case_insensitive_filename(filename)
        with open(filename, mode='r', encoding='utf8') as f:
            file_contents = f.read().splitlines()
    # number of lines in the file
    file_lines = len(file_contents)
    # compile regular expression operator for extracting numerical values
    # from input ascii files of spatial data
    regex_pattern = r'[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[EeD][+-]?\d+)?'
    rx = re.compile(regex_pattern, re.VERBOSE)
    # check if header has a known format
    if (str(kwargs['header']).upper() == 'YAML'):
        # counts the number of lines in the header
        YAML = False
        count = 0
        # Reading over header text
        while (YAML is False) & (count < file_lines):
            # file line at count
            line = file_contents[count]
            # if End of YAML Header is found: set YAML flag
            YAML = bool(re.search(r"\# End of YAML header", line))
            # add 1 to counter
            count += 1
        # parse the YAML header (specifying yaml loader)
        YAML_HEADER = yaml.load('\n'.join(file_contents[:count]),
           Loader=yaml.BaseLoader)
        # output spatial data and attributes
        dinput = {}
        # copy global attributes
        dinput['attributes'] = YAML_HEADER['header']['global_attributes']
        # allocate for each variable and copy variable attributes
        for c in columns:
            if (c == 'time') and kwargs['parse_dates']:
                dinput[c] = np.zeros((file_lines-count), dtype='datetime64[ns]')
            else:
                dinput[c] = np.zeros((file_lines-count))
            dinput['attributes'][c] = YAML_HEADER['header']['variables'][c]
        # update number of file lines to skip for reading data
        header = int(count)
    else:
        # allocate for each variable and variable attributes
        dinput = {}
        header = int(kwargs['header'])
        for c in columns:
            if (c == 'time') and kwargs['parse_dates']:
                dinput[c] = np.zeros((file_lines-header), dtype='datetime64[ns]')
            else:
                dinput[c] = np.zeros((file_lines-header))
        dinput['attributes'] = {c:dict() for c in columns}
    # extract spatial data array
    # for each line in the file
    for i, line in enumerate(file_contents[header:]):
        # extract columns of interest and assign to dict
        # convert fortran exponentials if applicable
        if kwargs['delimiter']:
            column = {c:l.replace('D', 'E') for c, l in
                zip(columns, line.split(kwargs['delimiter']))}
        else:
            column = {c:r.replace('D', 'E') for c, r in
                zip(columns, rx.findall(line))}
        # copy variables from column dict to output dictionary
        for c in columns:
            if (c == 'time') and kwargs['parse_dates']:
                dinput[c][i] = timescale.time.parse(column[c])
            else:
                dinput[c][i] = np.float64(column[c])
    # convert to masked array if fill values
    if 'data' in dinput.keys() and '_FillValue' in dinput['attributes']['data'].keys():
        dinput['data'] = np.ma.asarray(dinput['data'])
        dinput['data'].fill_value = dinput['attributes']['data']['_FillValue']
        dinput['data'].mask = (dinput['data'].data == dinput['data'].fill_value)
    # return the spatial variables
    return dinput

def from_netCDF4(filename: str, **kwargs):
    """
    Read data from a netCDF4 file

    Parameters
    ----------
    filename: str
        full path of input netCDF4 file
    compression: str or NoneType, default None
        file compression type
    group: str or NoneType, default None
        netCDF4 variable group
    timename: str, default 'time'
        name for time-dimension variable
    xname: str, default 'lon'
        name for x-dimension variable
    yname: str, default 'lat'
        name for y-dimension variable
    varname: str, default 'data'
        name for data variable
    field_mapping: dict, default {}
        mapping between output variables and input netCDF4
    """
    # set default keyword arguments
    kwargs.setdefault('compression', None)
    kwargs.setdefault('group', None)
    kwargs.setdefault('timename', 'time')
    kwargs.setdefault('xname', 'lon')
    kwargs.setdefault('yname', 'lat')
    kwargs.setdefault('varname', 'data')
    kwargs.setdefault('field_mapping', {})
    # read data from netCDF4 file
    # Open the NetCDF4 file for reading
    if (kwargs['compression'] == 'gzip'):
        # read as in-memory (diskless) netCDF4 dataset
        with gzip.open(case_insensitive_filename(filename), 'r') as f:
            fileID = netCDF4.Dataset(uuid.uuid4().hex, memory=f.read())
    elif (kwargs['compression'] == 'bytes'):
        # read as in-memory (diskless) netCDF4 dataset
        fileID = netCDF4.Dataset(uuid.uuid4().hex, memory=filename.read())
    else:
        # read netCDF4 dataset
        fileID = netCDF4.Dataset(case_insensitive_filename(filename), 'r')
    # Output NetCDF file information
    logging.info(fileID.filepath())
    logging.info(list(fileID.variables.keys()))
    # create python dictionary for output variables and attributes
    dinput = {}
    dinput['attributes'] = {}
    # get attributes for the file
    for attr in ['title', 'description', 'projection']:
        # try getting the attribute
        try:
            ncattr, = [s for s in fileID.ncattrs() if re.match(attr, s, re.I)]
            dinput['attributes'][attr] = fileID.getncattr(ncattr)
        except (ValueError, AttributeError):
            pass
    # list of attributes to attempt to retrieve from included variables
    attributes_list = ['description', 'units', 'long_name', 'calendar',
        'standard_name', 'grid_mapping', '_FillValue']
    # mapping between netCDF4 variable names and output names
    if not kwargs['field_mapping']:
        kwargs['field_mapping']['x'] = copy.copy(kwargs['xname'])
        kwargs['field_mapping']['y'] = copy.copy(kwargs['yname'])
        if kwargs['varname'] is not None:
            kwargs['field_mapping']['data'] = copy.copy(kwargs['varname'])
        if kwargs['timename'] is not None:
            kwargs['field_mapping']['time'] = copy.copy(kwargs['timename'])
    # check if reading from root group or sub-group
    group = fileID.groups[kwargs['group']] if kwargs['group'] else fileID
    # for each variable
    for key, nc in kwargs['field_mapping'].items():
        # Getting the data from each NetCDF variable
        dinput[key] = group.variables[nc][:]
        # get attributes for the included variables
        dinput['attributes'][key] = {}
        for attr in attributes_list:
            # try getting the attribute
            try:
                ncattr, = [s for s in group.variables[nc].ncattrs()
                    if re.match(attr, s, re.I)]
                dinput['attributes'][key][attr] = \
                    group.variables[nc].getncattr(ncattr)
            except (ValueError, AttributeError):
                pass
    # get projection information if there is a grid_mapping attribute
    if 'data' in dinput.keys() and 'grid_mapping' in dinput['attributes']['data'].keys():
        # try getting the attribute
        grid_mapping = dinput['attributes']['data']['grid_mapping']
        # get coordinate reference system attributes
        dinput['attributes']['crs'] = {}
        for att_name in group[grid_mapping].ncattrs():
            dinput['attributes']['crs'][att_name] = \
                group.variables[grid_mapping].getncattr(att_name)
        # get the spatial projection reference information from wkt
        # and overwrite the file-level projection attribute (if existing)
        osgeo.osr.UseExceptions()
        srs = osgeo.osr.SpatialReference()
        srs.ImportFromWkt(dinput['attributes']['crs']['crs_wkt'])
        dinput['attributes']['projection'] = srs.ExportToProj4()
    # convert to masked array if fill values
    if 'data' in dinput.keys() and '_FillValue' in dinput['attributes']['data'].keys():
        dinput['data'] = np.ma.asarray(dinput['data'])
        dinput['data'].fill_value = dinput['attributes']['data']['_FillValue']
        dinput['data'].mask = (dinput['data'].data == dinput['data'].fill_value)
    # Closing the NetCDF file
    fileID.close()
    # return the spatial variables
    return dinput

def from_HDF5(filename: str | pathlib.Path, **kwargs):
    """
    Read data from a HDF5 file

    Parameters
    ----------
    filename: str
        full path of input HDF5 file
    compression: str or NoneType, default None
        file compression type
    group: str or NoneType, default None
        netCDF4 variable group
    timename: str, default 'time'
        name for time-dimension variable
    xname: str, default 'lon'
        name for x-dimension variable
    yname: str, default 'lat'
        name for y-dimension variable
    varname: str, default 'data'
        name for data variable
    field_mapping: dict, default {}
        mapping between output variables and input HDF5
    """
    # set default keyword arguments
    kwargs.setdefault('compression', None)
    kwargs.setdefault('group', None)
    kwargs.setdefault('timename', 'time')
    kwargs.setdefault('xname', 'lon')
    kwargs.setdefault('yname', 'lat')
    kwargs.setdefault('varname', 'data')
    kwargs.setdefault('field_mapping', {})
    # read data from HDF5 file
    # Open the HDF5 file for reading
    if (kwargs['compression'] == 'gzip'):
        # read gzip compressed file and extract into in-memory file object
        with gzip.open(case_insensitive_filename(filename), 'r') as f:
            fid = io.BytesIO(f.read())
        # set filename of BytesIO object
        fid.filename = filename.name
        # rewind to start of file
        fid.seek(0)
        # read as in-memory (diskless) HDF5 dataset from BytesIO object
        fileID = h5py.File(fid, 'r')
    elif (kwargs['compression'] == 'bytes'):
        # read as in-memory (diskless) HDF5 dataset
        fileID = h5py.File(filename, 'r')
    else:
        # read HDF5 dataset
        fileID = h5py.File(case_insensitive_filename(filename), 'r')
    # Output HDF5 file information
    logging.info(fileID.filename)
    logging.info(list(fileID.keys()))
    # create python dictionary for output variables and attributes
    dinput = {}
    dinput['attributes'] = {}
    # get attributes for the file
    for attr in ['title', 'description', 'projection']:
        # try getting the attribute
        try:
            dinput['attributes'][attr] = fileID.attrs[attr]
        except (KeyError, AttributeError):
            pass
    # list of attributes to attempt to retrieve from included variables
    attributes_list = ['description', 'units', 'long_name', 'calendar',
        'standard_name', 'grid_mapping', '_FillValue']
    # mapping between HDF5 variable names and output names
    if not kwargs['field_mapping']:
        kwargs['field_mapping']['x'] = copy.copy(kwargs['xname'])
        kwargs['field_mapping']['y'] = copy.copy(kwargs['yname'])
        if kwargs['varname'] is not None:
            kwargs['field_mapping']['data'] = copy.copy(kwargs['varname'])
        if kwargs['timename'] is not None:
            kwargs['field_mapping']['time'] = copy.copy(kwargs['timename'])
    # check if reading from root group or sub-group
    group = fileID[kwargs['group']] if kwargs['group'] else fileID
    # for each variable
    for key, h5 in kwargs['field_mapping'].items():
        # Getting the data from each HDF5 variable
        dinput[key] = np.copy(group[h5][:])
        # get attributes for the included variables
        dinput['attributes'][key] = {}
        for attr in attributes_list:
            # try getting the attribute
            try:
                dinput['attributes'][key][attr] = group[h5].attrs[attr]
            except (KeyError, AttributeError):
                pass
    # get projection information if there is a grid_mapping attribute
    if 'data' in dinput.keys() and 'grid_mapping' in dinput['attributes']['data'].keys():
        # try getting the attribute
        grid_mapping = dinput['attributes']['data']['grid_mapping']
        # get coordinate reference system attributes
        dinput['attributes']['crs'] = {}
        for att_name, att_val in group[grid_mapping].attrs.items():
            dinput['attributes']['crs'][att_name] = att_val
        # get the spatial projection reference information from wkt
        # and overwrite the file-level projection attribute (if existing)
        osgeo.osr.UseExceptions()
        srs = osgeo.osr.SpatialReference()
        srs.ImportFromWkt(dinput['attributes']['crs']['crs_wkt'])
        dinput['attributes']['projection'] = srs.ExportToProj4()
    # convert to masked array if fill values
    if 'data' in dinput.keys() and '_FillValue' in dinput['attributes']['data'].keys():
        dinput['data'] = np.ma.asarray(dinput['data'])
        dinput['data'].fill_value = dinput['attributes']['data']['_FillValue']
        dinput['data'].mask = (dinput['data'].data == dinput['data'].fill_value)
    # Closing the HDF5 file
    fileID.close()
    # return the spatial variables
    return dinput

def from_geotiff(filename: str, **kwargs):
    """
    Read data from a geotiff file

    Parameters
    ----------
    filename: str
        full path of input geotiff file
    compression: str or NoneType, default None
        file compression type
    bounds: list or NoneType, default bounds
        extent of the file to read: ``[xmin, xmax, ymin, ymax]``
    """
    # set default keyword arguments
    kwargs.setdefault('compression', None)
    kwargs.setdefault('bounds', None)
    # Open the geotiff file for reading
    if (kwargs['compression'] == 'gzip'):
        # read as GDAL gzip virtual geotiff dataset
        mmap_name = f"/vsigzip/{str(case_insensitive_filename(filename))}"
        ds = osgeo.gdal.Open(mmap_name)
    elif (kwargs['compression'] == 'bytes'):
        # read as GDAL memory-mapped (diskless) geotiff dataset
        mmap_name = f"/vsimem/{uuid.uuid4().hex}"
        osgeo.gdal.FileFromMemBuffer(mmap_name, filename.read())
        ds = osgeo.gdal.Open(mmap_name)
    else:
        # read geotiff dataset
        ds = osgeo.gdal.Open(str(case_insensitive_filename(filename)),
            osgeo.gdalconst.GA_ReadOnly)
    # print geotiff file if verbose
    logging.info(str(filename))
    # create python dictionary for output variables and attributes
    dinput = {}
    dinput['attributes'] = {c:dict() for c in ['x', 'y', 'data']}
    # get the spatial projection reference information
    srs = ds.GetSpatialRef()
    dinput['attributes']['projection'] = srs.ExportToProj4()
    dinput['attributes']['wkt'] = srs.ExportToWkt()
    # get dimensions
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    bsize = ds.RasterCount
    # get geotiff info
    info_geotiff = ds.GetGeoTransform()
    dinput['attributes']['spacing'] = (info_geotiff[1], info_geotiff[5])
    # calculate image extents
    xmin = info_geotiff[0]
    ymax = info_geotiff[3]
    xmax = xmin + (xsize-1)*info_geotiff[1]
    ymin = ymax + (ysize-1)*info_geotiff[5]
    # x and y pixel center coordinates (converted from upper left)
    x = xmin + info_geotiff[1]/2.0 + np.arange(xsize)*info_geotiff[1]
    y = ymax + info_geotiff[5]/2.0 + np.arange(ysize)*info_geotiff[5]
    # if reducing to specified bounds
    if kwargs['bounds'] is not None:
        # reduced x and y limits
        xlimits = (kwargs['bounds'][0], kwargs['bounds'][1])
        ylimits = (kwargs['bounds'][2], kwargs['bounds'][3])
        # Specify offset and rows and columns to read
        xoffset = int((xlimits[0] - xmin)/info_geotiff[1])
        yoffset = int((ymax - ylimits[1])/np.abs(info_geotiff[5]))
        xcount = int((xlimits[1] - xlimits[0])/info_geotiff[1]) + 1
        ycount = int((ylimits[1] - ylimits[0])/np.abs(info_geotiff[5])) + 1
        # reduced x and y pixel center coordinates
        dinput['x'] = x[slice(xoffset, xoffset + xcount, None)]
        dinput['y'] = y[slice(yoffset, yoffset + ycount, None)]
        # read reduced image with GDAL
        dinput['data'] = ds.ReadAsArray(xoff=xoffset, yoff=yoffset,
            xsize=xcount, ysize=ycount)
        # reduced image extent (converted back to upper left)
        xmin = np.min(dinput['x']) - info_geotiff[1]/2.0
        xmax = np.max(dinput['x']) - info_geotiff[1]/2.0
        ymin = np.min(dinput['y']) - info_geotiff[5]/2.0
        ymax = np.max(dinput['y']) - info_geotiff[5]/2.0
    else:
        # x and y pixel center coordinates
        dinput['x'] = np.copy(x)
        dinput['y'] = np.copy(y)
        # read full image with GDAL
        dinput['data'] = ds.ReadAsArray()
    # image extent
    dinput['attributes']['extent'] = (xmin, xmax, ymin, ymax)
    # set default time to zero for each band
    dinput.setdefault('time', np.zeros((bsize)))
    # check if image has fill values
    dinput['data'] = np.ma.asarray(dinput['data'])
    dinput['data'].mask = np.zeros_like(dinput['data'], dtype=bool)
    if ds.GetRasterBand(1).GetNoDataValue():
        # mask invalid values
        dinput['data'].fill_value = ds.GetRasterBand(1).GetNoDataValue()
        # create mask array for bad values
        dinput['data'].mask[:] = (dinput['data'].data == dinput['data'].fill_value)
        # set attribute for fill value
        dinput['attributes']['data']['_FillValue'] = dinput['data'].fill_value
    # close the dataset
    ds = None
    # return the spatial variables
    return dinput

def from_parquet(filename: str, **kwargs):
    """
    Read data from a parquet file

    Parameters
    ----------
    filename: str
        full path of input parquet file
    index: str, default 'time'
        name of index column
    columns: list or None, default None
        column names of parquet file
    primary_column: str, default 'geometry'
        default geometry column in geoparquet files
    geometry_encoding: str, default 'WKB'
        default encoding for geoparquet files
    """
    # set default keyword arguments
    kwargs.setdefault('index', 'time')
    kwargs.setdefault('columns', None)
    kwargs.setdefault('primary_column', 'geometry')
    kwargs.setdefault('geometry_encoding', 'WKB')
    filename = case_insensitive_filename(filename)
    # read input parquet file
    dinput = pd.read_parquet(filename)
    # reset the dataframe index if not a range index
    if not isinstance(dinput.index, pd.RangeIndex):
        dinput.reset_index(inplace=True, names=kwargs['index'])
    # output parquet file information
    attr = {}
    # get parquet file metadata
    metadata = pyarrow.parquet.read_metadata(filename).metadata
    # decode parquet metadata from JSON
    for att_name, val in metadata.items():
        try:
            att_val = json.loads(val.decode('utf-8'))
            attr[att_name.decode('utf-8')] = att_val
        except Exception as exc:
            pass
    # check if parquet file contains geometry metadata
    attr['geoparquet'] = False
    primary_column = kwargs['primary_column']
    encoding = kwargs['geometry_encoding']
    if 'geo' in attr.keys():
        # extract crs and encoding from geoparquet metadata
        primary_column = attr['geo']['primary_column']
        crs_metadata = attr['geo']['columns'][primary_column]['crs']
        encoding = attr['geo']['columns'][primary_column]['encoding']
        attr['geometry_encoding'] = encoding
        # create spatial reference object from PROJJSON
        osgeo.osr.UseExceptions()
        srs = osgeo.osr.SpatialReference()
        srs.SetFromUserInput(json.dumps(crs_metadata))
        # add projection information to attributes
        attr['projection'] = srs.ExportToProj4()
        attr['wkt'] = srs.ExportToWkt()
    elif 'pyTMD' in attr.keys():
        # extract crs from pyTMD metadata
        crs_metadata = attr['pyTMD']['crs']
        # create spatial reference object from PROJJSON
        osgeo.osr.UseExceptions()
        srs = osgeo.osr.SpatialReference()
        srs.SetFromUserInput(json.dumps(crs_metadata))
        # add projection information to attributes
        attr['projection'] = srs.ExportToProj4()
        attr['wkt'] = srs.ExportToWkt()
    # extract x and y coordinates
    if (encoding == 'WKB') and (primary_column in dinput.keys()):
        # set as geoparquet file
        attr['geoparquet'] = True
        # decode geometry column from WKB
        geometry = shapely.from_wkb(dinput[primary_column].values)
        dinput['x'] = shapely.get_x(geometry)
        dinput['y'] = shapely.get_y(geometry)
    # remap columns to default names
    if kwargs['columns'] is not None:
        field_mapping = default_field_mapping(kwargs['columns'])
        remap = inverse_mapping(field_mapping)
        dinput.rename(columns=remap, inplace=True)
    # return the data and attributes
    dinput.attrs = copy.copy(attr)
    return dinput

def to_file(
        output: dict,
        attributes: dict,
        filename: str | pathlib.Path,
        format: str,
        **kwargs
    ):
    """
    Wrapper function for writing data to an output format

    Parameters
    ----------
    output: dict
        python dictionary of output data
    attributes: dict
        python dictionary of output attributes
    filename: str or pathlib.Path,
        full path of output file
    format: str
        format of output file
    **kwargs: dict
        Keyword arguments for file writer
    """
    # read input file to extract spatial coordinates and data
    if (format == 'ascii'):
        to_ascii(output, attributes, filename, **kwargs)
    elif (format == 'netCDF4'):
        to_netCDF4(output, attributes, filename, **kwargs)
    elif (format == 'HDF5'):
        to_HDF5(output, attributes, filename, **kwargs)
    elif format in ('GTiff','cog'):
        to_geotiff(output, attributes, filename, **kwargs)
    elif (format == 'parquet'):
        to_parquet(output, attributes, filename, **kwargs)
    else:
        raise ValueError(f'Invalid format {format}')

def to_ascii(
        output: dict,
        attributes: dict,
        filename: str | pathlib.Path,
        **kwargs
    ):
    """
    Write data to an ascii file

    Parameters
    ----------
    output: dict
        python dictionary of output data
    attributes: dict
        python dictionary of output attributes
    filename: str or pathlib.Path
        full path of output ascii file
    delimiter: str, default ','
        delimiter for output spatial file
    columns: list, default ['time', 'y', 'x', 'data']
        column names of ascii file
    header: bool, default False
        create a YAML header with data attributes
    """
    # set default keyword arguments
    kwargs.setdefault('delimiter', ',')
    kwargs.setdefault('columns', ['time', 'lat', 'lon', 'tide'])
    kwargs.setdefault('header', False)
    # get column names
    columns = copy.copy(kwargs['columns'])
    # output filename
    filename = pathlib.Path(filename).expanduser().absolute()
    logging.info(str(filename))
    # open the output file
    fid = filename.open(mode='w', encoding='utf8')
    # create a column stack arranging data in column order
    data_stack = np.c_[[output[col] for col in columns]]
    ncol, nrow = np.shape(data_stack)
    # print YAML header to top of file
    if kwargs['header']:
        fid.write('{0}:\n'.format('header'))
        # data dimensions
        fid.write('\n  {0}:\n'.format('dimensions'))
        fid.write('    {0:22}: {1:d}\n'.format('time', nrow))
        # non-standard attributes
        fid.write('  {0}:\n'.format('non-standard_attributes'))
        # data format
        fid.write('    {0:22}: ({1:d}f0.8)\n'.format('formatting_string', ncol))
        fid.write('\n')
        # global attributes
        fid.write('\n  {0}:\n'.format('global_attributes'))
        today = datetime.datetime.now().isoformat()
        fid.write('    {0:22}: {1}\n'.format('date_created', today))
        # print variable descriptions to YAML header
        fid.write('\n  {0}:\n'.format('variables'))
        # print YAML header with variable attributes
        for i, v in enumerate(columns):
            fid.write('    {0:22}:\n'.format(v))
            for atn, atv in attributes[v].items():
                fid.write('      {0:20}: {1}\n'.format(atn, atv))
            # add precision and column attributes for ascii yaml header
            fid.write('      {0:20}: double_precision\n'.format('precision'))
            fid.write('      {0:20}: column {1:d}\n'.format('comment', i+1))
        # end of header
        fid.write('\n\n# End of YAML header\n')
    # write to file for each data point
    for line in range(nrow):
        line_contents = [f'{d}' for d in data_stack[:, line]]
        print(kwargs['delimiter'].join(line_contents), file=fid)
    # close the output file
    fid.close()

def to_netCDF4(
        output: dict,
        attributes: dict,
        filename: str | pathlib.Path,
        **kwargs
    ):
    """
    Wrapper function for writing data to a netCDF4 file

    Parameters
    ----------
    output: dict
        python dictionary of output data
    attributes: dict
        python dictionary of output attributes
    filename: str or pathlib.Path
        full path of output netCDF4 file
    mode: str, default 'w'
        NetCDF file mode
    data_type: str, default 'drift'
        Input data type

            - ``'time series'``
            - ``'drift'``
            - ``'grid'``
    """
    # default arguments
    kwargs.setdefault('mode', 'w')
    kwargs.setdefault('data_type', 'drift')
    # opening NetCDF file for writing
    filename = pathlib.Path(filename).expanduser().absolute()
    fileID = netCDF4.Dataset(filename, kwargs['mode'], format="NETCDF4")
    if kwargs['data_type'] in ('drift',):
        kwargs.pop('data_type')
        _drift_netCDF4(fileID, output, attributes, **kwargs)
    elif kwargs['data_type'] in ('grid',):
        kwargs.pop('data_type')
        _grid_netCDF4(fileID, output, attributes, **kwargs)
    elif kwargs['data_type'] in ('time series',):
        kwargs.pop('data_type')
        _time_series_netCDF4(fileID, output, attributes, **kwargs)
    # add attribute for date created
    fileID.date_created = datetime.datetime.now().isoformat()
    # add attributes for software information
    fileID.software_reference = grounding_zones.version.project_name
    fileID.software_version = grounding_zones.version.full_version
    # add file-level attributes if applicable
    if 'ROOT' in attributes.keys():
        # Defining attributes for file
        for att_name, att_val in attributes['ROOT'].items():
            fileID.setncattr(att_name, att_val)
    # Output NetCDF structure information
    logging.info(str(filename))
    logging.info(list(fileID.variables.keys()))
    # Closing the NetCDF file
    fileID.close()

def _drift_netCDF4(fileID, output: dict, attributes: dict, **kwargs):
    """
    Write drift data variables to a netCDF4 file object

    Parameters
    ----------
    fileID: obj
        open netCDF4 file object
    output: dict
        python dictionary of output data
    attributes: dict
        python dictionary of output attributes
    """

    # Defining the NetCDF dimensions
    fileID.createDimension('time', len(np.atleast_1d(output['time'])))
    # defining the NetCDF variables
    nc = {}
    for key, val in output.items():
        if key in fileID.variables:
            nc[key] = fileID.variables[key]
        elif '_FillValue' in attributes[key].keys():
            nc[key] = fileID.createVariable(key, val.dtype, ('time',),
                fill_value=attributes[key]['_FillValue'], zlib=True)
            attributes[key].pop('_FillValue')
        elif val.shape:
            nc[key] = fileID.createVariable(key, val.dtype, ('time',))
        else:
            nc[key] = fileID.createVariable(key, val.dtype, ())
        # filling NetCDF variables
        nc[key][:] = val
        # Defining attributes for variable
        for att_name, att_val in attributes[key].items():
            nc[key].setncattr(att_name, att_val)

def _grid_netCDF4(fileID, output: dict, attributes: dict, **kwargs):
    """
    Write gridded data variables to a netCDF4 file object

    Parameters
    ----------
    fileID: obj
        open netCDF4 file object
    output: dict
        python dictionary of output data
    attributes: dict
        python dictionary of output attributes
    """
    # output data fields
    dimensions = ['t', 'time', 'lon', 'longitude', 'x', 'lat', 'latitude', 'y']
    crs = ['crs', 'crs_wkt', 'crs_proj4', 'projection']
    fields = sorted(set(output.keys()) - set(dimensions) - set(crs))
    # Defining the NetCDF dimensions
    reference_fields = [v for v in fields if output[v].ndim == 3]
    ny, nx, nt = output[reference_fields[0]].shape
    fileID.createDimension('y', ny)
    fileID.createDimension('x', nx)
    fileID.createDimension('time', nt)
    # defining the NetCDF variables
    nc = {}
    for key, val in output.items():
        if key in fileID.variables:
            nc[key] = fileID.variables[key]
        elif '_FillValue' in attributes[key].keys():
            nc[key] = fileID.createVariable(key, val.dtype, ('y', 'x', 'time'),
                fill_value=attributes[key]['_FillValue'], zlib=True)
            attributes[key].pop('_FillValue')
        elif (val.ndim == 3):
            nc[key] = fileID.createVariable(key, val.dtype, ('y', 'x', 'time'))
        elif (val.ndim == 2):
            nc[key] = fileID.createVariable(key, val.dtype, ('y', 'x'))
        elif val.shape and (len(val) == ny):
            nc[key] = fileID.createVariable(key, val.dtype, ('y',))
        elif val.shape and (len(val) == nx):
            nc[key] = fileID.createVariable(key, val.dtype, ('x',))
        elif val.shape and (len(val) == nt):
            nc[key] = fileID.createVariable(key, val.dtype, ('time',))
        else:
            nc[key] = fileID.createVariable(key, val.dtype, ())
        # filling NetCDF variables
        nc[key][:] = val
        # Defining attributes for variable
        for att_name, att_val in attributes[key].items():
            nc[key].setncattr(att_name, att_val)

def _time_series_netCDF4(fileID, output: dict, attributes: dict, **kwargs):
    """
    Write time series data variables to a netCDF4 file object

    Parameters
    ----------
    fileID: obj
        open netCDF4 file object
    output: dict
        python dictionary of output data
    attributes: dict
        python dictionary of output attributes
    """
    # output data fields
    dimensions = ['t', 'time', 'lon', 'longitude', 'x', 'lat', 'latitude', 'y']
    crs = ['crs', 'crs_wkt', 'crs_proj4', 'projection']
    fields = sorted(set(output.keys()) - set(dimensions) - set(crs))
    # Defining the NetCDF dimensions
    reference_fields = [v for v in fields if output[v].ndim == 2]
    nstation, nt = output[reference_fields[0]].shape
    fileID.createDimension('station', nstation)
    fileID.createDimension('time', nt)
    # defining the NetCDF variables
    nc = {}
    for key, val in output.items():
        if key in fileID.variables:
            nc[key] = fileID.variables[key]
        elif '_FillValue' in attributes[key].keys():
            nc[key] = fileID.createVariable(key, val.dtype, ('station', 'time'),
                fill_value=attributes[key]['_FillValue'], zlib=True)
            attributes[key].pop('_FillValue')
        elif (val.ndim == 2):
            nc[key] = fileID.createVariable(key, val.dtype, ('station', 'time'))
        elif val.shape and (len(val) == nt):
            nc[key] = fileID.createVariable(key, val.dtype, ('time',))
        elif val.shape and (len(val) == nstation):
            nc[key] = fileID.createVariable(key, val.dtype, ('station',))
        else:
            nc[key] = fileID.createVariable(key, val.dtype, ())
        # filling NetCDF variables
        nc[key][:] = val
        # Defining attributes for variable
        for att_name, att_val in attributes[key].items():
            nc[key].setncattr(att_name, att_val)

def to_HDF5(
        output: dict,
        attributes: dict,
        filename: str | pathlib.Path,
        **kwargs
    ):
    """
    Write data to a HDF5 file

    Parameters
    ----------
    output: dict
        python dictionary of output data
    attributes: dict
        python dictionary of output attributes
    filename: str or pathlib.Path
        full path of output HDF5 file
    mode: str, default 'w'
        HDF5 file mode
    """
    # set default keyword arguments
    kwargs.setdefault('mode', 'w')
    # opening HDF5 file for writing
    filename = pathlib.Path(filename).expanduser().absolute()
    fileID = h5py.File(filename, mode=kwargs['mode'])
    # Defining the HDF5 dataset variables
    h5 = {}
    for key, val in output.items():
        if key in fileID:
            fileID[key][...] = val[:]
        elif '_FillValue' in attributes[key].keys():
            h5[key] = fileID.create_dataset(key, val.shape, data=val,
                dtype=val.dtype, fillvalue=attributes[key]['_FillValue'],
                compression='gzip')
            attributes[key].pop('_FillValue')
        elif val.shape:
            h5[key] = fileID.create_dataset(key, val.shape, data=val,
                dtype=val.dtype, compression='gzip')
        else:
            h5[key] = fileID.create_dataset(key, val.shape,
                dtype=val.dtype)
        # Defining attributes for variable
        for att_name, att_val in attributes[key].items():
            h5[key].attrs[att_name] = att_val
    # add attribute for date created
    fileID.attrs['date_created'] = datetime.datetime.now().isoformat()
    # add attributes for software information
    fileID.attrs['software_reference'] = grounding_zones.version.project_name
    fileID.attrs['software_version'] = grounding_zones.version.full_version
    # add file-level attributes if applicable
    if 'ROOT' in attributes.keys():
        # Defining attributes for file
        for att_name, att_val in attributes['ROOT'].items():
            fileID.attrs[att_name] = att_val
    # Output HDF5 structure information
    logging.info(str(filename))
    logging.info(list(fileID.keys()))
    # Closing the HDF5 file
    fileID.close()

def to_geotiff(
        output: dict,
        attributes: dict,
        filename: str | pathlib.Path,
        **kwargs
    ):
    """
    Write data to a (cloud optimized) geotiff file

    Parameters
    ----------
    output: dict
        python dictionary of output data
    attributes: dict
        python dictionary of output attributes
    filename: str or pathlib.Path
        full path of output geotiff file
    varname: str, default 'data'
        output variable name
    driver: str, default 'cog'
        GDAL driver

            - ``'GTiff'``: GeoTIFF
            - ``'cog'``: Cloud Optimized GeoTIFF
    dtype: obj, default osgeo.gdal.GDT_Float64
        GDAL data type
    options: list, default ['COMPRESS=LZW']
        GDAL driver creation options
    """
    # set default keyword arguments
    kwargs.setdefault('varname', 'data')
    kwargs.setdefault('driver', 'cog')
    kwargs.setdefault('dtype', osgeo.gdal.GDT_Float64)
    kwargs.setdefault('options', ['COMPRESS=LZW'])
    varname = copy.copy(kwargs['varname'])
    # verify grid dimensions to be iterable
    output = expand_dims(output, varname=varname)
    # grid shape
    ny, nx, nband = np.shape(output[varname])
    # output as geotiff or specified driver
    driver = osgeo.gdal.GetDriverByName(kwargs['driver'])
    # set up the dataset with creation options
    filename = pathlib.Path(filename).expanduser().absolute()
    ds = driver.Create(str(filename), nx, ny, nband,
        kwargs['dtype'], kwargs['options'])
    # top left x, w-e pixel resolution, rotation
    # top left y, rotation, n-s pixel resolution
    xmin, xmax, ymin, ymax = attributes['extent']
    dx, dy = attributes['spacing']
    ds.SetGeoTransform([xmin, dx, 0, ymax, 0, dy])
    # set the spatial projection reference information
    osgeo.osr.UseExceptions()
    srs = osgeo.osr.SpatialReference()
    srs.ImportFromWkt(attributes['wkt'])
    # export
    ds.SetProjection( srs.ExportToWkt() )
    # for each band
    for band in range(nband):
        # set fill value for band
        if '_FillValue' in attributes[varname].keys():
            fill_value = attributes[varname]['_FillValue']
            ds.GetRasterBand(band+1).SetNoDataValue(fill_value)
        # write band to geotiff array
        ds.GetRasterBand(band+1).WriteArray(output[varname][:, :, band])
    # print filename if verbose
    logging.info(str(filename))
    # close dataset
    ds.FlushCache()

def to_parquet(
        output: dict,
        attributes: dict,
        filename: str | pathlib.Path,
        **kwargs
    ):
    """
    Write data to a (geo)parquet file

    Parameters
    ----------
    output: dict
        python dictionary of output data
    attributes: dict
        python dictionary of output attributes
    filename: str or pathlib.Path,
        full path of output parquet file
    crs: int, default None
        coordinate reference system EPSG code
    index: bool, default None
        write index to parquet file
    compression: str, default 'snappy'
        file compression type
    geoparquet: bool, default False
        write geoparquet file
    geometry_encoding: str, default 'WKB'
        default encoding for geoparquet geometry
    primary_column: str, default 'geometry'
        default column name for geoparquet geometry
    """
    # set default keyword arguments
    kwargs.setdefault('crs', None)
    kwargs.setdefault('index', None)
    kwargs.setdefault('compression', 'snappy')
    kwargs.setdefault('schema_version', '1.1.0')
    kwargs.setdefault('geoparquet', False)
    kwargs.setdefault('geometry_encoding', 'WKB')
    kwargs.setdefault('primary_column', 'geometry')
    # convert to pandas dataframe
    df = pd.DataFrame(output)
    attrs = df.attrs.copy()
    # add coordinate reference system to attributes
    if kwargs['crs'] and isinstance(kwargs['crs'], int):
        # create spatial reference object from EPSG code
        osgeo.osr.UseExceptions()
        srs = osgeo.osr.SpatialReference()
        srs.ImportFromEPSG(kwargs['crs'])
        # add projection information to attributes
        attributes['crs'] = json.loads(srs.ExportToPROJJSON())
    elif kwargs['crs'] and isinstance(kwargs['crs'], dict):
        # create spatial reference object from PROJJSON
        osgeo.osr.UseExceptions()
        srs = osgeo.osr.SpatialReference()
        srs.SetFromUserInput(json.dumps(kwargs['crs']))
        # add projection information to attributes
        attributes['crs'] = copy.copy(kwargs['crs'])
    # convert spatial coordinates to WKB encoded geometry
    if kwargs['geoparquet'] and (kwargs['geometry_encoding'] == 'WKB'):
        # get geometry columns
        primary_column = kwargs['primary_column']
        geometries = ['lon', 'longitude', 'x', 'lat', 'latitude', 'y']
        geom_vars = [v for v in geometries if v in output.keys()]
        # convert to shapely geometry
        points = shapely.points(df[geom_vars[0]], df[geom_vars[1]])
        df.drop(columns=geom_vars, inplace=True)
        df[primary_column] = shapely.to_wkb(points)
        # get bounding box of total set of points
        bbox = shapely.MultiPoint(points).bounds
        # drop attributes for geometry columns
        [attributes.pop(v) for v in geom_vars if v in attributes]
        # add attributes for geoparquet
        attrs[b"geo"] = attrs.get(b"geo", {})
        attrs[b"geo"]["version"] = kwargs['schema_version']
        attrs[b"geo"]["primary_column"] = primary_column
        attrs[b"geo"]["columns"] = {primary_column: {
                "encoding": 'WKB',
                "crs": json.loads(srs.ExportToPROJJSON()),
                "bbox": bbox,
                "covering": {
                    "bbox": collections.OrderedDict(
                        xmin=[bbox[0], "xmin"],
                        ymin=[bbox[1], "ymin"],
                        xmax=[bbox[2], "xmax"],
                        ymax=[bbox[3], "ymax"]
                    )
                }
            }
        }
    elif kwargs['geoparquet'] and (kwargs['geometry_encoding'] == 'point'):
        raise ValueError('geoarrow encodings are currently unsupported')
    # add attribute for date created
    attributes['date_created'] = datetime.datetime.now().isoformat()
    # add attributes for software information
    attributes['software_reference'] = grounding_zones.version.project_name
    attributes['software_version'] = grounding_zones.version.full_version
    # dump the attributes to encoded JSON-format
    attr_metadata = {b"pyTMD": json.dumps(attributes).encode('utf-8')}
    for att_name, att_val in attrs.items():
        attr_metadata[att_name] = json.dumps(att_val).encode('utf-8')
    # convert dataframe to arrow table
    table = pyarrow.Table.from_pandas(df,
        preserve_index=kwargs['index'])
    # update parquet metadata
    metadata = table.schema.metadata
    metadata.update(attr_metadata)
    # replace schema metadata with updated
    table = table.replace_schema_metadata(metadata)
    # write arrow table to (geo)parquet file
    filename = pathlib.Path(filename).expanduser().absolute()
    logging.info(str(filename))
    pyarrow.parquet.write_table(table, filename,
        compression=kwargs['compression']
    )

def expand_dims(obj: dict, varname: str = 'data'):
    """
    Add a singleton dimension to a spatial dictionary if non-existent

    Parameters
    ----------
    obj: dict
        python dictionary of data
    varname: str, default data
        variable name to expand
    """
    # change time dimensions to be iterableinformation
    try:
        obj['time'] = np.atleast_1d(obj['time'])
    except:
        pass
    # output spatial with a third dimension
    if isinstance(varname, list):
        for v in varname:
            obj[v] = np.atleast_3d(obj[v])
    elif isinstance(varname, str):
        obj[varname] = np.atleast_3d(obj[varname])
    # return reformed spatial dictionary
    return obj

def default_field_mapping(variables: list | np.ndarray):
    """
    Builds field mappings from a variable list


    Parameters
    ----------
    variables: list
        Variables from argument parser

            - ``time``
            - ``yname``
            - ``xname``
            - ``varname``

    Returns
    -------
    field_mapping: dict
        Field mappings for netCDF4/HDF5 read functions
    """
    # get each variable name and add to field mapping dictionary
    field_mapping = {}
    for i, var in enumerate(['time', 'y', 'x', 'data']):
        try:
            field_mapping[var] = copy.copy(variables[i])
        except IndexError as exc:
            pass
    # return the field mapping
    return field_mapping

def inverse_mapping(field_mapping):
    """
    Reverses the field mappings of a dictionary

    Parameters
    ----------
    field_mapping: dict
        Field mappings for netCDF4/HDF5 read functions
    """
    return field_mapping.__class__(map(reversed, field_mapping.items()))
