#!/usr/bin/env python
u"""
utilities.py
Written by Tyler Sutterley (05/2022)
Download and management utilities for syncing time and auxiliary files
Adds additional modules to the icesat2_toolkit utilities

PYTHON DEPENDENCIES:
    lxml: processing XML and HTML in Python
        https://pypi.python.org/pypi/lxml

UPDATE HISTORY:
    Updated 05/2022: updated docstrings to numpy documentation format
    Updated 03/2021: add data path function for this set of utilities
    Written 01/2021
"""
# extend icesat2_toolkit utilities
from icesat2_toolkit.utilities import *

def get_data_path(relpath):
    """
    Get the absolute path within a package from a relative path

    Parameters
    ----------
    relpath: str,
        relative path
    """
    # current file path
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    filepath = os.path.dirname(os.path.abspath(filename))
    if isinstance(relpath,list):
        # use *splat operator to extract from list
        return os.path.join(filepath,*relpath)
    elif isinstance(relpath,str):
        return os.path.join(filepath,relpath)

# PURPOSE: list a directory on Polar Geospatial Center https server
def pgc_list(HOST, timeout=None, context=ssl.SSLContext(),
    parser=lxml.etree.HTMLParser(), format='%Y-%m-%d %H:%M',
    pattern='', sort=False):
    """
    List a directory on Polar Geospatial Center (PGC) servers

    Parameters
    ----------
    HOST: str or list
        remote https host path
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    context: obj, default ssl.SSLContext()
        SSL context for url opener object
    parser: obj, default lxml.etree.HTMLParser()
        HTML parser for lxml
    formatt: str, default '%Y-%m-%d %H:%M'
        format for input time string
    pattern: str, default ''
        regular expression pattern for reducing list
    sort: bool, default False
        sort output list

    Returns
    -------
    colnames: list
        column names in a directory
    collastmod: list
        last modification times for items in the directory
    colerror: list
        notification for list error
    """
    # verify inputs for remote http host
    if isinstance(HOST, str):
        HOST = url_split(HOST)
    # try listing from https
    try:
        # Create and submit request.
        request=urllib2.Request(posixpath.join(*HOST))
        response=urllib2.urlopen(request,context=context,timeout=timeout)
    except (urllib2.HTTPError, urllib2.URLError):
        colerror = 'List error from {0}'.format(posixpath.join(*HOST))
        return (False,False,colerror)
    else:
        # read and parse request for files (column names and modified times)
        tree = lxml.etree.parse(response,parser)
        colnames = [i.replace(posixpath.sep,'')
            for i in tree.xpath('//tr/td[not(@*)]//a/@href')]
        # get the Unix timestamp value for a modification time
        lastmod = [get_unix_time(i,format=format)
            for i in tree.xpath('//tr/td[@align="right"][1]/text()')]
        # reduce using regular expression pattern
        if pattern:
            i = [i for i,f in enumerate(colnames) if re.search(pattern,f)]
            # reduce list of column names and last modified times
            colnames = [colnames[indice] for indice in i]
            lastmod = [lastmod[indice] for indice in i]
        # sort the list
        if sort:
            i = [i for i,j in sorted(enumerate(colnames), key=lambda i: i[1])]
            # sort list of column names and last modified times
            colnames = [colnames[indice] for indice in i]
            lastmod = [lastmod[indice] for indice in i]
        # return the list of column names and last modified times
        return (colnames,lastmod,None)
