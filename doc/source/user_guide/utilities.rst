============
utilities.py
============

Download and management utilities for syncing time and auxiliary files

 - Adds additional modules to the ICESat-2 `icesat2_toolkit utilities <https://github.com/tsutterley/read-ICESat-2/blob/main/icesat2_toolkit/utilities.py>`__


`Source code`__

.. __: https://github.com/tsutterley/ICESat-2-Grounding-Zones/blob/main/grounding_zones/utilities.py


General Methods
===============

.. method:: grounding_zones.utilities.get_data_path(relpath)

    Get the absolute path within a package from a relative path

    Arguments:

        ``relpath``: local relative path as list or string

.. method:: grounding_zones.utilities.pgc_list(HOST,timeout=None,context=ssl.SSLContext(),parser=None,format='%Y-%m-%d %H:%M',pattern='',sort=False)

    List a directory on `Polar Geospatial Center (PGC) <https://www.pgc.umn.edu/data/>`_ servers

    Arguments:

        `HOST`: remote http host path split as list

    Keyword arguments:

        `timeout`: timeout in seconds for blocking operations

        `context`: SSL context for url opener object

        `parser`: HTML parser for lxml

        `pattern`: regular expression pattern for reducing list

        `format`: format for input time string

        `sort`: sort output list

    Returns:

        `colnames`: list of column names in a directory

        `collastmod`: list of last modification times for items in the directory

        `colerror`: notification for list error

