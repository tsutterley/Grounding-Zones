======================
MPI_interpolate_DEM.py
======================

- Determines which digital elevation model tiles for an input file (ascii, netCDF4, HDF5, geotiff)
- Reads 3\ |times|\ 3 array of tiles for points within bounding box of central mosaic tile
- Interpolates digital elevation model to coordinates

- ArcticDEM 2m digital elevation model tiles

    * `http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v3.0/ <http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v3.0/>`_
    * `http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/indexes/ <http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/indexes/>`_

- REMA 8m digital elevation model tiles

    * `http://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v1.1/ <http://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v1.1/>`_
    * `http://data.pgc.umn.edu/elev/dem/setsm/REMA/indexes/ <http://data.pgc.umn.edu/elev/dem/setsm/REMA/indexes/>`_

- GIMP 30m digital elevation model tiles

    * `https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0645.001/ <https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0645.001/>`_

`Source code`__

.. __: https://github.com/tsutterley/Grounding-Zones/blob/main/DEM/MPI_interpolate_DEM.py

Calling Sequence
################

.. argparse::
    :filename: MPI_interpolate_DEM.py
    :func: arguments
    :prog: MPI_interpolate_DEM.py
    :nodescription:
    :nodefault:

    --variables : @after
        * for csv files: the order of the columns within the file
        * for HDF5 and netCDF4 files: time, y, x and data variable names

    --type -t : @after
        * ``'drift'``: drift buoys or satellite/airborne altimetry (time per data point)
        * ``'grid'``: spatial grids or images (single time for all data points)

    --projection : @after
        * ``4326``: latitude and longitude coordinates on WGS84 reference ellipsoid

.. |times|      unicode:: U+00D7 .. MULTIPLICATION SIGN
