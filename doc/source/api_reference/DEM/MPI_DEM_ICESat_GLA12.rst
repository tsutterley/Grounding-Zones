=======================
MPI_DEM_ICESat_GLA12.py
=======================

- Determines which digital elevation model tiles to read for a given GLA12 file
- Reads 3\ |times|\ 3 array of tiles for points within bounding box of central mosaic tile
- Interpolates digital elevation model to locations of ICESat/GLAS L2 GLA12 Antarctic and Greenland Ice Sheet elevation data

- ArcticDEM 2m digital elevation model tiles

    * `http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v3.0/ <http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v3.0/>`_
    * `http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/indexes/ <http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/indexes/>`_

- REMA 8m digital elevation model tiles

    * `http://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v1.1/ <http://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v1.1/>`_
    * `http://data.pgc.umn.edu/elev/dem/setsm/REMA/indexes/ <http://data.pgc.umn.edu/elev/dem/setsm/REMA/indexes/>`_

- GIMP 30m digital elevation model tiles

    * `https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0645.001/ <https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0645.001/>`_


`Source code`__

.. __: https://github.com/tsutterley/Grounding-Zones/blob/main/DEM/MPI_DEM_ICESat_GLA12.py

Calling Sequence
################

.. argparse::
    :filename: MPI_DEM_ICESat_GLA12.py
    :func: arguments
    :prog: MPI_DEM_ICESat_GLA12.py
    :nodescription:
    :nodefault:

.. |times|      unicode:: U+00D7 .. MULTIPLICATION SIGN
