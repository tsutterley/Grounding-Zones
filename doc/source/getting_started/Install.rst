======================
Setup and Installation
======================

Dependencies
############
This software is dependent on open source programs that can be installed using OS-specific package management systems,
`conda <https://anaconda.org/conda-forge/repo>`_ or from source:

- `MPI <https://www.open-mpi.org/>`_
- `GDAL <https://gdal.org/index.html>`_
- `GEOS <https://trac.osgeo.org/geos>`_
- `PROJ <https://proj.org/>`_
- `HDF5 <https://www.hdfgroup.org>`_
- `netCDF <https://www.unidata.ucar.edu/software/netcdf>`_
- `libxml2 <http://xmlsoft.org/>`_
- `libxslt <http://xmlsoft.org/XSLT/>`_

Installation
############
Presently the software is only available for use as a
`GitHub repository <https://github.com/tsutterley/Grounding-Zones>`_.
The contents of the repository can be download as a
`zipped file <https://github.com/tsutterley/Grounding-Zones/archive/main.zip>`_  or cloned.

To use this repository, please fork into your own account and then clone onto your system:

.. code-block:: bash

    git clone https://github.com/tsutterley/Grounding-Zones.git

Can then install using ``setuptools``:

.. code-block:: bash

    python3 setup.py install

or ``pip``

.. code-block:: bash

    python3 -m pip install --user .

Alternatively can install the programs directly from GitHub with ``pip``:

.. code-block:: bash

    python3 -m pip install --user git+https://github.com/tsutterley/Grounding-Zones.git
