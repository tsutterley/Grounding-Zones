=============================
calculate_GZ_ICESat2_ATL11.py
=============================

- Calculates ice sheet grounding zones using ICESat-2 annual land ice height data following:

    * `Brunt et al., Annals of Glaciology, 51(55), 2010 <https://doi.org/10.3189/172756410791392790>`_
    * `Fricker et al. Geophysical Research Letters, 33(15), 2006 <https://doi.org/10.1029/2006GL026907>`_
    * `Fricker et al. Antarctic Science, 21(5), 2009 <https://doi.org/10.1017/S095410200999023X>`_
- Outputs an HDF5 file of flexure scaled to match the downstream tide model
- Outputs the grounding zone location, time and spatial uncertainty

`Source code`__

.. __: https://github.com/tsutterley/Grounding-Zones/blob/main/scripts/calculate_GZ_ICESat2_ATL11.py

Calling Sequence
################

.. argparse::
    :filename: ../scripts/calculate_GZ_ICESat2_ATL11.py
    :func: arguments
    :prog: calculate_GZ_ICESat2_ATL11.py
    :nodescription:
    :nodefault:

    --reanalysis -R : @after

    * `ERA-Interim <http://apps.ecmwf.int/datasets/data/interim-full-moda>`_
    * `ERA5 <http://apps.ecmwf.int/data-catalogues/era5/?class=ea>`_
    * `MERRA-2 <https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/>`_
