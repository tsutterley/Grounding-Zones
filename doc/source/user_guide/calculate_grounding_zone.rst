===========================
calculate_grounding_zone.py
===========================

- Calculates ice sheet grounding zones following:
    * `Brunt et al., Annals of Glaciology, 51(55), 2010 <https://doi.org/10.3189/172756410791392790>`_
    * `Fricker et al. Geophysical Research Letters, 33(15), 2006 <https://doi.org/10.1029/2006GL026907>`_
    * `Fricker et al. Antarctic Science, 21(5), 2009 <https://doi.org/10.1017/S095410200999023X>`_

`Source code`__

.. __: https://github.com/tsutterley/Grounding-Zones/blob/main/scripts/calculate_grounding_zone.py

Calling Sequence
################

.. argparse::
    :filename: ../scripts/calculate_grounding_zone.py
    :func: arguments
    :prog: calculate_grounding_zone.py
    :nodescription:
    :nodefault:

    --variables : @after
        * for csv files: the order of the columns within the file
        * for HDF5 and netCDF4 files: time, y, x and data variable names

    --projection : @after
        * ``4326``: latitude and longitude coordinates on WGS84 reference ellipsoid
