===========================
calculate_grounding_zone.py
===========================

- Calculates ice sheet grounding zones following [Brunt2010]_ [Fricker2006]_ [Fricker2009]_

`Source code`__

.. __: https://github.com/tsutterley/Grounding-Zones/blob/main/GZ/calculate_grounding_zone.py

Calling Sequence
################

.. argparse::
    :filename: calculate_grounding_zone.py
    :func: arguments
    :prog: calculate_grounding_zone.py
    :nodescription:
    :nodefault:

    --variables : @after
        * for csv files: the order of the columns within the file
        * for HDF5 and netCDF4 files: time, y, x and data variable names

    --projection : @after
        * ``4326``: latitude and longitude coordinates on WGS84 reference ellipsoid

References
##########

.. [Brunt2010] K. M. Brunt, H. A. Fricker, L. Padman, T. A. Scambos, and S. O'Neel, "Mapping the grounding zone of the Ross Ice Shelf, Antarctica, using ICESat laser altimetry", *Annals of Glaciology*, 51(55), 71--79, (2010). `doi: 10.3189/172756410791392790 <https://doi.org/10.3189/172756410791392790>`_

.. [Fricker2006] H. A. Fricker and L. Padman, "Ice shelf grounding zone structure from ICESat laser altimetry", *Geophysical Research Letters*, 33(15), L15502, (2006). `doi: 10.1029/2006GL02690 <https://doi.org/10.1029/2006GL026907>`_

.. [Fricker2009] H. A. Fricker, R. Coleman, L. Padman, T. A. Scambos, J. Bohlander, and K. M. Brunt, "Mapping the grounding zone of the Amery Ice Shelf, East Antarctica using InSAR, MODIS and ICESat", *Antarctic Science*, 21(5), 515--532, (2009). `doi: 10.1017/S095410200999023X <https://doi.org/10.1017/S095410200999023X>`_
