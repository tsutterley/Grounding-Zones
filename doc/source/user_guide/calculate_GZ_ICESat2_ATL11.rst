=============================
calculate_GZ_ICESat2_ATL11.py
=============================

- Calculates ice sheet grounding zones using ICESat-2 annual land ice height data following [Brunt2010]_ [Fricker2006]_ [Fricker2009]_
- Outputs an HDF5 file of flexure scaled to match the downstream tide model
- Outputs the grounding zone location, time and spatial uncertainty

`Source code`__

.. __: https://github.com/tsutterley/Grounding-Zones/blob/main/GZ/calculate_GZ_ICESat2_ATL11.py

Calling Sequence
################

.. argparse::
    :filename: ../../GZ/calculate_GZ_ICESat2_ATL11.py
    :func: arguments
    :prog: calculate_GZ_ICESat2_ATL11.py
    :nodescription:
    :nodefault:

    --reanalysis -R : @after

    * `ERA-Interim <http://apps.ecmwf.int/datasets/data/interim-full-moda>`_
    * `ERA5 <http://apps.ecmwf.int/data-catalogues/era5/?class=ea>`_
    * `MERRA-2 <https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/>`_

References
##########

.. [Brunt2010] K. M. Brunt, H. A. Fricker, L. Padman, T. A. Scambos, and S. O'Neel, "Mapping the grounding zone of the Ross Ice Shelf, Antarctica, using ICESat laser altimetry", *Annals of Glaciology*, 51(55), 71--79, (2010). `doi:10.3189/172756410791392790 <https://doi.org/10.3189/172756410791392790>`_

.. [Fricker2006] H. A. Fricker and L. Padman, "Ice shelf grounding zone structure from ICESat laser altimetry", *Geophysical Research Letters*, 33(15), L15502, (2006). `doi:10.1029/2006GL02690 <https://doi.org/10.1029/2006GL026907>`_

.. [Fricker2009] H. A. Fricker, R. Coleman, L. Padman, T. A. Scambos, J. Bohlander, and K. M. Brunt, "Mapping the grounding zone of the Amery Ice Shelf, East Antarctica using InSAR, MODIS and ICESat", *Antarctic Science*, 21(5), 515--532, (2009). `doi:10.1017/S095410200999023X <https://doi.org/10.1017/S095410200999023X>`_
