==============================
MPI_median_elevation_filter.py
==============================

- Filters elevation change rates from triangulated Operation IceBridge data using an interquartile range algorithm described by [Pritchard2009]_ and a robust dispersion estimator (RDE) filter described in [Smith2017]_

`Source code`__

.. __: https://github.com/tsutterley/Grounding-Zones/blob/main/scripts/MPI_median_elevation_filter.py

Calling Sequence
################

.. argparse::
    :filename: MPI_median_elevation_filter.py
    :func: arguments
    :prog: MPI_median_elevation_filter.py
    :nodescription:
    :nodefault:

References
##########

.. [Pritchard2009] H. D. Pritchard, R. J. Arthern, D. G. Vaughan, and L. A. Edwards, "Extensive dynamic thinning on the margins of the Greenland and Antarctic ice sheets", *Nature*, 461(7266), 971--975, (2009). `doi: 10.1038/nature08471 <https://doi.org/10.1038/nature08471>`_

.. [Smith2017] B. E. Smith, N. Gourmelen, A. Huth, and I. Joughin, "Connected subglacial lake drainage beneath Thwaites Glacier, West Antarctica", *The Cryosphere*, 11(1), 451--467, (2017). `doi: 10.5194/tc-11-451-2017 <https://doi.org/10.5194/tc-11-451-2017>`_
