MPI_reduce_ICESat2_ATL03_grounding_zone.py
==========================================

- Create masks for reducing ICESat-2 geolocated photon height data to within a buffer region near the ice sheet grounding zone
- Used to calculate a more definite grounding zone from the ICESat-2 data

#### Calling Sequence
```bash
mpiexec -np <processes> python3 MPI_reduce_ICESat2_ATL03_ice_shelves.py <path_to_ATL03_file>
```
[Source code](https://github.com/tsutterley/ICESat-2-Grounding-Zones/blob/main/scripts/MPI_reduce_ICESat2_ATL03_grounding_zone.py)

#### Inputs
1. `ATL03_file`: full path to ATL03 file

#### Command Line Options
- `-D X`, `--directory X`: Working data directory for ice shelf shapefiles
- `-B X`, `--buffer X`: Distance in kilometers to buffer ice shelves mask
- `-V`, `--verbose`: output module information for process
- `-M X`, `--mode X`: permissions mode of output HDF5 datasets
