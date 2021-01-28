MPI_interpolate_DEM.py
======================

- Determines which digital elevation model tiles for an input file (ascii, netCDF4, HDF5, geotiff)
- Reads 3x3 array of tiles for points within bounding box of central mosaic tile
- Interpolates digital elevation model to coordinates

- ArcticDEM 2m digital elevation model tiles
    * [http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v3.0/](http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v3.0/)
    * [http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/indexes/](http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/indexes/)

- REMA 8m digital elevation model tiles
    * [http://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v1.1/](http://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v1.1/)
    * [http://data.pgc.umn.edu/elev/dem/setsm/REMA/indexes/](http://data.pgc.umn.edu/elev/dem/setsm/REMA/indexes/)

- GIMP 30m digital elevation model tiles
    * [https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0645.001/](https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0645.001/)

#### Calling Sequence
```bash
mpiexec -np <processes> python3 MPI_interpolate_DEM.py --model <model> \
    --format <format> --verbose input_file output_file
```
[Source code](https://github.com/tsutterley/ICESat-2-Grounding-Zones/blob/main/scripts/MPI_interpolate_DEM.py)

#### Inputs
 1. `input_file`: name of input file
 2. `output_file`: name of output file

#### Command Line Options
- `-D X`, `--directory X`: Working data directory
- `-m X`, `--model X`: Digital elevation model to run
    * `'REMA'`
    * `'ArcticDEM'`
    * `'GIMP'`
- `--format X`: input and output data format
    * `'csv'` (default)
    * `'netCDF4'`
    * `'HDF5'`
    * `'geotiff'`
- `--variables X`: variable names of data in csv, HDF5 or netCDF4 file
    * for csv files: the order of the columns within the file
    * for HDF5 and netCDF4 files: time, y, x and data variable names
- `-H X`, `--header X`: number of header lines for csv files
- `-t X`, `--type X`: input data type
    * `'drift'`: drift buoys or satellite/airborne altimetry (time per data point)
    * `'grid'`: spatial grids or images (single time for all data points)
- `--projection X`: spatial projection as EPSG code or PROJ4 string
    * `4326`: latitude and longitude coordinates on WGS84 reference ellipsoid
- `-V`, `--verbose`: Verbose output of processing run
- `-M X`, `--mode X`: Permission mode of output file
