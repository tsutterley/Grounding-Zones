pgc_rema_sync.py
================

 - Syncs [Reference Elevation Map of Antarctica (REMA) DEM tar files](http://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic) from the [Polar Geospatial Center (PGC)](https://www.pgc.umn.edu/data/)

#### Calling Sequence
```bash
python3 pgc_rema_sync.py --directory <outgoing> --version v1.1 \
    --resolution 8m --mode 0o775
```
[Source code](https://github.com/tsutterley/ICESat-2-Grounding-Zones/blob/main/scripts/pgc_rema_sync.py)

#### Command Line Options
 - `-D X`, `--directory`: working data directory
 - `-v X`, `--version X:` REMA DEM version
    * `'v1.0'`
    * `'v1.1'`
 - `-r X`, `--resolution X`: REMA DEM spatial resolution
    * `'8m'`
    * `'100m'`
    * `'200m'`
    * `'1km'`
 - `-t X`, `--tile X`: REMA DEM tiles to sync
 - `--list`: print files to be transferred, but do not execute transfer
 - `--log`: output log of files downloaded
 - `-C`, `--clobber`: Overwrite existing data in transfer
 - `-M X`, `--mode X`: Local permissions mode of the directories and files synced
