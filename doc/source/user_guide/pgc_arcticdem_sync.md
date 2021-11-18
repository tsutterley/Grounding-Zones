pgc_arcticdem_sync.py
=====================

 - Syncs [ArcticDEM tar files](http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic) from the [Polar Geospatial Center (PGC)](https://www.pgc.umn.edu/data/)

#### Calling Sequence
```bash
python3 pgc_arcticdem_sync.py --directory <outgoing> --version v3.0 \
    --resolution 2m --mode 0o775
```
[Source code](https://github.com/tsutterley/Grounding-Zones/blob/main/scripts/pgc_arcticdem_sync.py)

#### Command Line Options
 - `-D X`, `--directory`: working data directory
 - `-v X`, `--version X:` ArcticDEM version
    * `'v2.0'`
    * `'v3.0'`
 - `-r X`, `--resolution X`: ArcticDEM spatial resolution
    * `'2m'`
    * `'10m'`
    * `'32m'`
    * `'100m'`
    * `'500m'`
    * `'1km'`
 - `-t X`, `--tile X`: ArcticDEM tiles to sync
 - `--list`: print files to be transferred, but do not execute transfer
 - `--log`: output log of files downloaded
 - `-C`, `--clobber`: Overwrite existing data in transfer
 - `-M X`, `--mode X`: Local permissions mode of the directories and files synced
