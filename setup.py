import os
from setuptools import setup

# list of all scripts to be included with package
scripts=[]
for s in ['scripts','subset','DAC','DEM','geoid','GZ','SL','tides']:
    scripts.extend([os.path.join(s,f) for f in os.listdir(s) if f.endswith('.py')])

setup(
    name='grounding-zones',
    scripts=scripts,
)
