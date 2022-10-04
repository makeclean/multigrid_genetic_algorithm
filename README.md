Overview
========

This is a prototype for a Multiscale Grid Genetic Algorithm based on that of [this PhD thesis by Stephen Asbury](https://deepblue.lib.umich.edu/bitstream/handle/2027.42/91388/stasbury_1.pdf?sequence=1&isAllowed=y).

The idea is to optimise OpenMC geometries for given quanities of interest at increasing resolution.

Prerequisites
=============
 - OpenMC: Please follow the installation instructions [here.](https://docs.openmc.org/en/stable/usersguide/install.html). 

Usage
=====
```
python make_geom.py [-h] [--initialise] [--generate GENERATE] --input INPUT

required:
--input INPUT  JSON file containing input settings

optional arguments:
  -h, --help     show this help message and exit
  --initialise   In this run mode, initialise the first generation population
  --generate     In this run mode, initialise the next generation population
```

