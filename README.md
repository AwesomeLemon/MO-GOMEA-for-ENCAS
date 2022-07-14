# MO-GOMEA

Modified version of the official implementation (https://homepages.cwi.nl/~bosman/source_code.php)

I introduced the following changes:
1) using fitness functions written in Python. Note that it's not very general (i.e. restricted to my NAS use cases), but should be relatively easy to adapt to your use case; 
2) each variable can take more than 2 values, and different variables may have different alphabet sizes;
3) setting random seed to a specific value;
4) executing a precise (user-provided) amount of evaluations.

## How to compile

1) set `LD_LIBRARY_PATH`:
`export LD_LIBRARY_PATH="/usr/lib64:path_to_conda_lib"`,
where `path_to_conda_lib` is something like `/export/scratch1/home/aleksand/miniconda3/envs/supernetevo/lib`
2) Set `LIBRARY_PATH` to `path_to_conda_lib` (which is the same as in the previous step)
3) In CMakeLists.txt change ``target_include_directories`` to refer to the header files of your Python distribution.

4) run
```
/usr/bin/cmake -DCMAKE_BUILD_TYPE=Release -G "CodeBlocks - Unix Makefiles" /export/scratch3/aleksand/MO_GOMEA/ -B/export/scratch3/aleksand/MO_GOMEA/cmake-build-release-remote
/usr/bin/cmake --build /export/scratch3/aleksand/MO_GOMEA/cmake-build-release-remote --target MO_GOMEA -- -j 6
```
where you should replace `/export/scratch3/aleksand/MO_GOMEA/` with the path to where you cloned the repo

## How to use

Refer to ENCAS repo, it has a Python interface to this executable. 

To be clear, the relationship with Python is bi-directional. 
1) Python code calls C executable to perform search, and gets the results from it.
2) Within the C executable, Python fitness functions are called.

For proper communication between Python and C, you need to set `PYTHONPATH`: `export PYTHONPATH=PATH_TO_ENCAS`,
where `PATH_TO_ENCAS` is the path where you cloned ENCAS (e.g. `"/export/scratch3/aleksand/encas"`)

## Citations

If you found this implementation helpful, please consider citing the following papers:

MO-GOMEA:
```
@inproceedings{luong2014multi,
  title={Multi-objective gene-pool optimal mixing evolutionary algorithms},
  author={Luong, Ngoc Hoang and La Poutr{\'e}, Han and Bosman, Peter A N},
  booktitle={Proceedings of the 2014 Annual Conference on Genetic and Evolutionary Computation},
  pages={357--364},
  year={2014}
}
```

ENCAS:
```
@inproceedings{10.1145/3512290.3528749,
	author = {Chebykin, Alexander and Alderliesten, Tanja and Bosman, Peter A. N.},
	title = {Evolutionary Neural Cascade Search across Supernetworks},
	year = {2022},
	isbn = {9781450392372},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	url = {https://doi.org/10.1145/3512290.3528749},
	doi = {10.1145/3512290.3528749},
	booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
	pages = {1038–1047},
	numpages = {10},
	series = {GECCO '22}
}
```
