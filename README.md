# Use

Build code with
```cpp
make
```

This will generate 3 files inside `build` directory: `kmeansSequential`, `kmeansCuda`, `kmeansParallel` (openmp implementation).

You can try them passing the dataset and starting centroids csv files as an argument:
Example:

```cpp
./build/kmeansSequential datasetUtils/generatedDatasets/1000_5.csv datasetUtils/generatedDatasets/1000_5_centroids.csv
```

This will display the elapsed time.


## Display final centroids
To display final centroids build with `PRINT_FINAL_CENTROIDS` flag enabled:

```cpp
make PRINT_FINAL_CENTROIDS=1
```

## Export points to csv
To enable export of the end result (\*) (as csv) at the end of execution, build with `EXPORT_FINAL_RESULT` flag enabled:
```cpp
make PRINT_FINAL_CENTROIDS=1
```

(\*) The csv file will have the following form
```cpp
point0_x, point0_y, point0_cluster
point1_x, point1_y, point1_cluster
...
```
You can view the result using the `visualize_result.py` file inside `datasetUtils` directory.


## Generate new dataset
You can generate new datasets running the `create_dataset.py` file inside `datasetUtils` directory.
You can change the number of point and cluster number inside the file.


## Run test
The `runAnalysisUtils` directory contains some python files to automatically run tests.
`run_test.py` for speedup analysis, `run_test_threadperblock.py` to compare thread per block change effect.