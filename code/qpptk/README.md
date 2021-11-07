# qpptk - Query Performance Prediction Toolkit

This package was written with python 3.7, it wasn't tested in any other versions.
Currently the package uses a non-standard python package for [timing](https://github.com/nadavo/Timer),

It should be installed by simply running:\
`pip install -e git+https://github.com/Zendelo/qpptk` \
[more about the -e option](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs) 

Or by cloning the repo and installing with:\
`pip install -e [path to qpptk repo]`

If the Timer package isn't installed by pip, it can be installed manually by running:\
`pip install git+https://github.com/nadavo/Timer.git`

The configurations are located in config.toml, some are also available through the command line.

To run prediction or retrieval see the usage example below:

##### To run retrieval and predictions on all the ciff indexes and matching ciff queries:
Modify the paths and run the script:\
`python3.7 generate_all_script.py`

#### For specific results:

###### run the module qpptk_main.py
`python3.7 qpptk/qpptk_main.py --help`:

 
```
usage: qpptk_main.py [-h] [-ti INDEX | -ci INDEX] [-tq QUERIES | -cq QUERIES]
                     [--retrieve] [--predict]

Run QL retrieval or Query Performance Prediction

optional arguments:
  -h, --help            show this help message and exit
  -ti INDEX, --text_index INDEX
                        path to text index dir
  -ci INDEX, --ciff_index INDEX
                        path to ciff index file
  -tq QUERIES, --text_queries QUERIES
                        path to text queries file
  -cq QUERIES, --ciff_queries QUERIES
                        path to ciff queries file
  --retrieve            add this flag to run retrieval
  --predict             add this flag to run predictions
```
###### In order to run predictions (given that the index and queries paths are set in the `config.toml` file)

`python3.7 qpptk/qpptk_main.py --predict`
```
28/04/2020 23:39:48 INFO: Started initialize_text_index
28/04/2020 23:39:51 INFO: initialize_text_index took 2.73 seconds to complete
28/04/2020 23:39:51 INFO: Started initialize_text_queries
28/04/2020 23:39:51 INFO: initialize_text_queries took 5.44 ms to complete
28/04/2020 23:39:51 INFO: Started run_prediction_process
301 finished
28/04/2020 23:39:51 INFO: run_prediction_process took 0.49 ms to complete
28/04/2020 23:39:51 INFO: Started run_prediction_process
302 finished
28/04/2020 23:39:51 INFO: run_prediction_process took 0.24 ms to complete
28/04/2020 23:39:51 INFO: Started run_prediction_process
303 finished
28/04/2020 23:39:51 INFO: run_prediction_process took 0.25 ms to complete
...
...
699 finished
28/04/2020 23:39:51 INFO: run_prediction_process took 0.20 ms to complete
28/04/2020 23:39:51 INFO: Started run_prediction_process
700 finished
28/04/2020 23:39:51 INFO: run_prediction_process took 0.19 ms to complete

```

Running retrieval or predictions will generate new files with the results,
the file names will begin with the index name.
All the results will be saved in the directory that is set with the `results_dir` in the `config.toml` file.