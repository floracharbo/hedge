# HEDGE: Home energy data generator
Contact: Flora Charbonnier, flora.charbonnier@eng.ox.ac.uk


## What is it?

The aim of the HEDGE tool is to randomly generate realistic home photovoltaic (PV) generation, electricity consumption, and electric vehicle consumption and availability daily profiles, based on UK historical datasets. 
It can for example generate data that can be used to train neural networks.

More details on the aims and methodology in [related paper here](https://arxiv.org/abs/2310.01661).

This README guides you through (A) how to prepare the tool dataset, (B) how to use the tool


## Where to get it
The source code for **HEDGE** is currently hosted on GitHub at: https://github.com/floracharbo/HEDGE

## Usage


### A. Preparing the HEDGE tool dataset [skip if using hourly resolution]
Perform this step once to prepare the data that will be used by the HEDGE tool
1. If you wish to generate car data, 
- Find the dataset Study Number (SN) 5340 "National Travel Survey, 2002-2020" on the UK Data Service website
https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=5340
- Register with the UK Data Service, add the dataset to a project, and download the data in TAB format
- Move the 'UKDA-5340-tab' folder to the `data/data_preparation_inputs` folder
3. If you wish to generate PV data,
- Find the dataset "Enhanced profiling of domestic customers with solar photovoltaics (PV)" on the Customer-Led Network Revolution website http://www.networkrevolution.co.uk/resources/project-data/
- Download and move the 'TC5' folder to the `data/data_preparation_inputs` folder
4. If you wish to generate household demand data,
- Find the dataset "Basic profiling of domestic smart meter customers" on the Customer-Led Network Revolution website http://www.networkrevolution.co.uk/resources/project-data/
- Download and move the 'TC1a' folder to the `data/data_preparation_inputs` folder
5. From the terminal, clone the `hedge` GitHub repository 
6. Input personal settings in `config_parameters/data_preparation_config.yaml`
7. Create virtual environment: 
```sh
python3 -m venv my_venv
```
8. Activate virtual environment: 
```sh
source my_venv/bin/activate
```
9. Install requirements: 
```sh
make install
```
10. Run the data preparation programme 
```sh
python3 -m src/prepare_data.py
```
Note that this step takes some time due to the large size of the datasets (e.g. ~1 day for hourly resolution data, as performed on an Intel(R) Core(TM) i7-9800X CPU @ 3.80 GHz). 

If you hit RAM issues and the code is unable to run on your machine, you may find that reducing `max_size_chunk` in `config_parameters/data_preparation_config.yaml` may help (though this will increase run time). Alternatively, you can run the code for one data type at a time (setting this in `config_parameters/data_preparation_config.yaml`, `data_types`).

### B. Using the HEDGE tool (example in `test_hedge.py`)
1. Inputs personal parameters in `config_parameters/hedge_config.yaml`
2. Import home energy data generator object
`from src.hedge import HEDGE`
3. Create home energy data generator object instance, with at least the number of homes as an input
`data_generator = HEDGE(n_homes)`.

other optional inputs are:
- factors0; the initial scaling factors in the format factors0[data_type][home]
  - where data_type is in ['car', 'PV', 'loads']
  - where home is an integer in [0, n_homes[
- clusters0; the initial behaviour clusters in the format factors0[data_type][home]
  - where data_type is in ['car', 'loads']
  - where home is an integer in [0, n_homes[
- data_types; the types of data the user would like to produce, it has to list one, two of three of the strings ['car', 'PV', 'loads']; if left blank all three types will be generated
4. For each new day to be generated, call hedge.
`day = data_generator.make_next_day()`

where day contains the following entries:
- `day['avail_car']` (`n_homes` x `n`) integers 0 for unavailable, 1 for available]
  where n was defined in run_config.yaml as the number of time intervals per day
- `day['loads_car']` (`n_homes` x `n`) car consumption in kWh for each time interval of the day
- `day['gen']` (`n_homes` x `n`) PV generation in kWh for each time interval of the day
- `day['loads']` (`n_homes` x `n`) household consumption in kWh for each time interval of the day

an optional parameter to this method is `plotting`, which will save plots of the data that has been generated for each home.

Note that the code was tested on macOS Monterey 12.6, Python 3.9 and on Ubuntu 20.04.5 LTS (Focal Fossa), Python 3.8.10.