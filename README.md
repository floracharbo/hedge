# HEDGE: Home energy data generator
Contact: Flora Charbonnier, flora.charbonnier@eng.ox.ac.uk

The aim of the HEDGE tool is to randomly generate realistic home photovoltaic (PV) generation, electricity consumption, and electric vehicle (EV) consumption and availability daily profiles.
More details on the aims and methodology here:

This README guides you through (A) how to prepare the tool dataset, (B) how to use the tool


# A. Preparing the HEDGE tool dataset
Perform this step once to prepare the data that will be used by the HEDGE tool
1. Create an input_data folder at the location of your choice. You will populate this folder with input data for any combination of data you are intersted in (demand, PV, EV)
2. If you wish to generate EV data, 
- Find the dataset Study Number (SN) 5340 "National Travel Survey, 2002-2020" on the UK Data Service website
https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=5340
- Register with the UK Data Service, add the dataset to a project, and download the data in TAB format
- Move the 'UKDA-5340-tab' folder to your input data folder
3. If you wish to generate PV data,
- Find the dataset "Enhanced profiling of domestic customers with solar photovoltaics (PV)" on the Customer-Led Network Revolution website http://www.networkrevolution.co.uk/resources/project-data/
- Download and move the 'TC5' folder to your input data folder
4. If you wish to generate household demand data,
- Find the dataset "Basic profiling of domestic smart meter customers" on the Customer-Led Network Revolution website http://www.networkrevolution.co.uk/resources/project-data/
- Download and move the 'TC1a' folder to your input data folder
5. From the terminal, clone the GitHub to your folder of choice 
6. Input personal settings and paths in inputs/run_config.yaml
7. In the terminal, create a virtual environment in the location of your choice
 python3 -m venv data_venv
8. activate the virtual environment
source data_venv/bin/activate
9. Navigate to the data_preparation cloned repository directory
cd /path/to/data_preparation
10. Install the required packages
python3 -m pip install -r requirements.txt
11. Run the programme python3 -m main.py
This may take some time (e.g. xxx mins on )

# B. Using the HEDGE tool
1. Make sure the 'results' folder obtain from (A) is in the 'results_path_HEDGE' directory in run_config.yaml
2. In your Python script, import the HEDGE object:
   from HEDGE import HEDGE
3. Create HEDGE object, with at least the number of homes as an input
   hedge = HEDGE(n_homes)
    other optional inputs are:
- path_inputs_path; the path to the 'inputs' folder containing the .yaml input files, if it is different to the current working directory
- factors0; the initial scaling factors in the format factors0[data_type][home]
  - where data_type is in ['EV', 'PV', 'loads']
  - where home is an integer in [0, n_homes[
- clusters0; the initial behaviour clusters in the format factors0[data_type][home]
  - where data_type is in ['EV', 'loads']
  - where home is an integer in [0, n_homes[
- day_week0; integer between 1 to 7 denoting the day of the week if the user wants to start on a specific day
- data_types; the types of data the user would like to produce, it has to list one, two of three of the strings ['EV', 'PV', 'loads']; if left blank all three types will be generated
4. for each new day, call hedge.
    day = hedge.load_next_day()
    where day contains the following entries:
- day['avail_EV'] (n_homes, n) integers 0 for unavailable, 1 for available]
  where n was defined in run_config.yaml as the number of time intervals per day
- day['loads_EV'] (n_homes, n) EV consumption in kWh for each time interval of the day
- day['gen'] (n_homes, n) PV generation in kWh for each time interval of the day
- day['loads'] (n_homes, n) household consumption in kWh for each time interval of the day