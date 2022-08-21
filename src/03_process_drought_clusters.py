"""
This script processes the output of the drought cluster identification algorithm 
carried out by the script named 02_calculate_drought_clusters_parallel.py.

This script only needs to be run once and it will save the dictionary with all the 
drought clusters' data that we can then go and analyze.. 

Written by Julio E. Herrera Estrada, Ph.D.
"""

# Import Python libraries
import yaml
import numpy as np
import cPickle as pickle
from netCDF4 import Dataset
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Import custom libraries
import drought_clusters_utils as dclib

##################################################################################
############################ SET PATHS AND DEFINITIONS ###########################
##################################################################################

# Load all the definitions needed to run this file
with open("definitions.yaml") as f:
    definitions = yaml.load(f, Loader=yaml.FullLoader)

# Name of the dataset used to calculate gridded drought metric
dataset = definitions["dataset"]

# Region where this analysis is carried out
region = definitions["region"]

# Name of the drought metric
drought_metric = definitions["drought_metric"]

# Threshold for drought definition
drought_threshold = definitions["drought_threshold"]
drought_threshold_name = str(drought_threshold)

# Start and end years for the timer period for which we will identify the drought clusters
start_year = definitions["start_year"]
end_year = definitions["end_year"]

# Path and file name of the NetCDF file with the drought metric
drought_metric_path = definitions["drought_metric_path"]
drought_metric_file_name = definitions["drought_metric_file_name"]

# Names of the variables in the NetCDF file with the drought metric
lat_var = definitions["lat_var"]
lon_var = definitions["lon_var"]

# Path where the drought clusters will be saved
clusters_partial_path = definitions["clusters_partial_path"]
clusters_full_path = (
    clusters_partial_path
    + "/"
    + dataset
    + "/"
    + region
    + "/"
    + drought_metric
    + "/"
    + drought_threshold_name
    + "/"
)

##################################################################################
####################### TRACK DROUGHT CLUSTERS THROUGH TIME ######################
##################################################################################

# Start and end date and number of time steps in between
start_date = datetime(start_year, 1, 1)
nt = (end_year - start_year + 1) * 12
end_date = start_date + relativedelta(months=nt - 1)

# Load coordinates
f = Dataset(drought_metric_path + drought_metric_file_name)
lons = f.variables[lon_var][:]
lats = f.variables[lat_var][:]
f.close()

# Track drought clusters through time (Note: only need to run once after calculating drought clusters)
dclib.track_clusters_and_save(
    clusters_full_path,
    start_date,
    end_date,
    nt,
    lons,
    lats,
    drought_threshold_name,
    dataset,
)
print("Done tracking drought clusters.")
