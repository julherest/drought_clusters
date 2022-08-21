"""
This script is used to identify 2D drought clusters for each time step separately. It parallelizes the for-loop through the
different time steps using mpi4py.

Here is an example of how this code should be run: 
mpirun -np 4 python 02_caculate_drought_clusters_parallel.py

Written by Julio E. Herrera Estrada, Ph.D.
"""

# Import libraries
import yaml
import numpy as np
from mpi4py import MPI
import cPickle as pickle
from netCDF4 import Dataset
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Import custom libraries
import drought_clusters_utils as dclib

# Initiate communicator and determine the core number and number of cores
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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

# Set boolean variable of whether to treat the right/left edges of the map as periodic
periodic_bool = definitions["periodic_bool"]

# Path and file name of the NetCDF file with the drought metric
drought_metric_path = definitions["drought_metric_path"]
drought_metric_file_name = definitions["drought_metric_file_name"]

# Names of the variables in the NetCDF file with the drought metric
lat_var = definitions["lat_var"]
lon_var = definitions["lon_var"]
metric_var = definitions["metric_var"]

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

# Threshold for minimum cluster area (km^2)
minimum_area_threshold = definitions["minimum_area_threshold"]

######################## DONE SETTING PATHS AND DEFINTIONS #######################

# Load the 3D array with the drought metric (t, lat, lon)
f = Dataset(drought_metric_path + drought_metric_file_name)
drought_metric = f.variables[metric_var][:]
lons = f.variables[lon_var][:]
lats = f.variables[lat_var][:]
f.close()

# Set date time objects and the number of time steps
start_date = datetime(start_year, 1, 1)
nsteps = (end_year - start_year + 1) * 12
date_temp = start_date

# Spatial resolution of dataset in each direction
resolution_lon = np.mean(lons[1:] - lons[:-1])
resolution_lat = np.mean(lats[1:] - lats[:-1])

##################################################################################
#################### IDENTIFY DROUGHT CLUSTERS (PER TIME STEP) ###################
##################################################################################

# Function to carry out analysis in parallel. Each core is given a chunk of the time steps to analyze.
def find_clusters(chunk):

    # Length of the chunk
    chunk_length = len(chunk)

    # Repeat analysis for each time step within the assigned chunck
    for i in range(0, chunk_length):

        # Current date
        current_date = start_date + relativedelta(months=int(chunk[i]))

        # STEP 1: GET DATA FOR THE CURRENT TIME STEP
        current_data_slice = drought_metric[int(chunk[i]), :, :]

        # STEP 2: APPLY MEDIAN FILTER TO THE TIME STEP IN EACH FIELD TO SMOOTH OUT NOISE
        filtered_slice = dclib.median_filter(current_data_slice)

        # STEP 3: APPLY DROUGHT THRESHOLD DEFINITION (e.g. 20th percentile)
        droughts = dclib.filter_non_droughts(filtered_slice, drought_threshold)

        # STEP 4: IDENTIFY DROUGHT CLUSTERS PER TIME STEP
        print(
            "Rank "
            + str(rank + 1)
            + ": Identifying clusters for time step "
            + str(int(chunk[i]) + 1)
            + " of "
            + str(nsteps)
            + " ("
            + str(i + 1)
            + "/"
            + str(chunk_length)
            + ")..."
        )
        cluster_count, cluster_dictionary = dclib.find_drought_clusters(
            droughts, lons, lats, resolution_lon, resolution_lat, periodic_bool
        )

        # STEP 5: FILTER DROUGHT CLUSTERS BY AREA AND IF THE CENTROID LIES IN THE SAHARA
        droughts, cluster_count, cluster_dictionary = dclib.filter_drought_clusters(
            droughts, cluster_count, cluster_dictionary, minimum_area_threshold
        )

        # STEP 6: SAVE THE DROUGHT CLUSTERS FOR CURRENT TIME STEP

        # Paths and file names for saving data
        f_name_slice = (
            clusters_full_path + "cluster-matrix_" + str(current_date) + ".pck"
        )
        f_name_dictionary = (
            clusters_full_path + "cluster-dictionary_" + str(current_date) + ".pck"
        )
        f_name_count = (
            clusters_full_path + "cluster-count_" + str(current_date) + ".pck"
        )

        # Save the data in pickle format
        pickle.dump(droughts, open(f_name_slice, "wb"), pickle.HIGHEST_PROTOCOL)
        pickle.dump(
            cluster_dictionary, open(f_name_dictionary, "wb"), pickle.HIGHEST_PROTOCOL
        )
        pickle.dump(cluster_count, open(f_name_count, "wb"), pickle.HIGHEST_PROTOCOL)
        print(
            "Rank "
            + str(rank + 1)
            + ": Saved data for time step "
            + str(int(chunk[i]) + 1)
            + " of "
            + str(nsteps)
            + " ("
            + str(i + 1)
            + "/"
            + str(chunk_length)
            + ")."
        )

    return


# Number of steps for each processor
offset = 0
h = np.ceil(nsteps / np.float32(size - offset))

# Number of steps that each process will be required to do
if rank >= offset and rank < size - 1:
    chunk = np.arange((rank - offset) * h, (rank - offset) * h + h)
elif rank == size - 1:
    chunk = np.arange((rank - offset) * h, nsteps)

# Identify drought clusters for the current chunk of data
find_clusters(chunk)
