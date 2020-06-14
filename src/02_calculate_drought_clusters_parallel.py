'''
This script is used to test parallelization of a for-loop process.
Example: mpirun -np 4 python 02_caculate_drought_clusters_parallel.py
'''
# Import library
import numpy as np
from mpi4py import MPI
import cPickle as pickle
from netCDF4 import Dataset
from datetime import datetime
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

# Import custom libraries
import drought_clusters_utils as dclib

#Initiate communicator and determine the core number and number of cores
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

##################################################################################
############################ SET PATHS AND DEFINITIONS ###########################
##################################################################################

# ** Name of the reanalysis dataset to process (options: "ERA-Interim", "MERRA2", "CFSR", "NCEP-DOE-R2")
reanalysis = 'CFSR'

# ** Region
region = 'World'

# ** Window of the cumulative anomalies of precipitation minus evapotranspiration (months)
window = 12
anomalies_name = 'pme_' + str(window) + 'months'

# ** Threshold for drought definition (percentile/100)
drought_threshold = 0.2
drought_threshold_name = str(int(100*drought_threshold)) + 'th_percentile'

# ** Version of the analysis
version = 'v4'

# Start and end years for the data of the chosen reanalysis (add extra year to the start given the cumulative analysis)
if reanalysis in ['ERA-Interim', 'NCEP-DOE-R2', 'CFSR']:
    start_year = 1980
    end_year = 2018
elif reanalysis in ['MERRA2']:
    start_year = 1981
    end_year = 2018

# ** Set boolean variable of whether to treat the right/left edges of the map as periodic
periodic_bool = True

# ** Path where the percentiles are saved
percentiles_path = '/oak/stanford/groups/omramom/group_members/jehe/Ocean_Clusters/' + version + '/' + reanalysis + '/' + region + '/' + anomalies_name + '/percentiles_' + reanalysis + '_' + str(start_year) + '-' + str(end_year) + '.nc'

# ** Path where the drought clusters will be saved
clusters_path = '/oak/stanford/groups/omramom/group_members/jehe/Ocean_Clusters/' + version + '/' + reanalysis + '/' + region + '/' + anomalies_name + '/' + drought_threshold_name + '/'

############################ DONE EDITING INFORMATION ############################

# Load the percentiles matrix 
f = Dataset(percentiles_path)
percentiles_matrix = f.variables['percentiles'][:]
lons = f.variables['lon'][:]
lats = f.variables['lat'][:]
f.close()

# Dimensions of the dataset
nsteps, nlats, nlons = percentiles_matrix.shape

# Set dates
start_date = datetime(start_year,1,1)
date_temp = start_date
end_date = start_date + relativedelta(months=nsteps-1)

# Resolution of dataset in the longitudes and latitudes
resolution_lon = np.mean(lons[1:]-lons[:-1])
resolution_lat = np.mean(lats[1:]-lats[:-1])

# Threshold for minimum cluster area (km^2)
area_threshold = 10000	

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

            # Current slice
    	    current_data_slice = percentiles_matrix[int(chunk[i]),:,:]
	
	    # STEP 1: APPLY MEDIAN FILTER TO THE TIME STEP IN EACH FIELD TO SMOOTH OUT NOISE
	    filtered_slice = dclib.median_filter(current_data_slice)

	    # STEP 2: APPLY DROUGHT THRESHOLD DEFINITION (e.g. 20th percentile)
	    droughts = dclib.filter_non_droughts(filtered_slice, drought_threshold)
    
	    # STEP 3: IDENTIFY DROUGHT CLUSTERS PER TIME STEP
            print('Rank ' + str(rank+1) + ': Identifying clusters for time step ' + str(int(chunk[i])+1) + ' of ' + str(nsteps) + ' (' + str(i+1) + '/' + str(chunk_length) + ')...')
	    cluster_count, cluster_dictionary = dclib.find_drought_clusters(droughts, lons, lats, resolution_lon, resolution_lat, periodic_bool)

	    # STEP 4: FILTER DROUGHT CLUSTERS BY AREA AND IF THE CENTROID LIES IN THE SAHARA
	    droughts, cluster_count, cluster_dictionary = dclib.filter_drought_clusters(droughts, cluster_count, cluster_dictionary, area_threshold)

	    # STEP 5: SAVE THE DROUGHT CLUSTERS FOR CURRENT TIME STEP
	
	    # Paths and file names for saving data
	    f_name_slice = clusters_path + 'cluster-matrix_' + str(current_date) + '.pck'
	    f_name_dictionary = clusters_path + 'cluster-dictionary_' + str(current_date) + '.pck'
	    f_name_count = clusters_path + 'cluster-count_' + str(current_date) + '.pck'

	    # Save the data in pickle format
	    pickle.dump(droughts,open(f_name_slice,"wb"),pickle.HIGHEST_PROTOCOL)
	    pickle.dump(cluster_dictionary,open(f_name_dictionary,"wb"),pickle.HIGHEST_PROTOCOL)
	    pickle.dump(cluster_count,open(f_name_count,"wb"),pickle.HIGHEST_PROTOCOL)
            print('Rank ' + str(rank+1) + ': Saved data for time step ' + str(int(chunk[i])+1) + ' of ' + str(nsteps) + ' (' + str(i+1) + '/' + str(chunk_length) + ').')

        return 
    
# Number of steps for each processor
offset = 0
h = np.ceil(nsteps/np.float32(size-offset))

# Number of steps that each process will be required to do
if rank >= offset and rank < size-1:
	chunk = np.arange((rank-offset)*h,(rank-offset)*h+h)
elif rank == size-1:
	chunk = np.arange((rank-offset)*h,nsteps)

# Identify drought clusters for the current chunk of data 
find_clusters(chunk)
