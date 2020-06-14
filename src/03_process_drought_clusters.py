'''
This script processes the output of the drought cluster identification algorithm carried out by the script named calculate_drought_clusters_parallel_v2.py 
if the identification was done in parallel or or calculate_drought_clusters_series_v2.py otherwise. This script needs to be run once for each of the 
reanalysis datasets used. The dictionaries produced by this script will then be further analyzed by the script named paper_figures_multiple_v2.py
to produce the results and figures to be used in the manuscript.

Credit: Julio E. Herrera Estrada, Stanford University, 2019. E-mail: jehe@stanford.edu
'''

# Import Python libraries
import numpy as np
import cPickle as pickle
from netCDF4 import Dataset
from datetime import datetime
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

# Import custom libraries
import paper_calculations_lib_v4 as pclib
import drought_clusters_lib_v4 as dclib

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

# ** Start and end years for the data of the chosen reanalysis (add extra year to the start given the cumulative analysis)
start_year = 1981
end_year = 2018

# ** Path where the data for this reanalysis is saved
common_path = '/oak/stanford/groups/omramom/group_members/jehe/Ocean_Clusters/' + version + '/' + reanalysis + '/' + region + '/' 

# ** Path where the drought clusters are saved
clusters_path = common_path + anomalies_name + '/' + drought_threshold_name + '/'

# ** Path where the land-sea mask is saved
mask_path = common_path + 'land_mask_' + region + '.nc'

# ** Path where the normalized cumulatives of P-E are saved
if reanalysis  in ['ERA-Interim', 'CFSR']:
    anomalies_start_year = 1980
else:
    anomalies_start_year = 1981    
norm_cumulative_anomalies_path = common_path + anomalies_name + '/normalized_cumultive_anomalies_pme_' + str(window) + 'months_' + region + '_' + str(anomalies_start_year) + '-' + str(end_year) + '.nc'

# ** Area threshold for defining a landfalling drought over land (km^2)
land_area_threshold = 100000.

# ** Minimum duration threshold (months)
duration_threshold = 3

##################################################################################
#################################### LOAD DATA ###################################
##################################################################################

print 'Processing data for ' + reanalysis + ', version: ' + version, ', window:', window, ', threshold:', drought_threshold

# Start and end date and number of time steps in between
start_date = datetime(start_year,1,1)
nt = (end_year - start_year + 1)*12	
end_date = start_date + relativedelta(months=nt-1)

# Load land sea mask
f = Dataset(mask_path)
mask = f.variables['lsm'][:]
lons = f.variables['lon'][:]
lats = f.variables['lat'][:]
f.close()
print 'Land-sea mask loaded'

# Resolution of dataset in the longitudes and latitudes
resolution_lon = np.mean(lons[1:]-lons[:-1])
resolution_lat = np.mean(lats[1:]-lats[:-1])

# Load normalized P-E cumulative anomalies (Begin one year later for ERA-Interim and CFSR)
f = Dataset(norm_cumulative_anomalies_path)
if reanalysis in ['ERA-Interim', 'CFSR']:
    normalized_anomalies = f.variables['pme'][12:,:,:]
else:
    normalized_anomalies = f.variables['pme'][:]
f.close()
print 'Normalized anomalies are loaded'

##################################################################################
################ EXTRACT INFORMATION ON THE DROUGHT CLUSTER TRACKS ###############
##################################################################################

# Track drought clusters through time (Note: only need to run once after calculating drought clusters)
dclib.track_clusters_and_save(clusters_path, start_date, end_date, nt, lons, lats, drought_threshold_name, reanalysis)
print 'Done tracking the clusters.'

# Open dictionary of drought clusters
cluster_data_dictionary = pickle.load(open(clusters_path + 'tracked_clusters_dictionary_' + str(start_date.year) + '-' + str(end_date.year) + '.pck',"rb"))

# Extract tracks
tracks_dictionary = dclib.extract_tracks(cluster_data_dictionary)

##################################################################################
########################### CLASSIFY DROUGHT CLUSTERS ############################
##################################################################################

# Create dictionary of landfalling drought clusters and save
bool_save = True
pclib.classify_drought_clusters(cluster_data_dictionary, tracks_dictionary, clusters_path, lons, lats, duration_threshold, land_area_threshold, mask, normalized_anomalies, start_date, end_date, resolution_lon, resolution_lat, bool_save)    # Calculate and save
dictionary_cluster_classification = pickle.load(open(clusters_path + 'classified_drought_clusters_' + str(start_date.year) + '-' + str(end_date.year) + '.pck',"rb")) # Load
print 'Classification dictionary has been loaded.'

# Verify that all the clusters were sorted correctly
pclib.check_cluster_classification(dictionary_cluster_classification, land_area_threshold)
print 'Doneso!'
