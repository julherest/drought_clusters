"""
This script processes reanalysis data to be used in the clustering algorithm. The output is a 3D matrix of the percentiles
of cumulative anomalies of precipitation minus evaporation.

Note that the "**" at the beginning of a comment show the lines where the user will need to make modifications, 
either by writing the path to the file or by making sure the variable names correspond to those in the NetCDF file.

Written by Julio E. Herrera Estrada, Ph.D.
"""

# Import Python libraries
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Import custom libraries
import drought_clusters_utils as dclib

##################################################################################
############################ SET PATHS AND DEFINITIONS ###########################
##################################################################################

# ** Definitions
dataset = "MERRA2"
start_date = datetime(1980, 1, 1)

# ** Full path with file name for monthly precipitation and evaporation NetCDF files
prcp_path = ""
et_path = ""

# ** Full path with file name for the NetCDF file with the percentiels
percentiles_path = ""

# ** Open NetCDF files containing the monthly precipitation ane evaporation from MERRA-2
f = Dataset(prcp_path)
prcp = f.variables["prcp"][:]
lons = f.variables["lon"][:]
lats = f.variables["lat"][:]
f.close()

f = Dataset(et_path)
et = f.variables["et"][:]
f.close()

##################################################################################
############################# CARRY OUT CALCULATIONS #############################
##################################################################################

# Calculate precipitation minus evaporation (P-E)
pme = prcp - et

# Calculate monthly anomalies of P-E
anomalies = dclib.calculate_anomalies_matrix(pme)

# Calculate cumulative anomalies of P-E over a given accumulation period
accumulation_window = 12  # months
cumulative_anomalies = dclib.calculate_cumulative_anomalies_matrix(
    anomalies, accumulation_window
)
new_start_date = start_date + relativedelta(years=1)

# Calculate percentiles of cumulative anomalies of P-E
seasonality_bool = False
percentiles_matrix = dclib.calculate_percentiles_matrix(
    cumulative_anomalies, seasonality_bool
)

# Information to save the percentiles_matrix
units = "percentiles"
var_name_percentiles = "percentiles"
var_info = "Percentiles of P-E cumulative anomalies from " + dataset + "."

# Save percentiles matrix
dclib.save_netcdf_file(
    percentiles_matrix,
    lons,
    lats,
    units,
    var_name_percentiles,
    var_info,
    percentiles_path,
    new_start_date,
)
print("Done calculating and saving percentiles.")
