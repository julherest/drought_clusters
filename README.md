# Identifying and tracking droughts through space and time

## Summary
The code in this repository can be used to identify and track drought clusters, allowing the user to study the characteristics and behaviors of droughts in space and time within a given region or globally. This code is intended to be use for analyzing gridded datasets (e.g. reanalyses, climate model outputs) and not over point-observations. 

The code is found under the `src` directory and contains the following files:
 - `01_data_preprocessing.py`: This code can be used to understand the type of data pre-processing needed prior to using the clustering algorithm.
 - `02_calculate_drought_clusters_parallel.py`: This code contains the workflow to calculate droughts clusters for a given time slice. The code loops through the time steps given by the user and is set up to run in prallel using the `mpi4py` library. However, the code can be easily modified by the user to run in series (i.e. no parallelization) or to implement the parallelization using a different library.
 - `03_process_drought_clusters.py`: After the 2D drought clusters have been identified for each individual time step in the time period of interest, the use can run this code to link the drought 2D drought clusters through time. 
- `drought_clusters_utils.py`: This file contains all the necessary functions used to identify and track the drought clusters. 

For further information on the algorithm, please read the references provided in the **How to cite this code** and **References** sections.

## Data pre-processing
Prior to identifying and tracking drought clusters using this code, the user needs to create a NetCDF file that contains a 3D array (time, lat, lon) that contains the gridded, normalized drought metric to use. Further, the longitudinal coordinates must be given in the form (-180, 180) instead of (0, 360). 

This code was initially written to analyze percentiles of soil moisture in Herrera-Estrada et al. (2017) and later used to analyze percentiles of 12-month cumulative anomalies of precipitation-minus-evaporation in Herrera-Estrada and Diffenbaugh, *under review*.

The `01_data_preprocessing.py` provides an example of how the user may calculate these percentiles using sample data from the Modern-Era Retrospective Analysis for Research and Applications version 2 (MERRA2), a reanalysis dataset produced by NASA. 

However, the user can apply this clustering algorithm to other reanalysies datasets (e.g. ECMWF's ERA5), as well as climate model simulations (e.g. CMIP6). Further, the user can also identify and track drought clusters based on more traditional drought indices such as SPI, PDSI, and SPEI so long as they are provided as a gridded product. Note that the user may need to make slight adjustments to the names of the variables in the NetCDF file with the drought metric to make sure the coordinates and the metric are loaded correctly.

## Identifying drought clusters in 2D
Once the user has created the NetCDF file with the 3D array (time, lat, lon) containing the gridded and normalized drought metric, the user can run the `02_calculate_drought_clusters_parallel.py` file to identify the 2D drought clusters in every time step (we will join the drought clusters through time in the next step). This file has been designed to run in parallel using the `mpi4py` Python library, so the user must make sure that they have this library and required dependencies installed.

To run this file on 4 processors, the user would time the following into the command line:
`mpirun -np 4 python 02_caculate_drought_clusters_parallel.py`

The maximum number of processors that the user can use will be dictated by the processors available to the user in their computer or computer cluster.

The code in `02_caculate_drought_clusters_parallel.py` will carry out the following operations for each time step:
1. Extract the 2D array for the given time step
2. Apply a median filter over the 2D array to smooth out noise
3. Apply drougth threshold definition. As mentioned previously, this code was written to work with percentiles (e.g. of soil moisture) so a drought is definied by a percentile threshold (e.g. 20th percentile). If a user were to use this code on drought metrics such as SPI, PDSI, or SPEI, they will need to set the drought definition specific to those metrics (e.g. 0 for any negative anomaly or -1 to focus on more severe droughts). Since the drought clusters are identified in 2D first and connected later, the user is encouraged to pick a less strict drought definition at this stage (e.g. 20th percentile when using percentiles or -0.5 when using SPI, PDSI or SPEI) to improve the temporal connectivity of the drought clusters, in case there is some intermitent recovery of drought conditions.
4. Identify all the 2D drougth clusters for that time step. If the user intends to calculate drought clusters that may cross the right/left boundaries of the map, then they must set the Boolean variable `periodic_bool` to `True`. This will make sure the clustering algorithm links clusters along that cross the right/left boundary. Otherwise, if the user is doing a regional analysis, they must set `periodic_bool` to `False`.
5. Filter drought clusters by area and if the centroid lies within the Sahara Desert. The pevious step will include single grid cells that meet the drought definition as an individual cluster. To make the code more efficient, the user can set a minimum area threshold using the `area_threshold` variable so as not to include clusters that are too small. 
6. Save the drought clusters for the current time step. Three files will be saved:
	- `cluster-matrix_' + str(current_date) + '.pck`: This is a 2D NumPy array of the same dimensions as the initial 2D array but where all the grid cells not belonging to a drought cluster are masked out.
	- `cluster-dictionary_' + str(current_date) + '.pck`: This is the dicionary for the current time step containing all the characteristics of the 2D drought clusters identified.
	- `cluster-count_' + str(current_date) + '.pck`: This is the number (integer) of drought clusters identified in the current time step.
	(The `current_data` variable is of the type `datetime` and will be needed in the following step to combine all the 2D drought clusters calculated for each time step individually in parallel.)

## Tracking drought clusters through time


## Analyzing drought clusters


## How to cite this code


## References


