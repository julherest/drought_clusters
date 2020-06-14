# Identifying and tracking droughts through space and time

## Summary
The code in this repository can be used to identify and track drought clusters allowing the user to study the characteristics and behaviors of droughts in space and time within a given region or globally. This code is intended to be use for analyzing gridded datasets (e.g. reanalyses, climate model outputs) and not over point-observations. 

The code is found under the `/src/` directory and is ordered as follows:
 - `01_process_data.py`: This code can be used to understand the type of data pre-processing needed prior to using the clustering algorithm.
 - `02_calculate_drought_clusters_parallel.py`: This code contains the workflow to calculate droughts clusters for a given time slice. The code loops through the time steps given by the user and is set up to run in prallel using the `mpi4py` library. However, the code can be easily modified by the user to run in series (i.e. no parallelization) or to implement the parallelization using a different library.
 - `03_process_drought_clusters.py`: After the 2D drought clusters have been identified for each individual time step in the time period of interest, the use can run this code to link the drought 2D drought clusters through time. 

For further information on the algorithm, please read the references provided in the **How to cite this code** and **References** sections.

## Data pre-processing
Prior to identifying and tracking drought clusters using this code

## Identifying drought clusters in 2D


## Tracking drought clusters through time


## Analyzing drought clusters


## How to cite this code


## References


