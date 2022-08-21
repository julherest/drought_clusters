"""
This file contains the functions needed to calculate, track, and analyze the drought clusters.

Written by Julio E. Herrera Estrada, Ph.D.
"""

# Import libraries
import numpy as np
import cPickle as pickle
from netCDF4 import Dataset
from calendar import monthrange
from datetime import timedelta
from dateutil.relativedelta import relativedelta


#############################################################################################################
######################################### DATA PRE-PROCESSING TOOLS #########################################
#############################################################################################################


def calculate_anomalies_matrix(data_matrix):
    """
    This function calculates monthly anomalies from the given 3D matrix. Assumes the data starts in January.

    Arguments:
    - data_matrix = 3D NumPy array (time, lats, lons) with the data from which we want to calculate anomalies

    Returns:
    - anomalies = 3D NumPy array (time, lats, lons) with the calculated anomalies
    """

    # Dimensions of data
    nt, nlats, nlons = data_matrix.shape

    # Initialize array to save cumulative anomalies
    anomalies = np.zeros([nt, nlats, nlons])

    # Calculate anomalies
    month_idx = 0
    for i in range(0, nt):

        # Index of current months
        idx = np.arange(month_idx, nt, 12)

        # Subtract monthly means from current value
        anomalies[i, :, :] = data_matrix[i, :, :] - np.nanmean(
            data_matrix[idx, :, :], axis=0
        )

        # Update month index
        if month_idx == 11:  # December given start at 0
            month_idx = 0
        else:
            month_idx = month_idx + 1

    return anomalies


def calculate_cumulative_anomalies_matrix(anomalies, window):
    """
    This function calculates the cumulative anomalies over a given window from the anomalies array.
    We assume the data starts on January.

    Arguments:
    - anomalies: 3D NumPy array (time, lats, lons) with the monthly anomalies.
    - window: Integer of the accumulation period to use in months

    Returns:
    - cumulative_anomalies: 3D NumPy array (time, lats, lons) with the cumulative anomalies. Note
        that because of the accumulation period, this array will start the following January.
    """

    # Dimensions of data
    nt, nlats, nlons = anomalies.shape

    # Initialize array to save cumulative anomalies
    cumulative_anomalies = np.zeros([nt - window, nlats, nlons])

    # Calculate cumulative anomalies
    for i in range(window, nt):
        cumulative_anomalies[i - window, :, :] = np.nansum(
            anomalies[i - window : i, :, :], 0
        )

    # Crop data to start in January of the next year
    cumulative_anomalies = cumulative_anomalies[12 - window :, :, :]

    return cumulative_anomalies


def percentiles_from_Weibull(data_array):
    """
    This function takes in a 1D time series and uses the Weibull plotting positions
    to create an empirical cumulative distributions function used to replace each value
    in the time series with its respective percentile.

    Arguments:
    - data_array: 1D time series of values for a given month (e.g. all January values from 1980-2015)

    Returns:
    - percentile_array: 1D array with the percentiles corresponding to each value for the given month in
                        the input time series.
    """

    # Finding the rank of each entry in the data array
    temp = data_array.argsort()
    ranks = np.empty(len(data_array), int)
    ranks[temp] = np.arange(len(data_array))

    # Calculating the percentiles using Weibull plotting position
    percentile_array = ranks / float((len(data_array) + 1))

    return percentile_array


def find_percentiles_single_month(data_array):
    """
    This function goes through a time series from a given grid cell, and calculates
    the percentile values from the Weibull plotting positions for each individual
    month within the time series.

    Arguments:
    - data_array: 1D time series for a given grid cell (e.g. January 1980 - December 2015)

    Returns:
    - percentile_array: 1D array with the percentiles corresponding to each value for all the months in
                        the input time series.
    """

    # Number of observations and years
    n = len(data_array)
    nsteps = 12
    nyears = int(len(data_array) / float(nsteps))

    # Initiate array to store results
    percentiles_array = np.zeros(n)

    # Calculate percentiles
    for i in range(0, nsteps):
        # Indices of the current month of year
        current = np.arange(i, n, nsteps)

        # Array of the all the values for the current month
        current_data = data_array[current]

        # Calculate percentiles from data of the three time steps
        current_percentiles = percentiles_from_Weibull(current_data)

        # Store in percentiles array
        percentiles_array[current] = current_percentiles

    return percentiles_array


def calculate_percentiles_matrix(data_matrix, seasonality_bool):
    """
    This function will take in the full 3D matrix of data (e.g. soil moisture or cumulative anomalies
    of precipitation minus evaporation) in the form (time, lat, lon) and create a similar matrix with
    each value replaced by its respective percentile according to the Weibull position.

    Argument:
    - data_matrix: 3D matrix of original data (e.g. soil moisture) in the form (time, lat, lon)
    - seasonality_bool: Boolean variable. True if the input data has seasonality. False if the seasonality
        has been removed (e.g. by calculating anomalies)

    Returns:
    - percentile_matrix: 3D matrix of the same dimensions and orientation as the input matrix,
                         but where each value entry is replaced by its respective monthly percentile.
    """

    # Dimensions
    n, nlat, nlon = data_matrix.shape

    # Initialize results matrix
    percentile_matrix = np.zeros([n, nlat, nlon])

    # Do analysis for each grid cell
    for i in range(0, nlat):
        for j in range(0, nlon):

            # Current grid cell's data
            current_data = data_matrix[:, i, j]

            # Find percentiles
            if np.isnan(np.mean(current_data)) == False:
                if seasonality_bool:
                    percentile_array = find_percentiles_single_month(
                        current_data
                    )  # This is for data with seasonality
                else:
                    percentile_array = percentiles_from_Weibull(
                        current_data
                    )  # This is for data without seasonality (e.g. anomalies)
            else:
                percentile_array = np.empty(n)
                percentile_array.fill(np.nan)

            # Save to percentiles matrix
            percentile_matrix[:, i, j] = percentile_array

    return percentile_matrix


#############################################################################################################
####################################### IDENTIFYING DROUGHT CLUSTERS  #######################################
#############################################################################################################


def median_filter(data_matrix):
    """
    This function applies a median filter on a 2D matrix, presumeably a map.

    Argument:
    - data_matrix: 2D matrix (lat, lon) of percentile values for a given time step.

    Return:
    - filtered_matrix: 2D matrix of the same dimensions and orientation as the input
                       but smoothed out by the median filter.
    """

    # Dimensions
    n, m = data_matrix.shape

    # Initialize matrix where results will be stored
    filtered_matrix = np.zeros([n, m])

    # Loop through each grid cell
    for i in range(0, n):
        for j in range(0, m):

            # Check if grid cell has a real value
            if np.isnan(data_matrix[i, j]) == False:

                # Check all the surrounding grid cells
                window_values = []
                if i < n - 1:
                    window_values.append(data_matrix[i + 1, j])
                if j < m - 1:
                    window_values.append(data_matrix[i, j + 1])
                if i < n - 1 and j < m - 1:
                    window_values.append(data_matrix[i + 1, j + 1])
                if i > 0:
                    window_values.append(data_matrix[i - 1, j])
                if j > 0:
                    window_values.append(data_matrix[i, j - 1])
                if i > 0 and j > 0:
                    window_values.append(data_matrix[i - 1, j - 1])
                if i < n - 1 and j > 0:
                    window_values.append(data_matrix[i + 1, j - 1])
                if i > n and j < m - 1:
                    window_values.append(data_matrix[i - 1, j + 1])

                # Mask any nans that might have been picked up
                window_values = np.array(window_values)
                window_values = window_values[~np.isnan(window_values)]

                # Check if grid cell is isolated, otherwise apply filter
                if len(window_values) == 0:
                    filtered_matrix[i, j] = data_matrix[i, j]
                else:
                    filtered_matrix[i, j] = np.median(window_values)
            else:

                filtered_matrix[i, j] = np.nan

    return filtered_matrix


def filter_non_droughts(data_matrix, threshold):
    """
    This function will filter out all pixels in a map that are above the given
    threshold.

    Argument:
    - data_matrix: 2D matrix (lat, lon) of (smoothed) percentile values for a given time step.
    - threshold: Percentile threshold used to filter droughts.

    Return:
    - droughts_matrix: 2D matrix of the same dimensions and orientation as the input matrix but where
                        all the grid cells that have values above the threshold are filtered out.
    """

    # Ignore floating-point errors
    np.seterr(invalid="ignore")

    # Make copy of data matrix
    droughts_matrix = np.array(data_matrix[:])

    # Mask out pixels above a given threshold
    idx = np.where(droughts_matrix > threshold)
    droughts_matrix[idx] = np.nan

    return droughts_matrix


def check_drought_in_surroundings(
    current_pixel, linked_indices, data_matrix, periodic_bool
):
    """
    This function will look around the neighbouring cells for a pixel under drought and return their indices.

    Arguments:
    - current_pixel: Coordinates of the pixel/gridcell of interest
    - linked_indices: List of all the coordinates of gridcells/pixes that have been identified as being under drought.
    - data_matrix: 2D matrix for the given time step from where the gridcells under drought have been identified.
    - periodic_bool: Boolean variable, True if we want to treat the left/right edges of the array as periodic (e.g. if
        we are calculating clusters over a global map), False otherwise (e.g. if we are identifying clusters within a
        smaller region).

    Returns:
    - surrounding_drought_pixels: list of pixels directly surrounding the current_pixel that are also under drought.
    """

    # Dimensions of matrix
    nlats, nlons = data_matrix.shape

    # Array of surrounding indices under drought
    surrounding_drought_pixels = []

    # Individual coordinates of current pixel
    n, m = current_pixel

    # Coordinates we will check
    for x, y in [
        (n + i, m + j) for i in (-1, 0, 1) for j in (-1, 0, 1) if i != 0 or j != 0
    ]:

        if periodic_bool:

            # Check around the left and right edges to allow for continuity of a cluster
            if y < 0:
                y = nlons - 1
            elif y > nlons - 1:
                y = 0

        # If it is under drought, add it to the list
        if (x, y) in linked_indices:
            surrounding_drought_pixels.append((x, y))
            value = data_matrix[x, y]

    return surrounding_drought_pixels


def find_gridcell_area(center_lon, center_lat, resolution_lon, resolution_lat):
    """
    This function finds the area of an individual grid cell on a sphere.

    Argument:
    - center_lon: Longitude at the center of the grid cell in degrees
    - center_lat: Latitude at the center of the grid cell in degrees
    - resolution_lon, resolution_lat: Resolution of the dataset in the longitudinal and latitudinal directions (e.g. 0.5 degrees)

    Returns:
    - area: Area of the given grid cell in km^2
    """

    # Earth's radius (km)
    R = 6371.0

    # Find gridcell edges
    lon_min = center_lon - resolution_lon / 2.0
    lon_max = center_lon + resolution_lon / 2.0
    lat_min = center_lat - resolution_lat / 2.0
    lat_max = center_lat + resolution_lat / 2.0

    # Convert coordinates to radians
    lon_min = (360 + lon_min) * np.pi / 180.0
    lon_max = (360 + lon_max) * np.pi / 180.0
    lat_min = lat_min * np.pi / 180.0
    lat_max = lat_max * np.pi / 180.0

    # Calculate area
    area = (R**2) * (lon_max - lon_min) * (np.sin(lat_max) - np.sin(lat_min))

    return area


def find_cluster_area(coordinates, lons, lats, resolution_lon, resolution_lat):
    """
    This function calculates the area of a cluster by adding the areas of the individual grid cells.

    Argument:
    - coordinates: Coordinates that belong to a given drought cluster.
    - lons: 1D array of longitudes for the data_matrix
    - lats: 1D array of latitudes for the data_matrix
    - resolution_lon, resolution_lat: Resolution of the dataset in the longitudinal and latitudinal directions (e.g. 0.5 degrees)

    Returns:
    - running_sum: Estimated area of the given drought cluster (km^2)
    """

    # Number of grid cells
    n = len(coordinates)

    # Initialize running sum
    running_sum = 0

    # Go through each coordinate set
    for i in range(0, n):
        # Obtain lon and lat
        center_lat_idx, center_lon_idx = coordinates[i]
        center_lat = lats[center_lat_idx]
        center_lon = lons[center_lon_idx]

        # Calculate area
        area = find_gridcell_area(
            center_lon, center_lat, resolution_lon, resolution_lat
        )

        # Update running sum
        running_sum = running_sum + area

    return running_sum


def find_weighed_centroid(lats_array, lons_array, intensities, lons, lats):
    """
    This function finds the centroid of the cluster from the given coordinates of each drought pixel. The
    centroid is weighed using the intensities-related matric (1 - percentile values) of each pixel such that
    the centroid is closer to or located at the most intense part of the drought cluster (i.e. with lower
    percentile values).

    Argument:
    - lats_array: 1D array of the latitudes in degrees of each gridcell contained in the drought cluster.
    - lons_array: 1D array of the longitudes in degrees of each gridcell contained in the drought cluster.
        These must be in the form (-180,180) and NOT in the form (0, 360)
    - intensities: 1D array with the intensities for each gridcell contained in the drought cluster.

    Return:
    - centroid_lat: Latitude in degrees of the cluster centroid.
    - centroid_lon: Longitude in degrees of the cluster centroid.
    """

    # Number of pixels in cluster
    npixels = len(lons_array)

    # Calculate the number of cells with negative and positive longitudes
    npositive = len(np.where(lons_array >= 0)[0])
    nnegative = len(np.where(lons_array < 0)[0])

    # Initialize flag for how to calculate centroid
    flag = "normal"

    # Dig deeper if we suspect the cluster wraps around the edge
    if npositive > 0 and nnegative > 0:

        # Create 1D histogram of longitudes in the cluster
        hist, bin_edges = np.histogram(lons_array, bins=lons)

        # Find if the cluster wraps around
        if hist[0] > 0 and hist[-1] > 0:  # If it does, check if it's a polar cluster

            # Check if the cluster is potentially a polar one
            if len(np.where(hist == 0)[0]) == 0:

                # Need to dig deeper to see if it's just a complex wrapped cluster or if it's a polar cluster
                # Calculate 2D histogram
                hist2d, lats_edges, lons_edges = np.histogram2d(
                    lats_array, lons_array, bins=(lats, lons)
                )

                # Calculate sum over all longitudes
                sum_hist = np.sum(hist2d, 1)
                if len(np.where(sum_hist >= len(lons))[0]) > 0:
                    flag = "polar"
                else:
                    flag = "wrapped_complex"

            elif len(np.where(hist == 0)[0]) > 0:

                # It's not a polar cluster, it's just wrapped around the Pacific
                flag = "wrapped"

    # Calculate the cluster centroid depending on the type of cluster it was found to be
    if flag == "normal" or flag == "polar":

        # Find centroid coordinates
        centroid_lat = np.average(lats_array, weights=intensities)
        centroid_lon = np.average(lons_array, weights=intensities)

    elif flag == "wrapped" or flag == "wrapped_complex":

        # Minimum and maximum longitude of the matrix
        min_lon = np.min(lons)
        max_lon = np.max(lons)

        if flag == "wrapped":

            # Find all the zero instances
            idx_zeros = np.where(hist == 0)[0]

        elif flag == "wrapped_complex":

            # Find the latitude with the narrowest gap
            idx_lat = np.argmax(sum_hist)

            # Find all the zero instances for this latitude
            idx_zeros = np.where(hist2d[idx_lat, :] == 0)[0]

        # Longitude of the last zero
        idx_last_zero = idx_zeros[-1]
        lon_shift = lons[idx_last_zero]

        # Shift all longitudes in the cluster by this value
        lons_array_temp = np.array(lons_array[:])
        lons_array_temp[lons_array > lon_shift] = (
            lons_array_temp[lons_array > lon_shift] - lon_shift
        )
        lons_array_temp[lons_array <= lon_shift] = (
            lons_array_temp[lons_array <= lon_shift] + 360 - lon_shift
        )

        # Find centroid coordinates on new reference grid and shift back
        centroid_lat = np.average(lats_array, weights=intensities)
        centroid_lon_temp = np.average(lons_array_temp, weights=intensities)
        if centroid_lon_temp + lon_shift > max_lon:
            centroid_lon = centroid_lon_temp - 360 + lon_shift
        else:
            centroid_lon = centroid_lon_temp + lon_shift

    return centroid_lat, centroid_lon


def find_drought_clusters(
    data_matrix, lons, lats, resolution_lon, resolution_lat, periodic_bool
):
    """
    This function will find the individual drought clusters for a given time step. Drought clusters
    are defined as spatially contiguous areas under drought. This algorithm was inspired by the algorithm
    describied in Andreadis et al. (2005), Journal of Hydrometeorology.

    Arguments:
    - data_matrix: 2D matrix for a given time step with the non-drought pixels filtered out.
    - lons: 1D array of longitudes in degrees for the data_matrix. These must be in the form (-180,180) and NOT in the form (0, 360)
    - lats: 1D array of latitudes in degrees for the data_matrix
    - resolution_lon, resolution_lat: Resolution of the dataset in the longitudinal and latitudinal directions (e.g. 0.5 degrees)
    - periodic_bool: Boolean variable, True if we want to treat the left/right edges of the array as periodic (e.g. if
        we are calculating clusters over a global map), False otherwise (e.g. if we are identifying clusters within a
        smaller region).

    Returns:
    - cluster_count: Number of drought clusters above the area threshold identified for the current
                     time step.
    - cluster_dictionary: Dictionary containing the characteritics of each drought cluster identified
                          in this current time step.
    """

    # Find the indices in each dimension of the pixels under drought
    idx_lat, idx_lon = np.where(np.isfinite(data_matrix))

    # Link them so they become coordinates
    linked_indices = zip(idx_lat, idx_lon)

    # Number of pixels under drought
    npixels = len(idx_lat)

    # Initialize cluster count
    cluster_count = 0

    # Initialize cluster dictionary
    cluster_dictionary = {}

    # Initialize list of pixels that have been clustered
    clustered_pixels = []

    # Go through every drought pixel to make sure none is left out
    for i in range(0, npixels):

        # Current pixel used to start cluster (i.e. seed pixel)
        seed_pixel = linked_indices[i]

        # Make sure current pixel hasn't already been assigned to a cluster
        if seed_pixel not in clustered_pixels:

            # Increase cluster count by 1
            cluster_count = cluster_count + 1

            # Add current unclustered pixel to the dictionary entry of the new
            # cluster and to the list of pixels that have been assigned to a cluster
            cluster_dictionary[cluster_count] = {
                "coordinates": [],
                "area": [],
                "intensity": [],
                "centroid": [],
            }
            cluster_dictionary[cluster_count]["coordinates"].append(seed_pixel)
            clustered_pixels.append(seed_pixel)

            # Initialize the list of new pixels added to a cluster and add current pixel
            new_additions = []
            new_additions.append(seed_pixel)

            # While more pixels keep being added to the current cluster with each iteration, keep going
            while len(new_additions) > 0:

                # Make copy of new additions list to sweep through, and reset list so that new pixels are
                # added in this coming iteration
                additions_to_check = list(new_additions)
                new_additions = []

                # Go through each new addition and compare it to the pixels in its surroundings
                for current_pixel in additions_to_check:

                    # Finding those pixels surrounding the current pixel that are also under drought
                    positive_matches = check_drought_in_surroundings(
                        current_pixel, linked_indices, data_matrix, periodic_bool
                    )

                    # Check there was at least one positive match
                    if len(positive_matches) > 0:

                        # Go through each surrounding pixel under drought and check if they have already been clustered
                        for pixel in positive_matches:

                            # Check if it has already been assigned
                            if pixel not in clustered_pixels:
                                # Add pixel to current cluster dictionary, to the list of clustered pixels,
                                # and the list of new additions
                                cluster_dictionary[cluster_count]["coordinates"].append(
                                    pixel
                                )
                                clustered_pixels.append(pixel)
                                new_additions.append(pixel)

            ## Once the cluster has been finalized, find its approximate area and intensity and centroid

            # List of coordinate indices for current cluster
            current_cluster_coordinates = cluster_dictionary[cluster_count][
                "coordinates"
            ]

            # Clean up in case there are any duplicates and resave
            current_cluster_coordinates = list(set(current_cluster_coordinates))
            cluster_dictionary[cluster_count][
                "coordinates"
            ] = current_cluster_coordinates

            # Initialize arrays of intensities and actual coordinates
            pixels_in_cluster = len(current_cluster_coordinates)
            intensities_array = np.zeros(pixels_in_cluster)
            lons_array = np.zeros(pixels_in_cluster)
            lats_array = np.zeros(pixels_in_cluster)

            # Sweep through the coordinates and extract each element
            for j in range(0, pixels_in_cluster):
                # Indices of current pixel
                current_lat_idx, current_lon_idx = current_cluster_coordinates[j]

                # Extract intensity of current pixel
                intensities_array[j] = 1 - data_matrix[current_lat_idx, current_lon_idx]

                # Coordinates of current pixel
                lons_array[j] = lons[current_lon_idx]
                lats_array[j] = lats[current_lat_idx]

            # Find mean and standard deviation of cluster intensity
            mean_cluster_intensity = np.mean(intensities_array)
            std_cluster_intensity = np.std(intensities_array)

            # Find cluster's centroid
            centroid_lat, centroid_lon = find_weighed_centroid(
                lats_array, lons_array, intensities_array, lons, lats
            )

            # Save cluster's characteristics
            cluster_dictionary[cluster_count]["intensity"] = mean_cluster_intensity
            cluster_dictionary[cluster_count]["variability"] = std_cluster_intensity
            cluster_dictionary[cluster_count]["centroid"] = (centroid_lon, centroid_lat)

            # Finding the cluster's approximate area
            cluster_area = find_cluster_area(
                current_cluster_coordinates, lons, lats, resolution_lon, resolution_lat
            )

            # Save cluster area
            cluster_dictionary[cluster_count]["area"] = cluster_area

    return cluster_count, cluster_dictionary


def cluster_in_Sahara(centroid_lon, centroid_lat):
    """
    This function determines whether the current drought cluster centroid falls within the Sahara desert
    during the current time step.

    Argument:
    - centroid_lat: Latitude in degrees of the cluster centroid.
    - centroid_lon: Longitude in degrees of the cluster centroid.

    Returns:
    - Boolean variable False if the centroid does not fall in the Sahara desert and True otherwise.
    """

    if (
        centroid_lat >= 20
        and centroid_lat <= 25
        and centroid_lon >= -17
        and centroid_lon <= 34
    ):
        return True
    else:
        return False


def filter_drought_clusters(
    data_matrix, cluster_count, cluster_dictionary, area_threshold
):
    """
    This function will take in all the drought clusters and remove those that are smaller
    than the given area threshold. If a drought cluster does not meet the critertion
    then they will be deleted from the drought field. Drought clusters whose centroids fall within
    the Sahara desert are also removed. This function works using the dictionary created by the
    find_drought_clusters function.

    Argument:
    - data_matrix: 2D matrix for a given time step with the non-drought pixels filtered out.
    - cluster_count: Number of drought clusters above the area threshold identified for the current
                     time step.
    - cluster_dictionary: Dictionary containing the characteritics of each drought cluster identified
                          in this current time step.
    - area_threshold: Minimum area threshold (km^2) for the drought clusters. Any clusters that are
                      smaller than this threshold will not be included.

    Return:
    - data_matrix: Updated input data matrix where the gridcells belonging to small clusters have been
                   filtered out.
    - filtered_cluster_count: Updated count of drought clusters in the current time step (having removed
                              small drought clusters).
    - filtered_cluster_dictionary: Updated dictionary of clusters and their characteristics for the current
                                   time step.
    """

    # Initiate new dictionary where filtered results will be stored
    filtered_cluster_dictionary = {}

    # Initialize count of clusters that meet the area criterion
    filtered_cluster_count = 0

    # Check every cluster
    for i in range(1, cluster_count + 1):

        # Cluster area and centroid
        cluster_area = cluster_dictionary[i]["area"]
        centroid_lon, centroid_lat = cluster_dictionary[i]["centroid"]

        # Check whether cluster is in the Sahara Desert
        in_sahara = cluster_in_Sahara(centroid_lon, centroid_lat)

        # Compare to given threshold and check if it's in the Sahara
        if cluster_area >= area_threshold and not in_sahara:

            # Add a count to the number of filtered clusters
            filtered_cluster_count = filtered_cluster_count + 1

            # Save list of pixels for this cluster and the cluster's characteristics to new dictionary
            filtered_cluster_dictionary[filtered_cluster_count] = {
                "coordinates": [],
                "area": [],
                "intensity": [],
                "centroid": [],
            }
            filtered_cluster_dictionary[filtered_cluster_count][
                "coordinates"
            ] = cluster_dictionary[i]["coordinates"]
            filtered_cluster_dictionary[filtered_cluster_count][
                "area"
            ] = cluster_dictionary[i]["area"]
            filtered_cluster_dictionary[filtered_cluster_count][
                "intensity"
            ] = cluster_dictionary[i]["intensity"]
            filtered_cluster_dictionary[filtered_cluster_count][
                "variability"
            ] = cluster_dictionary[i]["variability"]
            filtered_cluster_dictionary[filtered_cluster_count][
                "centroid"
            ] = cluster_dictionary[i]["centroid"]

        else:

            # List of coordinate indices for current cluster
            current_cluster_coordinates = cluster_dictionary[i]["coordinates"]

            # Sweep through the coordinates and extract each element
            for j in range(0, len(current_cluster_coordinates)):
                # Indices of current pixel
                current_lat_idx, current_lon_idx = current_cluster_coordinates[j]

                # Delete pixel from drought field
                data_matrix[current_lat_idx, current_lon_idx] = np.nan

    return data_matrix, filtered_cluster_count, filtered_cluster_dictionary


#############################################################################################################
#################################### TRACK DROUGHT CLUSTERS THROUGH TIME ####################################
#############################################################################################################


def load_drought_cluster_data(data_path, start_date, tsteps, nlons, nlats):
    """
    This function loads the individual fields saved for each time step to create the full data
    structures used in the analysis phase.

    Arguments:
    - data_path: Path where the clusters data was saved to
    - start_date: Date of the first clusters calculated (datetime type)
    - tsteps: Number of time steps for which the clusters were calculated
    - nlats, nlons: Number of latitudes and longitudes of the initial data matrix

    Returns:
    - droughts_cluster_matrix: Matrix with the same dimensions of the original data but with
                               the drought clusters identified for each time step.
    - drought_clusters_dictionary: Dictionary that includes all the cluster dictionaries (one per time step)
    """

    # Initialize matrix and dictionaries to save info about drought clusters
    drought_clusters_matrix = np.zeros([tsteps, nlats, nlons])
    drought_clusters_dictionary = {}
    for i in range(0, tsteps):
        drought_clusters_dictionary[i] = {
            "clusters_characteristics": {},
            "cluster_count": [],
        }

    # Sweep through each date to obtain the data
    current_date = start_date
    for i in range(0, tsteps):
        # Data paths for the individual cluster files for the current time step
        f_name_slice = data_path + "cluster-matrix_" + str(current_date) + ".pck"
        f_name_dictionary = (
            data_path + "cluster-dictionary_" + str(current_date) + ".pck"
        )
        f_name_count = data_path + "cluster-count_" + str(current_date) + ".pck"

        # Load current data field, dictionary, and count of drought clusters
        droughts = pickle.load(open(f_name_slice, "rb"))
        cluster_dictionary = pickle.load(open(f_name_dictionary, "rb"))
        cluster_count = pickle.load(open(f_name_count, "rb"))

        # Save current drought cluster field
        drought_clusters_matrix[i, :, :] = droughts
        drought_clusters_dictionary[i]["clusters_characteristics"] = cluster_dictionary
        drought_clusters_dictionary[i]["cluster_count"] = cluster_count

        # Update date
        current_date = current_date + relativedelta(months=1)

    return drought_clusters_matrix, drought_clusters_dictionary


def track_clusters_and_save(
    data_path, start_date, end_date, tsteps, lons, lats, drought_threshold, dataset
):
    """
    This function implements the tracking algorithm for the drought clusters and saves the resulting
    array of tracked clusters and a dictionary with all the characteristics for each tracked cluster.

    Arguments:
    - data_path: Path where the clusters data was saved to
    - start_date: Date of the first clusters calculated (datetime type)
    - end_date: Date of the last calculation of clusters (datetime type)
    - tsteps: Number of time steps for which the clusters were calculated
    - lats, lons: Latitudes and longitudes of the initial data matrix
    - drought_threshold: Percentile threshold used to calculate the drought clusters
    - dataset: Name of the dataset used for these calculations
    """

    # Number of lons and lats
    nlons = len(lons)
    nlats = len(lats)

    # Load cluster data
    drought_clusters_matrix, drought_clusters_dictionary = load_drought_cluster_data(
        data_path, start_date, tsteps, nlons, nlats
    )

    # Tracking the clusters
    cluster_data_dictionary = track_clusters(
        drought_clusters_dictionary, drought_clusters_matrix, start_date, end_date
    )

    # Saving the tracked clusters
    f_name = (
        data_path
        + "tracked_clusters_dictionary_"
        + str(start_date.year)
        + "-"
        + str(end_date.year)
        + ".pck"
    )
    pickle.dump(cluster_data_dictionary, open(f_name, "wb"), pickle.HIGHEST_PROTOCOL)

    # Save matrix with the drought clusters
    units = "percentiles"
    var_name = "clusters"
    var_info = (
        "Identified drought clusters from percentiles using a "
        + drought_threshold
        + " threshold."
    )
    fname = (
        data_path
        + "drought_clusters_"
        + dataset
        + "_"
        + drought_threshold
        + "_"
        + str(start_date.year)
        + "-"
        + str(end_date.year)
        + ".nc"
    )
    save_netcdf_file(
        drought_clusters_matrix,
        lons,
        lats,
        units,
        var_name,
        var_info,
        fname,
        start_date,
    )


def find_geo_distance(lon1, lat1, lon2, lat2):
    """
    This function calculates the distance between two coordinate points in km.
    """

    # Import libraries
    from math import sin, cos, sqrt, atan2, radians

    # Earth's radius
    R = 6371.0

    # Convert to radians
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    # Calculate distance
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat / 2)) ** 2 + cos(lat1) * cos(lat2) * (sin(dlon / 2)) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance


def track_clusters(drought_cluster_dictionary, drought_matrix, start_date, end_date):
    """
    This function takes in all the data for the drought clusters found (post filtered)
    and tracks each cluster. It will generate a new dictionary with consisten numbering
    such that, for example, drought cluster 1 at t = 1 is labeled the same in t = 2.

    Arguments:
    - drought_clusters_dictionary: Dictionary that includes all the cluster dictionaries (one per time step)
    - droughts_matrix: Matrix with the same dimensions of the original data but with
                       the drought clusters identified for each time step.
    - start_date: Date of the first clusters calculated (datetime type)
    - end_date: Date of the last calculation of clusters (datetime type)

    Returns:
    - cluster_data_dictionary: Dictionary with the information of all the clusters tracked across time.
    """

    # Make copy of 3D matrix containing the filtered droughts
    drought_matrix_copy = np.array(drought_matrix[:])

    # Number of time steps
    nsteps, _, _ = drought_matrix.shape

    # Initialize time index
    t_idx = start_date

    # Initialize the dictionary where we will store all the cluster information
    cluster_data_dictionary = {}
    cluster_data_dictionary[t_idx] = []

    # Initialize cluster count for the cluster IDs
    cluster_ID_count = 0

    # Load the dictionary of the clusters in the first time step
    # as well as the cluster count
    first_dictionary = drought_cluster_dictionary[0]["clusters_characteristics"]
    first_count = drought_cluster_dictionary[0]["cluster_count"]

    # Initialize a cluster object for each of the clusters in the first time step
    for i in range(1, first_count + 1):
        # Update cluster count
        cluster_ID_count = cluster_ID_count + 1

        # Save properties of current cluster
        cluster_data_dictionary[cluster_ID_count] = {}
        cluster_data_dictionary[cluster_ID_count]["start"] = t_idx
        cluster_data_dictionary[cluster_ID_count]["end"] = []
        cluster_data_dictionary[cluster_ID_count]["parent_of"] = []
        cluster_data_dictionary[cluster_ID_count]["child_of"] = []
        cluster_data_dictionary[cluster_ID_count]["splitting_events"] = []
        cluster_data_dictionary[cluster_ID_count]["splitting_events_by_date"] = []
        cluster_data_dictionary[cluster_ID_count]["merging_events"] = []
        cluster_data_dictionary[cluster_ID_count]["merging_events_by_date"] = []
        cluster_data_dictionary[cluster_ID_count][
            t_idx
        ] = {}  # Each property is a function of time
        cluster_data_dictionary[cluster_ID_count][t_idx]["area"] = first_dictionary[i][
            "area"
        ]
        cluster_data_dictionary[cluster_ID_count][t_idx][
            "intensity"
        ] = first_dictionary[i]["intensity"]
        cluster_data_dictionary[cluster_ID_count][t_idx]["centroid"] = first_dictionary[
            i
        ]["centroid"]
        cluster_data_dictionary[cluster_ID_count][t_idx][
            "coordinates"
        ] = first_dictionary[i]["coordinates"]

        # Also save the clusters present in the first time step
        cluster_data_dictionary[t_idx].append(cluster_ID_count)

    # Loop over the time steps to track the clusters
    for i in range(0, nsteps - 1):

        # Load the cluster IDs for current time step
        current_IDs = cluster_data_dictionary[t_idx]

        # Create copy of list of current clusters to help identify clusters that disappear in the future time step
        current_clusters_copy = current_IDs[:]

        # Extract cluster characteristics for current and future time steps
        current_dictionary = drought_cluster_dictionary[i]["clusters_characteristics"]
        future_dictionary = drought_cluster_dictionary[i + 1][
            "clusters_characteristics"
        ]

        # Cluster counts for current and future time steps
        current_count = drought_cluster_dictionary[i]["cluster_count"]
        future_count = drought_cluster_dictionary[i + 1]["cluster_count"]

        # Create copy of list of future clusters to help identify clusters that appear in future time step
        future_clusters_copy = list(np.arange(1, future_count + 1))

        # Tomorrow's date
        next_date = t_idx + relativedelta(months=1)

        # Initialize list of clusters in future time step in the cluster data dictionary
        cluster_data_dictionary[next_date] = []

        # Initialize graph of future clusters associated with current clusters to identify splits
        current_to_future_graph = {}

        # Initialize graph of current clusters associated with future clusters to identify mergers
        future_to_current_graph = {}
        for k in range(1, future_count + 1):
            future_to_current_graph[k] = {}
            future_to_current_graph[k][
                "local"
            ] = []  # Keeps track of the numbering between two time steps
            future_to_current_graph[k][
                "global"
            ] = []  # Keeps track of the running cluster count

        # Go through each cluster in the current time step and link it to cluster(s) in the next time step
        for j in range(0, current_count):

            # Current cluster ID
            cluster_ID = current_IDs[j]

            # Extract coordinates of current cluster and make them a set
            current_cluster_coordinates = set(
                cluster_data_dictionary[cluster_ID][t_idx]["coordinates"]
            )

            # Initalize graph for current cluster to identify splits
            current_to_future_graph[cluster_ID] = []

            # Compare current cluster to the clusters in next time step
            for k in range(1, future_count + 1):

                # Coordinates for comparison cluster
                future_cluster_coordinates = set(future_dictionary[k]["coordinates"])

                # Calculate the overlap of the current and future cluster coordinates
                overlap = len(
                    current_cluster_coordinates & future_cluster_coordinates
                ) / np.float(len(current_cluster_coordinates))

                # Linking the clusters from both time steps with graphs
                if overlap > 0.0:
                    # Add this cluster to the graphs
                    current_to_future_graph[cluster_ID].append(k)
                    future_to_current_graph[k]["local"].append(j + 1)
                    future_to_current_graph[k]["global"].append(cluster_ID)

        # Rearange the future clusters in order of area
        future_areas = np.zeros(future_count)
        for k in range(1, future_count + 1):
            future_areas[k - 1] = future_dictionary[k]["area"]
        copy_future = np.arange(1, future_count + 1)
        sorted_future = [
            x for (y, x) in sorted(zip(future_areas, copy_future), reverse=True)
        ]

        # Start by listing all the clusters in the future time step that need to be assigned
        for k in sorted_future:

            # Extract the graph of clusters associated with current cluster in the previous time step
            clusters_to_link = future_to_current_graph[k]["global"]
            nclusters = len(clusters_to_link)

            # If no clusters to link then this is a new cluster that appeared in future time step
            if nclusters == 0:

                # Update cluster count to create new IDs
                cluster_ID_count = cluster_ID_count + 1

                # Create entry
                cluster_data_dictionary[cluster_ID_count] = {}
                cluster_data_dictionary[cluster_ID_count]["start"] = next_date
                cluster_data_dictionary[cluster_ID_count]["end"] = []
                cluster_data_dictionary[cluster_ID_count]["parent_of"] = []
                cluster_data_dictionary[cluster_ID_count]["child_of"] = []
                cluster_data_dictionary[cluster_ID_count]["splitting_events"] = []
                cluster_data_dictionary[cluster_ID_count][
                    "splitting_events_by_date"
                ] = []
                cluster_data_dictionary[cluster_ID_count]["merging_events"] = []
                cluster_data_dictionary[cluster_ID_count]["merging_events_by_date"] = []
                cluster_data_dictionary[cluster_ID_count][next_date] = {}
                cluster_data_dictionary[cluster_ID_count][next_date][
                    "area"
                ] = future_dictionary[k]["area"]
                cluster_data_dictionary[cluster_ID_count][next_date][
                    "intensity"
                ] = future_dictionary[k]["intensity"]
                cluster_data_dictionary[cluster_ID_count][next_date][
                    "centroid"
                ] = future_dictionary[k]["centroid"]
                cluster_data_dictionary[cluster_ID_count][next_date][
                    "coordinates"
                ] = future_dictionary[k]["coordinates"]

                # Add this cluster to the list of clusters in the next time step
                cluster_data_dictionary[next_date].append(cluster_ID_count)
                print(
                    next_date,
                    "[Appear]: Cluster " + str(cluster_ID_count) + " appeared.",
                )

            elif nclusters > 0:

                # There is at least one cluster to link this future cluster

                # Go through each cluster from the previous time step that is linked
                largest_area = 0
                largest_cluster = 0
                for j in range(0, nclusters):

                    # ID of cluster in current/past (as in, not future) time step
                    current_ID = clusters_to_link[j]

                    # Extract the cluster's area
                    area = cluster_data_dictionary[current_ID][t_idx]["area"]

                    # Check if it's larger than the largest area found and that this ID hasn't been assigned already
                    if area > largest_area and (
                        current_ID not in cluster_data_dictionary[next_date]
                    ):
                        # Update area and ID of largest cluster found so far
                        largest_area = area
                        largest_cluster = current_ID

                # In the case that all the linked clusters have already inherited their ID to another cluster, create new one
                if largest_cluster == 0:

                    # Update cluster count to create new IDs
                    cluster_ID_count = cluster_ID_count + 1

                    # Create entry
                    cluster_data_dictionary[cluster_ID_count] = {}
                    cluster_data_dictionary[cluster_ID_count]["start"] = next_date
                    cluster_data_dictionary[cluster_ID_count]["end"] = []
                    cluster_data_dictionary[cluster_ID_count]["parent_of"] = []
                    cluster_data_dictionary[cluster_ID_count]["child_of"] = []
                    cluster_data_dictionary[cluster_ID_count]["splitting_events"] = []
                    cluster_data_dictionary[cluster_ID_count][
                        "splitting_events_by_date"
                    ] = []
                    cluster_data_dictionary[cluster_ID_count]["merging_events"] = []
                    cluster_data_dictionary[cluster_ID_count][
                        "merging_events_by_date"
                    ] = []
                    cluster_data_dictionary[cluster_ID_count][next_date] = {}
                    cluster_data_dictionary[cluster_ID_count][next_date][
                        "area"
                    ] = future_dictionary[k]["area"]
                    cluster_data_dictionary[cluster_ID_count][next_date][
                        "intensity"
                    ] = future_dictionary[k]["intensity"]
                    cluster_data_dictionary[cluster_ID_count][next_date][
                        "centroid"
                    ] = future_dictionary[k]["centroid"]
                    cluster_data_dictionary[cluster_ID_count][next_date][
                        "coordinates"
                    ] = future_dictionary[k]["coordinates"]

                    # Add this cluster to the list of clusters in the next time step
                    cluster_data_dictionary[next_date].append(cluster_ID_count)

                    # Record geneology
                    for j in range(0, nclusters):
                        # Append geneology
                        cluster_data_dictionary[clusters_to_link[j]][
                            "parent_of"
                        ].append(cluster_ID_count)
                        cluster_data_dictionary[cluster_ID_count]["child_of"].append(
                            clusters_to_link[j]
                        )

                else:
                    # One current cluster was found to be the largest and that will inherit it's ID

                    # Create entry for the biggest linked cluster that hasn't been assigned already
                    cluster_data_dictionary[largest_cluster][next_date] = {}
                    cluster_data_dictionary[largest_cluster][next_date][
                        "area"
                    ] = future_dictionary[k]["area"]
                    cluster_data_dictionary[largest_cluster][next_date][
                        "intensity"
                    ] = future_dictionary[k]["intensity"]
                    cluster_data_dictionary[largest_cluster][next_date][
                        "centroid"
                    ] = future_dictionary[k]["centroid"]
                    cluster_data_dictionary[largest_cluster][next_date][
                        "coordinates"
                    ] = future_dictionary[k]["coordinates"]

                    # Add this cluster to the list of clusters in the next time step
                    cluster_data_dictionary[next_date].append(largest_cluster)

                    # Remove from copy of current clusters
                    current_clusters_copy.remove(largest_cluster)

                    # Record geneology if there is more than one cluster involved
                    if nclusters > 1:
                        for j in range(0, nclusters):

                            # Record the merging event for the cluster that carried on the ID
                            if clusters_to_link[j] == largest_cluster:

                                # Record merging event
                                print(
                                    next_date,
                                    "[Merger]: Cluster "
                                    + str(largest_cluster)
                                    + " continues on",
                                )

                            elif clusters_to_link[j] != largest_cluster:

                                # Append geneology for the other clusters that merged into the largest one found above
                                cluster_data_dictionary[clusters_to_link[j]][
                                    "parent_of"
                                ].append(largest_cluster)
                                cluster_data_dictionary[largest_cluster][
                                    "child_of"
                                ].append(clusters_to_link[j])
                                print(
                                    next_date,
                                    "[Merger]: Cluster "
                                    + str(clusters_to_link[j])
                                    + " merged into "
                                    + str(largest_cluster),
                                )

                            # Record merging event
                            cluster_data_dictionary[clusters_to_link[j]][
                                "merging_events"
                            ].append(i + 1)
                            cluster_data_dictionary[clusters_to_link[j]][
                                "merging_events_by_date"
                            ].append(next_date)

        # Find splitting events
        for cluster_ID in current_IDs:

            # Future cluster(s) associated to this cluster
            linked_clusters = current_to_future_graph[cluster_ID]

            # Number of clusters to link
            nclusters = len(linked_clusters)

            # If there's more than one then there was a split
            if nclusters > 1:
                # Record time of event
                cluster_data_dictionary[cluster_ID]["splitting_events"].append(i + 1)
                cluster_data_dictionary[cluster_ID]["splitting_events_by_date"].append(
                    next_date
                )
                print(
                    next_date,
                    "[Split]: Cluster "
                    + str(cluster_ID)
                    + " split into "
                    + str(nclusters)
                    + " clusters.",
                )

            # Also record end dates of merged clusters
            if cluster_ID not in cluster_data_dictionary[next_date]:
                # Record their ending
                cluster_data_dictionary[cluster_ID]["end"] = t_idx
                print(
                    next_date,
                    "[Disappear]: Cluster "
                    + str(cluster_ID)
                    + " disappeared or merged.",
                )

        # Tie loose ends
        if next_date.year == end_date.year and next_date.month == end_date.month:
            for cluster_ID in cluster_data_dictionary[next_date]:
                cluster_data_dictionary[cluster_ID]["end"] = next_date

        # Update time
        t_idx = next_date

    # Save the total number of clusters
    cluster_data_dictionary["cluster_count"] = cluster_ID_count

    return cluster_data_dictionary


#############################################################################################################
######################################### ANALYZE DROUGHT CLUSTERS ##########################################
#############################################################################################################


def moving_average(time_series, window):
    """
    This function calculates a moving average based on a given window.

    Arguments:
    - time_series: Array of time series data
    - window: Period of time used for the moving average

    Returns:
    - time_series_filtered: Resulting filtered time series
    """

    # Length of time series
    n = len(time_series)

    # Array to save moving average
    time_series_filtered = np.zeros(n)

    # Calculate moving average
    for i in range(0, n - window):
        time_series_filtered[i] = np.mean(time_series[i : i + window])
    time_series_filtered[n - window :] = time_series[n - window :]

    return time_series_filtered


def find_clusters_displacements(centroid_lons, centroid_lats):
    """
    This function finds the displacements of each cluster.

    Arguments:
    - centroid_lons, centroid_lats: Longitudes and latitudes of the centroids of a given cluster

    Returns:
    - total_displacement: Total displacement of the drought cluster (integral over cluster path)
    - end_points: Distance between start and end point of the centroid for a given cluster.
    - displacements: Array of the displacements at each time step, filtered by a moving window
    """

    # Number of time steps that the current cluster lasted
    tsteps = len(centroid_lats)

    # Check that the cluster lasted more than one time step
    if tsteps > 1:

        # Initialize array of displacements
        displacements = np.zeros(tsteps - 1)

        # Calculate speeds
        for i in range(0, tsteps - 1):
            # Calculate distance between consecutive centroids (km)
            displacements[i] = find_geo_distance(
                centroid_lons[i],
                centroid_lats[i],
                centroid_lons[i + 1],
                centroid_lats[i + 1],
            )

        # Filtering displacement using a moving window
        window = 3
        if len(displacements) > window:
            displacements = moving_average(displacements, window)

        # Calculate total displacement (integral over path)
        total_displacement = np.sum(displacements)

        # Calculate distance between start and end points of the centroids
        end_points = find_geo_distance(
            centroid_lons[0], centroid_lats[0], centroid_lons[-1], centroid_lats[-1]
        )
    else:
        displacements = [0]
        total_displacement = 0
        end_points = 0

    return total_displacement, end_points, displacements


def diff_month(start_date, end_date):
    """
    This function calcualtes the number of months between two dates.

    Arguments:
    - start_date: Datetime object with the start date
    - end_date: Datetime object with the end date

    Returns:
    - Integer with the number of months between the two months.
    """

    return (end_date.year - start_date.year) * 12 + end_date.month - start_date.month


def extract_tracks(cluster_data_dictionary):
    """
    This function extracts in the tracks and characteristics of each of the clusters.

    Arguments:
    - cluster_data_dictionary: Dictionary of the tracked clusters containing all the iformation for each cluster.

    Returns:
    - tracks_dictionary: Dictionary with data regarding the tracks, areas, and intensities of the drought clusters.
    """

    # Find the number of clusters
    nclusters = cluster_data_dictionary["cluster_count"]
    print("Number of clusters:", nclusters)

    # Initialize dictionary where the data will be saved
    tracks_dictionary = {}

    # Go through each cluster and extract the coordinates of the centroids through time
    for cluster_ID in range(1, nclusters + 1):

        # Find cluster's beginning and end
        cluster_start = cluster_data_dictionary[cluster_ID]["start"]
        cluster_end = cluster_data_dictionary[cluster_ID]["end"]
        duration = diff_month(cluster_start, cluster_end) + 1

        # Initialize entries for current cluster
        tracks_dictionary["cluster_count"] = nclusters
        tracks_dictionary[cluster_ID] = {}
        tracks_dictionary[cluster_ID]["duration"] = duration
        tracks_dictionary[cluster_ID]["lons"] = []
        tracks_dictionary[cluster_ID]["lats"] = []
        tracks_dictionary[cluster_ID]["start_year"] = cluster_start.year
        tracks_dictionary[cluster_ID]["end_year"] = cluster_end.year
        tracks_dictionary[cluster_ID]["start_month"] = cluster_start.month
        tracks_dictionary[cluster_ID]["end_month"] = cluster_end.month
        tracks_dictionary[cluster_ID]["areas"] = []
        tracks_dictionary[cluster_ID]["intensities"] = []

        # Sweep through the time the cluster existed and extract centroid
        date_temp = cluster_start
        while date_temp <= cluster_end:
            # Current centroid
            centroid_lon, centroid_lat = cluster_data_dictionary[cluster_ID][date_temp][
                "centroid"
            ]

            # Save coordinates
            tracks_dictionary[cluster_ID]["lons"].append(centroid_lon)
            tracks_dictionary[cluster_ID]["lats"].append(centroid_lat)

            # Save characteristics
            tracks_dictionary[cluster_ID]["areas"].append(
                cluster_data_dictionary[cluster_ID][date_temp]["area"]
            )
            tracks_dictionary[cluster_ID]["intensities"].append(
                cluster_data_dictionary[cluster_ID][date_temp]["intensity"]
            )

            # Update date
            date_temp = date_temp + relativedelta(months=1)

        # Calculate centroid's displacements and speeds
        total_displacement, end_points, displacements = find_clusters_displacements(
            tracks_dictionary[cluster_ID]["lons"], tracks_dictionary[cluster_ID]["lats"]
        )

        tracks_dictionary[cluster_ID]["total_displacement"] = total_displacement
        tracks_dictionary[cluster_ID]["end_points"] = end_points
        tracks_dictionary[cluster_ID]["individual_displacements"] = displacements

    return tracks_dictionary


#############################################################################################################
############################################ CREATE NETCDF FILES ############################################
#############################################################################################################


def save_netcdf_file(
    data, lons, lats, units, var_name, var_info, file_name, start_date
):
    """
    This function saves the given data into a netcdf file.
    """

    # Dimensions
    nt, _, _ = data.shape

    # Put together data for netcdf file
    dims = {}
    dims["lons"] = lons
    dims["lats"] = lats
    dims["res"] = lons[1] - lons[0]
    dims["tres"] = "months"
    dims["time"] = np.arange(0, nt)
    dims["var_units"] = units

    # Create NetCDF file
    fp = Create_NETCDF_File(dims, file_name, var_name, var_info, data, start_date)


def Create_NETCDF_File(dims, file, var, var_info, data, tinitial):
    """
    This function creates a netcdf file to save a 3D matrix (t, lat, lon):

    Arguments:
    - dims: dictionary that contains the longitudes, latitudes, spatial and temporal resolutiosn, and
            number of time steps.
    - file: the full pat and file name of the netcdf file.
    - var: the name of the variable to be saved (short)
    - var_info: the long name and any information on the variable being saved.
    - data: the 3D data matrix (numpy array)
    - tinitial: a datetime object with the start date of the data

    Returns
    - f: netcdf file object created.
    """

    # Extract info
    lons = dims["lons"]
    lats = dims["lats"]
    res = dims["res"]
    nlon = len(lons)
    nlat = len(lats)
    tstep = dims["tres"]
    t = dims["time"]

    # Prepare the netcdf file
    # Create file
    f = Dataset(file, "w")

    # Define dimensions
    f.createDimension("lon", nlon)
    f.createDimension("lat", nlat)
    f.createDimension("t", len(t))

    # Longitude
    f.createVariable("lon", "d", ("lon",))
    f.variables["lon"][:] = lons
    f.variables["lon"].units = "degrees_east"
    f.variables["lon"].long_name = "Longitude"
    f.variables["lon"].res = res

    # Latitude
    f.createVariable("lat", "d", ("lat",))
    f.variables["lat"][:] = lats
    f.variables["lat"].units = "degrees_north"
    f.variables["lat"].long_name = "Latitude"
    f.variables["lat"].res = res

    # Time
    times = f.createVariable("t", "d", ("t",))
    f.variables["t"][:] = t
    f.variables["t"].units = "%s since %04d-%02d-%02d %02d:00:00.0" % (
        tstep,
        tinitial.year,
        tinitial.month,
        tinitial.day,
        tinitial.hour,
    )
    f.variables["t"].long_name = "Time"

    # Data
    f.createVariable(var, "f", ("t", "lat", "lon"), fill_value=-9.99e08)
    f.variables[var].long_name = var_info
    f.variables[var][:] = data

    return f
