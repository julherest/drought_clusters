'''
This script processes reanalysis data to be used in the clustering algorithm. The output is a 3D matrix of the percentiles
of cumulative anomalies of precipitation minus evaporation and a 2D matrix of the land-sea mask for the given reanalysis.
'''

# Import Python libraries
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
import matplotlib.pyplot as plt

# Import custom libraries
import data_analysis_lib_v4 as dalib
import drought_clusters_lib_v4 as dclib

###############################################################################################################################################
################################################################## FUNCTIONS ##################################################################
###############################################################################################################################################

def get_coordinate_indices(data_path, region, reanalysis):
	'''
	This function finds the indices of the region of interests within the coordinate
	grid of a given dataset.
	'''

	# Load the grid of the dataset
        if reanalysis not in ['Hadley', 'ESA']:
	    f = Dataset(data_path + 'invariant/land_sea_mask.nc')
        elif reanalysis == 'ESA':
            f = Dataset(data_path + 'sea_surface_salinity/ESACCI-SEASURFACESALINITY-L4-SSS-MERGED_OI_Monthly_CENTRED_15Day_25km-20181101-fv1.8.nc')
        else:
            f = Dataset(data_path + 'sea_surface_salinity/EN.4.2.1.f.analysis.g10.201812.nc')
        if reanalysis == 'ERA-Interim':
	    lons = f.variables['longitude'][:]
	    lats = f.variables['latitude'][:]
            shift_lons = 180
        elif reanalysis == 'MERRA2':
            lons = f.variables['lon'][:]
	    lats = f.variables['lat'][:]
            shift_lons = 0
        elif reanalysis == 'NCEP-DOE-R2':
            lons = f.variables['lon'][:]
	    lats = f.variables['lat'][:]
            shift_lons = 180
        elif reanalysis == 'CFSR':
            lons = f.variables['lon'][:]
            lats = f.variables['lat'][:]
            shift_lons = 180
        elif reanalysis == 'Hadley':
            lons = f.variables['lon'][:]
            lats = f.variables['lat'][:]
            shift_lons = 180
        elif reanalysis == 'ESA':
            lons = f.variables['lon'][:]
            lats = f.variables['lat'][:]
            shift_lons = 0
	f.close()
        
        # Select the coordinate limits
        if region == 'North_America':
                min_lon = -170
                max_lon = -40
                min_lat = 0
                max_lat = 70
        elif region == 'World':
                min_lon = np.min(lons)-shift_lons
                max_lon = np.max(lons)-shift_lons
                min_lat = np.min(lats)
                max_lat = np.max(lats)

	# Shift longitudes if necessary
	lons = lons - shift_lons
        
	# Find the indices for the data
	ilon_min = dalib.find_index(lons, min_lon)
	ilon_max = dalib.find_index(lons, max_lon)	
	ilat_min = dalib.find_index(lats, min_lat)
	ilat_max = dalib.find_index(lats, max_lat)
	
	# Find the new latitudes and longitudes
	ilons = np.arange(ilon_min,ilon_max+1)
        if reanalysis == 'ERA-Interim' or reanalysis == 'NCEP-DOE-R2' or reanalysis == 'CFSR':
	    ilats = np.arange(ilat_max,ilat_min+1)	
        else:
            ilats = np.arange(ilat_min,ilat_max+1)
	lons = lons[ilons]
	lats = lats[ilats] 
        
	return ilons, ilats, lons, lats

def flip_matrix(data_matrix):
	tsteps,nlats,nlons = data_matrix.shape
	data_matrix_new = np.zeros([tsteps, nlats, nlons])
	for i in range(0, tsteps):
		data_matrix_new[i,:,:] = np.flipud(data_matrix[i,:,:])
	return data_matrix_new

def shift_matrix(data_mat, lons, reanalysis):
	'''
	This function shifts the 2-D matrix to have longitudes ranging from -170W - 190E
        '''
        
        # Determine where the data will be pivoted
        if reanalysis == 'ERA-Interim' or reanalysis == 'NCEP-DOE-R2' or reanalysis == 'CFSR' or reanalysis == 'Hadley':
            lon_pivot = 190
            lons_shift = 170
        elif reanalysis == 'MERRA2' or reanalysis == 'ESA':
            lon_pivot = -170
            lons_shift = -10

        # Find the index of the pivot value
        idx= dalib.find_index(lons, lon_pivot)
        
        # Number of lons
	nlats, nlons = data_mat.shape

        # Make an empty array to save the data
        data_copy = np.zeros((nlats, nlons))

        # Shift matrix
        data_copy[:, 0:nlons-idx] = data_mat[:, idx:]
        data_copy[:, nlons-idx:] = data_mat[:, :idx]
        
        # Shift lons
        new_lons = lons - lons_shift

        return data_copy, new_lons
        
def extract_land_mask(data_path, save_path, ilats, ilons, region, reanalysis, start_year):
	'''
	This function extracts the land mask for the selected data
        '''

	# Load the mask
	f = Dataset(data_path + 'invariant/land_sea_mask.nc')
        if reanalysis == 'ERA-Interim':
            data = f.variables['lsm'][0, ilats, ilons]
            lons = f.variables['longitude'][ilons]
	    lats = f.variables['latitude'][ilats]

            # Flip the matrix
	    data = np.flipud(data)
	    lats = np.flipud(lats)

        elif reanalysis == 'MERRA2':
            data1 = f.variables['FRLAND'][0, ilats, ilons]
            data2 = f.variables['FRLANDICE'][0, ilats, ilons]
            lons = f.variables['lon'][ilons]
	    lats = f.variables['lat'][ilats]

            # Add the two masks
            data = data1 + data2
            
            # Convert from fraction to integers
            data[data>=0.5] = 1
            data[data<0.5] = 0

        elif reanalysis == 'NCEP-DOE-R2':
            data = f.variables['land'][0, ilats, ilons]
            lons = f.variables['lon'][ilons]
	    lats = f.variables['lat'][ilats]

            # Flip the matrix
	    data = np.flipud(data)
	    lats = np.flipud(lats)

        elif reanalysis == 'CFSR':
            data = f.variables['LAND_L1_Avg'][0, ilats, ilons]
            lons = f.variables['lon'][ilons]
	    lats = f.variables['lat'][ilats]

            # Flip the matrix
	    data = np.flipud(data)
	    lats = np.flipud(lats)

        # Close file
	f.close()
        
        # Shift global map
        if region == 'World':
                data, lons = shift_matrix(data, lons, reanalysis)
        print lons
        print
        print lats
        plt.imshow(data, origin = 'lower')
        plt.show()
 
        # Put together data for netcdf file
	dims = {}
	start_year = datetime(start_year,1,1)
	dims['lons'] = lons
	dims['lats'] = lats
	dims['res'] = lons[1]-lons[0]
    	var_name = 'lsm'
	var_info = 'Land-sea mask.'
	file_name = save_path + 'land_mask_' + region + '.nc' 
	
        # Create NetCDF file
	fp = dalib.Create_NETCDF_File_invariant(dims,file_name,var_name,var_info,data)
	print 'Done saving land mask for ' + reanalysis + '.' 
        
def calculate_p_minus_e(data_path, save_path, file_name, ilons, ilats, region, reanalysis, start_year, end_year, var_name):
        '''
        This function calculates P, E, or P-E for a given region. 
        '''

        # Dimensions
        nlons = len(ilons)
        nlats = len(ilats)
        nt = (end_year - start_year + 1)*12

        # Number of days per month
        month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        
        # Array containing leap years
        leap_years = np.arange(1980,2024,4)
        
        # Initialize array with all the data
        processed_data = np.zeros([nt, nlats, nlons])

        # Constant to convert latent heat flux to evaporation
        lhf_constant = 0.03526531

	# Index to keep track of location in array
	idx = 0

        # Select the appropriate algorithm for data extraction depending on the reanalysis
        if reanalysis == 'ERA-Interim' or reanalysis == 'MERRA2':
	
            # Loop through each year and month
	    for year in range(start_year,end_year+1):
		    for month in range(1, 13):
                    
			# Month name
			monthstr = "0"+str(month)
			monthstr = monthstr[-2:]

			# Date
			date = str(year) + monthstr
			print date

			# File paths
                        if reanalysis == 'ERA-Interim':
                            
                            # Variables are stored in separate files
                            prcp_path = data_path + 'total_precipitation/monthly/' + date + '.nc'
                            et_path = data_path + 'evaporation/monthly/' + date + '.nc'

                            # Open precipitation data (in m day-1)
			    f = Dataset(prcp_path)
                            prcp = f.variables['tp'][0,ilats, ilons]
			    lons = f.variables['longitude'][ilons]
			    lats = f.variables['latitude'][ilats]
			    f.close()
                            
                            # Open evaporation data (in m day-1)
                            f = Dataset(et_path)
                            et = f.variables['e'][0,ilats, ilons]
			    f.close()
                            
                            # Correct negative values to zeros
                            prcp[prcp<0] = 0

                            # Correct sign for ET
                            et = -et

                            # Flip the matrix
	                    prcp = np.flipud(prcp)
                            et = np.flipud(et)
	                    lats = np.flipud(lats)

                            # ERA-Interim data is in m/day so convert to mm/month
                            if year in leap_years and month == 2:
                                month_length = 29
                            else:
                                month_length = month_lengths[month-1]
                            prcp = prcp*1000*month_length
                            et = et*1000*month_length
                            
                        elif reanalysis == 'MERRA2':
                            
                            # Variables are stored in the same file 
                            if year >=1980 and year < 1992:
                                some_constant = '100'
                            elif year >= 1992 and year < 2001:
                                some_constant = '200'
                            elif year >= 2001 and year < 2011:
                                some_constant = '300'
                            elif year >= 2011:
                                some_constant = '400'
                            prcp_et_path = data_path + 'monthly/MERRA2_' + some_constant + '.tavgM_2d_flx_Nx.' + date + '.SUB.nc' 

			    # Open precipitation and evaporation data (in kg m-2 s-1)
			    f = Dataset(prcp_et_path)
			    prcp = f.variables['PRECTOT'][0,ilats, ilons]
                            et = f.variables['EVAP'][0,ilats, ilons]
			    lons = f.variables['lon'][ilons]
			    lats = f.variables['lat'][ilats]
			    f.close()

                            # MERRA-2 data is in mm/s so convert to mm/month
                            if year in leap_years and month == 2:
                                month_length = 29
                            else:
                                month_length = month_lengths[month-1]
                            prcp = prcp*60*60*24*month_length
                            et = et*60*60*24*month_length

                        # Save P, ET, or P-ET
                        if var_name == 'prcp':
                            data = prcp
                        elif var_name == 'et':
                            data = et
                        elif var_name == 'pme':
                            data = prcp-et

                        # Shift global map
                        if region == 'World':
                            data, lons = shift_matrix(data, lons, reanalysis)
                        
                        # Save data
                        processed_data[idx,:,:] = data
                        
			# Update index
			idx = idx + 1
        
        elif reanalysis == 'CFSR':

            # Aggregate prcp and et, bias correct, and save
            if var_name == 'prcp' or var_name == 'et':

                # Initialize temporary array
                temp_data_matrix = np.zeros([nt, nlats, nlons])

                # Loop through each year
	        for year in range(start_year,end_year+1):

                    # Extract data from CFSR from 1979-2010 (data is grouped by year)
                    if year < 2011:
                        print year

                        # Variables are stored in separate files (for CFSR we have latent heat flux instead of evaporation)
                        if var_name == 'prcp':
                            prcp_path = data_path + 'monthly/total_precipitation/pgbh06.gdas.' + str(year) + '.grb2.nc'

                            # Open precipitation data (in mm)
		            f = Dataset(prcp_path)
                            prcp = f.variables['A_PCP_L1_AccumAvg'][:,ilats, ilons]
                            lons = f.variables['lon'][:]
                            lats = f.variables['lat'][:]
		            f.close()

                            # Convert from 6-hourly totals to daily totals
                            prcp = prcp*4.
                            
                            # Save 
                            data = prcp

                        elif var_name == 'et':
                            lhf_path = data_path + 'monthly/latent_heat_flux/pgbh06.gdas.' + str(year) + '.grb2.nc' 
                    
                            # Open evaporation data (in W m-2)
                            f = Dataset(lhf_path)
                            lhf = f.variables['LHTFL_L1_FcstAvg'][:,ilats, ilons]
                            lons = f.variables['lon'][:]
                            lats = f.variables['lat'][:]
		            f.close()

                            # Calculate ET from latent heat flux (in mm/day)
                            et = lhf*lhf_constant

                            # Save P, ET
                            data = et

                        # Save within the larger matrix
                        temp_data_matrix[idx:idx+12,:,:] = data

                        # Update index 
                        idx = idx + 12

                        # Record when one is done analyzing CFSR data
                        if year == 2010:
                            idx_cfsr = idx
                            print idx_cfsr
                                
                    # Extract data from CFSv2 from 2011-2018 (data is stored for each month)
                    elif year >= 2011:
                                                    
                        for month in range(1, 13):
                    
			    # Month name
			    monthstr = "0"+str(month)
		            monthstr = monthstr[-2:]

		            # Date
			    date = str(year) + monthstr
			    print date

                            # Variables are stored in separate files (for CFSR we have latent heat flux instead of evaporation)
                            if var_name == 'prcp':
                                prcp_path = '/oak/stanford/groups/omramom/group_members/jehe/CFSv2/monthly/total_precipitation/pgbh.gdas.' + date + '.grb2.nc'

                                # Open precipitation data (in kg m-2)
		                f = Dataset(prcp_path)
                                prcp = f.variables['A_PCP_L1_AccumAvg'][:,ilats, ilons]
		                f.close()

                                # Convert from 6-hourly totals to daily totals
                                prcp = prcp*4

                                # Save data
                                data = prcp

                            elif var_name == 'et':
                                
                                lhf_path = '/oak/stanford/groups/omramom/group_members/jehe/CFSv2/monthly/latent_heat_flux/pgbh.gdas.' + date + '.grb2.nc' 
                                
                                # Open evaporation data (in W m-2)
                                f = Dataset(lhf_path)
                                lhf = f.variables['LHTFL_L1_FcstAvg'][:,ilats, ilons]
		                f.close()

                                # Calculate ET from latent heat flux 
                                et = lhf*lhf_constant

                                # Save data
                                data = et

                            # Save within the larger matrix
                            temp_data_matrix[idx,:,:] = data

                            # Update index 
                            idx = idx + 1

                print 'Data gathering complete. Now to shift and flip maps.'
            
                # Flip and shift the matrix, and convert to mm/month
                idx_month = 0; year = start_year
                for i in range(0, nt):
                  
                    # Current time step
                    temp = temp_data_matrix[i,:,:]

                    # Flip matrix
                    temp = np.flipud(temp)

                    # Shift matrix
                    if region == 'World':
                        temp, new_lons = shift_matrix(temp, lons, reanalysis)

                    # Convert to mm/month and save
                    if year in leap_years and idx_month + 1 == 2:
                        processed_data[i,:,:] = temp*29
                    else:
                        processed_data[i,:,:] = temp*month_lengths[idx_month]

                    # Update index
                    if idx_month < 11:
                        idx_month = idx_month + 1
                    else:
                        idx_month = 0
                        year = year + 1

                # Flip the latitudes
                lats = np.flipud(lats)
                   
                # Save shifted longitudes
                lons = new_lons

                # Calculate bias correction of CFSR using quantile matching
                processed_data = bias_correction_cfsr(processed_data, idx_cfsr)
                print 'Done doing bias correction'
                
            elif var_name == 'pme':

                # Paths
                prcp_path = save_path + 'prcp_monthly_' + region + '_' + str(start_year) + '-' + str(end_year) + '.nc'
                et_path = save_path + 'et_monthly_' + region + '_' + str(start_year) + '-' + str(end_year) + '.nc'

                # Load precipitation
                f = Dataset(prcp_path)
                prcp = f.variables['prcp'][:,ilats, ilons]
                lons = f.variables['lon'][:]
                lats = f.variables['lat'][:]
		f.close()

                # Load evaporation
                f = Dataset(et_path)
                et = f.variables['et'][:,ilats, ilons]
		f.close()

                # Calculate P-E
                processed_data = prcp - et
                idx_cfsr = 384
                
        elif reanalysis == 'CFSR_old':
                
            # Initialize temporary array
            temp_data_matrix = np.zeros([nt, nlats, nlons])
            temp_et_matrix = np.zeros([nt, nlats, nlons])

            # Loop through each year
	    for year in range(start_year,end_year+1):

                # Extract data from CFSR from 1979-2010 (data is grouped by year)
                if year < 2011:
                    print year

                    # Variables are stored in separate files (for CFSR we have latent heat flux instead of evaporation)
                    prcp_path = data_path + 'monthly/total_precipitation/pgbh06.gdas.' + str(year) + '.grb2.nc'
                    lhf_path = data_path + 'monthly/latent_heat_flux/pgbh06.gdas.' + str(year) + '.grb2.nc' 
                    
                    # Open precipitation data (in mm)
		    f = Dataset(prcp_path)
                    prcp = f.variables['A_PCP_L1_AccumAvg'][:,ilats, ilons]
		    lons = f.variables['lon'][ilons]
		    lats = f.variables['lat'][ilats]
		    f.close()

                    # Open evaporation data (in W m-2)
                    f = Dataset(lhf_path)
                    lhf = f.variables['LHTFL_L1_FcstAvg'][:,ilats, ilons]
		    f.close()

                    # Convert from 6-hourly totals to daily totals
                    prcp = prcp*4.

                    # Calculate ET from latent heat flux (in mm/day)
                    et = lhf*lhf_constant

                    # Save P, ET, or P-ET
                    if var_name == 'prcp':
                        data = prcp
                    elif var_name == 'et':
                        data = et
                    elif var_name == 'pme':
                        data = prcp-et

                    # Save within the larger matrix
                    temp_data_matrix[idx:idx+12,:,:] = data
                    temp_et_matrix[idx:idx+12,:,:] = et

                    if year == 2010:
                        # Mean of et so far
                        #max_et_cfsr = np.max(temp_et_matrix[:idx+12,:,:], axis = 0, keepdims=False)
                        mean_et_cfsr = np.mean(temp_et_matrix[:idx+12,:,:], axis = 0, keepdims=False)
                        std_et_cfsr = np.std(temp_et_matrix[:idx+12,:,:], axis = 0, keepdims=False)
                        idx_cfsr = idx+12

                    # Update index 
                    idx = idx + 12
                                
                # Extract data from CFSv2 from 2011-2018 (data is stored for each month)
                elif year >= 2011:
                                                    
                    for month in range(1, 13):
                    
			# Month name
			monthstr = "0"+str(month)
			monthstr = monthstr[-2:]

			# Date
			date = str(year) + monthstr
			print date

                        # Variables are stored in separate files (for CFSR we have latent heat flux instead of evaporation)
                        prcp_path = '/oak/stanford/groups/omramom/group_members/jehe/CFSv2/monthly/total_precipitation/pgbh.gdas.' + date + '.grb2.nc'
                        lhf_path = '/oak/stanford/groups/omramom/group_members/jehe/CFSv2/monthly/latent_heat_flux/pgbh.gdas.' + date + '.grb2.nc' 
                        
                        # Open precipitation data (in kg m-2)
		        f = Dataset(prcp_path)
                        prcp = f.variables['A_PCP_L1_AccumAvg'][:,ilats, ilons]
		        f.close()

                        # Open evaporation data (in W m-2)
                        f = Dataset(lhf_path)
                        lhf = f.variables['LHTFL_L1_FcstAvg'][:,ilats, ilons]
		        f.close()

                        # Convert from 6-hourly totals to daily totals
                        prcp = prcp*4

                        # Calculate ET from latent heat flux 
                        et = lhf*lhf_constant

                        # Save P, ET, or P-ET
                        if var_name == 'prcp':
                            data = prcp
                        elif var_name == 'et':
                            data = et
                        elif var_name == 'pme':
                            data = prcp-et
                            
                        # Save within the larger matrix
                        temp_data_matrix[idx,:,:] = data
                        temp_et_matrix[idx,:,:] = et

                        # Update index 
                        idx = idx + 1

            print 'Data gathering complete. Now to shift and flip maps.'
            
            # Calculate the mean of ET during CFSv2 period and calculate the difference
            # with the mean ET during CFSR to identify grid cells with discontinuities
            mean_et_cfsv2 = np.mean(temp_et_matrix[idx_cfsr:,:,:], axis = 0, keepdims=False)
            std_et_cfsv2 = np.std(temp_et_matrix[idx_cfsr:,:,:], axis = 0, keepdims=False)
            diff_cfsv2_cfsr = mean_et_cfsv2-mean_et_cfsr
            ratio_cfsv2_cfsr_2 = std_et_cfsv2/std_et_cfsr
            
            # Create mask for the pixels with discontinuities
            mask = np.zeros([nlats, nlons])
            mask[np.abs(diff_cfsv2_cfsr) > 2.5*std_et_cfsr] = 1
            mask[ratio_cfsv2_cfsr_2 > 2] = 1
            print 'Percent of grid cells with discontinuities:', 100*len(np.where(mask>0)[0])/np.float(nlats*nlons)
   
            # Flip and shift the matrix, and convert to mm/month
            idx_month = 0; year = start_year
            for i in range(0, nt):
                  
                    # Current time step
                    temp = temp_data_matrix[i,:,:]

                    # Mask the grid cells with discontinuities in ET
                    #if var_name != 'prcp':
                    #temp[mask > 0] = np.nan
                    
                    # Flip matrix
                    temp = np.flipud(temp)

                    # Shift matrix
                    if region == 'World':
                        temp, new_lons = shift_matrix(temp, lons, reanalysis)

                    # Convert to mm/month and save
                    if year in leap_years and idx_month + 1 == 2:
                        processed_data[i,:,:] = temp*29
                    else:
                        processed_data[i,:,:] = temp*month_lengths[idx_month]

                    # Update index
                    if idx_month < 11:
                        idx_month = idx_month + 1
                    else:
                        idx_month = 0
                        year = year + 1

            # Flip the latitudes
            lats = np.flipud(lats)
                   
            # Save shifted longitudes
            lons = new_lons

            # Calculate bias correction of CFSR using quantile matching
            processed_data = bias_correction_cfsr(processed_data, idx_cfsr)
            print 'Done doing bias correction'
       
        bla = processed_data[:,342, 217]; print np.mean(bla[:idx_cfsr]), np.mean(bla[idx_cfsr:]), np.std(bla[:idx_cfsr]), np.std(bla[idx_cfsr:])
        bla = processed_data[:,319, 137]; print np.mean(bla[:idx_cfsr]), np.mean(bla[idx_cfsr:]), np.std(bla[:idx_cfsr]), np.std(bla[idx_cfsr:])
        bla = processed_data[:,324, 125]; print np.mean(bla[:idx_cfsr]), np.mean(bla[idx_cfsr:]), np.std(bla[:idx_cfsr]), np.std(bla[idx_cfsr:])
        plt.subplot(4,1,1); plt.imshow(processed_data[450,:,:], origin='lower', cmap = 'plasma_r'); plt.colorbar(); plt.clim(0,15*30)
        plt.subplot(4,1,2); plt.plot(processed_data[:,342, 217]); 
        plt.subplot(4,1,3); plt.plot(processed_data[:,319, 137]);
        plt.subplot(4,1,4); plt.plot(processed_data[:,10, 10])
        plt.show()
        
        #plt.imshow(np.mean(processed_data,0), origin = 'lower', cmap = 'seismic_r'); plt.colorbar(); plt.clim(-150,150)
        plt.imshow(np.nanmean(processed_data,0), origin = 'lower', cmap = 'plasma_r'); plt.colorbar(); plt.clim(0,15*30)
        plt.title('Mean ' + var_name)
        plt.show()
        
        # Put together data for netcdf file
	dims = {}
	start_year = datetime(start_year,1,1)
	dims['lons'] = lons
	dims['lats'] = lats
	dims['res'] = lons[1]-lons[0]
	dims['tres'] = 'months'
	dims['time'] = np.arange(0,nt)
	dims['var_units'] = "mm/month" 
        if var_name == 'prcp':
            var_name = 'prcp'
	    var_info = 'Monthly P.'
        elif var_name == 'et':
            var_name = 'et'
	    var_info = 'Monthly ET.'
        elif var_name == 'pme':
	    var_name = 'pme'
	    var_info = 'Monthly accumulation of P-ET.'
        
	# Create NetCDF file
	fp = dalib.Create_NETCDF_File(dims, save_path + file_name, var_name, var_info, processed_data, start_year)
	print 'Done processing ' + var_name + ' from monthly data.'

def bias_correction_cfsr(data_matrix, idx):
        '''
        This function does a bias correction of CFSR part of the data using CFSv2.
        '''

        # Dimensions of dataset
        nt, nlats, nlons = data_matrix.shape

        # Array to save results (copy of the input array)
        bias_corrected = np.array(data_matrix[:])

        for i in range(0, nlats):
            for j in range(0, nlons):

                # Current array
                data_array = data_matrix[:, i, j]

                # Separate by periods
                array_cfsr = data_array[:idx]
                array_cfs = data_array[idx:]

                # Calculate bias corrected CFSv2 data
                array_cfsr_corrected = bias_correction(array_cfsr, array_cfs)

                # Save
                bias_corrected[:idx, i, j] = array_cfsr_corrected

        return bias_corrected

def bias_correction(biased_array, reference_array):
        '''
        This function calculates the bias correction for a single time series.
        '''

        # Definitions
	nbins = 50
        
        # Sort the arrays
        biased_sorted = np.sort(biased_array)
        reference_sorted = np.sort(reference_array)
        
        # Calculate the min and max value and define bins.
	min_value = np.min([np.min(biased_array), np.min(reference_array)])
	max_value = np.max([np.max(biased_array), np.max(reference_array)]) 
	xbins = np.linspace(min_value,max_value,nbins)

        # Create PDFs of each dataset
	pdf_ref, bins = np.histogram(reference_sorted, bins = xbins, density = True)
	pdf_biased, bins = np.histogram(biased_sorted, bins = xbins, density = True)

        # Create CDF with zero in first entry.
	cdf_ref = np.insert(np.cumsum(pdf_ref),0,0.0)
	cdf_biased = np.insert(np.cumsum(pdf_biased),0,0.0)

        # Calculate exact CDF values of model data using linear interpolation
        cdf_biased_data = np.interp(biased_array, xbins, cdf_biased)

        # Now use interpol again to invert the observations CDF, hence reversed x,y
	array_corrected = np.interp(cdf_biased_data, cdf_ref, xbins)

        '''
        plt.subplot(3,1,1); plt.plot(reference_array)
        plt.subplot(3,1,2); plt.plot(biased_array)
        plt.subplot(3,1,3); plt.plot(array_corrected)
        plt.show()
        exit()
        '''
        return array_corrected

def calculate_cumulative_anomalies(data_path, data_file_name, save_path, save_file_name1, save_file_name2, window, start_year):
        '''
        This function calculates cumulative anomalies of P-E over a given window.
        '''

        # Load P-E data
        f = Dataset(data_path + data_file_name)
        data = f.variables['pme'][:]
        lons = f.variables['lon'][:]
        lats = f.variables['lat'][:]
        f.close()
        
        # Calculate cumulative anomalies
        cumulative_anomalies = dalib.calculate_cumulative_anomalies_matrix(data, window)

        # Mask
        temp = data[12:,:,:]
        cumulative_anomalies[np.isnan(temp)] = np.nan

        # Calculate maximum range for each location
        range_values = dalib.calculate_range_values(data)

        # Calculate normalized cumulative anomalies
        normalized_cumulative_anomalies = cumulative_anomalies/range_values

        # Plot to check
        plt.subplot(6,1,1); plt.imshow(np.mean(data,0), origin = 'lower', interpolation = 'nearest'); plt.colorbar()
        plt.subplot(6,1,2); plt.imshow(cumulative_anomalies[0,:,:], origin = 'lower', interpolation = 'nearest'); plt.colorbar()
        plt.subplot(6,1,3); plt.imshow(range_values, origin = 'lower', interpolation = 'nearest'); plt.colorbar( )
        plt.subplot(6,1,4); plt.imshow(np.max(np.abs(normalized_cumulative_anomalies),0), origin = 'lower', interpolation = 'nearest'); plt.colorbar( )
        plt.subplot(6,1,5); plt.plot(cumulative_anomalies[:,26, 665])
        plt.subplot(6,1,6); plt.plot(normalized_cumulative_anomalies[:,26, 665])
        print 'Maximum normalized cumulative anomalies:', np.nanmax(abs(normalized_cumulative_anomalies))
        plt.show()
        
        # Dimensions of data (starting now one year later)
        nt, nlats, nlons = cumulative_anomalies.shape
        
        # Start date of processed data data
        start_date = datetime(start_year,1,1)

        # Put together data for netcdf file of cumulative anomalies
	dims = {}
	dims['lons'] = lons
	dims['lats'] = lats
	dims['res'] = lons[1]-lons[0]
	dims['tres'] = 'months'
	dims['time'] = np.arange(0,nt)
	dims['var_units'] = 'mm per ' + str(window) + ' months' 
	var_name = 'pme'
	var_info = 'Cumulative anomalies of P-ET over ' + str(window) + ' months.'
        
	# Create NetCDF file
	fp = dalib.Create_NETCDF_File(dims, save_path + save_file_name1, var_name, var_info, cumulative_anomalies, start_date)
	print 'Done saving cumulative anomalies (' + str(window) + ' months) of P-E from monthly data.'

        # Put together data for netcdf file of normalized cumulative anomalies
	dims = {}
	dims['lons'] = lons
	dims['lats'] = lats
	dims['res'] = lons[1]-lons[0]
	dims['tres'] = 'months'
	dims['time'] = np.arange(0,nt)
	dims['var_units'] = 'mm per ' + str(window) + ' months/mm per ' + str(window) + ' months' 
	var_name = 'pme'
	var_info = 'Normalized cumulative anomalies of P-ET over ' + str(window) + ' months.'
        
	# Create NetCDF file
	fp = dalib.Create_NETCDF_File(dims, save_path + save_file_name2, var_name, var_info, normalized_cumulative_anomalies, start_date)
	print 'Done saving normalized cumulative anomalies (' + str(window) + ' months) of P-E from monthly data.'

def calculate_percentiles(data_path, data_file_name, percentiles_file_name, reanalysis, start_year):
    '''
    This function calculates the percentiles of P-E cumulative anomalies to be able to run the cluster code in parallel. 
    '''
    
    # Load P-E anomalies 
    f = Dataset(data_path + data_file_name)
    data_matrix = f.variables['pme'][:]
    lons = f.variables['lon'][:]
    lats = f.variables['lat'][:]
    f.close()

    plt.imshow(data_matrix[0,:,:], origin='lower')
    plt.show()

    # Calculate percentiles of 3D matrix (time, lat, lon)
    percentiles_matrix = dclib.obtain_percentiles_matrix(data_matrix)

    # Pass on the same mask as the cumulative anomalies
    percentiles_matrix[np.isnan(data_matrix)==True] = np.nan

    # Information to save the percentiles_matrix
    start_date = datetime(start_year,1,1)
    units = 'percentiles'
    var_name_percentiles = 'percentiles'
    var_info = 'Percentiles of P-E cumulative anomalies from ' + reanalysis + '.'

    # Save percentiles matrix 
    dalib.save_netcdf_file(percentiles_matrix, lons, lats, units, var_name_percentiles, var_info, data_path + percentiles_file_name, start_date)
    print 'Done calculating and saving percentiles.'

    return

def check_land_mask(land_mask_path):
    '''
    This functions plots the land mask for the given reanalysis.
    '''
    # Load data
    f = Dataset(land_mask_path + 'land_mask_' + region + '.nc')
    data = f.variables['lsm'][:]
    lons = f.variables['lon'][:]
    lats = f.variables['lat'][:]
    f.close()
    
    print lons
    print
    print lats
    print
    
    # Plot
    plt.imshow(data, origin='lower')
    plt.show()

def check_percentiles(percentiles_path_file_name):
    '''
    This function is used to visualize the percentiles calculated by drought_clusters.py.
    '''

    # Load data
    f = Dataset(percentiles_path_file_name)
    data = f.variables['percentiles'][:]
    lons = f.variables['lon'][:]
    lats = f.variables['lat'][:]
    f.close()

    print lons
    print
    print lats
    print
    
    # Plot it
    for i in range(0,2):
        current_data = data[i*10,:,:]
        plt.subplot(2,1,1); plt.imshow(current_data, origin='lower'); plt.clim(0,1)
        current_data[current_data>0.2] = np.nan
        plt.subplot(2,1,2); plt.imshow(current_data, origin='lower'); plt.clim(0,1)

        plt.show()

def calculate_environmental_anomalies(data_path, save_path, iltas, ilons, var_name, anomalies_type, window, reanalysis, start_year, end_year):
        '''
        This function calculates the monthly anomalies for the given variable. 
        '''
    
        # Dimensions
        nlons = len(ilons)
        nlats = len(ilats)
        nt = (end_year - start_year + 1)*12
        if reanalysis == 'ESA':
            nt = nt - 1

        # Initialize array with all the data
        processed_data = np.zeros([nt, nlats, nlons])

	# Index to keep track of location in array
	idx = 0

	# Loop through each month
	for year in range(start_year, end_year + 1):
		for month in range(1, 13):

                    # Month name
		    monthstr = "0"+str(month)
		    monthstr = monthstr[-2:]

		    # Date
		    date = str(year) + monthstr
		    #print date

                    if reanalysis == 'ERA-Interim':
			
			# File paths
                        if var_name == 'geopotential':
                            full_path = data_path + 'geopotential_500mb/monthly/' + date + '.nc'
                            var_code = 'z'
                        elif var_name == 'SST':
                            full_path = data_path + 'SST/monthly/' + date + '.nc'
                            var_code = 'sst'
                        elif var_name == 'sea_level_pressure':
                            full_path = data_path + 'mean_sea_level_pressure/monthly/' + date + '.nc'
                            var_code = 'msl'
                        elif var_name == 'Q_flux_eastward':
                            full_path = data_path + 'vertical_integral_of_eastward_vapor_flux/monthly/' + date + '.nc'
                            var_code = 'p71.162'
                        elif var_name == 'Q_flux_northward':
                            full_path = data_path + 'vertical_integral_of_northward_vapor_flux/monthly/' + date + '.nc'
                            var_code = 'p72.162'
                        elif var_name == 'u_wind_10m':
                            full_path = data_path + 'u_wind_10m/monthly/' + date + '.nc'
                            var_code = 'u10'
                        elif var_name == 'v_wind_10m':
                            full_path = data_path + 'v_wind_10m/monthly/' + date + '.nc'
                            var_code = 'v10'
                        elif var_name == 'u_wind_500mb':
                            full_path = data_path + 'u_wind_500mb/monthly/' + date + '.nc'
                            var_code = 'u'
                        elif var_name == 'v_wind_500mb':
                            full_path = data_path + 'v_wind_500mb/monthly/' + date + '.nc'
                            var_code = 'v'
                        elif var_name == 'soil_moisture':
                            full_path = data_path + 'volumetric_soil_moisture/monthly/' + date + '.nc'
                            var_code1 = 'swvl1'
                            var_code2 = 'swvl2'
                            var_code3 = 'swvl3'

                            # Thickness of soil moisture layers in ERA-interim
                            thickness = np.array([0.07, 0.21, 0.72])

                        # Open data
                        if var_name == 'soil_moisture':
			    f = Dataset(full_path)
			    data_layer1 = f.variables[var_code1][0,ilats, ilons]
                            data_layer2 = f.variables[var_code2][0,ilats, ilons]
                            data_layer3 = f.variables[var_code3][0,ilats, ilons]
                            data = thickness[0]*data_layer1 + thickness[1]*data_layer2 + thickness[2]*data_layer3
                        else:
			    f = Dataset(full_path)
			    data = f.variables[var_code][0,ilats, ilons]
			lons = f.variables['longitude'][ilons]
			lats = f.variables['latitude'][ilats]
			f.close()

                        # Mask out land areas for SSTs
                        if var_name == 'SST':
                            data[data<0] = np.nan
                        elif var_name == 'soil_moisture':
                            data[data<1e-10] = np.nan

                        # Shift global map and flip it
                        if region == 'World':
                            data, lons = shift_matrix(data, lons, reanalysis)
                        data = np.flipud(data)
                        lats = np.flipud(lats)
                                        
			# Save data
			processed_data[idx,:,:] = data
                        #plt.imshow(data);
                        #plt.show()
                        #exit()
                        
			# Update index
			idx = idx + 1

                    elif reanalysis == 'Hadley':
                    
                        # File paths
                        if var_name == 'sea_surface_salinity':
                            full_path = data_path + 'sea_surface_salinity/EN.4.2.1.f.analysis.g10.' + date + '.nc'
                            var_code = 'salinity'
                            var_code2 = 'salinity_observation_weights'

                        # Data quality threshold
                        quality_threshold = 0.95

                        # Open data
                        idx_depth = 10
			f = Dataset(full_path)
                        depths = f.variables['depth'][:idx_depth]
                        data = f.variables[var_code][0,:idx_depth,ilats, ilons]    # Get data down to 9.83 meters
                        qaqc = f.variables[var_code2][0,:idx_depth,ilats, ilons]
			lons = f.variables['lon'][ilons]
			lats = f.variables['lat'][ilats]
			f.close()

                        # Mask values with poor quality
                        data = np.ma.masked_where(qaqc < quality_threshold, data)
                        
                        # Average data across depths
                        data = np.nanmean(data, axis = 0)
                                                                
                        # Shift global map
                        if region == 'World':
                            data, lons = shift_matrix(data, lons, reanalysis)
                        
                        # Mask out land areas
                        #data[data<0] = np.nan

                        #plt.imshow(data); plt.colorbar()
                        #plt.show()
                        #exit()

			# Save data
			processed_data[idx,:,:] = data
			
                        # Update index
			idx = idx + 1

                    elif reanalysis == 'ESA':
                      
                      if date != '201812':

                        if var_name == 'sea_surface_salinity':
                            fname = 'ESACCI-SEASURFACESALINITY-L4-SSS-MERGED_OI_Monthly_CENTRED_15Day_25km-' + date + '01-fv1.8.nc'
                            full_path = data_path + 'sea_surface_salinity/' + fname
                            var_code = 'sss'
                          
                        # Open data for current month
                        f = Dataset(full_path)
                        data = f.variables[var_code][0,ilats, ilons]
			lons = f.variables['lon'][ilons]
			lats = f.variables['lat'][ilats]
			f.close()

                        # Shift global map
                        if region == 'World':
                            data, lons = shift_matrix(data, lons, reanalysis)
                        
                        # Mask out land areas
                        data[data==0] = np.nan

                        #plt.imshow(data); plt.colorbar()
                        #plt.show()

			# Save data
			processed_data[idx,:,:] = data
			
                        # Update index
			idx = idx + 1

        # Detrend SSS data
        #if reanalysis == 'Hadley':
            
        #    processed_data = dalib.detrend_data_matrix(processed_data)
        #    processed_data = np.ma.masked_where(processed_data==0, processed_data)

        # Calculate anomalies
        if anomalies_type == 'anomalies':
            anomalies = dalib.calculate_anomalies_matrix(processed_data)
        
            # Constrain to the following year to match the data clusters
            if reanalysis != 'ESA':
                anomalies = anomalies[12:,:,:]
            
        elif anomalies_type == 'cumulative_anomalies':
            anomalies = dalib.calculate_cumulative_anomalies_matrix(processed_data, window)

        elif anomalies_type == 'normal':
            
            # Constrain to the following year to match the data clusters
            if reanalysis != 'ESA':
                anomalies = processed_data[12:,:,:]
            else:
                anomalies = processed_data

        # Plot to check
        print anomalies.shape
        plt.subplot(4,1,1); plt.imshow(processed_data[100,:,:], origin = 'lower', interpolation = 'nearest'); plt.colorbar(); #plt.clim(280,300)
        plt.subplot(4,1,2); plt.imshow(anomalies[100,:,:], origin = 'lower', interpolation = 'nearest'); plt.colorbar()
        plt.subplot(4,1,3); plt.plot(processed_data[:,81,170])
        plt.subplot(4,1,4); plt.plot(anomalies[:,81,170])
        plt.show()
        
        # Dimensions of data (starting now in 1980)
        nt, nlats, nlons = anomalies.shape
        print var_name, anomalies_type
        # Put together data for netcdf file
	dims = {}
	start_date = datetime(start_year + 1, 1, 1)
	dims['lons'] = lons
	dims['lats'] = lats
	dims['res'] = lons[1]-lons[0]
	dims['tres'] = 'months'
	dims['time'] = np.arange(0,nt)
        if var_name == 'geopotential':
	    dims['var_units'] = 'm**2 s**-2' 
    	    var_info = 'Monthly anomalies of geopotential height at 500mb.'
	    file_name = save_path + anomalies_type + '_geopotential_500mb_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc' 
        elif var_name == 'SST':
	    dims['var_units'] = 'K' 
    	    var_info = 'Monthly anomalies of sea surface temperatures.'
	    file_name = save_path + anomalies_type + '_SST_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
        elif var_name == 'sea_level_pressure':
            dims['var_units'] = 'Pa' 
    	    var_info = 'Monthly anomalies of sea level pressure.'
	    file_name = save_path + anomalies_type + '_sea_level_pressure_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
        elif var_name == 'Q_flux_eastward':
            dims['var_units'] = 'kg m**-1 s**-1' 
    	    var_info = 'Vertical integral of eastward water vapour flux'
	    if anomalies_type == 'normal':
	        file_name = save_path + 'Q_flux_eastward_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
            else:
                file_name = save_path + anomalies_type + '_Q_flux_eastward_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
        elif var_name == 'Q_flux_northward':
            dims['var_units'] = 'kg m**-1 s**-1' 
    	    var_info = 'Vertical integral of northward water vapour flux'
            if anomalies_type == 'normal':
	        file_name = save_path + 'Q_flux_northward_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
            else:
                file_name = save_path + anomalies_type + '_Q_flux_northward_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
        elif var_name == 'u_wind_10m':
            dims['var_units'] = 'm s**-1' 
    	    var_info = '10 metre U wind component'
	    if anomalies_type == 'normal':
	        file_name = save_path + 'u_wind_10m_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
            else:
                file_name = save_path + anomalies_type + '_u_wind_10m_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
        elif var_name == 'v_wind_10m':
            dims['var_units'] = 'm s**-1' 
    	    var_info = '10 metre V wind component'
	    if anomalies_type == 'normal':
	        file_name = save_path + 'v_wind_10m_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
            else:
                file_name = save_path + anomalies_type + '_v_wind_10m_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
        elif var_name == 'u_wind_500mb':
            dims['var_units'] = 'm s**-1' 
    	    var_info = 'U wind component at 500mb'
	    if anomalies_type == 'normal':
	        file_name = save_path + 'u_wind_500mb_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
            else:
                file_name = save_path + anomalies_type + '_u_wind_500mb_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
        elif var_name == 'v_wind_500mb':
            dims['var_units'] = 'm s**-1' 
    	    var_info = 'V wind component at 500mb'
	    if anomalies_type == 'normal':
	        file_name = save_path + 'v_wind_500mb_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
            else:
                file_name = save_path + anomalies_type + '_v_wind_500mb_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
        elif var_name == 'sea_surface_salinity' and reanalysis == 'Hadley':
	    dims['var_units'] = 'psu' 
    	    var_info = 'Monthly sea surface salinity down to ' + str(np.max(depths)) + ' meters.'
	    if anomalies_type == 'normal':
	        file_name = save_path + 'sss_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
            else:
                file_name = save_path + anomalies_type + '_sss_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
        elif var_name == 'sea_surface_salinity' and reanalysis == 'ESA':
	    dims['var_units'] = 'psu' 
    	    var_info = 'Monthly sea surface salinity'
	    if anomalies_type == 'normal':
	        file_name = save_path + 'sss_' + region + '_' + str(start_year) + '-' + str(end_year) + '.nc'
            else:
                file_name = save_path + anomalies_type + '_sss_' + region + '_' + str(start_year) + '-' + str(end_year) + '.nc'
        elif var_name == 'soil_moisture':
	    dims['var_units'] = 'm**3 m**-3' 
    	    var_info = 'Monthly volumetric soil water (0-1m)'
            var_code = 'vsw'
            if anomalies_type == 'normal':
	        file_name = save_path + 'sm_' + region + '_' + str(start_year+1) + '-' + str(end_year) + '.nc'
            else:
                file_name = save_path + anomalies_type + '_sm_' + region + '_' + str(start_year+1) + '-' + str(end_year) + '.nc'
            
	# Create NetCDF file
	fp = dalib.Create_NETCDF_File(dims, file_name, var_code, var_info, anomalies, start_date)
	print file_name
        print 'Done saving anomalies for ' + var_name 

def calculate_scalar_moisture_fluxes(data_path, save_path, iltas, ilons, reanalysis, start_year, end_year):
    '''
    This function calculates the magnitude of the moisture flux vectors.
    '''

    # Dimensions
    nlons = len(ilons)
    nlats = len(ilats)
    nt = (end_year - start_year + 1)*12

    # Initialize array with all the data
    processed_data = np.zeros([nt, nlats, nlons])

    # Index to keep track of location in array
    idx = 0

    # Loop through each month
    for year in range(start_year, end_year + 1):
	for month in range(1, 13):

                if reanalysis == 'ERA-Interim':

		    # Month name
		    monthstr = "0"+str(month)
		    monthstr = monthstr[-2:]

		    # Date
		    date = str(year) + monthstr
		    print date

                    # Load U component
                    f = Dataset(data_path + 'vertical_integral_of_eastward_vapor_flux/monthly/' + date + '.nc')
		    u_data = f.variables['p71.162'][0,ilats, ilons]
		    lons = f.variables['longitude'][ilons]
		    lats = f.variables['latitude'][ilats]
		    f.close()

                    # Load V component
                    f = Dataset(data_path + 'vertical_integral_of_northward_vapor_flux/monthly/' + date + '.nc')
		    v_data = f.variables['p72.162'][0,ilats, ilons]
		    f.close()

                    # Shift global map and flip it
                    if region == 'World':
                        u_data, lons_shifted = shift_matrix(u_data, lons, reanalysis)
                        v_data, _ = shift_matrix(v_data, lons, reanalysis)
                    u_data = np.flipud(u_data)
                    v_data = np.flipud(v_data)
                    lats = np.flipud(lats)
                                        
		    # Save data
		    processed_data[idx,:,:] = np.sqrt(u_data**2 + v_data**2)

                    # Update index
		    idx = idx + 1

    # Calculate anomalies
    anomalies = dalib.calculate_anomalies_matrix(processed_data)
        
    # Constrain to the following year to match the data clusters
    processed_data = processed_data[12:,:,:]
    anomalies = anomalies[12:,:,:]

    # Plot to check
    plt.subplot(4,1,1); plt.imshow(processed_data[300,:,:], origin = 'lower', interpolation = 'nearest'); plt.colorbar(); #plt.clim(280,300)
    plt.subplot(4,1,2); plt.imshow(anomalies[300,:,:], origin = 'lower', interpolation = 'nearest'); plt.colorbar()
    plt.subplot(4,1,3); plt.plot(processed_data[:,72,204])
    plt.subplot(4,1,4); plt.plot(anomalies[:,72,204])
    plt.show()

    # Dimensions of data (starting now in 1980)
    nt, nlats, nlons = processed_data.shape
        
    # Put together data for netcdf file
    dims = {}
    start_date = datetime(start_year + 1, 1, 1)
    dims['lons'] = lons_shifted
    dims['lats'] = lats
    dims['res'] = lons[1]-lons[0]
    dims['tres'] = 'months'
    dims['time'] = np.arange(0,nt)
    dims['var_units'] = 'kg m**-1 s**-1' 
    var_code = 'moisture_flux'
    var_info1 = 'Monthly magnitudes of vertically integrated moisture fluxes.'
    file_name1 = save_path + '_moisture_flux_magnitude_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
            
    # Create NetCDF file
    fp = dalib.Create_NETCDF_File(dims, file_name1, var_code, var_info1, processed_data, start_date)

    # Information for the anomalies file
    var_info2 = 'Monthly anomalies of magnitudes of vertically integrated moisture fluxes.'
    file_name2 = save_path + 'anomalies_moisture_flux_magnitude_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
            
    # Create NetCDF file
    fp = dalib.Create_NETCDF_File(dims, file_name2, var_code, var_info2, anomalies, start_date)

    print 'Done saving scalar moisture fluxes.' 
    
###############################################################################################################################################
################################################################ EXECUTING CODE ###############################################################
###############################################################################################################################################

##################################################### DEFINITIONS, PATHS, AND FILE NAMES ######################################################

# ** Chose the reanalysis (options: "ERA-Interim", "MERRA2", "CFSR", "NCEP-DOE-R2", "Hadley", "ESA")
reanalysis = 'CFSR'

# ** Type of analysis
analysis = 'pme'

# ** Define the region to use (options: "World")
region = 'World'

# ** Window of cumulative anomalies of precipitation minus evaporation (abbreviated as "pme" in the paths and file names below)
window = 12

# ** Version of the analysis (matching the ending of the current file)
version = 'v4'

#### Done editing ####

# Start and end years for the data of the chosen reanalysis  
if reanalysis in ['ERA-Interim', 'CFSR', 'NCEP-DOE-R2', 'Hadley']:
    start_year = 1979
    end_year = 2018
elif reanalysis in ['MERRA2']:
    start_year = 1980
    end_year = 2018
elif reanalysis == 'ESA':
    start_year = 2010
    end_year = 2018

# Path where the raw reanalysis data is stored 
reanalysis_path = '/oak/stanford/groups/omramom/group_members/jehe/' + reanalysis + '/'

# Path where the processed data will be stored
processed_path = '/oak/stanford/groups/omramom/group_members/jehe/Ocean_Clusters/' + version + '/' + reanalysis + '/' + region + '/'

# Path where the files of the cumulative anomalies and the percentiles matrix will be stored
pme_path = processed_path + 'pme_' + str(window) + 'months/'

# Name of the file of precipitation minus evaporation 
pme_file_name = analysis + '_monthly_' + region + '_' + str(start_year) + '-' + str(end_year) + '.nc'

# Name of the file of cumulative anomalies and of normalized cumulative anomalies of precipitation minus evaporation 
cumulative_anomalies_file_name = 'cumultive_anomalies_pme_' + str(window) + 'months_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
norm_cumulative_anomalies_file_name = 'normalized_cumultive_anomalies_pme_' + str(window) + 'months_' + region + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'
  
# Name of the file of the percentiles matrix of cumulative anomalies of precipitation minus evaporation
percentiles_file_name = 'percentiles_' + reanalysis + '_' + str(start_year + 1) + '-' + str(end_year) + '.nc'

print 'Calculating ' + analysis + ' for ' + reanalysis + ' using a window of ' + str(window) + ' months.'

######################################################## CARRYING OUT THE CALCULATIONS ########################################################

# Obtain the indices of the region of interest
ilons, ilats, lons, lats = get_coordinate_indices(reanalysis_path, region, reanalysis)

# Carry out the respective analysis given the type of analysis chosen above
if analysis in ['pme', 'prcp', 'et']:

    # STEP 1: Extract land-sea mask for chosen region
    #extract_land_mask(reanalysis_path, processed_path, ilats, ilons, region, reanalysis, start_year)

    # STEP 2: Calculate precipitation minus evaporation
    #calculate_p_minus_e(reanalysis_path, processed_path, pme_file_name, ilons, ilats, region, reanalysis, start_year, end_year, analysis)
    
    # STEP 3: Calculate cumulative anomalies of precipitation minus evaporation
    #calculate_cumulative_anomalies(processed_path, pme_file_name, pme_path, cumulative_anomalies_file_name, norm_cumulative_anomalies_file_name, window, start_year + 1)
    
    # STEP 4: Calculate percentiles of the cumulative anomalies of precipitation minus evaporation
    calculate_percentiles(pme_path, cumulative_anomalies_file_name, percentiles_file_name, reanalysis, start_year + 1)
    
    # CHECK: Check the files of the land mask and the percentiles
    #check_land_mask(processed_path)
    #check_percentiles(pme_path + percentiles_file_name)

elif analysis == 'moisture_flux_scalar':

    # Calculate scalar quantity of moisture fluxes and anomalies
    calculate_scalar_moisture_fluxes(reanalysis_path, processed_path, ilats, ilons, reanalysis, start_year, end_year)

else:
    
    # Aggregate the data into a single file
    calculate_environmental_anomalies(reanalysis_path, processed_path, ilats, ilons, analysis, 'normal', 'na', reanalysis, start_year, end_year)

    # Calculate anomalies
    calculate_environmental_anomalies(reanalysis_path, processed_path, ilats, ilons, analysis, 'anomalies', 'na', reanalysis, start_year, end_year)

    # Calculate cumulative anomalies
    #calculate_environmental_anomalies(reanalysis_path, processed_path, ilats, ilons, analysis, 'cumulative_anomalies', window, reanalysis, start_year, end_year)

print 'Done processing ' + reanalysis + ' data for ' + analysis + '!'
