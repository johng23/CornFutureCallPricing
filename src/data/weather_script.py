import rasterio
from rasterio.transform import xy
from pyproj import Transformer
from rasterio.transform import rowcol
import scipy
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import lru_cache
import pickle
import os

def pixelToLatitudeLongitude(src, row, col):

    transform = src.transform  # Get the affine transformation
    
    x_proj, y_proj = xy(transform, row, col)
    
    # Define projection transformer
    transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    
    # Convert projected (X, Y) to (Lon, Lat)
    lon, lat = transformer.transform(x_proj, y_proj)
    
    return (lat, lon)

def latitudeLongitudeToPixel(src, lat, lon, m):
    transform = src.transform  # Get affine transform
    crs = src.crs  # Get coordinate system
    # Define the transformer from WGS84 (EPSG:4326) to NAD83 / Conus Albers (EPSG:5070)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    
    # Convert to projected coordinates
    x_proj, y_proj = transformer.transform(lon, lat)
    
    # print(f"Projected X: {x_proj}, Projected Y: {y_proj}")
    row,col = rowcol(transform, x_proj, y_proj)
    return (int(row//m), int(col//m))


import heapq
def getLatLonClimateFromMatrixAndList(arr, k, stationDict, rmax = 9999999):
    # Initialize missing variables that weren't in the original code
    done = [[False for _ in range(len(arr[0]))] for _ in range(len(arr))]
    results_map = {}
    trueResults = []
    l = list(stationDict.keys())
    for r in range(rmax):
        indexShift = find_integer_pairs_optimized(r)
        hotIndices = []
        # print(r)
        for (a,b) in l:
            relevant = False
            for (i,j) in indexShift:
                a1 = a+i
                b1 = b+j
                #bounds check
                if a1 < 0 or a1 >= len(arr) or b1 < 0 or b1 >= len(arr[0]):
                    continue
                if done[a1][b1]:
                    continue
                relevant = True
                hotIndices.append((a1,b1))
                if (a1,b1) not in results_map:
                    results_map[(a1,b1)] = []
                heapq.heappush(results_map[(a1,b1)], (np.sqrt(i**2+j**2),stationDict[(a,b)]))
            if not relevant:
                l.remove((a,b))
                
        for (a1,b1) in hotIndices:
            # Ensure the key exists in results_map
            if (a1,b1) in results_map:
                while len(results_map[(a1,b1)]) > k:
                    heapq.heappop(results_map[(a1,b1)])
                if len(results_map[(a1,b1)]) == k:
                    done[a1][b1] = True
                    if arr[a1][b1] != 0:
                        trueResults.append((arr[a1][b1],results_map[(a1,b1)]))
                    # Fixed remove operation - should be del instead of remove
                    del results_map[(a1,b1)]

        if len(l) == 0:
            break
    for (a,b) in results_map.keys():
        if arr[a][b] != 0:
            trueResults.append((arr[a][b],results_map[(a,b)]))
        # print(trueResults)
    return trueResults
                
        
@lru_cache(maxsize=None)
def find_integer_pairs_optimized(r):
    """
    Optimized version that uses geometric properties to reduce the search space.
    We only need to search in a ring-shaped region.
    
    Args:
        r (int): The lower bound of the distance range
        
    Returns:
        list: List of tuples (x, y) satisfying r <= sqrt(x^2 + y^2) < r+1
    """
    result = []
    
    # For optimization, we can limit y based on each x value
    # If x^2 + y^2 is between r^2 and (r+1)^2, then
    # y must be between sqrt(r^2 - x^2) and sqrt((r+1)^2 - x^2)
    for x in range(-r-1, r+2):
        # Skip impossible x values
        if abs(x) > r+1:
            continue
            
        # Calculate y bounds for this x
        min_y_squared = max(0, r**2 - x**2)
        max_y_squared = (r+1)**2 - x**2
        
        # Skip if no valid y exists for this x
        if min_y_squared > max_y_squared:
            continue
            
        min_y = int(min_y_squared**0.5)
        max_y = int(max_y_squared**0.5) + 1
        
        # Check each potential y value
        for y in range(min_y, max_y + 1):
            dist_squared = x**2 + y**2
            if r**2 <= dist_squared < (r+1)**2:
                result.append((x, y))
                # We can also add the symmetric point in the other quadrants
                if y != 0:
                    result.append((x, -y))
                if x != 0:
                    result.append((-x, y))
                if x != 0 and y != 0:
                    result.append((-x, -y))
        
    # Remove duplicates that might have been added in the symmetric additions
    return list(set(result))

import numpy as np

def apply_convolution(matrix, kernel, cropValue):
    """
    Applies a kxk convolution filter with a stride of k on a matrix.
    
    Parameters:
        matrix (2D numpy array): The input matrix.
        kernel (2D numpy array): The kxk filter to apply.
        k (int): The kernel size and stride.
    
    Returns:
        2D numpy array: The downsampled convolved matrix.
    """

    # Filter out other crops
    np.equal(matrix, cropValue, out=matrix)

    # Get dimensions
    m, n = matrix.shape
    km, kn = kernel.shape

    # Compute output size
    out_m = m // km
    out_n = n // kn

    # Create output matrix
    output = np.zeros((out_m, out_n))

    # Perform convolution with stride k
    output = scipy.ndimage.convolve(matrix, kernel, mode='constant', cval=0)
    # for i in range(out_m):
    #     for j in range(out_n):
    #         patch = matrix[i*km : i*km + km, j*kn : j*kn + kn]  # Extract kxk patch
    #         mask = (patch == cropValue)
    #         masked_matrix_patch = np.where(mask, np.uint8(1),np.uint8(0))
    #         output[i, j] = np.sum(masked_matrix_patch * kernel)  # Apply convolution

    return output
    
from datetime import date, timedelta
import requests
import json
import pandas as pd
from geopy.distance import geodesic
from datetime import datetime
from datetime import timedelta

def get_climate_data(state, start_date, end_date):
    stns = []
    stndata = get_station_data(state, start_date, end_date)["meta"]
    for i in range(len(stndata)):
        stns.append(stndata[i]["sids"][0])
    
    url = "http://data.rcc-acis.org/MultiStnData"
    params = {
        # "sid": sid,  # Station ID
        "sdate": start_date,  # Start date
        "edate": end_date,  # End date
        # "date": date,
        "sids": stns,
        "elems":["maxt","mint","avgt","pcpn","snow"],
        "output": "json"
    }
    
    response = requests.post(url, json=params)
    
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Error fetching data:", response.status_code, response.text)

def get_station_data(state, start_date, end_date):
    url = "http://data.rcc-acis.org/StnMeta"
    # current_date = str(datetime.now().date())
    # start_date = str((datetime.now()-timedelta(days = 15*1)).date())
    params = {
        # "sid": sid,  # Station ID
        "sdate": start_date,  # Start date
        "edate": end_date,  # End date
        # "date": date,
        "state": state,
        "elems":["avgt"],
        "output": "json"
    }
    
    response = requests.post(url, json=params)
    
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Error fetching data:", response.status_code, response.text)

def daterange(start_date: date, end_date: date):
    
    days = int((end_date - start_date).days)
    for n in range(days):
        yield start_date + timedelta(n)

def get_station_data_by_date(start_date, end_date, state):
    start_date2 = datetime.fromisoformat(start_date)
    end_date2 = datetime.fromisoformat(end_date)
    
    lonlatlistlist = []
    for date in daterange(start_date2, end_date2):
        lonlatlist = []
        d = get_station_data(str(date)[0:10], state)
        for i in range(len(d["data"])):
            if "ll" in d.get("data")[i]["meta"].keys():
                lonlatlist.append(d.get("data")[i]["meta"]["ll"])
        lonlatlistlist.append([str(date)[0:10],lonlatlist])

    return lonlatlistlist
    
def climate_data_to_dict(climate_data):
    ll_to_data = {}
    climate_data2 = climate_data["data"]
    for i in range(len(climate_data2)):
        if "ll" in climate_data2[i]["meta"].keys():
            l1, l2 = climate_data2[i]["meta"]["ll"]
            ll_to_data[(l1, l2)] = climate_data2[i]["data"]
    return ll_to_data

# def weightedSum(KNearestLocationDataValue, ll_to_data, days):
#     retval = np.zeros(5)
#     #5 is the number of elems
#     for i in range(5):
#         valArr = []
#         distArr = []
#         for j in range(len(KNearestLocationDataValue)):
#
#             if ll_to_data[KNearestLocationDataValue[j][1]][days][i] == 'M' or ll_to_data[KNearestLocationDataValue[j][1]][days][i] == 'S' or ll_to_data[KNearestLocationDataValue[j][1]][days][i][-1] == 'A':
#                 continue
#             if ll_to_data[KNearestLocationDataValue[j][1]][days][i] == 'T':
#                 valArr.append(0)
#                 distArr.append(KNearestLocationDataValue[j][0])
#             else:
#                 valArr.append(ll_to_data[KNearestLocationDataValue[j][1]][days][i])
#                 distArr.append(KNearestLocationDataValue[j][0])
#         if len(distArr) == 0:
#             return [-99999]
#         if distArr[0] == 0:
#             retval[i] = valArr[0]
#         else:
#             totInvDist = sum(1/distArr[j] for j in range(len(distArr)))
#             retval[i] = sum((1/distArr[j])/totInvDist * float(valArr[j]) for j in range(len(valArr)))
#     return retval

def weightedSum_vec(KNearestLocationDataValue, ll_to_data, day_offsets):
    """
    KNearestLocationDataValue: list of (distance, key) for a single area
    ll_to_data: dict mapping key -> list of lists per day
    day_offsets: list of day indices to process
    Returns: array of shape (num_days, 5)
    """
    num_days = len(day_offsets)
    retval = np.zeros((num_days, 5), dtype=float)

    for i in range(5):  # 5 elements
        vals_matrix = []
        dist_matrix = []

        for dist, key in KNearestLocationDataValue:
            values = [ll_to_data[key][day][i] for day in day_offsets]

            # Mask invalid entries
            values_clean = []
            for val in values:
                if val == 'M' or val == 'S' or (isinstance(val, str) and val.endswith('A')):
                    values_clean.append(np.nan)
                elif val == 'T':
                    values_clean.append(0.0)
                else:
                    values_clean.append(float(val))
            vals_matrix.append(values_clean)
            dist_matrix.append([dist] * num_days)

        vals_matrix = np.array(vals_matrix, dtype=float)  # shape: (num_neighbors, num_days)
        dist_matrix = np.array(dist_matrix, dtype=float)

        # Compute weighted sum per day
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_dists = 1 / dist_matrix
            weights = inv_dists / np.nansum(inv_dists, axis=0)
            weighted_vals = np.nansum(vals_matrix * weights, axis=0)
            # If first distance is zero, use that value directly
            zero_mask = dist_matrix[0] == 0
            weighted_vals[zero_mask] = vals_matrix[0, zero_mask]

        # If all distances invalid, mark as -99999
        all_nan_mask = np.isnan(weighted_vals)
        weighted_vals[all_nan_mask] = -99999

        retval[:, i] = weighted_vals

    return retval
# def weightedSum_vec(KNearestLocationDataValue, ll_to_data, days):
#     retval = np.zeros(5)
#
#     for i in range(5):
#         vals = []
#         dists = []
#
#         for dist, key in KNearestLocationDataValue:
#             value = ll_to_data[key][days][i]
#
#             # Skip invalid entries
#             if value == 'M' or value == 'S' or (isinstance(value, str) and value.endswith('A')):
#                 continue
#
#             # Treat 'T' as 0
#             if value == 'T':
#                 vals.append(0)
#             else:
#                 vals.append(float(value))
#             dists.append(dist)
#
#         if len(dists) == 0:
#             return [-99999]
#
#         vals = np.array(vals, dtype=float)
#         dists = np.array(dists, dtype=float)
#
#         # If distance is zero, use the first value directly
#         if dists[0] == 0:
#             retval[i] = vals[0]
#         else:
#             inv_dists = 1 / dists
#             weights = inv_dists / inv_dists.sum()
#             retval[i] = np.dot(vals, weights)
#
#     return retval
# def heavy_task(x):
#     print(f"Running {x}", flush=True)
#     return x * x


#src is something like src = "../data/2023_30m_cdls.tif" aka HUGE array
#m is the downscale factor for the src array
#cropValue is the "categorization code" here https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php#what.7
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from datetime import datetime



from rasterio.windows import Window


class bigArrayParser:
    def __init__(self, src_path, m, cropValue, block_size=4096, n_workers=None):
        """
        Multiprocessing downsampling of raster by counting cropValue pixels in m×m tiles.
        """
        self.src_path = src_path
        self.src = rasterio.open(src_path)
        self.m = m
        self.cropValue = cropValue
        self.block_size = block_size
        self.n_workers = n_workers or cpu_count()

        # Open once to get dimensions
        with rasterio.open(src_path) as src_file:
            self.H, self.W = src_file.height, src_file.width

        self.out_h, self.out_w = self.H // m, self.W // m
        self.arr = np.zeros((self.out_h, self.out_w), dtype=np.uint32)

        # Perform the downsampling
        self._downsample()

    def _downsample(self):
        """Orchestrates multiprocessing over raster blocks and collects results."""
        H, W = self.H, self.W
        m = self.m
        block_size = self.block_size

        # Prepare block arguments
        h_blocks = (H + block_size - 1) // block_size
        w_blocks = (W + block_size - 1) // block_size
        block_args = []
        for hi in range(h_blocks):
            for wi in range(w_blocks):
                row_off = hi * block_size
                col_off = wi * block_size
                height = min(block_size, H - row_off)
                width = min(block_size, W - col_off)
                block_args.append((self.src_path, self.cropValue, m,
                                   row_off, col_off, height, width))

        # Process blocks in parallel
        with Pool(self.n_workers) as pool:
            for row_idx, col_idx, tile_sums in tqdm(pool.imap_unordered(self._process_block, block_args),
                                                    total=len(block_args), desc="Downsampling"):
                if tile_sums.size == 0:
                    continue
                h, w = tile_sums.shape
                self.arr[row_idx:row_idx+h, col_idx:col_idx+w] = tile_sums

    @staticmethod
    def _process_block(args):
        """Static method to process a single block in a worker process."""
        src_path, cropValue, m, row_off, col_off, height, width = args
        with rasterio.open(src_path) as src:
            block = src.read(1, window=Window(col_off, row_off, width, height))
        mask = (block == cropValue)

        h_tiles = height // m
        w_tiles = width // m
        if h_tiles == 0 or w_tiles == 0:
            return (row_off//m, col_off//m, np.zeros((0,0), dtype=np.uint32))

        mask = mask[:h_tiles*m, :w_tiles*m]
        tiles = mask.reshape(h_tiles, m, w_tiles, m)
        tile_sums = np.sum(tiles, axis=(1,3))
        return (row_off//m, col_off//m, tile_sums)

    def close(self):
        self.src.close()
        
    #lonlatlistlist is a list of lists of longitude and latitudes (of weather stations), probably one list for each date
    #k is the k nearest neighbors
    #r is some notion of distance that you can basically think that scales linearly i.e. a unit like miles or kilometers. It is the search radius.
    #for each nonzero amount of 30 by 30 meters of land of cropValue within r pixel radius of some station in a list in lonlatlistlist, 
    # it returns how many 30 by 30 meters of crop production there is, along with k pairs (distance to station, lonlat of station) 
    # associated to the k nearest stations. The algorithm downsamples by a factor of m from src.
    # def getKNearestLocations(self, lonlatlistlist, k, r):
    #     retval = []
    #     for i in range(len(lonlatlistlist)):
    #         dat = self.getKNearestLocationsHelper(lonlatlistlist[i], k, r, m)
    #         retval.append(dat)
    #     return retval

    # weatherStations is a list of tuples (lat, lon)
    def getKNearestLocationsHelper(self, weatherStations, k, rmax, m):
        # arr = src.read(1)
        stationDict = {}
        for (lon, lat) in weatherStations:
            # print(lat,lon)
            stationDict[latitudeLongitudeToPixel(self.src, lat, lon, m)] = (lon, lat)
        results = getLatLonClimateFromMatrixAndList(self.arr, k, stationDict, rmax)
        return results
    
    # Assuming these are already defined: 
    # daterange, start_date2, end_date2, KNearestLocationData, ll_to_data, weightedSum
    def get_area_with_climate(self, k, r, m, state, start_date, end_date):
        climate_cache = "cached_climate.pkl"
        station_cache = "cached_station.pkl"
        if os.path.exists(climate_cache) and os.path.exists(station_cache):
            with open(climate_cache, "rb") as f:
                ll_to_data = pickle.load(f)
            with open(station_cache, "rb") as f:
                KNearestLocationData = pickle.load(f)
        else:
            # Load and convert raw climate data
            raw_data = get_climate_data(state, start_date, end_date)
            ll_to_data_raw = climate_data_to_dict(raw_data)

            # Clean the data: convert to floats, handle 'T', 'M', 'S', 'A'
            ll_to_data = {}
            for key, records in ll_to_data_raw.items():
                ll_to_data[key] = []
                for day_record in records:
                    clean_record = []
                    for val in day_record:
                        if val == 'M' or val == 'S' or (isinstance(val, str) and val.endswith('A')):
                            clean_record.append(np.nan)  # mark as missing
                        elif val == 'T':
                            clean_record.append(0.0)  # treat trace as 0
                        else:
                            clean_record.append(float(val))
                    ll_to_data[key].append(clean_record)
            with open(climate_cache, "wb") as f:
                pickle.dump(ll_to_data, f)
            # Get nearest stations
            KNearestLocationData = self.getKNearestLocationsHelper(list(ll_to_data.keys()), k, r, m)
            with open(station_cache, "wb") as f:
                pickle.dump(KNearestLocationData, f)

        start_date2 = datetime.fromisoformat(start_date)
        end_date2 = datetime.fromisoformat(end_date)
        day_offsets = [(date - start_date2).days for date in daterange(start_date2, end_date2)]

        # Prepare arguments for starmap
        args_list = [(area, KNearestLocationDataValue, {idx: ll_to_data[idx] for dist, idx in KNearestLocationDataValue}, day_offsets)
                     for a, (area, KNearestLocationDataValue) in enumerate(KNearestLocationData) if a]

        climate_area_data_over_time = {date: {} for date in daterange(start_date2, end_date2)}
        # 3. Process each area in parallel
        with Pool(processes=cpu_count()) as pool:
            for area_results, area in tqdm(pool.imap_unordered(self._process_area, args_list, chunksize=8), total=len(args_list), desc="Obtaining weather distribution over area"):
                # area_results: dict[day_offset] -> climate_key
                for day_offset, climate_key in area_results.items():
                    date = start_date2 + timedelta(days=day_offset)
                    if climate_key not in climate_area_data_over_time[date]:
                        climate_area_data_over_time[date][climate_key] = area
                    else:
                        climate_area_data_over_time[date][climate_key] += area

        return climate_area_data_over_time

    @staticmethod
    def _process_area(args):
        """
        Vectorized computation of climate for a single area across all days.
        ll_to_data_array: shape (num_stations, num_days, 5)
        KNearestLocationDataValue: list of (distance, station_idx)
        Returns: dict[day_offset] -> climate_key, includes 'area'
        """
        area, KNearestLocationDataValue, ll_to_data, day_offsets = args

        # raise Exception("ENTERED PROCESS AREA")
        area_results = {}
        if len(KNearestLocationDataValue) == 0:
            return area_results, area  # no stations

        # Extract distances and station indices
        distances = np.array([dist for dist, idx in KNearestLocationDataValue], dtype=float)
        station_indices = [idx for dist, idx in KNearestLocationDataValue]

        # Handle zero distance (just pick first station for that day)
        zero_mask = distances == 0
        if zero_mask.any():
            inv_distances = np.zeros_like(distances)
            inv_distances[zero_mask] = 1.0  # weight only the zero-distance station
        else:
            inv_distances = 1 / distances

        # Normalize weights
        weights = inv_distances / np.nansum(inv_distances)

        # Collect station data (all days, 5 variables) into a NumPy array
        # Shape: (num_stations, num_days, 5)
        station_data_list = []
        for key in station_indices:  # loop over nearest stations
            station_data_list.append(ll_to_data[key])  # each is shape (num_days, 5)

        station_data = np.array(station_data_list, dtype=float)  # shape: (num_stations, num_days, 5)

        # Compute weighted sum over stations for all days at once
        # weighted_data: shape (num_days, 5)
        weighted_data = np.nansum(station_data * weights[:, np.newaxis, np.newaxis], axis=0)

        # Fill area_results per day_offset
        for i, day_offset in enumerate(day_offsets):
            vals = weighted_data[i, :]  # [maxt, mint, avgt, prcp, snow]
            if np.isnan(vals).all():
                continue  # skip if no valid data

            # Round first three (temperature) to nearest integer
            temps = np.round(vals[:3]).astype(np.int16)

            # Round last two (precipitation, snow) to nearest 0.2
            prcp_snow = np.round(vals[3:] / 0.2) * 0.2
            prcp_snow = prcp_snow.astype(np.float16)

            climate_key = tuple(np.concatenate([temps, prcp_snow]))
            area_results[day_offset] = climate_key

        return area_results, area
    # @staticmethod
    # def process_area(area_idx, area, KNearestLocationDataValue, ll_to_data, day_offsets):
    #     """
    #     Compute weighted sums for a single area across all days.
    #     Returns: dict mapping day_offset -> climate_key
    #     """
    #     area_results = {}
    #
    #     for i, day_offset in enumerate(day_offsets):
    #         retval = weightedSum_vec(KNearestLocationDataValue, ll_to_data, day_offset)
    #         if len(retval) == 1:  # Skip if no valid data
    #             continue
    #         maxt, mint, avgt, prcp, snow = retval
    #         climate_key = (round(maxt, 1), round(mint, 1), round(avgt, 1), round(prcp, 1), round(snow, 1))
    #         area_results[day_offset] = climate_key
    #
    #     return area_results, area
        # for date in daterange(start_date2, end_date2):
        #     print(date)
        #     climate = {}
        #     for i in range(len(KNearestLocationData)):
        #         area = KNearestLocationData[i][0]
        #         retval = weightedSum(KNearestLocationData[i][1], ll_to_data, (date-start_date2).days)
        #         if len(retval) == 1:
        #             continue
        #         maxt, mint, avgt, prcp, snow = retval
        #         if (round(maxt, 1), round(mint, 1), round(avgt, 1), round(prcp, 1), round(snow, 1)) not in climate.keys():
        #             climate[(round(maxt, 1), round(mint, 1), round(avgt, 1), round(prcp, 1), round(snow, 1))] = area
        #         else:
        #             climate[(round(maxt, 1), round(mint, 1), round(avgt, 1), round(prcp, 1), round(snow, 1))] += area
        #     climate_area_data_over_time[date] = climate
        return climate_area_data_over_time

def get_projections(cadot):
    proj = {}
    for date in cadot:
        dictArr = [{} for i in range(5)]
        for k in cadot[date].keys():
            area = cadot[date][k]
            for i in range(5):
                if k[i] not in dictArr[i].keys():
                    dictArr[i][k[i]] = 0
                dictArr[i][k[i]] += area
        proj[date] = dictArr
    return proj

def process_date_for_projection(args):
    date, cadot = args
    dictArr = [{} for _ in range(5)]
    for k, area in cadot.items():
        for i in range(5):
            dictArr[i][k[i]] = dictArr[i].get(k[i], 0) + area
    return date, dictArr

def get_projections_multithreaded(cadot):
    proj = {}
    dates = list(cadot.keys())
    args_list = [(date, cadot[date]) for date in dates]

    with Pool(processes=cpu_count()) as pool:
        # imap_unordered yields results as soon as a worker finishes
        for date, dictArr in tqdm(pool.imap_unordered(process_date_for_projection, args_list, chunksize = 8), total=len(dates), desc="Processing projections"):
            proj[date] = dictArr

    return proj

def get_weather_features(proj):
    """
    Given a dictionary `proj` where each key is a date and each value is a list
    of five dictionaries corresponding to:
      0: average_temperature_distribution,
      1: maximum_temperature_distribution,
      2: minimum_temperature_distribution,
      3: precipitation_distribution,
      4: snow_distribution,
    this function computes weighted statistics for each distribution and
    returns a nested dictionary of features.
    """
    features = {}
    for date in proj:
        average_temperature_distribution = proj[date][0]
        maximum_temperature_distribution = proj[date][1]
        minimum_temperature_distribution = proj[date][2]
        precipitation_distribution = proj[date][3]
        snow_distribution = proj[date][4]
        
        # Compute stats for each distribution using compute_weighted_stats()
        avg_stats = compute_weighted_stats(average_temperature_distribution)
        max_stats = compute_weighted_stats(maximum_temperature_distribution)
        min_stats = compute_weighted_stats(minimum_temperature_distribution)
        prec_stats = compute_weighted_stats(precipitation_distribution)
        snow_stats = compute_weighted_stats(snow_distribution)
        
        # Create a nested dictionary for the current date with descriptive keys
        features[date] = {
            "average_temperature_distribution_weighted_mean": avg_stats["Weighted Mean"],
            "average_temperature_distribution_weighted_variance": avg_stats["Weighted Variance"],
            "average_temperature_distribution_weighted_std": avg_stats["Weighted Standard Deviation"],
            "average_temperature_distribution_weighted_skewness": avg_stats["Weighted Skewness"],
            "average_temperature_distribution_weighted_kurtosis": avg_stats["Weighted Kurtosis"],
            "average_temperature_distribution_weighted_median": avg_stats["Weighted Median"],
            "average_temperature_distribution_min_value": avg_stats["Min Value"],
            "average_temperature_distribution_max_value": avg_stats["Max Value"],
            
            "maximum_temperature_distribution_weighted_mean": max_stats["Weighted Mean"],
            "maximum_temperature_distribution_weighted_variance": max_stats["Weighted Variance"],
            "maximum_temperature_distribution_weighted_std": max_stats["Weighted Standard Deviation"],
            "maximum_temperature_distribution_weighted_skewness": max_stats["Weighted Skewness"],
            "maximum_temperature_distribution_weighted_kurtosis": max_stats["Weighted Kurtosis"],
            "maximum_temperature_distribution_weighted_median": max_stats["Weighted Median"],
            "maximum_temperature_distribution_min_value": max_stats["Min Value"],
            "maximum_temperature_distribution_max_value": max_stats["Max Value"],
            
            "minimum_temperature_distribution_weighted_mean": min_stats["Weighted Mean"],
            "minimum_temperature_distribution_weighted_variance": min_stats["Weighted Variance"],
            "minimum_temperature_distribution_weighted_std": min_stats["Weighted Standard Deviation"],
            "minimum_temperature_distribution_weighted_skewness": min_stats["Weighted Skewness"],
            "minimum_temperature_distribution_weighted_kurtosis": min_stats["Weighted Kurtosis"],
            "minimum_temperature_distribution_weighted_median": min_stats["Weighted Median"],
            "minimum_temperature_distribution_min_value": min_stats["Min Value"],
            "minimum_temperature_distribution_max_value": min_stats["Max Value"],
            
            "precipitation_distribution_weighted_mean": prec_stats["Weighted Mean"],
            "precipitation_distribution_weighted_variance": prec_stats["Weighted Variance"],
            "precipitation_distribution_weighted_std": prec_stats["Weighted Standard Deviation"],
            "precipitation_distribution_weighted_skewness": prec_stats["Weighted Skewness"],
            "precipitation_distribution_weighted_kurtosis": prec_stats["Weighted Kurtosis"],
            "precipitation_distribution_weighted_median": prec_stats["Weighted Median"],
            "precipitation_distribution_min_value": prec_stats["Min Value"],
            "precipitation_distribution_max_value": prec_stats["Max Value"],
            
            "snow_distribution_weighted_mean": snow_stats["Weighted Mean"],
            "snow_distribution_weighted_variance": snow_stats["Weighted Variance"],
            "snow_distribution_weighted_std": snow_stats["Weighted Standard Deviation"],
            "snow_distribution_weighted_skewness": snow_stats["Weighted Skewness"],
            "snow_distribution_weighted_kurtosis": snow_stats["Weighted Kurtosis"],
            "snow_distribution_weighted_median": snow_stats["Weighted Median"],
            "snow_distribution_min_value": snow_stats["Min Value"],
            "snow_distribution_max_value": snow_stats["Max Value"],
        }
    return features
        
import numpy as np
import statsmodels.api as sm

def compute_weighted_stats(data_dict):
    # Convert dictionary keys and values to numpy arrays
    values = np.array(list(data_dict.keys()))
    weights = np.array(list(data_dict.values()))
    
    # Use statsmodels to compute weighted statistics (mean, variance, std)
    weighted_stats = sm.stats.DescrStatsW(values, weights=weights)
    weighted_mean = weighted_stats.mean
    weighted_variance = weighted_stats.var
    weighted_std_dev = weighted_stats.std

    # Compute weighted skewness manually:
    # Formula: sum(weights * (x - mean)^3) / (sum(weights * (x - mean)^2)^(3/2))
    mean_centered = values - weighted_mean
    weighted_skewness = np.sum(weights * mean_centered**3) / (np.sum(weights * mean_centered**2)**(3/2))

    # Compute weighted kurtosis manually:
    # Formula: sum(weights * (x - mean)^4) / (sum(weights * (x - mean)^2)^2) - 3
    weighted_kurtosis = np.sum(weights * mean_centered**4) / (np.sum(weights * mean_centered**2)**2) - 3

    # Compute weighted median manually:
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)
    total_weight = np.sum(sorted_weights)
    median_index = np.searchsorted(cumulative_weights, total_weight / 2)
    weighted_median = sorted_values[median_index]

    # Compute min and max values
    min_val = np.min(values)
    max_val = np.max(values)

    # Return all computed statistics as a dictionary
    return {
        "Weighted Mean": weighted_mean,
        "Weighted Variance": weighted_variance,
        "Weighted Standard Deviation": weighted_std_dev,
        "Weighted Skewness": weighted_skewness,
        "Weighted Kurtosis": weighted_kurtosis,
        "Weighted Median": weighted_median,
        "Min Value": min_val,
        "Max Value": max_val
    }

from concurrent.futures import ThreadPoolExecutor

def process_weather_for_date(date, distributions):
    average_temperature_distribution = distributions[0]
    maximum_temperature_distribution = distributions[1]
    minimum_temperature_distribution = distributions[2]
    precipitation_distribution = distributions[3]
    snow_distribution = distributions[4]

    avg_stats = compute_weighted_stats(average_temperature_distribution)
    max_stats = compute_weighted_stats(maximum_temperature_distribution)
    min_stats = compute_weighted_stats(minimum_temperature_distribution)
    prec_stats = compute_weighted_stats(precipitation_distribution)
    snow_stats = compute_weighted_stats(snow_distribution)

    return date, {
        "average_temperature_distribution_weighted_mean": avg_stats["Weighted Mean"],
        "average_temperature_distribution_weighted_variance": avg_stats["Weighted Variance"],
        "average_temperature_distribution_weighted_std": avg_stats["Weighted Standard Deviation"],
        "average_temperature_distribution_weighted_skewness": avg_stats["Weighted Skewness"],
        "average_temperature_distribution_weighted_kurtosis": avg_stats["Weighted Kurtosis"],
        "average_temperature_distribution_weighted_median": avg_stats["Weighted Median"],
        "average_temperature_distribution_min_value": avg_stats["Min Value"],
        "average_temperature_distribution_max_value": avg_stats["Max Value"],

        "maximum_temperature_distribution_weighted_mean": max_stats["Weighted Mean"],
        "maximum_temperature_distribution_weighted_variance": max_stats["Weighted Variance"],
        "maximum_temperature_distribution_weighted_std": max_stats["Weighted Standard Deviation"],
        "maximum_temperature_distribution_weighted_skewness": max_stats["Weighted Skewness"],
        "maximum_temperature_distribution_weighted_kurtosis": max_stats["Weighted Kurtosis"],
        "maximum_temperature_distribution_weighted_median": max_stats["Weighted Median"],
        "maximum_temperature_distribution_min_value": max_stats["Min Value"],
        "maximum_temperature_distribution_max_value": max_stats["Max Value"],

        "minimum_temperature_distribution_weighted_mean": min_stats["Weighted Mean"],
        "minimum_temperature_distribution_weighted_variance": min_stats["Weighted Variance"],
        "minimum_temperature_distribution_weighted_std": min_stats["Weighted Standard Deviation"],
        "minimum_temperature_distribution_weighted_skewness": min_stats["Weighted Skewness"],
        "minimum_temperature_distribution_weighted_kurtosis": min_stats["Weighted Kurtosis"],
        "minimum_temperature_distribution_weighted_median": min_stats["Weighted Median"],
        "minimum_temperature_distribution_min_value": min_stats["Min Value"],
        "minimum_temperature_distribution_max_value": min_stats["Max Value"],

        "precipitation_distribution_weighted_mean": prec_stats["Weighted Mean"],
        "precipitation_distribution_weighted_variance": prec_stats["Weighted Variance"],
        "precipitation_distribution_weighted_std": prec_stats["Weighted Standard Deviation"],
        "precipitation_distribution_weighted_median": prec_stats["Weighted Median"],
        "precipitation_distribution_min_value": prec_stats["Min Value"],
        "precipitation_distribution_max_value": prec_stats["Max Value"],

        "snow_distribution_weighted_mean": snow_stats["Weighted Mean"],
        "snow_distribution_weighted_variance": snow_stats["Weighted Variance"],
        "snow_distribution_weighted_std": snow_stats["Weighted Standard Deviation"],
        "snow_distribution_weighted_median": snow_stats["Weighted Median"],
        "snow_distribution_min_value": snow_stats["Min Value"],
        "snow_distribution_max_value": snow_stats["Max Value"],
    }

def get_weather_features_multithreaded(proj):
    features = {}
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_weather_for_date, date, proj[date]) for date in proj]
        for future in futures:
            date, data = future.result()
            features[date] = data
    return features

def safe_filename(date_str):
    # replace ':' and spaces with underscores
    return date_str.replace(":", "-").replace(" ", "_")

def save_cadot_by_date(cadot, cache_dir="cached_cadot"):
    os.makedirs(cache_dir, exist_ok=True)
    for date, dictArr in cadot.items():
        filename = safe_filename(str(date)) + ".pkl"
        path = os.path.join(cache_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(dictArr, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_cadot_by_date(cache_dir="cached_cadot"):
    cadot = {}
    for filename in os.listdir(cache_dir):
        if not filename.endswith(".pkl"):
            continue
        path = os.path.join(cache_dir, filename)
        # recover date by reversing safe_filename
        date_str = filename.replace(".pkl", "").replace("_", " ").replace("-", ":")
        with open(path, "rb") as f:
            cadot[date_str] = pickle.load(f)
    return cadot