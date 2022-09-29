import geopandas as gpd
from rasterstats import zonal_stats as zs
import rasterio as rio
import os
from rasterio import features
import numpy as np
import shapely
from shapely.geometry import shape
import glob
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
import math
from tabulate import tabulate
from math import trunc
from shapely import geometry
from geopandas import GeoDataFrame as gdf
from shapely.geometry import Point, Polygon, LineString

# Define paths
data_path = os.getenv('DATA_PATH', '/data')
inputs_path = os.path.join(data_path, 'inputs')
outputs_path = os.path.join(data_path, 'outputs')
if not os.path.exists(outputs_path):
    os.mkdir(outputs_path)
mastermap = glob.glob(os.path.join(inputs_path, 'mastermap', '*.gpkg'))[0]
udm_buildings = os.path.join(inputs_path, 'buildings', 'urban_fabric.gpkg')
output_file = os.path.join(outputs_path, 'features.gpkg')
uprn_lookup = glob.glob(os.path.join(inputs_path, 'uprn', '*.csv'))
dd_curves = os.path.join(inputs_path, 'dd-curves')

threshold = float(os.getenv('THRESHOLD'))

# Set buffer around buildings
buffer = 5

with rio.open(os.path.join(inputs_path, 'run/max_depth.tif')) as max_depth,\
        rio.open(os.path.join(inputs_path, 'run/max_vd_product.tif')) as max_vd_product:
    # Read MasterMap data
    buildings = gpd.read_file(mastermap, bbox=max_depth.bounds)

    if os.path.exists(udm_buildings):
        udm_buildings = gpd.read_file(udm_buildings, bbox=max_depth.bounds)
        udm_buildings['toid'] = 'udm' + udm_buildings.index.astype(str)
        udm_buildings['building_use'] = 'residential'
        buildings = buildings.append(udm_buildings)


    # Read flood depths and vd_product
    depth = max_depth.read(1)
    vd_product = max_vd_product.read(1)

    # Find flooded areas
    flooded_areas = gpd.GeoDataFrame(
        geometry=[shape(s[0]) for s in features.shapes(
            np.ones(depth.shape, dtype=rio.uint8), mask=np.logical_and(depth >= threshold, max_depth.read_masks(1)),
            transform=max_depth.transform)], crs=max_depth.crs)

    # Store original areas for damage calculation
    buildings['original_area'] = buildings.area


    # Buffer buildings
    buildings['geometry'] = buildings.buffer(buffer)

    # Extract maximum depth and vd_product for each building
    buildings['depth'] = [row['max'] for row in
                          zs(buildings, depth, affine=max_depth.transform, stats=['max'],
                             all_touched=True, nodata=max_depth.nodata)]

    # Filter buildings
    buildings = buildings[buildings['depth'] > threshold]

    # Calculate depth above floor level
    buildings['depth'] = buildings.depth - threshold
    

    if len(buildings) == 0:
        with open(os.path.join(outputs_path, 'buildings.csv'), 'w') as f:
            f.write('')
        exit(0)

    buildings['vd_product'] = [row['max'] for row in
                               zs(buildings, vd_product, affine=max_vd_product.transform, stats=['max'],
                                  all_touched=True, nodata=max_vd_product.nodata)]                       
                                   
    #Calculate damage
    residential = pd.read_csv(os.path.join(dd_curves, 'residential_original.csv'))
    nonresidential = pd.read_csv(os.path.join(dd_curves, 'nonresidential_original.csv'))

    buildings['damage'] = (np.interp(
        buildings.depth, residential.depth, residential.damage) * buildings.original_area).round(0)
    buildings['damage'] = buildings['damage'].where(
        buildings.building_use != 'residential', (np.interp(
            buildings.depth, nonresidential.depth, nonresidential.damage
        ) * buildings.original_area).round(0)).astype(int)

    # Create a new data frame called centres which is a copy of buildings
    building_centroid=buildings.filter(['building_use','geometry','damage','depth'])

    # Get the flooded perimeter length for each building
    flooded_perimeter = gpd.overlay(gpd.GeoDataFrame({'toid': buildings.toid}, geometry=buildings.geometry.boundary,
                                                     crs=buildings.crs), flooded_areas)
    flooded_perimeter['flooded_perimeter'] = flooded_perimeter.geometry.length.round(2)

    buildings['perimeter'] = buildings.geometry.length

    buildings = buildings.merge(flooded_perimeter, on='toid', how='left')
    buildings['flooded_perimeter'] = buildings.flooded_perimeter.divide(
        buildings.perimeter).fillna(0).multiply(100).round(0).astype(int)

    # Lookup UPRN if available
    if len(uprn_lookup) > 0:
        uprn = pd.read_csv(uprn_lookup[0], usecols=['IDENTIFIER_1', 'IDENTIFIER_2'],
                           dtype={'IDENTIFIER_1': str}).rename(columns={'IDENTIFIER_1': 'uprn',
                                                                        'IDENTIFIER_2': 'toid'})
        buildings = buildings.merge(uprn, how='left')
    # Save to CSV
    buildings[['toid', *['uprn' for _ in uprn_lookup[:1]], 'depth', 'damage', 'vd_product', 'flooded_perimeter']].to_csv(
        os.path.join(outputs_path, 'buildings.csv'), index=False,  float_format='%g')

# Use the limits of the tif containing the maximum depths
bbox=max_depth.bounds
# Identify the 1km OS grid cells contained within the chosen area
x_min=math.floor(bbox[0]/1000)*1000
y_min=math.floor(bbox[1]/1000)*1000
x_max=math.ceil(bbox[2]/1000)*1000
y_max=math.ceil(bbox[3]/1000)*1000
length = 1000
width = 1000
# Define the x and y coordinates of the corners of each grid cell
cols = list(np.arange(x_min,x_max+1000,1000))
rows=list(np.arange(y_min,y_max+1000,1000))

# Create a list of polygons related to each grid cell
polygons=[]
for x in cols[:-1]:
    for y in rows[:-1]:
        polygons.append(Polygon([(x,y),(x+width,y),(x+width,y+length),(x,y+length)]))

# Create a geo dataframe, with the newly created polygons as the geometry field
grid=gpd.GeoDataFrame({geometry:polygons})
grid.rename(columns={list(grid)[0]:'geometry'},inplace=True)

# Add columns to the dataframe of the information we would like to know for each grid cell
grid['Residential_Count']=[0 for n in range(len(grid))]
grid['Non_Residential_Count']=[0 for n in range(len(grid))]
grid['Mixed_Count']=[0 for n in range(len(grid))]
grid['Unclassified_Count']=[0 for n in range(len(grid))]
grid['Total_Cost']=[0 for n in range(len(grid))]
dataframe=pd.DataFrame(grid)

# Apply the centroid function to the geometry column to determin the centre of each polygon
building_centroid.geometry=building_centroid['geometry'].centroid
# Ensure buildings and the new data frame have the same coordinate system
#building_centroid.crs=buildings.crs
# Redefine the index layer with sequential numbers
building_centroid=building_centroid.assign(Index=range(len(building_centroid))).set_index('Index')

# For each grid cell, determine how many buildings fall within the cell
for i in range(0,len(grid)-1):
    for j in range (0, len(building_centroid)-1):
        if building_centroid.geometry[j].within(grid.geometry[i]):
            # Establish total cost of damage per cell
            grid.Total_Cost[i] +=building_centroid.damage[j]
            # Establish the number of building types within the cell
            if building_centroid.building_use[j]=='residential':
                grid.Residential_Count[i] +=1
            if building_centroid.building_use[j]=='non-residential':
                grid.Non_Residential_Count[i] +=1
            if building_centroid.building_use[j]=='mixed':
                grid.Mixed_Count[i] +=1
            if building_centroid.building_use[j]=='unclassified':
                grid.Unclassified_Count[i] +=1

# Now we need to find the average max depth per grid cell
depth=[]
depth=pd.DataFrame(depth)
depth['grid_num']=[0 for n in range(len(building_centroid))]
depth['Water_depth']=[0 for n in range(len(building_centroid))]
depth['Water_depth']=depth['Water_depth'].astype(float)

# For each building determine which cell the building falls within and the max depth for that building
for a in range(0,len(building_centroid)-1):
    for b in range (0, len(grid)-1):
        if building_centroid.geometry[a].within(grid.geometry[b]):
            depth.grid_num[a]=b
            depth.Water_depth[a]=building_centroid.depth[a]

# There may be some empty cells and so we need to identify the list of cells with no buildings and assign a zero
# value for the depth
mylist = ['Total_Cost']
selection = grid.loc[grid['Total_Cost']==0]
depth1=pd.DataFrame()
depth1['grid_num']=selection.index
depth1['Water_depth']=0

# Combine the building/grid cell/ depth data with the list of empty cells
combined=depth.append(depth1, ignore_index=True)
# Find the average max depth per cell
averages=combined.groupby(['grid_num'])['Water_depth'].mean()
averages=pd.DataFrame(averages)
# Merge the average depth data with the building type and total cost
All_Information=pd.merge(grid, averages, left_index=True, right_index=True)


All_Information.to_csv(
        os.path.join(outputs_path, '1km_Grid_Cell_Information.csv'), index=False,  float_format='%g')
