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
buildings_path = os.path.join(inputs_path,'buildings')
uprn_lookup = glob.glob(os.path.join(inputs_path, 'uprn', '*.csv'))
dd_curves = os.path.join(inputs_path, 'dd-curves')
outputs_path = os.path.join(data_path, 'outputs')
if not os.path.exists(outputs_path):
    os.mkdir(outputs_path)

threshold = float(os.getenv('THRESHOLD'))
print('threshold:',threshold)

# Set buffer around buildings
buffer = 5

# Read in the buildings data from the shapefile
all_buildings = gpd.read_file(os.path.join(buildings_path,'all_buildings.shp'))
print('Buildings shape file read in correctly')

#Need to change the fid column from real to integer (renamed and replaced)
all_buildings.rename(columns={"fid":"Check"}, inplace=True)
all_buildings['fid'] = np.arange(all_buildings.shape[0])
    
#Output a gpkg file with all of the buildings to a seperate folder
all_buildings.to_file(os.path.join(buildings_path,'all_buildings.gpkg'),driver='GPKG')
print('Buildings gpkg created')

with rio.open(os.path.join(inputs_path, 'run/max_depth.tif')) as max_depth:
    print('max_depth_opened')
        #rio.open(os.path.join(inputs_path, 'run/max_vd_product.tif')) as max_vd_product:
    # Read MasterMap data
    
    buildings = os.path.join(buildings_path, 'all_buildings.gpkg') 
    buildings = gpd.read_file(buildings, bbox=max_depth.bounds)
    buildings['toid_new'] = buildings['toid_numbe'].astype(str)+buildings['toid'].astype(str)
    buildings.pop('toid')
    buildings.pop('toid_numbe')
    buildings.toid_new = buildings.toid_new.str.strip('nan')
    buildings.toid_new = buildings.toid_new.str.strip('None')   
    buildings.rename(columns = {'toid_new':'toid'}, inplace = True)
    buildings['toid'] = 'osgb' + buildings['toid'].astype(str)
    print('buildings data set amended and columns renamed')

    # Read flood depths and vd_product
    depth = max_depth.read(1)

    # Find flooded areas
    flooded_areas = gpd.GeoDataFrame(
        geometry=[shape(s[0]) for s in features.shapes(
            np.ones(depth.shape, dtype=rio.uint8), mask=np.logical_and(depth >= threshold, max_depth.read_masks(1)),
            transform=max_depth.transform)], crs=max_depth.crs)

    # Store original areas for damage calculation
    buildings['original_area'] = buildings.area
    print('buildings_area_calculated')

    # Buffer buildings
    buildings['geometry'] = buildings.buffer(buffer)
    print('buildings_buffer_calculated')

    # Extract maximum depth and vd_product for each building
    buildings['depth'] = [row['max'] for row in
                          zs(buildings, depth, affine=max_depth.transform, stats=['max'],
                             all_touched=True, nodata=max_depth.nodata)]

    # Filter buildings
    buildings = buildings[buildings['depth'] > threshold]
    print('buildings_filtered')

    # Calculate depth above floor level
    buildings['depth'] = buildings.depth - threshold
    print('depths above threshold calculated')

    if len(buildings) == 0:
        with open(os.path.join(outputs_path, 'buildings.csv'), 'w') as f:
            f.write('')
        exit(0)
                 
                                   
    #Calculate damage using Pauls dd curves
    residential = pd.read_csv(os.path.join(dd_curves, 'residential.csv'))
    nonresidential = pd.read_csv(os.path.join(dd_curves, 'nonresidential.csv'))
    print('depth-damage curves read in')

    buildings['damage'] = (np.interp(
        buildings.depth, residential.depth, residential.damage) * buildings.original_area).round(0)
    buildings['damage'] = buildings['damage'].where(
        buildings.building_use != 'residential', (np.interp(
            buildings.depth, nonresidential.depth, nonresidential.damage
        ) * buildings.original_area).round(0)).astype(int)

    # Create a new data frame called centres which is a copy of buildings
    building_centroid=buildings.filter(['building_u','geometry','damage','depth'])
    building_centroid.crs=buildings.crs
    print('New data frame created')

    # Save to CSV
    building_centroid.to_csv(
        os.path.join(outputs_path,'building_centroids.csv'), index=False,  float_format='%g') 
    print('building centroids saved to csv')

    # Get the flooded perimeter length for each building
    flooded_perimeter = gpd.overlay(gpd.GeoDataFrame({'toid': buildings.toid}, geometry=buildings.geometry.boundary,
                                                     crs=buildings.crs), flooded_areas)
    flooded_perimeter['flooded_perimeter'] = flooded_perimeter.geometry.length.round(2)
    print('flooded perimeter calculated')

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
    print('urpn determined')

    # Save to CSV
    buildings[['toid', *['uprn' for _ in uprn_lookup[:1]], 'depth', 'damage', 'flooded_perimeter','building_u']].to_csv(
        os.path.join(outputs_path, 'buildings_1000m.csv'), index=False,  float_format='%g')
    print('building data saved to csv')

# Use the limits of the tif containing the maximum depths
bbox=max_depth.bounds
# Identify the 1km OS grid cells contained within the chosen area
x_min=math.floor(bbox[0]/1000)*1000
y_min=math.floor(bbox[1]/1000)*1000
x_max=math.ceil(bbox[2]/1000)*1000
y_max=math.ceil(bbox[3]/1000)*1000
length = 1000
width = 1000
print('Bounding boxes calculated')


# Define the x and y coordinates of the corners of each grid cell
cols = list(np.arange(x_min,x_max+1000,1000))
rows=list(np.arange(y_min,y_max+1000,1000))

# Create a list of polygons related to each grid cell
polygons=[]
for x in cols[:-1]:
    for y in rows[:-1]:
        polygons.append(Polygon([(x,y),(x+width,y),(x+width,y+length),(x,y+length)]))
print('Grid cell polygons created')

# Create a geo dataframe, with the newly created polygons as the geometry field
grid=gpd.GeoDataFrame({geometry:polygons})
grid.rename(columns={list(grid)[0]:'geometry'},inplace=True)
print('new geodataframe')

# Add columns to the dataframe of the information we would like to know for each grid cell
grid['Residential_Count']=[0 for n in range(len(grid))]
grid['Non_Residential_Count']=[0 for n in range(len(grid))]
grid['Mixed_Count']=[0 for n in range(len(grid))]
grid['Unclassified_Count']=[0 for n in range(len(grid))]
grid['Unknown_Count']=[0 for n in range(len(grid))]
grid['Total_Cost']=[0 for n in range(len(grid))]
dataframe=pd.DataFrame(grid)
print('new column headers for required information')

# Apply the centroid function to the geometry column to determin the centre of each polygon
building_centroid.geometry=building_centroid['geometry'].centroid
# Ensure buildings and the new data frame have the same coordinate system
#building_centroid.crs=buildings.crs
# Redefine the index layer with sequential numbers
building_centroid=building_centroid.assign(Index=range(len(building_centroid))).set_index('Index')
print('building centroids calculated')

# For each grid cell, determine how many buildings fall within the cell
for i in range(0,len(grid)-1):
    for j in range (0, len(building_centroid)-1):
            if building_centroid.geometry[j].within(grid.geometry[i]):
            # Establish total cost of damage per cell
                grid.Total_Cost[i] +=building_centroid.damage[j]
            # Establish the number of building types within the cell
            if building_centroid.building_u[j]=='residential':
                grid.Residential_Count[i] +=1
            elif building_centroid.building_u[j]=='non-residential':
                grid.Non_Residential_Count[i] +=1
            elif building_centroid.building_u[j]=='mixed':
                grid.Mixed_Count[i] +=1
            elif building_centroid.building_u[j]=='unclassified':
                grid.Unclassified_Count[i] +=1
            else:
                grid.Unknown_Count[i] +=1

print('count on impacted buildings per cell')

# Now we need to find the average max depth per grid cell
depth=[]
depth=pd.DataFrame(depth)
depth['grid_num']=[0 for n in range(len(building_centroid))]
depth['Water_depth']=[0 for n in range(len(building_centroid))]
depth['Water_depth']=depth['Water_depth'].astype(float)
print('average max depth calculated')

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
print('unimpacted cells defined as zero')

# Combine the building/grid cell/ depth data with the list of empty cells
combined=depth.append(depth1, ignore_index=True)
# Find the average max depth per cell
averages=combined.groupby(['grid_num'])['Water_depth'].mean()
averages=pd.DataFrame(averages)
# Merge the average depth data with the building type and total cost
All_Information=pd.merge(grid, averages, left_index=True, right_index=True)


All_Information.to_csv(
        os.path.join(outputs_path, '1km_Grid_Cell_Information.csv'), index=False,  float_format='%g')
print('all new information saved to csv')
