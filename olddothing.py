import chunk
import tifffile as tiff
import matplotlib.pyplot as plt

topleft = (43, -124)
galice = (42.57326195385712, -123.59671965219182)

# inputfile = "C:/dev/projects/heightmap/n42_w124_1arc_v3.tif"
# inputfile = "C:/Users/dillerm/Downloads/USGS_1M_10_x44y465_OR_RogueSiskiyouNF_2019_B19.tif"
inputfile = "best_tst.tif"

[-123.72447836999999, 41.90966911100003, -123.60288968199995, 42.00042782300005]

# tfile = tiff.imread(inputfile)
# tfile.shape
# tiff.imshow(tfile)


# raster = rasterio.open(inputfile)

# print(raster.res)
# point = ((1 / raster.res[0]) * abs(galice[0] - topleft[0]),
#          (1/ raster.res[1]) * (galice[1] - topleft[1]))

# plt.plot(point[0], point[1], marker="o", markersize=2,
#          markeredgecolor="red", markerfacecolor="green")

# plt.savefig("out.png")




# packages
# - imagecodecs
# - tifffile
# - matplotlib

# # Which band are you interested.
# # 1 if there is only one band
# band_of_interest = 1

# # Row and Columns of the raster you want to know
# # the value
# row_of_interest = 30
# column_of_interest = 50

# # open the raster and close it automatically
# # See https://stackoverflow.com/questions/1369526
# with rasterio.open(inputfile) as dataset:
# 	band = dataset.read(band_of_interest)
# 	value_of_interest = band[row_of_interest, column_of_interest]
# 	print(value_of_interest)



import math
import numpy as np
import rasterio
import json
from collections import namedtuple


threshold_radius_meters = 200
radius_of_earth_meters = 6378100 # meters

with open("cache/river_points.json", "r") as f:
	river_points = json.loads(f.read())

# coords = list(map(lambda p: (p["lat"], p["lon"]), river_points))

# the two below from here: https://stackoverflow.com/questions/7477003/calculating-new-longitude-latitude-from-old-n-meters
def add_lon(lat, lon, meters):
	return lon + (meters / radius_of_earth_meters) * (180 / math.pi) / math.cos(lat * math.pi/180)
def add_lat(lat, lon, meters):
	return lat  + (meters / radius_of_earth_meters) * (180 / math.pi);

def lat_long_distance(lat1, lon1, lat2, lon2):
	dlon = lon2 - lon1
	dlat = lat2 - lat1

	# haversine formula
	a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
	return c * radius_of_earth_meters

print("prebuilding chunks and bounding box...")
# build a bounding box for my points of interest
class BoundingBox:
	def __init__(self, points, chunk_amount):
		# get points
		self.min_lat = min(map(lambda p: p["lat"], points))
		self.min_lon = min(map(lambda p: p["lon"], points))
		self.max_lat = max(map(lambda p: p["lat"], points))
		self.max_lon = max(map(lambda p: p["lon"], points))

		# adjust points for buffer
		buffer_adjust = threshold_radius_meters * 2
		self.min_lat = add_lat(self.min_lat, self.min_lon, 0 - buffer_adjust)
		self.min_lon = add_lon(self.min_lat, self.min_lon, 0 - buffer_adjust)
		self.max_lat = add_lat(self.max_lat, self.max_lon, buffer_adjust)
		self.max_lon = add_lon(self.max_lat, self.max_lon, buffer_adjust)

		self.lat_size = self.max_lat - self.min_lat
		self.lon_size = self.max_lon - self.min_lon

		self.chunk_amount = chunk_amount
		self.chunk_x_helper = self.chunk_amount / self.lon_size
		self.chunk_y_helper = self.chunk_amount / self.lat_size
	
	def does_contain(self, lat, lon):
		return (lat > self.min_lat and lat < self.max_lat) and (lon > self.min_lon and lon < self.max_lon)
	
	def get_x_chunk(self, lon):
		return int(self.chunk_x_helper * (lon - bbox.min_lon))
		
	def get_y_chunk(self, lon):
		return int(self.chunk_x_helper * (lon - bbox.min_lon))

bbox = BoundingBox(river_points)
# (lat > bbox.min_lat and lat < bbox.max_lat) and (lon > bbox.min_lon and lon < bbox.max_lon)

# build a bunch of chunks that say whether or not they have a point in range
# each chunk represents a square inside the bounding box
chunk_amount = 10
close_chunks = [[False for x in range(chunk_amount)] for y in range(chunk_amount)]

for point in river_points:
	x = math.floor(chunk_amount * ((point["lon"] - bbox.min_lon) / bbox.lon_size))
	y = math.floor(chunk_amount * ((point["lat"] - bbox.min_lat) / bbox.lat_size))
	close_chunks[x][y] = True

for x in range(chunk_amount):
	line = ""
	for y in range(chunk_amount):
		if close_chunks[x][y]:
			line += "x"
		else:
			line += " "
	print(line)

# TODO: also mark chunks that are a bit close to the river

def is_valid_chunk(lat, lon):
	x = math.floor(chunk_amount * ((lon - bbox.min_lon) / bbox.lon_size))
	y = math.floor(chunk_amount * ((lat - bbox.min_lat) / bbox.lat_size))
	return close_chunks[x][y]

# returns whether or not the point is close enough to one of my river points
def is_point_relevant(lat, lon):
	if not ((lat > bbox.min_lat and lat < bbox.max_lat) and (lon > bbox.min_lon and lon < bbox.max_lon)):
		return False

	return is_valid_chunk(lat, lon)

	for point in river_points:
		dist = lat_long_distance(lat, lon, point["lat"], point["lon"])
		if dist < threshold_radius_meters:
			return True
	return False

print("loading image")
with rasterio.open(inputfile) as src:
	band1 = src.read(1)
	print('Band1 has shape', band1.shape)
	height = band1.shape[0]
	width = band1.shape[1]
	cols, rows = np.meshgrid(np.arange(width), np.arange(height))
	xs, ys = rasterio.transform.xy(src.transform, rows, cols)
	lons = np.array(xs)
	lats = np.array(ys)
	print('lons shape', lons.shape)
	print("cols length", len(cols))

	xpos = 10811
	ypos = 10811

	good_points = []
	for x in range(len(cols)):
		if (x % 100 == 0):
			print(x)
		if min(lats[x]) > bbox.max_lat or max(lats[x]) < bbox.min_lat:
			continue
		for y in range(len(rows)):
			if is_point_relevant(lats[x][y], lons[x][y]):
				good_points.append((lats[x][y], lons[x][y]))

	print(f"found {len(good_points)} points")

	# print("first point coord", rows[xpos][ypos], cols[xpos][ypos])
	# print("first point", lats[xpos][ypos], lons[xpos][ypos])
	# vals = src.sample([(lons[xpos][ypos], lats[xpos][ypos])])
	# for val in vals:
	# 	print(val[0])






# could make api to crawl for the 1m tiles, get bounding box from the json link
# https://www.sciencebase.gov/catalog/item/60d5633cd34ef0ccfc0c8604?format=json





# entity id SRTM1N42W124V3
# nw lat  43
# nw long -124
# se lat  42
# se long -123
