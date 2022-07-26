import chunk
import math
from re import M
import numpy as np
import rasterio
import json
from collections import namedtuple

tiff_files_list = "river_tiff_tiles.json"
# inputfile = "best_tst.tif"
threshold_radius_meters = 2500 # 900 before
radius_of_earth_meters = 6378100 # meters
chunk_amount = 300

# TODO: fill in more points to be closer together

with open("river_points.json", "r") as f:
	river_points = json.loads(f.read())

# the two below from here: https://stackoverflow.com/questions/7477003/calculating-new-longitude-latitude-from-old-n-meters
def add_lon(lat, lon, meters):
	return lon + (meters / radius_of_earth_meters) * (180 / math.pi) / math.cos(lat * math.pi/180)
def add_lat(lat, lon, meters):
	return lat  + (meters / radius_of_earth_meters) * (180 / math.pi)

# haversine formula: gets the distance (in meters) between 2 latlong points
def lat_long_distance(lat1, lon1, lat2, lon2):
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
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
	
	def get_chunk_x(self, lon):
		return int(self.chunk_x_helper * (lon - self.min_lon))

	def get_chunk_y(self, lat):
		return int(self.chunk_y_helper * (lat - self.min_lat))

	def xy_to_latlong(self, xy):
		return (
			((xy[0] / self.chunk_amount) * self.lon_size) + self.min_lon,
			((xy[1] / self.chunk_amount) * self.lat_size) + self.min_lat
		)

bbox = BoundingBox(river_points, chunk_amount)
# (lat > bbox.min_lat and lat < bbox.max_lat) and (lon > bbox.min_lon and lon < bbox.max_lon)

# build a bunch of chunks that say whether or not they have a point in range
# each chunk represents a square inside the bounding box
close_chunks = [[None for x in range(chunk_amount)] for y in range(chunk_amount)]
for point in river_points:
	buffer_amount = threshold_radius_meters * 2
	min_x = bbox.get_chunk_x(add_lon(point["lat"], point["lon"], 0 - buffer_amount))
	min_y = bbox.get_chunk_y(add_lat(point["lat"], point["lon"], 0 - buffer_amount))
	max_x = bbox.get_chunk_x(add_lon(point["lat"], point["lon"], buffer_amount))
	max_y = bbox.get_chunk_y(add_lat(point["lat"], point["lon"], buffer_amount))
	for x in range(min_x, max_x + 1):
		for y in range(min_y, max_y + 1):
			if x > 0 and x < chunk_amount and y > 0 and y < chunk_amount:
				close_chunks[x][y] = "partial"

# check for full coverage, and refine the partial ones
for x in range(chunk_amount):
	for y in range(chunk_amount):
		if close_chunks[x][y]:
			# check if all corners and center are covered
			points = [
				(x, y),
				(x + 1, y),
				(x, y + 1),
				(x + 1, y + 1),
				(x + 0.5, y + 0.5)
			]
			points = list(map(bbox.xy_to_latlong, points))
			covered_at_all = False
			covered_fully = True
			for point in points:
				point_is_covered = False
				for rpoint in river_points:
					# print(lat_long_distance(point[1], point[0], rpoint["lat"], rpoint["lon"]))
					if lat_long_distance(point[1], point[0], rpoint["lat"], rpoint["lon"]) < threshold_radius_meters:
						point_is_covered = True
						break
				if point_is_covered:
					covered_at_all = True
				else:
					covered_fully = False
			if covered_fully:
				close_chunks[x][y] = "full"
			if not covered_at_all:
				close_chunks[x][y] = None


# print chunks
# true_chunks = 0
# for x in range(chunk_amount):
# 	line = ""
# 	for y in range(chunk_amount):
# 		if close_chunks[x][y]:
# 			true_chunks += 1
# 			if close_chunks[x][y] == "full":
# 				line += "x"
# 			else:
# 				line += "."
# 		else:
# 			line += " "
# 	print(line)
# print(f"{true_chunks} included chunks")

# TODO: also mark chunks that are a bit close to the river

def is_valid_chunk(lat, lon):
	x = bbox.get_chunk_x(lon)
	y = bbox.get_chunk_y(lat)
	return close_chunks[x][y]

# returns whether or not the point is close enough to one of my river points
def is_point_relevant(lat, lon):
	if not ((lat > bbox.min_lat and lat < bbox.max_lat) and (lon > bbox.min_lon and lon < bbox.max_lon)):
		return False

	
	x = bbox.get_chunk_x(lon)
	y = bbox.get_chunk_y(lat)
	if close_chunks[x][y] is None:
		return False
	
	if close_chunks[x][y] == "full":
		return True

	for point in river_points:
		dist = lat_long_distance(lat, lon, point["lat"], point["lon"])
		if dist < threshold_radius_meters:
			return True
	return False

print("loading image(s)")
with open(tiff_files_list, "r") as f:
	image_paths = json.loads(f.read())

elevation_data = []
for image_path in image_paths:
	print("loading: " + image_path)
	with rasterio.open(image_path) as src:
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

		good_points = []
		for x in range(len(cols)):
			if (x % 100 == 0):
				print(x)
			if min(lats[x]) > bbox.max_lat or max(lats[x]) < bbox.min_lat:
				continue
			for y in range(len(rows)):
				if is_point_relevant(lats[x][y], lons[x][y]):
					good_points.append((lons[x][y], lats[x][y]))
					

		vals = src.sample(good_points)
		vals = list(map(lambda v: v[0], vals))

		# this is gonna be cartesian (+x is east, +y is north)
		mid_lat = (bbox.min_lat + bbox.max_lat) / 2
		mid_lon = (bbox.min_lon + bbox.max_lon) / 2
		meters_x = lat_long_distance(mid_lat, bbox.min_lon, mid_lat, bbox.max_lon) # how wide the bbox is in meters
		meters_y = lat_long_distance(bbox.min_lat, mid_lon, bbox.max_lat, mid_lon) # how tall the bbox is in meters

		for i in range(len(good_points)):
			lat = good_points[i][1]
			lon = good_points[i][0]
			x = ((lon - bbox.min_lon) / bbox.lon_size) * meters_x
			y = ((lat - bbox.min_lat) / bbox.lat_size) * meters_y
			elevation_data.append({
				"lat": lat,
				"lon": lon,
				"x": x,
				"y": y,
				"elevation": float(vals[i])
			})
	
print(f"found {len(elevation_data)} points")
with open("elevation_data_big.json", "w+") as f:
	f.write(json.dumps(elevation_data))


	# xpos = 10811
	# ypos = 10811
	# print("first point coord", rows[xpos][ypos], cols[xpos][ypos])
	# print("first point", lats[xpos][ypos], lons[xpos][ypos])
	# vals = src.sample([(lons[xpos][ypos], lats[xpos][ypos])])
	# for val in vals:
	# 	print(val[0])




# we have 10812 pixels accross 
# should be 108,120 meters

# actual lat distance is 111km
# actual lon distance is

# entity id SRTM1N42W124V3
# nw lat  43
# nw long -124
# se lat  42
# se long -123
