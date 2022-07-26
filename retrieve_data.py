import os
import json
import overpy
import requests
import time
import re

# get the points data from osm for the rogue river section i want

def get_cache_filename(name):
	return f"cache/{name}.json"


def get_cache(name):
	path = get_cache_filename(name)
	if os.path.isfile(path):
		with open(path, "r") as f:
			return json.loads(f.read())
	return None

def save_cache(name, data):
	path = get_cache_filename(name)
	with open(path, "w+") as f:
		f.write(json.dumps(data, indent="\t"))

def do_web_request(url, cache_name):
	# print(cache_name)
	if cache_name:
		data = get_cache(cache_name)
		if data:
			return data
	print(f"querying: {url} ...")
	response = requests.get(url)
	if response.status_code == 429:
		seconds_to_wait = 10
		print(f"being rate limited, waiting {seconds_to_wait} seconds...")
		time.sleep(seconds_to_wait)
		return do_web_request(url, cache_name)
	elif response.status_code != 200:
		print(response.headers)
		raise Exception(f"shit broke: {response.status_code}")
	data = response.json()
	if cache_name:
		save_cache(cache_name, data)
	return data

def get_river_points():
	river_points_file = "river_points.json"

	if os.path.exists(river_points_file):
		with open(river_points_file, "r") as f:
			return json.loads(f.read())
	
	river_id = 425061030
	start_node = 3052799227
	end_node = 3052799079

	api = overpy.Overpass()

	result = api.query(f"way({river_id});(._;>;);out;")

	points = []


	started = False
	for node in result.ways[0].nodes:
		if node.tags.get("whitewater:rapid_grade") is not None:
			print(node.tags.get("name") + f"[class {node.tags.get('whitewater:rapid_grade')}]")
		if node.id == start_node:
			started = True
		if started:
			point = {
				"lat": float(node.lat),
				"lon": float(node.lon)
			}
			if node.tags.get("name"):
				point["name"] = node.tags["name"]
			points.append(point)
		if node.id == end_node:
			break
	
	print(f"loaded {len(points)} points from overpass api")

	with open(river_points_file, "w+") as f:
		f.write(json.dumps(points))
	return points

def query_tile_data(base_url, points, containingtext=None):
	found_items_path = "cache/found_items.json"

	parent_id = base_url.replace("https://www.sciencebase.gov/catalog/item/", "")

	items_url = f"https://www.sciencebase.gov/catalog/items?parentId={parent_id}&format=json&offset=0"

	found_items = []
	if os.path.exists(found_items_path):
		with open(found_items_path, "r") as f:
			found_items = json.loads(f.read())

	while items_url is not None:
		match = re.search(r"offset=(\d+)", items_url)
		offset_number = match.group(1)

		items_data = do_web_request(
			items_url, f"children_{parent_id}_{offset_number}")

		start_found_count = len(found_items)
		for item in items_data["items"]:
			if item["id"] in found_items:
				continue # already found this one
			if containingtext and containingtext not in item["title"]:
				continue # isnt what we're lookin for
			
			data = do_web_request(item["link"]["url"] + "?format=json", f"item_{item['id']}")
			# check if any points in bounding box
			found = False
			bbox = data["spatial"]["boundingBox"]
			for point in points:
				if point["lat"] > bbox["minY"] and point["lat"] < bbox["maxY"] and point["lon"] > bbox["minX"] and point["lon"] < bbox["maxX"]:
					found = True
					break
			if found:
				found_items.append(item["id"])
				print(f"found matching tile {item['id']}: {item['title']}")
			
		if start_found_count != len(found_items):
			with open(found_items_path, "w+") as f:
				f.write(json.dumps(found_items))
		
		if items_data.get("nextlink") is None:
			print("reached end!")
			break

		items_url = items_data["nextlink"]["url"]
	
	return found_items


def download_tiff_files(found_items):
	out_info_path = "river_tiff_tiles.json"
	# valid_tags = [ "1/3 arc-second DEM" ]
	valid_tags = [ "13 arc-second DEM" ]

	tiff_paths = []
	for item in found_items:
		with open(f"cache/item_{item}.json") as f:
			data = json.loads(f.read())
		found_valid_tag = False
		for tag in data["tags"]:
			if tag["name"] in valid_tags:
				found_valid_tag = True
		if not found_valid_tag:
			continue # filters out any invalid/wrong dimension stuff
		url = None
		for link in data["webLinks"]:
			if link["type"] == "download" and link.get("title") == "TIFF":
				url = link["uri"]
		if url is None:
			continue # we only care about stuff with tiff images
		
		print("loading: " + data["title"] + " (" + data["link"]["url"] + ")")
		tiff_path = f"cache/tiff_{item}.tiff"
		tiff_paths.append(tiff_path)
		if os.path.exists(tiff_path):
			continue
		response = requests.get(url)
		if response.status_code != 200:
			print(f"bad response: {response.status_code}")
		with open(tiff_path, "wb+") as f:
			f.write(response.content)
	
	with open(out_info_path, "w+") as f:
		f.write(json.dumps(tiff_paths, indent="\t"))
	print(f"wrote output to: {out_info_path}")


	# https://www.sciencebase.gov/catalog/items?parentId=543e6b86e4b0fd76af69cf4c&format=json

# https://www.sciencebase.gov/catalog/item/4f70ab22e4b058caae3f8deb

river_points = get_river_points()

found_items = query_tile_data(
	"https://www.sciencebase.gov/catalog/item/4f70aa9fe4b058caae3f8de5", river_points)

download_tiff_files(found_items)