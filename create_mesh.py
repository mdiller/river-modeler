import numpy as np
import json
import math
import os
import typing
from utils import *

desired_length_mm = 450

class RichVertex(Vertex):
	index: int
	neighbors: typing.List[int]
	triangles: typing.List[int]

	# get the next vertex along the edge
	def get_edge_next(self):
		counts = [0] * len(self.neighbors)
		is_valid = [False] * len(self.neighbors)
		for t in self.triangles:
			tri = triangles[t]
			last_was_us = tri[2] == self.index
			for i in tri:
				if i in self.neighbors:
					index = self.neighbors.index(i)
					is_valid[index] = last_was_us
					counts[index] += 1
				last_was_us = i == self.index

		for i in range(len(counts)):
			if counts[i] == 1 and is_valid[i]:
				# print("index: ", self.index)
				# print("neighbors: ", self.neighbors)
				# print("triangles: ", list(map(lambda t: triangles[t], self.triangles)))
				# print("counts: ", counts)
				return self.neighbors[i]
		return None


print("loading data")
with open("elevation_data.json", "r") as f:
	data = json.loads(f.read())

xy_origin = data["origin"]
elevation_data = data["points"]


# give the xy coordinates of the given latlon location. (xy is just meters east&north from xy_origin)
def latlong_to_xy(point):
	x = lat_long_distance(point["lat"], xy_origin["lon"], point["lat"], point["lon"])
	y = lat_long_distance(xy_origin["lat"], point["lon"], point["lat"], point["lon"])
	return {
		"x": x,
		"y": y
	}

# data = data[:len(data) // 2]

elevation_modifier = 1

print("putting vertices in arrays")
flat_vertices = []
vertices = []
for point in elevation_data:
	vertices.append([
		point["x"],
		point["y"],
		point["elevation"] * elevation_modifier
	])
	flat_vertices.append([
		point["x"],
		point["y"]
	])


vertices = np.array(vertices)
flat_vertices = np.array(flat_vertices)

# SCIPY IMPLEMENTATION
triangle_file = f"cache/triangles_{len(vertices)}.json"

if not os.path.exists(triangle_file):
	print("generating triangles")
	from scipy.spatial import Delaunay
	triangle_data = Delaunay(flat_vertices)

	print("filtering triangles")
	# Separating small and large edges:
	thresh = 60  # user defined threshold
	small_edges = set()
	large_edges = set()
	filtered_triangles = []
	for tr in triangle_data.vertices:
		is_small = True
		for i in range(3):
			edge_idx0 = tr[i]
			edge_idx1 = tr[(i+1)%3]
			if (edge_idx1, edge_idx0) in small_edges:
				continue  # already visited this edge from other side
			if (edge_idx1, edge_idx0) in large_edges:
				is_small = False
				continue
			p0 = vertices[edge_idx0]
			p1 = vertices[edge_idx1]
			if np.linalg.norm(p1 - p0) <  thresh:
				small_edges.add((edge_idx0, edge_idx1))
			else:
				is_small = False
				large_edges.add((edge_idx0, edge_idx1))
		if is_small:
			filtered_triangles.append(tr.tolist())
	with open(triangle_file, "w+") as f:
		f.write(json.dumps(filtered_triangles))



# ADDING POINTS FOR UNDERSIDE AND EDGES
with open(triangle_file, "r") as f:
	triangles = json.loads(f.read())
	# triangles = np.array(json.loads(f.read())).astype(np.int32)



# print("computing edge lines")
# existing_edges = set()
# outside_edges = set()
# for tr in triangles:
# 	for i in range(3):
# 		edge_idx0 = tr[i]
# 		edge_idx1 = tr[(i + 1) % 3]
# 		edge = (edge_idx0, edge_idx1)
# 		edge2 = (edge_idx1, edge_idx0)
# 		if edge in outside_edges:
# 			outside_edges.remove(edge)
# 		if edge2 in outside_edges:
# 			outside_edges.remove(edge2)
# 		if (edge not in existing_edges) and (edge2 not in existing_edges):
# 			existing_edges.add(edge)
# 			outside_edges.add(edge)
# outside_edges = np.array(list(outside_edges))

# outside_points = set()
# for edge in outside_edges:
# 	outside_points.add(edge[0])
# 	outside_points.add(edge[1])
# print("edge verts: ", len(outside_edges))


# also grab these values
lowest_z = vertices[0][2]
highest_z = vertices[0][2]

print("building rich vertices")
rich_verts: typing.List[RichVertex]
rich_verts = []
for i in range(len(vertices)):
	v = RichVertex()
	v.x = vertices[i][0]
	v.y = vertices[i][1]
	v.z = vertices[i][2]
	v.index = i
	v.neighbors = []
	v.triangles = []
	rich_verts.append(v)
	if v.z < lowest_z:
		lowest_z = v.z
	if v.z > highest_z:
		highest_z = v.z
	

print("adding triangles info")
for i in range(len(triangles)):
	for vert in triangles[i]:
		rich_verts[vert].triangles.append(i)
		for vert2 in triangles[i]:
			if vert != vert2 and vert2 not in rich_verts[vert].neighbors:
				rich_verts[vert].neighbors.append(vert2)


print("convert river points to xy coordinates")
with open("river_points.json", "r") as f:
	river_points = json.loads(f.read())
river_points = list(map(latlong_to_xy, river_points))

print("do edge vertex work")

print("find an edge vertex")
start_edge_vertex = -1 # index of an edge vertex to start on
for vert in rich_verts:
	index = vert.get_edge_next()
	if index is not None:
		start_edge_vertex = index
		break

# go thru all edge vertexes and create points below em
print("iterating thru edge vertices to create walls")
base_z = lowest_z - ((highest_z - lowest_z) * 0.1)
v_count = len(rich_verts)
count = 0
index = start_edge_vertex
while True:
	vert = rich_verts[index]
	next_index = vert.get_edge_next()
	next_vert = rich_verts[next_index]

	new_index = v_count + count
	new_vert = RichVertex()
	new_vert.x = (vert.x + next_vert.x) / 2
	new_vert.y = (vert.y + next_vert.y) / 2
	new_vert.z = base_z
	new_vert.index = new_index
	new_vert.neighbors = [] # these just blank we dont care about em
	new_vert.triangles = [] # these just blank we dont care about em
	rich_verts.append(new_vert)

	triangles.append([ index, new_index, next_index ])
	if count != 0:
		triangles.append([ new_index, index, new_index - 1 ])

	if next_index == start_edge_vertex:
		triangles.append([ v_count, start_edge_vertex, new_index ])
		break
	index = next_index
	count += 1


print("creating base")

# edge verts organizing
smallest_dist = math.inf
smallest_dist_index = -1
river1 = river_points[0]
river2 = river_points[1]
for i in range(v_count, len(rich_verts)):
	vert = rich_verts[i]
	dist1 = point_distance(vert, river1)
	dist2 = point_distance(vert, river2)
	if dist1 < dist2 and dist1 < smallest_dist:
		smallest_dist = dist1
		smallest_dist_index = i
edge_points = rich_verts[smallest_dist_index:len(rich_verts)] + rich_verts[v_count:smallest_dist_index]

# base triangle creation
edge_index_1 = 0
edge_index_2 = len(edge_points) - 1
while edge_index_1 != edge_index_2:
	point1 = edge_points[edge_index_1]
	point2 = edge_points[edge_index_2]
	next_point1 = edge_points[edge_index_1 + 1]
	next_point2 = edge_points[edge_index_2 - 1]
	if point1.index == next_point2.index or point2.index == next_point1.index:
		break # we done
	if point_distance(point1, next_point2) > point_distance(point2, next_point1):
		triangles.append([ point1.index, point2.index, next_point1.index ])
		edge_index_1 += 1
	else:
		triangles.append([ point1.index, point2.index, next_point2.index ])
		edge_index_2 -= 1

print("find farthest apart base verts")
best_dist = 0
for i in range(len(edge_points)):
	for j in range(len(edge_points)):
		dist = point_distance(edge_points[i], edge_points[j])
		if dist > best_dist:
			best_dist = dist

print(f"resizing to be {desired_length_mm} mm long (assuming input units are meters)")
current_length_m = best_dist
scale = desired_length_mm / current_length_m
print(f"- scale is {scale}")
for vertex in rich_verts:
	vertex.x *= scale
	vertex.y *= scale
	vertex.z *= scale

print("stats: ")
print(f" - {len(rich_verts)} vertices")
print(f" - {len(triangles)} triangles")


print("exporting to json")
data = {
	"scale": scale,
	"origin": xy_origin,
	"vertices": list(map(lambda v: v.toJson(), rich_verts)),
	"triangles": list(map(lambda t: [t[0], t[1], t[2]], triangles)),
	"base_vertices": list(map(lambda v: v.index, edge_points))
}
with open("mesh_data.json", "w+") as f:
	f.write(json.dumps(data))

# exporting as an obj
print("exporting to obj")
dump_to_obj("mesh.obj", rich_verts, triangles)

exit(0)
