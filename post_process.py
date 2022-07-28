import numpy as np
import json
import math
import typing



def point_distance(p1, p2):
	# if isinstance(p1, RichVertex):
	# 	p1 = { "x": p1.x, "y": p1.y }
	# if isinstance(p2, RichVertex):
	# 	p2 = { "x": p2.x, "y": p2.y }
	a = p2["x"] - p1["x"]
	b = p2["y"] - p1["y"]
	return math.sqrt(a * a + b * b)

# PROLLY ALSO DUMP RICH VERTS INFORMATION IN THE MESH_DATA FILE


print("loading data")
with open("mesh_data.json", "r") as f:
	data = json.loads(f.read())

xy_origin = data["origin"]
vertices = data["vertices"]
triangles = data["triangles"]
edge_points = data["base_vertices"]

print("find farthest apart base verts")
best_dist = 0
best_pair = (0, 0)
for i in range(len(edge_points)):
	for j in range(len(edge_points)):
		dist = point_distance(vertices[edge_points[i]], vertices[edge_points[j]])
		if dist > best_dist:
			best_dist = dist
			best_pair = (edge_points[i], edge_points[j])

print("best pair: ")
print(vertices[best_pair[0]])
print(vertices[best_pair[1]])

