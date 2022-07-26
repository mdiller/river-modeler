import numpy as np
import json
import typing

print("loading data")
with open("elevation_data.json", "r") as f:
	data = json.loads(f.read())

# data = data[:len(data) // 2]

elevation_modifier = 1

print("putting vertexes in arrays")
flat_vertex_data = []
vertex_data = []
for point in data:
	vertex_data.append([
		point["x"],
		point["y"],
		point["elevation"] * elevation_modifier
	])
	flat_vertex_data.append([
		point["x"],
		point["y"]
	])


vertices = np.array(vertex_data)
flat_vertices = np.array(flat_vertex_data)

# SCIPY IMPLEMENTATION
triangle_file = "triangles.json"

if True: # not os.path.exists(triangle_file):
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

class RichVertex():
	x: float
	y: float
	z: float
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

# also grab these values
lowest_z = vertex_data[0][2]
highest_z = vertex_data[0][2]

print("building rich vertices")
rich_verts: typing.List[RichVertex]
rich_verts = []
for i in range(len(vertex_data)):
	v = RichVertex()
	v.x = vertex_data[i][0]
	v.y = vertex_data[i][1]
	v.z = vertex_data[i][2]
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

print("do edge vertex work")

print("find an edge vertex")
start_edge_vertex = -1 # index of an edge vertex to start on
for vert in rich_verts:
	index = vert.get_edge_next()
	if index is not None:
		start_edge_vertex = index



# go thru all edge vertexes and create points below em
print("iterating thru edge vertices")
base_z = lowest_z - ((highest_z - lowest_z) * 0.1)
v_count = len(rich_verts)
count = 0
index = start_edge_vertex
while True:
	# TODO: NOW WE DO STUFF HERE FOR EACH VERTEX, PROLLY JUST ADD A NEW ONE BELOW THIS AND THE NEXT ONE, AND THEN CREATE TRIANGLES FOR EM
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

print("rebuiling vertices and triangles for display")
vertices = np.array(list(map(lambda v: [v.x, v.y, v.z], rich_verts)))
triangles = np.array(triangles).astype(np.int32)

# print("adding new vertices to create a solid")
# new_vertices = []
# v_increment = 15
# base_elevation = (min(map(lambda p: p[2], vertex_data)) - 5) * elevation_modifier
# for i in range(len(vertex_data)):
# 	p = vertex_data[i]
# 	new_vertices.append((
# 		p[0],
# 		p[1],
# 		base_elevation
# 	))
# 	if i in outside_points: # add points up from the bottom
# 		y = base_elevation + v_increment
# 		while y < p[2]:
# 			new_vertices.append((
# 				p[0],
# 				p[1],
# 				y
# 			))
# 			y += v_increment
# vertex_data.extend(new_vertices)
# vertices = np.array(vertex_data)

print("stats: ")
print(f" - {len(vertices)} vertices")
print(f" - {len(triangles)} triangles")

# exporting as an obj
print("exporting to obj")
lines = []
lines.append("# river.obj")
lines.append("#")
lines.append("")
lines.append("g river")
lines.append("")
lines.extend(map(lambda v: f"v {v[0]} {v[1]} {v[2]}", vertices))
lines.append("")
lines.extend(map(lambda t: f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}", triangles))

text = "\n".join(lines)
with open("out.obj", "w+") as f:
	f.write(text)

exit(0)
