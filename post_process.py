import chunk
from functools import cache
import numpy as np
import orjson
import math
import typing
import os

from utils import *


chunk_amount = 2

def get_point_plane_intersect(p0: Vertex, p1: Vertex, plane_point: Vertex, plane_normal):
	result = isect_line_plane_v3(
		p0.toNumpy(),
		p1.toNumpy(),
		plane_point.toNumpy(),
		plane_normal
	)
	return Vertex(
		result[0],
		result[1],
		result[2]
	)

# percent is 0 to 1 of how far accross
def create_mid_percent_point(p0: Vertex, p1: Vertex, percent):
	return Vertex(
		p0.x + ((p1.x - p0.x) * percent),
		p0.y + ((p1.y - p0.y) * percent),
		p0.z + ((p1.z - p0.z) * percent)
	)

def create_midpoint(p0, p1):
	return create_mid_percent_point(p0, p1, 0.5)

def point_distance(p1, p2):
	a = p2.x - p1.x
	b = p2.y - p1.y
	return math.sqrt(a * a + b * b)

# relies on p2 being the middle point
def are_points_collinear(a: Vertex, b: Vertex, c: Vertex):
	ab = np.linalg.norm(a.toNumpy() - b.toNumpy())
	bc = np.linalg.norm(b.toNumpy() - c.toNumpy())
	ac = np.linalg.norm(a.toNumpy() - c.toNumpy())
	should_be_positive = (ab + bc) - ac
	return should_be_positive < 0.000000001

def normalize_vector(vector):
	return vector / np.sqrt(np.sum(vector**2))

# PROLLY ALSO DUMP RICH VERTS INFORMATION IN THE MESH_DATA FILE


print("loading data")
with open("test_mesh_data.json", "r") as f:
	data = orjson.loads(f.read())

vertices = list(map(lambda v: Vertex(**v), data["vertices"]))
triangles = data["triangles"]
xy_origin = data.get("origin")
edge_points = data.get("base_vertices")

if edge_points is None:
	edge_points = range(0, len(vertices))

base_z = vertices[edge_points[0]].z

print("checking for duplicate verts in triangles")
for triangle in triangles:
	for i in triangle:
		if triangle.count(i) > 1:
			print(triangle)


print("find farthest apart base verts")
best_dist = 0
best_pair = (0, 0)
for i in range(len(edge_points)):
	for j in range(len(edge_points)):
		dist = point_distance(vertices[edge_points[i]], vertices[edge_points[j]])
		if dist > best_dist:
			best_dist = dist
			best_pair = (edge_points[i], edge_points[j])

print("creating plane(s)")

start_point = vertices[best_pair[0]]
end_point = vertices[best_pair[1]]
plane_normal = np.array([
	end_point.x - start_point.x,
	end_point.y - start_point.y,
	end_point.z - start_point.z
])
plane_normal[2] = 0 # z component should be 0
plane_normal = normalize_vector(plane_normal)

plane_points = []
# create the number of planes we need
for i in range(1, chunk_amount):
	plane_points.append(create_mid_percent_point(start_point, end_point, i / chunk_amount))

print(plane_points)

start_point_np = start_point.toNumpy()
def get_dist_to_beginning(point):
	point_np = point.toNumpy()
	v = point_np - start_point_np
	return np.dot(v, plane_normal)

vertex_normal_distance_map = []
print("calculating vert normal distance")
for vert in vertices:
	vertex_normal_distance_map.append(get_dist_to_beginning(vert))

# insert a new vertex by bisecting a line between the 2 given verts
# cache it when we create it based on the source vert indexes, and return the new index
# use_midpoint is for if we should use the midpoint of the triangle instead
new_vert_cache = {}
def insert_vert_bisect_line(vert_i1, vert_i2, plane_point, use_midpoint=False):
	cache_id = f"{vert_i1}_{vert_i2}" if vert_i1 < vert_i2 else f"{vert_i2}_{vert_i1}"
	cache_id = f"{cache_id}_{use_midpoint}"
	if cache_id in new_vert_cache:
		return new_vert_cache[cache_id]
	new_index = len(vertices)
	if use_midpoint:
		vert = create_midpoint(vertices[vert_i1], vertices[vert_i2])
	else:
		vert = get_point_plane_intersect(vertices[vert_i1], vertices[vert_i2], plane_point, plane_normal)
	vertices.append(vert)
	new_vert_cache[cache_id] = new_index
	return new_index


# a list of triangle lists
all_chunk_triangles = []
all_plane_edge_triangles = []
all_plane_edge_vertices = []

# triangles to include on the next chunk (needed because of the way we generate chunks with planes)
next_chunk_triangles = []

print(f"splitting into {chunk_amount} chunks")
for chunk_i in range(0, chunk_amount):
	print(f"- generating for chunk {chunk_i}")
	is_last_chunk = chunk_i == (chunk_amount - 1)
	chunk_triangles = next_chunk_triangles
	next_chunk_triangles = []
	plane_edge_triangles = []
	plane_edge_vertices = []
	remaining_triangles = [] # triangles left after parsing for this chunk

	# for each invalid triangle, subdivide it into new triangles
	cutoff = float("inf") if is_last_chunk else get_dist_to_beginning(plane_points[chunk_i])
	for triangle in triangles:
		def is_vert_valid(vert_i):
			return is_last_chunk or vertex_normal_distance_map[vert_i] < cutoff
		valid_points = list(map(is_vert_valid, triangle))
		valid = all(valid_points)
		if valid:
			chunk_triangles.append(triangle)
		else:
			if is_last_chunk:
				continue # if we're on the last chunk, ignore all invalid ones, as we've already added the ones we care about
			plane_point = plane_points[chunk_i]
			invalid_count = valid_points.count(False)
			if invalid_count == 3:
				remaining_triangles.append(triangle)
				continue # triangle fully invalid, ignore it
			# subdivide this triangle
			# get a, b, c, where a and c are on one side, and b is on the other. These are indexes
			is_tip = invalid_count == 2
			alone_index = valid_points.index(is_tip)
			if alone_index == 0:
				b = triangle[0]
				c = triangle[1]
				a = triangle[2]
			elif alone_index == 1:
				a = triangle[0]
				b = triangle[1]
				c = triangle[2]
			elif alone_index == 2:
				c = triangle[0]
				a = triangle[1]
				b = triangle[2]

			ab = insert_vert_bisect_line(a, b, plane_point)
			bc = insert_vert_bisect_line(b, c, plane_point)
			ab_bc = insert_vert_bisect_line(ab, bc, plane_point, use_midpoint=True)
			tip_triangles = [
				[ ab, b, ab_bc ],
				[ ab_bc, b, bc ],
			]
			base_triangles = [
				[ a, ab, ab_bc ],
				[ a, ab_bc, c ],
				[ ab_bc, bc, c ]
			]

			plane_edge_vertices.extend([ab, bc, ab_bc])
			if is_tip:
				plane_edge_triangles.extend(tip_triangles)
				chunk_triangles.extend(tip_triangles)
				next_chunk_triangles.extend(base_triangles)
			else:
				plane_edge_triangles.extend(base_triangles)
				chunk_triangles.extend(base_triangles)
				next_chunk_triangles.extend(tip_triangles)
	triangles = remaining_triangles
	all_chunk_triangles.append(chunk_triangles)
	if len(plane_edge_triangles) != 0:
		all_plane_edge_triangles.append(plane_edge_triangles)
		all_plane_edge_vertices.append(plane_edge_vertices)

class NeighborInfo():
	left: int
	right: int
	def __init__(self):
		self.left = None
		self.right = None


# ok now we gotta create a thing that can build the plane via triangles
class PlaneBuilder():
	plane_point: Vertex
	edge_verts: typing.List[int] # list of vert indexes (these are the verts on the edges of our plane, sorted by z from highest to lowest)
	neighbor_info: typing.Dict[int, NeighborInfo] # neighbors of that vertex that are not currently hidden by a triangle
	wall_triangles: typing.List[typing.List[int]]
	# edge_triangles: typing.List[typing.List[int]] # list of triangles (these are the triangles that currently connect the edges to the rest of the mesh)
	def __init__(self, plane_point, edge_verts, edge_triangles):
		self.wall_triangles = []
		self.hidden_verts = []
		self.plane_point = plane_point
		self.edge_verts = sorted(list(set(edge_verts)), key=lambda v: vertices[v].z, reverse=True)
		self.edge_triangles = edge_triangles

		self.neighbor_info = {}
		triangle_map = {}
		for triangle in edge_triangles:
			for vert in triangle:
				if vert in edge_verts:
					if vert not in triangle_map:
						triangle_map[vert] = []
					triangle_map[vert].append(triangle)
		# build neighbors
		for vert in self.edge_verts:
			info = NeighborInfo()
			for triangle in triangle_map[vert]:
				i = triangle.index(vert)
				right_index = 0 if i == 2 else (i + 1)
				left_index = 2 if i == 0 else (i - 1)
				if triangle[right_index] in self.edge_verts:
					info.right = triangle[right_index]
				if triangle[left_index] in self.edge_verts:
					info.left = triangle[left_index]
			self.neighbor_info[vert] = info

	# returns true if a triangle or more is created
	def create_wall_triangle_if_possible(self, vert, z_threshold):
		if vert not in self.edge_verts:
			return False # edge of mesh, ignore
		neighbors = self.neighbor_info[vert]
		if neighbors.left is None or neighbors.right is None:
			return False # edge of mesh, ignore
		if vertices[vert].z < z_threshold or vertices[neighbors.left].z < z_threshold or vertices[neighbors.right].z < z_threshold:
			return False # can't create a triangle cuz one of the vertices is less than the threshold
		if are_points_collinear(vertices[neighbors.left], vertices[vert], vertices[neighbors.right]):
			return False # can't have a triangle with collinear points
		to_left = normalize_vector(vertices[neighbors.left].toNumpy() - vertices[vert].toNumpy())
		to_right = normalize_vector(vertices[neighbors.right].toNumpy() - vertices[vert].toNumpy())
		z_magnitude = (to_left + to_right)[2]
		if z_magnitude > 0: # if the angle of this joint doesn't point down, we can't create a triangle here
			return False
		self.wall_triangles.append([neighbors.right, vert, neighbors.left])
		self.neighbor_info[neighbors.left].right = neighbors.right
		self.neighbor_info[neighbors.right].left = neighbors.left
		self.edge_verts.remove(vert)
		self.create_wall_triangle_if_possible(neighbors.left, z_threshold)
		self.create_wall_triangle_if_possible(neighbors.right, z_threshold)
		return True

	def generate_triangle_wall(self):
		self.wall_triangles = []
		self.hidden_verts = []
		for vert in list(self.edge_verts):
			z_threshold = vertices[vert].z
			if z_threshold <= base_z:
				break # TODO: add a thing here to stop before we get to nubs
			self.create_wall_triangle_if_possible(vert, z_threshold)
			self.create_wall_triangle_if_possible(self.neighbor_info[vert].left, z_threshold)
			self.create_wall_triangle_if_possible(self.neighbor_info[vert].right, z_threshold)
		# TODO: add things for nubs




print("generating wall triangles")
for plane_i in range(0, len(plane_points)):
	builder = PlaneBuilder(plane_points[plane_i], all_plane_edge_vertices[plane_i], all_plane_edge_triangles[plane_i])
	builder.generate_triangle_wall()
	all_chunk_triangles[plane_i].extend(builder.wall_triangles)


print("checking for duplicate verts in triangles")
for triangle in all_chunk_triangles[0]:
	for i in triangle:
		if triangle.count(i) > 1:
			print(triangle)

print("exporting to obj(s)")
for i in range(0, chunk_amount):
	obj_path = f"out_{i}.obj"
	print(f"- writing to {obj_path}")
	dump_to_obj(obj_path, vertices, all_chunk_triangles[i])

os.startfile("out_0.obj")