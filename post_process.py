import chunk
from collections import namedtuple
from functools import cache
import numpy as np
import orjson
import math
import typing
import os

from utils import *

should_generate_nubs = True
NubDimensions = namedtuple("NubDimensions", "height width depth padding clearance chamfers")
nub_dimensions_mm = NubDimensions(2, 4, 6, 1, 0.2, 0.1)
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
def calc_triangle_size(a: Vertex, b: Vertex, c: Vertex):
	a = a.toNumpy()
	b = b.toNumpy()
	c = c.toNumpy()
	return np.linalg.norm(np.cross((b - a), (c - a))) / 2

def normalize_vector(vector):
	return vector / np.sqrt(np.sum(vector**2))

def normalize_vector3(vector):
	if np.linalg.norm(vector) == 0:
		raise Exception("bad things")
	return vector / np.linalg.norm(vector)


print("loading data")
with open("mesh_data.json", "r") as f:
	data = orjson.loads(f.read())

vertices = list(map(lambda v: Vertex(**v), data["vertices"]))
triangles = data["triangles"]
xy_origin = data.get("origin")
edge_points = data.get("base_vertices")

if edge_points is None:
	edge_points = range(0, len(vertices))

base_z = vertices[edge_points[0]].z

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


# this left/right is from the perspective of inside the open loop of verts looking out
class NeighborInfo():
	left: int
	right: int
	def __init__(self):
		self.left = None
		self.right = None


# ok now we gotta create a thing that can build the plane via triangles
class WallBuilder():
	plane_point: Vertex
	edge_verts: typing.List[int] # list of vert indexes (these are the verts on the edges of our plane, sorted by z from highest to lowest)
	neighbor_info: typing.Dict[int, NeighborInfo] # neighbors of that vertex that are not currently hidden by a triangle
	wall_triangles: typing.List[typing.List[int]]
	# edge_triangles: typing.List[typing.List[int]] # list of triangles (these are the triangles that currently connect the edges to the rest of the mesh)
	def __init__(self, plane_point, edge_verts, edge_triangles):
		self.wall_triangles = []
		self.plane_point = plane_point
		self.edge_verts = sorted(list(set(edge_verts)), key=lambda v: vertices[v].z, reverse=True)
		self.edge_triangles = edge_triangles
		self.invalid_triangle_list = []
		self.wall_triangles_side1 = []
		self.wall_triangles_side2 = []

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
	
	# generates the 4 corners of each nub
	def generate_all_nubs(self):
		print("generating nubs")
		# for generating nubs, right/left is from perspective of looking at the wall
		corner_right = None
		corner_left = None
		current = self.edge_verts[0]
		print("- finding corners")
		while corner_right is None or corner_left is None:
			neighbors = self.neighbor_info[current]
			if vertices[current].z == base_z and vertices[neighbors.right].z != base_z:
				corner_left = current
			if vertices[current].z == base_z and vertices[neighbors.left].z != base_z:
				corner_right = current
			current = neighbors.right
		
		dir_left = normalize_vector3(vertices[corner_left].toNumpy() - vertices[corner_right].toNumpy())
		dir_out = plane_normal
		dir_up = normalize_vector3(np.cross(dir_left, dir_out))
		self.generate_nub(corner_right, dir_left, dir_up, False)
		self.generate_nub(corner_left, -1 * dir_left, dir_up, True)

	def generate_nub(self, corner_vert_i, dir_across, dir_up, is_left):
		print(f"- generating {'left' if is_left else 'right'} nub")
		outline = []
		vert = vertices[corner_vert_i].toNumpy()
		vert = vert + (dir_across * nub_dimensions_mm.padding + dir_up * nub_dimensions_mm.padding)
		outline.append(vert)
		vert = vert + (dir_up * nub_dimensions_mm.height)
		outline.append(vert)
		vert = vert + (dir_across * nub_dimensions_mm.width)
		outline.append(vert)
		vert = vert + (-1 * dir_up * nub_dimensions_mm.height)
		outline.append(vert)
		vertices.extend(list(map(lambda v: Vertex(v[0], v[1], v[2]), outline)))
		outline_indexes = range(len(vertices) - len(outline), len(vertices))

		# create 2 triangles and setup neighbor stuff, so the rest of the wall can generate properly
		next_edge_vert = self.neighbor_info[corner_vert_i].left if is_left else self.neighbor_info[corner_vert_i].right
		new_triangles = [
			[ corner_vert_i, outline_indexes[0], next_edge_vert ],
			[ next_edge_vert, outline_indexes[0], outline_indexes[len(outline_indexes) - 1] ]
		]
		left_to_right = []
		left_to_right.append(corner_vert_i)
		left_to_right.extend(outline_indexes)
		left_to_right.append(next_edge_vert)
		if is_left:
			new_triangles = list(map(lambda t: list(reversed(t)), new_triangles))
			left_to_right = list(reversed(left_to_right))
		for i in range(len(left_to_right) - 1):
			current = left_to_right[i]
			next = left_to_right[i + 1]
			if next not in self.neighbor_info:
				self.neighbor_info[next] = NeighborInfo()
			self.neighbor_info[current].right = next
			self.neighbor_info[next].left = current
		self.edge_verts.extend(outline_indexes)
		self.wall_triangles.extend(new_triangles)

		# generate triangles that make up the actual nub. generate these for both sides of the wall, because they gon be different cuz of clearance
		self.wall_triangles_side1.extend(self.generate_nub_triangles(outline_indexes, is_left, False))
		self.wall_triangles_side2.extend(self.generate_nub_triangles(outline_indexes, is_left, True))

	def generate_nub_triangles(self, outline_indexes, is_left, is_side_2):
		nub_dir = plane_normal if is_left else (-1 * plane_normal)
		should_add_clearance = (is_left and (not is_side_2)) or ((not is_left) and is_side_2)
		should_reverse_triangles = should_add_clearance

		ring_vert_count = len(outline_indexes)
		outline = list(map(lambda i: vertices[i].toNumpy(), outline_indexes))
		# generate actual nubs
		nub_center = np.array([
			sum(map(lambda v: v[0], outline)) / ring_vert_count,
			sum(map(lambda v: v[1], outline)) / ring_vert_count,
			sum(map(lambda v: v[2], outline)) / ring_vert_count
		])
		to_center_dirs = list(map(lambda i: normalize_vector3(nub_center - outline[i]), range(ring_vert_count)))
		transforms = [
			# chamfer
			lambda i: (nub_dir * nub_dimensions_mm.chamfers) + (to_center_dirs[i] * nub_dimensions_mm.chamfers),
			# straight back
			lambda i: nub_dir * nub_dimensions_mm.depth,
			# chamfer again
			lambda i: (nub_dir * nub_dimensions_mm.chamfers) + (to_center_dirs[i] * nub_dimensions_mm.chamfers),
		]
		if should_add_clearance:
			transforms[1] = lambda i: nub_dir * (nub_dimensions_mm.depth - (2 * nub_dimensions_mm.clearance))
			transforms.insert(0, lambda i: to_center_dirs[i] * nub_dimensions_mm.clearance)

		# generate new points and triangles
		nub_vertices = []
		last_ring = outline
		for transform in transforms:
			ring = []
			for i in range(ring_vert_count):
				ring.append(last_ring[i] + transform(i))
			last_ring = ring
			nub_vertices.extend(ring)
		# generate triangles to connect rings
		nub_triangles = []
		base_i = len(vertices)
		for ring_i in range(len(transforms)):
			for i in range(ring_vert_count):
				i2 = i + 1 if i + 1 != ring_vert_count else 0
				ring1_i1 = base_i + ((ring_i - 1) * ring_vert_count) + i
				ring1_i2 = base_i + ((ring_i - 1) * ring_vert_count) + i2
				if ring_i == 0:
					ring1_i1 = outline_indexes[i]
					ring1_i2 = outline_indexes[i2]
				ring2_i1 = base_i + (ring_i * ring_vert_count) + i
				ring2_i2 = base_i + (ring_i * ring_vert_count) + i2
				nub_triangles.append([ ring1_i1, ring2_i2, ring2_i1 ])
				nub_triangles.append([ ring1_i2, ring2_i2, ring1_i1 ])
		# generate triangles for the end cap of the node
		end_vert_start_i = base_i + len(nub_vertices) - ring_vert_count
		for i in range(ring_vert_count):
			i2 = i + 1 if i + 1 != ring_vert_count else 0
			nub_triangles.append([ 
				end_vert_start_i + i2,
				end_vert_start_i + ring_vert_count,
				end_vert_start_i + i,
			])
		end_verts = nub_vertices[len(nub_vertices) - ring_vert_count:]
		nub_vertices.append(np.array([ # end_center vertex
			sum(map(lambda v: v[0], end_verts)) / ring_vert_count,
			sum(map(lambda v: v[1], end_verts)) / ring_vert_count,
			sum(map(lambda v: v[2], end_verts)) / ring_vert_count
		]))
		vertices.extend(list(map(lambda v: Vertex(v[0], v[1], v[2]), nub_vertices)))
		if should_reverse_triangles:
			nub_triangles = list(map(lambda t: list(reversed(list(t))), nub_triangles))
		return nub_triangles


	def is_triangle_valid(self, vert):
		neighbors = self.neighbor_info[vert]
		index1 = neighbors.left
		index2 = vert
		index3 = neighbors.right
		if index1 == index3 or index1 == index2 or index2 == index3:
			return False
		key = f"{index1}_{index2}_{index3}"
		if key in self.invalid_triangle_list:
			return False
		
		a = vertices[index1].toNumpy()
		b = vertices[index2].toNumpy()
		c = vertices[index3].toNumpy()
		ab = normalize_vector3(b - a)
		ac = normalize_vector3(c - a)
		vec = np.cross(ac, plane_normal)
		result = np.dot(vec, ab)
		is_valid = result < -0.00000001 # this will also check collinear lines

		if not is_valid:
			self.invalid_triangle_list.append(key)
		return is_valid

	# returns true if a triangle or more is created
	def create_wall_triangle(self, vert):
		neighbors = self.neighbor_info[vert]
		self.wall_triangles.append([neighbors.right, vert, neighbors.left])
		self.neighbor_info[neighbors.left].right = neighbors.right
		self.neighbor_info[neighbors.right].left = neighbors.left
		self.edge_verts.remove(vert)

	def generate_triangle_wall(self):
		print("generating triangle wall")
		TriangleWeight = namedtuple("TriangleWeight", "vert weight")
		triangle_weights: typing.List[TriangleWeight]
		triangle_weights = []
		def calc_weight(vert):
			neighbors = self.neighbor_info[vert]
			return calc_triangle_size(vertices[neighbors.left], vertices[vert], vertices[neighbors.right])
		def add_weight(vert):
			weight = calc_weight(vert)
			new_weight_info = TriangleWeight(vert, weight)
			for i in range(len(triangle_weights)): # could do binary search to make faster but we dont care
				if weight < triangle_weights[i].weight:
					triangle_weights.insert(i, new_weight_info)
					return
			triangle_weights.append(new_weight_info)
		
		# init weights
		for vert in self.edge_verts:
			if self.is_triangle_valid(vert):
				add_weight(vert)

		while len(self.edge_verts) >= 3 and len(triangle_weights) > 0:
			vert = triangle_weights[0].vert
			neighbors = self.neighbor_info[vert]
			self.create_wall_triangle(vert)
			vert_weights_to_remove = [ vert, neighbors.left, neighbors.right ]
			triangle_weights = list(filter(lambda tw: tw.vert not in vert_weights_to_remove, triangle_weights))
			if self.is_triangle_valid(neighbors.left):
				add_weight(neighbors.left)
			if self.is_triangle_valid(neighbors.right):
				add_weight(neighbors.right)
		
		if len(self.edge_verts) >= 3:
			print(f"WARNING: error building wall, still {len(self.edge_verts)} edge verts remaining")
		
		self.wall_triangles_side1.extend(self.wall_triangles)
		reversed_triangles = list(map(lambda t: list(reversed(list(t))), self.wall_triangles))
		self.wall_triangles_side2.extend(reversed_triangles)



print("generating wall triangles and nubs")
for plane_i in range(0, len(plane_points)):
	print(f"FOR PLANE {plane_i}:")
	builder = WallBuilder(plane_points[plane_i], all_plane_edge_vertices[plane_i], all_plane_edge_triangles[plane_i])
	if should_generate_nubs:
		builder.generate_all_nubs()
	builder.generate_triangle_wall()
	all_chunk_triangles[plane_i].extend(builder.wall_triangles_side1)
	all_chunk_triangles[plane_i + 1].extend(builder.wall_triangles_side2)

# print("checking for duplicate verts in triangles")
# for triangle in all_chunk_triangles[0]:
# 	for i in triangle:
# 		if triangle.count(i) > 1:
# 			print(triangle)

print("filtering vertices for each chunk")
final_triangles = []
final_vertices = []
for chunk_i in range(0, chunk_amount):
	final_chunk_vertices = []
	final_chunk_triangles = []
	vertex_map = {}
	for triangle in all_chunk_triangles[chunk_i]:
		new_triangle = []
		for vert in triangle:
			if vert not in vertex_map:
				vertex_map[vert] = len(final_chunk_vertices)
				final_chunk_vertices.append(vertices[vert])
			new_triangle.append(vertex_map[vert])
		final_chunk_triangles.append(new_triangle)
	final_vertices.append(final_chunk_vertices)
	final_triangles.append(final_chunk_triangles)


print("exporting chunks to objs")
for i in range(0, chunk_amount):
	obj_path = f"river_{i}.obj"
	print(f"- writing to {obj_path}")
	dump_to_obj(obj_path, final_vertices[i], final_triangles[i])

os.startfile("river_0.obj")