import math
import typing
import numpy as np

class Vertex():
	x: float
	y: float
	z: float

	def __init__(self, x=0, y=0, z=0):
		self.x = x
		self.y = y
		self.z = z

	def toJson(self):
		return {
			"x": float(self.x),
			"y": float(self.y),
			"z": float(self.z)
		}

	def toNumpy(self):
		return np.array([
			self.x,
			self.y,
			self.z
		])
	
	def __repr__(self):
		return f"(x: {self.x} y: {self.y} z: {self.z})"


def point_distance(p1, p2):
	if isinstance(p1, Vertex):
		p1 = { "x": p1.x, "y": p1.y }
	if isinstance(p2, Vertex):
		p2 = { "x": p2.x, "y": p2.y }
	a = p2["x"] - p1["x"]
	b = p2["y"] - p1["y"]
	return math.sqrt(a * a + b * b)

# haversine formula: gets the distance (in meters) between 2 latlong points
def lat_long_distance(lat1, lon1, lat2, lon2):
	radius_of_earth_meters = 6378100 # meters
	# convert decimal degrees to radians 
	lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
	# haversine formula 
	dlon = lon2 - lon1 
	dlat = lat2 - lat1 
	a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
	c = 2 * math.asin(math.sqrt(a)) 
	return c * radius_of_earth_meters


def dump_to_obj(filename, vertices, triangles):
	lines = []
	lines.append(f"# {filename}")
	lines.append("#")
	lines.append("")
	lines.append("g river")
	lines.append("")
	if isinstance(vertices[0], Vertex):
		lines.extend(map(lambda v: f"v {v.x} {v.y} {v.z}", vertices))
	else:
		lines.extend(map(lambda v: f"v {v['x']} {v['y']} {v['z']}", vertices))
	lines.append("")
	lines.extend(map(lambda t: f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}", triangles))
	text = "\n".join(lines)
	with open(filename, "w+") as f:
		f.write(text)


# https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
def isect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """

    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(p_no, u)

    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = sub_v3v3(p0, p_co)
        fac = -dot_v3v3(p_no, w) / dot
        u = mul_v3_fl(u, fac)
        return add_v3v3(p0, u)

    # The segment is parallel to plane.
    return None

# ----------------------
# generic math functions

def add_v3v3(v0, v1):
    return (
        v0[0] + v1[0],
        v0[1] + v1[1],
        v0[2] + v1[2],
    )


def sub_v3v3(v0, v1):
    return (
        v0[0] - v1[0],
        v0[1] - v1[1],
        v0[2] - v1[2],
    )


def dot_v3v3(v0, v1):
    return (
        (v0[0] * v1[0]) +
        (v0[1] * v1[1]) +
        (v0[2] * v1[2])
    )


def len_squared_v3(v0):
    return dot_v3v3(v0, v0)


def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
    )