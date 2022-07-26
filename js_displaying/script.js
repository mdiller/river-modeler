

var tiles = {
	"tiles": [
		{
			"data": [
				[1,2,5,5,5,5,5,5,5,2,1,2,5,5,5],
				[2,1,2,5,5,5,5,5,2,1,2,5,5,5,5],
				[5,2,1,2,5,5,5,2,1,2,5,5,5,5,5],
				[5,5,2,1,2,2,2,1,2,5,5,5,5,5,5],
				[5,8,5,2,1,1,1,2,5,8,5,5,5,5,5],
				[5,5,8,5,2,2,2,5,8,5,8,5,5,5,5],
				[5,5,5,8,8,5,5,5,5,8,5,5,5,5,5],
				[5,5,5,8,8,5,5,5,5,5,5,5,5,5,5],
				[5,5,5,8,8,5,5,5,5,5,5,5,5,5,5],
				[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
				[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
				[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
				[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
				[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
				[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
			]
		}
	]
}


function hexToRgb(hex) {
	var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
	return result ? {
		r: parseInt(result[1], 16),
		g: parseInt(result[2], 16),
		b: parseInt(result[3], 16)
	} : null;
}

function rgbToHex(rgb) {
	function componentToHex(c) {
		var hex = c.toString(16);
		return hex.length == 1 ? "0" + hex : hex;
	}
	return "#" + componentToHex(rgb.r) + componentToHex(rgb.g) + componentToHex(rgb.b);
}

// gradient is an array of colors representing the gradient. ie: [ "#ff0000", "#00ff00", "#0000ff" ]
function getColorFromGradient(percent, gradient) {
	if (typeof gradient === 'string') {
		gradient = [ gradient ];
	}
	if (gradient.length == 1) {
		gradient = [ gradient[0], gradient[0] ]
	}
	var index = Math.floor((percent * (gradient.length - 1)) - 0.0001);
	var min_color_rgb = hexToRgb(gradient[index]);
	var max_color_rgb = hexToRgb(gradient[index + 1]);
	var adjusted_percent = (percent - index * (1 / (gradient.length - 1))) * (gradient.length - 1);

	var color = {
		r: parseInt((max_color_rgb.r * adjusted_percent) + (min_color_rgb.r * (1 - adjusted_percent))),
		g: parseInt((max_color_rgb.g * adjusted_percent) + (min_color_rgb.g * (1 - adjusted_percent))),
		b: parseInt((max_color_rgb.b * adjusted_percent) + (min_color_rgb.b * (1 - adjusted_percent))),
	}
	return rgbToHex(color);
}


const COOL_GRADIENT = [ "#0000ff", "#00ff00", "#ffffff" ];


// document.getElementById("bitsoutput").innerHTML = bitsSet.join(", ");

// make pixel display thing
// make mesh display thing


var pixel_size = 20;
// draws the pixels on the given canvas, respecting the given bounds
function drawPixelCanvas(canvas, pixels, width, height) {
	var dimension = Math.max(width, height) * pixel_size;
	canvas.height = dimension;
	canvas.width = dimension;
	var ctx = canvas.getContext("2d");
	pixels.forEach(pixel => {
		ctx.fillStyle = pixel.color;
		ctx.fillRect(pixel.x * pixel_size, pixel.y * pixel_size, pixel_size, pixel_size);
	})
}

var divcontainer = document.getElementById("container");
var canvas = document.getElementById("thecanvas");

var pixels = [];
var data = tiles.tiles[0].data;
var max_z = Math.max.apply(null, data.map(row => Math.max.apply(Math, row)));
var width = data[0].length; // assume uniform row lengths
var height = data.length;
for (var i = 0; i < data.length; i++) {
	for (var j = 0; j < data[i].length; j++) {
		pixels.push({
			x: j,
			y: i,
			color: getColorFromGradient(data[i][j] / max_z, COOL_GRADIENT)
		})
	}
}

// drawPixelCanvas(canvas, pixels, width, height);


// CREATE MAH MESH
var vertices = [];
for (var i = 0; i < data.length; i++) {
	for (var j = 0; j < data[i].length; j++) {
		vertices.push({
			x: j,
			y: i,
			z: data[i][j]
		})
	}
}
function coordToVertIndex(x, y) {
	return (y * data.length) + x;
}
var triangles = [];
for (var y = 0; y < data.length - 1; y++) {
	for (var x = 0; x < data[y].length - 1; x++) {
		triangles.push([
			coordToVertIndex(x, y),
			coordToVertIndex(x + 1, y),
			coordToVertIndex(x, y + 1)
		])
		triangles.push([
			coordToVertIndex(x + 1, y),
			coordToVertIndex(x + 1, y + 1),
			coordToVertIndex(x, y + 1)
		])
	}
}


// CREATE OBJ FILE
var lines = [];
lines.push("# river.obj");
lines.push("#");
lines.push("");
lines.push("g river");
lines.push("");
vertices.forEach(v => lines.push(`v ${v.x} ${v.y} ${v.z}`));
lines.push("");
triangles.forEach(t => lines.push(`f ${t[0] + 1} ${t[1] + 1} ${t[2] + 1}`));

var objtext = lines.join("\n");
console.log(objtext);


import * as modelPlayer from "js-3d-model-viewer"
const viewerElement = document.getElementById("meshviewer")
const opts = {
  grid: true,
  trackball: false
}
const scene = modelPlayer.prepareScene(viewerElement, opts)
modelPlayer.loadObject(scene, "./river.obj") // Urls are fine here.