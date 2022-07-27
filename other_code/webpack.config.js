const path = require("path");

module.exports = {
	entry: "./script.js",
	devServer: {
	  static: {
		directory: path.join(__dirname, "public"),
	  },
	  compress: true,
	  port: 9000,
	},
	mode: "development",
	output: {
		filename: "script.js",
		path: path.resolve(__dirname, "dist")
	}
  };