// index.js
"use strict";

// Import the main MRZ scanning function:
const detectAndParseMrz = require("./build/detect-and-parse.js");
// Import the svm module which exports setModelPaths:
const svm = require("./build/svm.js");

// Attach the setModelPaths function onto the main export.
detectAndParseMrz.setModelPaths = svm.setModelPaths;

module.exports = detectAndParseMrz;
