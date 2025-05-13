// index.js
"use strict";

// Import the main function and the SVM module (which now includes setModelPaths)
const detectAndParseMrz = require("./build/detect-and-parse.js");
const svm = require("./build/svm.js");

// Attach the setModelPaths function to the exported function.
detectAndParseMrz.setModelPaths = svm.setModelPaths;

module.exports = detectAndParseMrz;
