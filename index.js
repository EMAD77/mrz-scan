// index.js
"use strict";

// Import the main MRZ scanning function:
const detectAndParseMrz = require("./build/detect-and-parse.js");
// Import the svm module which exports setModelPaths:
const svm = require("./build/svm.js");

// Attach the setModelPaths function onto the main export.
detectAndParseMrz.setModelPaths = svm.setModelPaths;

// Export both the main function and named functions for tree-shaking friendly imports
module.exports = detectAndParseMrz;
// For ESM and tree-shaking friendly imports - not needed for your current setup
// but might be useful for future compatibility
module.exports.setModelPaths = svm.setModelPaths;
