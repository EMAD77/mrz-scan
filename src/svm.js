'use strict';
const path = require('path');
const fs = require('fs');
const fsPromises = require('node:fs/promises');

const groupBy = require('lodash.groupby');
const hog = require('hog-features');
const Kernel = require('ml-kernel');
const range = require('lodash.range');
const uniq = require('lodash.uniq');
const BSON = require('bson');
const SVM = require('libsvm-js/asm');
const { readImages } = require('./util/readWrite');

// Global config for model paths (if set externally)
let externalModelPaths = null;

/**
 * Inject external model paths.
 * @param {Object} models - Should have properties: { descriptors, model }
 */
function setModelPaths(models) {
  if (
    !models ||
    typeof models !== 'object' ||
    !models.descriptors ||
    !models.model
  ) {
    throw new Error(
      'Model paths must be an object with both "descriptors" and "model" properties'
    );
  }
  externalModelPaths = models;
}

/**
 * Asynchronously load training data from a directory.
 * @param {string} dir - Directory path (relative to project root)
 * @returns {Promise<Array>} - Array of data objects
 */
async function loadData(dir) {
  const data = await readImages(path.resolve(path.join(__dirname, '..'), dir));

  for (let entry of data) {
    let { image } = entry;
    entry.descriptor = extractHOG(image);
    entry.height = image.height;
  }

  const groupedData = groupBy(data, (d) => d.card);
  for (let card in groupedData) {
    const heights = groupedData[card].map((d) => d.height);
    const maxHeight = Math.max(...heights);
    const minHeight = Math.min(...heights);
    for (let d of groupedData[card]) {
      // This bonus feature is important to differentiate numbers and letters.
      let bonusFeature = 1;
      if (minHeight !== maxHeight) {
        bonusFeature = (d.height - minHeight) / (maxHeight - minHeight);
      }
      d.descriptor.push(bonusFeature);
    }
  }
  return data;
}

/**
 * Extract Histogram of Oriented Gradients (HOG) features from an image.
 * @param {Object} image - The image object from image-js.
 * @returns {Array<number>} - HOG descriptor array.
 */
function extractHOG(image) {
  image = image.scale({ width: 20, height: 20 });
  image = image.pad({ size: 2 });
  const optionsHog = {
    cellSize: 5,
    blockSize: 2,
    blockStride: 1,
    bins: 4,
    norm: 'L2'
  };
  const hogFeatures = hog.extractHOG(image, optionsHog);
  return hogFeatures;
}

/**
 * Compute descriptors for an array of images.
 * @param {Array<Object>} images - Array of image objects.
 * @returns {Array<Array<number>>} - Array of descriptor arrays.
 */
function getDescriptors(images) {
  const result = [];
  for (let image of images) {
    result.push(extractHOG(image));
  }

  const heights = images.map((img) => img.height);
  const maxHeight = Math.max(...heights);
  const minHeight = Math.min(...heights);
  for (let i = 0; i < images.length; i++) {
    const img = images[i];
    let bonusFeature = 1;
    if (minHeight !== maxHeight) {
      bonusFeature = (img.height - minHeight) / (maxHeight - minHeight);
    }
    result[i].push(bonusFeature);
  }
  return result;
}

/**
 * Predict MRZ characters for an array of images.
 * @param {Array<Object>} images - Array of image objects.
 * @returns {Array<number>} - Prediction results.
 */
function predictImages(images) {
  const Xtest = getDescriptors(images);
  return applyModel(Xtest);
}

/**
 * Predict helper: Given an SVM classifier and training data.
 * @param {Object} classifier - SVM classifier.
 * @param {Array} Xtrain - Training descriptors.
 * @param {Array} Xtest - Test descriptors.
 * @param {Object} kernelOptions - Options for the kernel.
 * @returns {Array<number>} - Predictions.
 */
function predict(classifier, Xtrain, Xtest, kernelOptions) {
  const kernel = getKernel(kernelOptions);
  const Ktest = kernel
    .compute(Xtest, Xtrain)
    .addColumn(0, range(1, Xtest.length + 1));
  return classifier.predict(Ktest);
}

/**
 * Apply a trained SVM model to test descriptors.
 * @param {Array<Array<number>>} Xtest - Test descriptors.
 * @returns {Promise<Array<number>>} - Prediction result.
 */
async function applyModel(Xtest) {
  const { descriptors: descriptorsPath, model: modelPath } = getFilePath();

  try {
    const bson = new BSON();
    const file = await fsPromises.readFile(descriptorsPath);
    const { descriptors: Xtrain, kernelOptions } = bson.deserialize(file);

    const model = await fsPromises.readFile(modelPath, {
      encoding: 'utf8'
    });
    const classifier = await SVM.load(model);

    const prediction = predict(classifier, Xtrain, Xtest, kernelOptions);
    return prediction;
  } catch (error) {
    const errorInfo = `Error loading model files. Tried paths:
- descriptors: ${descriptorsPath}
- model: ${modelPath}
Original error: ${error.message}`;
    console.error(errorInfo);
    throw new Error(errorInfo);
  }
}

/**
 * Create (train) a new SVM model from provided letters.
 * @param {Array<Object>} letters - Array of training images (with descriptors & labels)
 * @param {Object} SVMOptions - Options for the SVM.
 * @param {Object} kernelOptions - Options for the kernel.
 * @returns {Promise<void>}
 */
async function createModel(letters, name, SVMOptions, kernelOptions) {
  const { descriptors: descriptorsPath, model: modelPath } = getFilePath();
  const { descriptors, classifier } = await train(letters, SVMOptions, kernelOptions);
  const bson = new BSON();

  try {
    await fsPromises.writeFile(
      descriptorsPath,
      bson.serialize({ descriptors, kernelOptions })
    );
    await fsPromises.writeFile(modelPath, classifier.serializeModel());
  } catch (e) {
    console.log(e);
  }
}

/**
 * Train an SVM model.
 * @param {Array<Object>} letters - Array of training images.
 * @param {Object} SVMOptions - Options for the SVM.
 * @param {Object} kernelOptions - Options for the kernel.
 * @returns {Promise<Object>} - Contains classifier, descriptors, and oneClass flag.
 */
async function train(letters, SVMOptions, kernelOptions) {
  const SVMOptionsOneClass = {
    type: SVM.SVM_TYPES.ONE_CLASS,
    kernel: SVM.KERNEL_TYPES.PRECOMPUTED,
    nu: 0.5,
    quiet: true
  };

  const SVMNormalOptions = {
    type: SVM.SVM_TYPES.C_SVC,
    kernel: SVM.KERNEL_TYPES.PRECOMPUTED,
    gamma: 1,
    quiet: true
  };

  const Xtrain = letters.map((s) => s.descriptor);
  const Ytrain = letters.map((s) => s.label);

  const uniqLabels = uniq(Ytrain);

  if (uniqLabels.length === 1) {
    // eslint-disable-next-line no-console
    console.log('training mode: ONE_CLASS');
    SVMOptions = Object.assign({}, SVMOptionsOneClass, SVMOptions, {
      kernel: SVM.KERNEL_TYPES.PRECOMPUTED
    });
  } else {
    SVMOptions = Object.assign({}, SVMNormalOptions, SVMOptions, {
      kernel: SVM.KERNEL_TYPES.PRECOMPUTED
    });
  }

  const oneClass = SVMOptions.type === SVM.SVM_TYPES.ONE_CLASS;
  const classifier = new SVM(SVMOptions);
  const kernel = getKernel(kernelOptions);

  const KData = kernel
    .compute(Xtrain)
    .addColumn(0, range(1, Ytrain.length + 1));
  classifier.train(KData, Ytrain);
  return { classifier, descriptors: Xtrain, oneClass };
}

/**
 * Determine file paths for descriptors and model files.
 * @returns {Object} - { descriptors: string, model: string }
 */
function getFilePath() {
  if (
    externalModelPaths &&
    externalModelPaths.descriptors &&
    externalModelPaths.model
  ) {
    return externalModelPaths;
  }

  // Check for production environment.
  const prodPath = '/var/task/static/mrz-models';
  if (fs.existsSync(path.join(prodPath, 'ESC-v2.svm.descriptors'))) {
    return {
      descriptors: path.join(prodPath, 'ESC-v2.svm.descriptors'),
      model: path.join(prodPath, 'ESC-v2.svm.model')
    };
  }

  // Fallback for local development.
  const localPath = path.join(process.cwd(), 'public/mrz-models');
  if (fs.existsSync(path.join(localPath, 'ESC-v2.svm.descriptors'))) {
    return {
      descriptors: path.join(localPath, 'ESC-v2.svm.descriptors'),
      model: path.join(localPath, 'ESC-v2.svm.model')
    };
  }

  throw new Error('MRZ model files not found in any expected locations');
}

/**
 * Create and return an instance of the Kernel.
 * @param {Object} options - Options for the kernel.
 * @returns {Kernel} - Kernel instance.
 */
function getKernel(options) {
  options = Object.assign({ type: 'linear' }, options);
  return new Kernel(options.type, options);
}

/**
 * Default MRZ scanner function.
 * Expects a Buffer (e.g. from reading an image file) and options.
 * This function loads the image (using image-js) and passes it for prediction.
 * @param {Buffer} buffer - Image buffer.
 * @param {Object} [options] - Optional MRZ scanning options.
 * @returns {Promise<Array<number>>} - Predictions.
 */
function mrzScanner(buffer, options) {
  const Image = require('image-js').Image;
  return Image.load(buffer).then((image) => {
    // For demonstration, predict on a single image.
    return predictImages([image]);
  });
}

// Attach helper methods as properties on the main function.
mrzScanner.applyModel = applyModel;
mrzScanner.createModel = createModel;
mrzScanner.train = train;
mrzScanner.predict = predict;
mrzScanner.extractHOG = extractHOG;
mrzScanner.predictImages = predictImages;
mrzScanner.loadData = loadData;
mrzScanner.setModelPaths = setModelPaths;

module.exports = mrzScanner;
