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

// Global variable to hold externally set model paths
let externalModelPaths = null;

/**
 * Inject external model paths.
 * @param {Object} models - Should have properties:
 *   - descriptors: Path to the descriptors file
 *   - model: Path to the model (classifier) file
 */
function setModelPaths(models) {
  externalModelPaths = models;
}

async function loadData(dir) {
  const data = await readImages(path.resolve(path.join(__dirname, '..'), dir));

  for (let entry of data) {
    const { image } = entry;
    entry.descriptor = extractHOG(image);
    entry.height = image.height;
  }

  const groupedData = groupBy(data, (d) => d.card);
  for (let card in groupedData) {
    const heights = groupedData[card].map((d) => d.height);
    const maxHeight = Math.max.apply(null, heights);
    const minHeight = Math.min.apply(null, heights);
    for (let d of groupedData[card]) {
      let bonusFeature = 1;
      if (minHeight !== maxHeight) {
        bonusFeature = (d.height - minHeight) / (maxHeight - minHeight);
      }
      d.descriptor.push(bonusFeature);
    }
  }
  return data;
}

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

// Get descriptors from an array of images
function getDescriptors(images) {
  const result = [];
  for (let image of images) {
    result.push(extractHOG(image));
  }

  const heights = images.map((img) => img.height);
  const maxHeight = Math.max.apply(null, heights);
  const minHeight = Math.min.apply(null, heights);
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

function predictImages(images) {
  const Xtest = getDescriptors(images);
  return applyModel(Xtest);
}

function predict(classifier, Xtrain, Xtest, kernelOptions) {
  const kernel = getKernel(kernelOptions);
  const Ktest = kernel
    .compute(Xtest, Xtrain)
    .addColumn(0, range(1, Xtest.length + 1));
  return classifier.predict(Ktest);
}

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
    console.log('training mode: ONE_CLASS');
    SVMOptions = Object.assign({}, SVMOptionsOneClass, SVMOptions, {
      kernel: SVM.KERNEL_TYPES.PRECOMPUTED
    });
  } else {
    SVMOptions = Object.assign({}, SVMNormalOptions, SVMOptions, {
      kernel: SVM.KERNEL_TYPES.PRECOMPUTED
    });
  }

  const classifier = new SVM(SVMOptions);
  const kernel = getKernel(kernelOptions);
  const KData = kernel
    .compute(Xtrain)
    .addColumn(0, range(1, Ytrain.length + 1));
  classifier.train(KData, Ytrain);
  return {
    classifier,
    descriptors: Xtrain,
    oneClass: SVMOptions.type === SVM.SVM_TYPES.ONE_CLASS
  };
}

/**
 * Determines the file paths for the model files.
 * Priority:
 * 1. Use environment variables if set
 * 2. Use externally provided model paths if set via setModelPaths().
 * 3. Check for production files under /var/task/static/mrz-models.
 * 4. Fallback to local development files in public/mrz-models.
 */
function getFilePath() {
  // 1. Use environment variables if set
  if (process.env.MRZ_DESCRIPTORS_PATH && process.env.MRZ_MODEL_PATH) {
    return {
      descriptors: process.env.MRZ_DESCRIPTORS_PATH,
      model: process.env.MRZ_MODEL_PATH,
    };
  }

  // 2. Use externally provided paths if set via setModelPaths()
  if (externalModelPaths?.descriptors && externalModelPaths?.model) {
    return externalModelPaths;
  }

  // 3. Use production “public” path first
  const prodPath = '/var/task/public/mrz-models';
  const descriptorsProd = path.join(prodPath, 'ESC-v2.svm.descriptors');
  const modelProd = path.join(prodPath, 'ESC-v2.svm.model');
  if (fs.existsSync(descriptorsProd)) {
    return { descriptors: descriptorsProd, model: modelProd };
  }

  // 4. Fallback to local development files in public/mrz-models
  const localPath = path.join(process.cwd(), 'public/mrz-models');
  const descriptorsLocal = path.join(localPath, 'ESC-v2.svm.descriptors');
  const modelLocal = path.join(localPath, 'ESC-v2.svm.model');
  if (fs.existsSync(descriptorsLocal)) {
    return { descriptors: descriptorsLocal, model: modelLocal };
  }

  throw new Error('MRZ model files not found in any expected locations');
}


  // 4. Fallback to local development files in public/mrz-models
  const localPath = path.join(process.cwd(), 'public/mrz-models');
  const descriptorsLocal = path.join(localPath, 'ESC-v2.svm.descriptors');
  const modelLocal = path.join(localPath, 'ESC-v2.svm.model');
  if (fs.existsSync(descriptorsLocal)) {
    return { descriptors: descriptorsLocal, model: modelLocal };
  }

  throw new Error('MRZ model files not found in any expected locations');
}


function getKernel(options) {
  options = Object.assign({ type: 'linear' }, options);
  return new Kernel(options.type, options);
}

module.exports = {
  applyModel,
  createModel,
  train,
  predict,
  extractHOG,
  predictImages,
  loadData,
  setModelPaths
};
