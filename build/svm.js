'use strict';
const path = require('path');
const fs = require('node:fs/promises');

const groupBy = require('lodash.groupby');
const hog = require('hog-features');
const Kernel = require('ml-kernel');
const range = require('lodash.range');
const uniq = require('lodash.uniq');
const BSON = require('bson');

const SVM = require('libsvm-js/asm');
// const SVMPromise = Promise.resolve(require('libsvm-js/asm'));
const { readImages } = require('./util/readWrite');

// ---------------------------------------------------------------------
// NEW: Modify getFilePath to accept external paths
// ---------------------------------------------------------------------
function getFilePath(externalPaths) {
  // If an external paths object is provided with both keys, use it.
  if (
    externalPaths &&
    externalPaths.descriptors &&
    externalPaths.model
  ) {
    return {
      descriptors: externalPaths.descriptors,
      model: externalPaths.model,
    };
  }
  // Fall back to the default paths otherwise.
  return {
    descriptors: require.resolve('mrz-scan/models/ESC-v2.svm.descriptors'),
    model: require.resolve('mrz-scan/models/ESC-v2.svm.model'),
  };
}

// ---------------------------------------------------------------------
// Rest of your code remains mostly unchanged...
// ---------------------------------------------------------------------

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

  let optionsHog = {
    cellSize: 5,
    blockSize: 2,
    blockStride: 1,
    bins: 4,
    norm: 'L2',
  };

  let hogFeatures = hog.extractHOG(image, optionsHog);
  return hogFeatures;
}

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
  // In the next line, we could optionally pass external model paths if needed.
  // For now, we call applyModel with only one argument.
  return applyModel(Xtest);
}

function predict(classifier, Xtrain, Xtest, kernelOptions) {
  const kernel = getKernel(kernelOptions);
  const Ktest = kernel
    .compute(Xtest, Xtrain)
    .addColumn(0, range(1, Xtest.length + 1));
  return classifier.predict(Ktest);
}

// ---------------------------------------------------------------------
// Modified: allow externalPaths to be passed to applyModel
// ---------------------------------------------------------------------
async function applyModel(Xtest, externalPaths) {
  const { descriptors: descriptorsPath, model: modelPath } =
    getFilePath(externalPaths);

  const bson = new BSON();
  const file = await fs.readFile(descriptorsPath);
  const { descriptors: Xtrain, kernelOptions } = bson.deserialize(file);
  const model = await fs.readFile(modelPath, { encoding: 'utf8' });
  const classifier = await SVM.load(model);
  const prediction = predict(classifier, Xtrain, Xtest, kernelOptions);
  return prediction;
}

// ---------------------------------------------------------------------
// Modified: allow externalPaths to be passed to createModel
// ---------------------------------------------------------------------
async function createModel(letters, name, SVMOptions, kernelOptions, externalPaths) {
  const { descriptors: descriptorsPath, model: modelPath } =
    getFilePath(externalPaths);
  const { descriptors, classifier } = await train(
    letters,
    SVMOptions,
    kernelOptions
  );
  const bson = new BSON();
  try {
    await fs.writeFile(
      descriptorsPath,
      bson.serialize({ descriptors, kernelOptions })
    );
    await fs.writeFile(modelPath, classifier.serializeModel());
  } catch (e) {
    console.log(e);
  }
}

async function train(letters, SVMOptions, kernelOptions) {
  let SVMOptionsOneClass = {
    type: SVM.SVM_TYPES.ONE_CLASS,
    kernel: SVM.KERNEL_TYPES.PRECOMPUTED,
    nu: 0.5,
    quiet: true,
  };

  let SVMNormalOptions = {
    type: SVM.SVM_TYPES.C_SVC,
    kernel: SVM.KERNEL_TYPES.PRECOMPUTED,
    gamma: 1,
    quiet: true,
  };

  const Xtrain = letters.map((s) => s.descriptor);
  const Ytrain = letters.map((s) => s.label);
  const uniqLabels = uniq(Ytrain);
  if (uniqLabels.length === 1) {
    console.log('training mode: ONE_CLASS');
    SVMOptions = Object.assign({}, SVMOptionsOneClass, SVMOptions, {
      kernel: SVM.KERNEL_TYPES.PRECOMPUTED,
    });
  } else {
    SVMOptions = Object.assign({}, SVMNormalOptions, SVMOptions, {
      kernel: SVM.KERNEL_TYPES.PRECOMPUTED,
    });
  }
  let oneClass = SVMOptions.type === SVM.SVM_TYPES.ONE_CLASS;
  var classifier = new SVM(SVMOptions);
  let kernel = getKernel(kernelOptions);
  const KData = kernel
    .compute(Xtrain)
    .addColumn(0, range(1, Ytrain.length + 1));
  classifier.train(KData, Ytrain);
  return { classifier, descriptors: Xtrain, oneClass };
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
};
