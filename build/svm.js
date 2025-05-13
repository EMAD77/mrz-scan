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

// Store global config that can be set by the user
let modelPathsConfig = null;

let externalModelPaths = null;

/**
 * Inject external model paths.
 * @param {Object} models - Should have { descriptors, model } properties.
 */
function setModelPaths(models) {
  externalModelPaths = models;
}

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
            // This last descriptor is very important to differentiate numbers and letters
            // Because with OCR-B font, numbers are slightly higher than numbers
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
    norm: 'L2'
  };
  let hogFeatures = hog.extractHOG(image, optionsHog);
  return hogFeatures;
}

// Get descriptors for images from 1 identity card
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
        // Add more detailed error information
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
        await fsPromises.writeFile(descriptorsPath, bson.serialize({
            descriptors,
            kernelOptions
        }));
        await fsPromises.writeFile(modelPath, classifier.serializeModel());
    } catch (e) {
        console.log(e);
    }
}

async function train(letters, SVMOptions, kernelOptions) {
    let SVMOptionsOneClass = {
        type: SVM.SVM_TYPES.ONE_CLASS,
        kernel: SVM.KERNEL_TYPES.PRECOMPUTED,
        nu: 0.5,
        quiet: true
    };

    let SVMNormalOptions = {
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

    let oneClass = SVMOptions.type === SVM.SVM_TYPES.ONE_CLASS;

    var classifier = new SVM(SVMOptions);
    let kernel = getKernel(kernelOptions);

    const KData = kernel
        .compute(Xtrain)
        .addColumn(0, range(1, Ytrain.length + 1));
    classifier.train(KData, Ytrain);
    return {
        classifier,
        descriptors: Xtrain,
        oneClass
    };
}

/**
 * Set custom model file paths for the SVM
 * @param {Object} paths - Object containing descriptors and model paths
 * @param {string} paths.descriptors - Path to the descriptors file
 * @param {string} paths.model - Path to the model file
 */
// function setModelPaths(paths) {
//     if (!paths || typeof paths !== 'object') {
//         throw new Error('Model paths must be an object with descriptors and model properties');
//     }
    
//     if (!paths.descriptors || !paths.model) {
//         throw new Error('Both descriptors and model paths are required');
//     }
    
//     modelPathsConfig = {
//         descriptors: paths.descriptors,
//         model: paths.model
//     };
// }

function getFilePath() {
  // Use externally provided model paths if available.
  if (
    externalModelPaths &&
    externalModelPaths.descriptors &&
    externalModelPaths.model
  ) {
    return externalModelPaths;
  }

  // Check for production environment.
  // (Your error shows the file is expected under "/var/task/static" in production.)
  const prodPath = '/var/task/static/mrz-models';
  if (fs.existsSync(path.join(prodPath, 'ESC-v2.svm.descriptors'))) {
    return {
      descriptors: path.join(prodPath, 'ESC-v2.svm.descriptors'),
      model: path.join(prodPath, 'ESC-v2.svm.model')
    };
  }

  // Fallback for local development — use the public folder.
  const localPath = path.join(process.cwd(), 'public/mrz-models');
  if (fs.existsSync(path.join(localPath, 'ESC-v2.svm.descriptors'))) {
    return {
      descriptors: path.join(localPath, 'ESC-v2.svm.descriptors'),
      model: path.join(localPath, 'ESC-v2.svm.model')
    };
  }

  throw new Error('MRZ model files not found in any expected locations');
}

function getKernel(options) {
    options = Object.assign({
        type: 'linear'
    }, options);
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
    setModelPaths, // Export the new function to set custom paths
};
