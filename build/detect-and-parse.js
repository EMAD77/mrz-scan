const ImageClass = require('image-js').Image;
const getMrz = require('./getMrz');
const mrzOcr = require('./internal/mrzOcr');
const {
  DateTime: luxon
} = require('luxon');
const parse = require('./mrz-relax');
const getOptions = value => typeof value === 'object' && !Array.isArray(value) && value !== null ? value : {};
module.exports = async function detectAndParseMrz(buffer, options) {
  try {
    const {
      original
    } = getOptions(options);
    const mrz = await getMrz(await ImageClass.load(buffer));
    const imageDataUrl = mrz.toDataURL();
    const toImage = await ImageClass.load(imageDataUrl);
    var {
      ocrResult
    } = await mrzOcr(toImage);
    const parsed = parse(ocrResult);
    const formattedResult = original ? parsed : {
      number: parsed.fields.documentNumber,
      validDate: luxon.fromFormat(parsed.fields.expirationDate, 'yyMMdd').toISODate(),
      birthDate: luxon.fromFormat(parsed.fields.birthDate, 'yyMMdd').toISODate(),
      name: parsed.fields.firstName.replace(/\s+/g, '').trim(),
      surname: parsed.fields.lastName.replace(/\s+/g, '').trim()
    };
    return formattedResult;
  } catch (e) {
    console.log(e);
  }
};