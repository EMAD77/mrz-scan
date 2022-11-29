'use strict';
module.exports = function groupRoisPerLine(rois) {
  const roisCopy = rois.slice();
  let total = roisCopy.reduce((a, b) => a + b.height, 0);
  const allowedShift = Math.round(total / roisCopy.length / 2);
  roisCopy.sort(function (a, b) {
    return a.minX - b.minX;
  });
  const lines = [];
  for (const roi of rois) {
    const x = roi.minX;
    const y = roi.minY;
    let currentLine;
    for (const line of lines) {
      if (Math.abs((line.y || 0) - y) <= allowedShift) {
        currentLine = line;
        break;
      }
    }
    if (!currentLine) {
      currentLine = {
        rois: []
      };
      lines.push(currentLine);
    }
    currentLine.y = y;
    currentLine.x = x;
    currentLine.rois.push(roi);
  }
  lines.sort((a, b) => Number(a.y) - Number(b.y));
  return lines;
};