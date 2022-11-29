const swap = (arr, x, y) => [arr[x], arr[y]] = [arr[y], arr[x]];
const calcMiddle = (x, y) => ~~((x + y) / 2);
function quickSelect(arr) {
  let low = 0;
  let high = arr.length - 1;
  let middle, ll, hh;
  let median = calcMiddle(low, high);
  do {
    if (high <= low) {
      return arr[median];
    }
    if (high == low + 1) {
      if (arr[low] > arr[high]) swap(arr, low, high);
      return arr[median];
    }
    middle = calcMiddle(low, high);
    if (arr[middle] > arr[high]) swap(arr, middle, high);
    if (arr[low] > arr[high]) swap(arr, low, high);
    if (arr[middle] > arr[low]) swap(arr, middle, low);
    swap(arr, middle, low + 1);
    ll = low + 1;
    hh = high;
    while (true) {
      do ll++; while (arr[low] > arr[ll]);
      do hh--; while (arr[hh] > arr[low]);
      if (hh < ll) break;
      swap(arr, ll, hh);
    }
    swap(arr, low, hh);
    if (hh <= median) {
      low = ll;
    }
    if (hh >= median) {
      high = hh - 1;
    }
  } while (true);
}
module.exports = quickSelect;