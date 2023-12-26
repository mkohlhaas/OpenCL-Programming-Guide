kernel void convolve(const global   uint *const input,
                           constant uint *const mask,
                           global   uint *const output,
                     const          int         inputWidth,
                     const          int         maskWidth) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int columns = get_global_size(0);

  uint sum = 0;
  for (int maskY = 0; maskY < maskWidth; maskY++) {
    for (int maskX = 0; maskX < maskWidth; maskX++) {
      sum += input[(y + maskY) * inputWidth + x + maskX] * mask[maskY * maskWidth + mx];
    }
  }
  output[y * columns + x] = sum;
}
