import numpy
import random

import img_lib


def verify_downscaling(w, h, new_w, new_h):
  random.seed(42)
  arr = numpy.zeros((h, w), dtype=int)
  for i in range(h):
    for j in range(w):
      arr[i, j] = random.randrange(256)

  naive_downscaled = img_lib.naive_downscale(arr, new_w, new_h)
  elementwise_downscaled = img_lib.ScalableImage(arr).elementwise_downscale(new_w, new_h)
  downscaled = img_lib.ScalableImage(arr).downscale(new_w, new_h)
  assert (naive_downscaled == downscaled).all()
  assert (elementwise_downscaled == downscaled).all()


def test_downscaling():
  yield verify_downscaling, 10, 10, 5, 5
  yield verify_downscaling, 17, 11, 17, 11
  yield verify_downscaling, 31, 19, 7, 9
  yield verify_downscaling, 40, 50, 1, 1
  yield verify_downscaling, 15, 20, 14, 19
