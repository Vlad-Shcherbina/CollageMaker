import Image
import numpy


def naive_downscale(arr, new_w, new_h):
    result = numpy.zeros((new_h, new_w), dtype=int)
    h, w = arr.shape
    assert new_w <= w
    assert new_h <= h
    for y in range(new_h):
        for x in range(new_w):
            s = 0
            for yy in range(h):
                for xx in range(w):
                    s += arr[(y * h + yy) // new_h, (x * w + xx) // new_w]
            result[y, x] = (s + h * w // 2) // (h * w)
    return result


def downscale(arr, new_w, new_h):
    h, w = arr.shape
    cum = numpy.zeros((h + 1, w + 1), dtype=int)
    cum[1:, 1:] = numpy.cumsum(numpy.cumsum(arr, axis=0), axis=1)
    print cum
    result = numpy.zeros((new_h, new_w), dtype=int)
    for y in range(new_h):
        for x in range(new_w):
            i1 = y * h // new_h + 1
            i2 = (y + 1) * h // new_h
            j1 = x * w // new_w + 1
            j2 = (x + 1) * w // new_w
            s = 0
            s += (cum[i2, j2] - cum[i1, j2] - cum[i2, j1] + cum[i1, j1]) * new_w * new_h

            s += (i1 * new_h - y * h) * (j1 * new_w - x * w) * arr[i1 - 1, j1 - 1]
            # TODO

            result[y, x] = (s + h * w // 2) // (w * h)

    return result


if __name__ == '__main__':
    img = Image.open('data/100px/1.png')
    print img
    arr = numpy.array(img.getdata()).reshape(img.size[1], img.size[0])
    print arr #[::10, ::10]
    print arr.shape
    #print downscale(arr, 10, 10)
    #print arr.sum()

    #arr = downscale(arr, 30, 30)
    #exit()
    arr = downscale(arr, 10, 10)
    print arr
    img = Image.fromarray(arr.astype(numpy.uint8))
    img.save("hz.png")
