import Image
import numpy


def load_image(path):
    img = Image.open(path)
    return numpy.array(img.getdata()).reshape(img.size[1], img.size[0])


def save_image(arr, path):
    Image.fromarray(arr.astype(numpy.uint8)).save(path)


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


class ScalableImage(object):
    def __init__(self, arr):
        self.arr = arr
        h, w = arr.shape

        self.cum = numpy.zeros((h + 1, w + 1), dtype=int)
        self.cum[1:, 1:] = numpy.cumsum(numpy.cumsum(arr, axis=0), axis=1)

    def rect_sum(self, i1, j1, i2, j2):
        cum = self.cum
        return cum[i2, j2] - cum[i1, j2] - cum[i2, j1] + cum[i1, j1]


    def downscale(self, new_w, new_h):
        arr = self.arr
        h, w = arr.shape
        assert 1 <= new_w <= w
        assert 1 <= new_h <= h

        result = numpy.zeros((new_h, new_w), dtype=int)
        for y in range(new_h):
            for x in range(new_w):
                i1 = y * h // new_h + 1
                i2 = ((y + 1) * h - 1) // new_h
                j1 = x * w // new_w + 1
                j2 = ((x + 1) * w - 1) // new_w
                assert y * h <= i1 * new_h
                assert i2 * new_h <= (y + 1) * h
                assert x * w <= j1 * new_w
                assert j2 * new_w <= (x + 1) * w

                # inner part
                s = self.rect_sum(i1, j1, i2, j2) * new_w * new_h

                # corners
                s += (i1 * new_h - y * h) * (j1 * new_w - x * w) * arr[i1 - 1, j1 - 1]
                s += (i1 * new_h - y * h) * ((x + 1) * w - j2 * new_w) * arr[i1 - 1, j2]
                s += ((y + 1) * h - i2 * new_h) * (j1 * new_w - x * w) * arr[i2, j1 - 1]
                s += ((y + 1) * h - i2 * new_h) * ((x + 1) * w - j2 * new_w) * arr[i2, j2]

                # edges
                s += (i1 * new_h - y * h) * new_w * self.rect_sum(i1 - 1, j1, i1, j2)
                s += ((y + 1) * h - i2 * new_h) * new_w * self.rect_sum(i2, j1, i2 + 1, j2)
                s += new_h * (j1 * new_w - x * w) * self.rect_sum(i1, j1 - 1, i2, j1)
                s += new_h * ((x + 1) * w - j2 * new_w) * self.rect_sum(i1, j2, i2, j2 + 1)

                result[y, x] = (s + h * w // 2) // (w * h)

        return result


if __name__ == '__main__':
    arr = load_image('data/100px/1.png')
    print arr.shape

    w, h = 20, 17
    small_arr = ScalableImage(arr).downscale(w, h)
    assert (naive_downscale(arr, w, h) == small_arr).all()
    save_image(small_arr, "hz.png")
