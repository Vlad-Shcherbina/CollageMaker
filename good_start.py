from math import sqrt
import random

import numpy

import img_lib


if __name__ == '__main__':
    pass

    target_arr = img_lib.load_image('data/300px/1.png')
    target = img_lib.TargetImage(target_arr)
    h, w = target_arr.shape

    sources_arr = [
        img_lib.load_image('data/100px/{}.png'.format(i)) for i in range(100)]


    result = numpy.zeros((h, w*2))
    result[:, :w] = target_arr
    result[:, w:] = target_arr

    random.seed(42)
    baseline = 49
    for _ in range(60, 65):
        print _

        best_save = 0
        best_pos = None

        #source_arr = random.choice(sources_arr)
        source_arr = sources_arr[_]

        q = img_lib.ScalableImage(source_arr)

        for _ in range(100):
            h1, w1 = source_arr.shape
            dw = random.randrange(50, w1 + 1)
            dh = random.randrange(50, h1 + 1)
            d = q.downscale(dw, dh)

            ia1 = img_lib.TargetImage(d).get_abstraction(0, 0, dw, dh)

            for _ in range(600):
                x1 = random.randrange(w - dw + 1)
                y1 = random.randrange(h - dh + 1)

                ia2 = target.get_abstraction(x1, y1, x1 + dw, y1 + dh)
                min_d = sqrt(
                    ia1.average_error(ia2) - ia1.noise - ia2.noise +
                    (sqrt(ia1.noise) - sqrt(ia2.noise))**2)
                if (baseline - min_d) * dw * dh < best_save:
                    continue

                delta = d - target_arr[y1:y1 + dh, x1:x1 + dw]

                save = sqrt(1.0 * (delta**2).sum() / delta.size)
                assert save >= min_d

                save = (baseline - save) * delta.size
                if save > best_save:
                    best_save = save
                    best_pos = d, x1, y1, x1 + dw, y1 + dh
                    print best_save


        #print best_pos

        if best_save > 0:
            patch, x1, y1, x2, y2 = best_pos
            result[y1:y2, x1+w:x2+w] = patch

    img_lib.save_image(result, 'good_start.png')
