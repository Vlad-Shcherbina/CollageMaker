import sys


def vertical_partitions(x1, y1, x2, y2):
    assert x2 - x1 >= 10
    result = []
    for x in range(x1 + 4, x2 - 4 + 1, 3):
        result.append(((x1, y1, x, y2), (x, y1, x2, y2)))
    return result


def horizontal_partitions(x1, y1, x2, y2):
    assert y2 - y1 >= 10
    result = []
    for y in range(y1 + 4, y2 - 4 + 1, 3):
        result.append(((x1, y1, x2, y), (x1, y, x2, y2)))
    return result


def partitions(x1, y1, x2, y2):
    result = []
    if x2 - x1 >= 10 and y2 - y1 <= 2 * (x2 - x1):
        result += vertical_partitions(x1, y1, x2, y2)
    if y2 - y1 >= 10 and x2 - x1 <= 2 * (y2 - y1):
        result += horizontal_partitions(x1, y1, x2, y2)
    return result


def area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)


def best_partition(target, ia, x1, y1, x2, y2):
    """ Return tuple (noise_delta, rect1, ia1, rect2, ia2). """
    n0 = ia.noise * area(x1, y1, x2, y2)

    if x2 - x1 <= 70 and y2 - y1 <= 100 or x2 - x1 <= 100 and y2 - y1 <= 70:
        size_bonus = 0
    else:
        size_bonus = 1e10

    best = -1, None, None
    for rect1, rect2 in partitions(x1, y1, x2, y2):
        ia1 = target.get_abstraction(*rect1)
        ia2 = target.get_abstraction(*rect2)

        delta = n0 - area(*rect1) * ia1.noise - area(*rect2) * ia2.noise
        delta += size_bonus
        if delta > best[0]:
            best = delta, rect1, ia1, rect2, ia2

    return best


def build_partition(target):
    h, w = target.arr.shape
    ia = target.get_abstraction(0, 0, w, h)
    items = [((0, 0, w, h), ia) + best_partition(target, ia, 0, 0, w, h)]
    for _ in range(10000):
        max_item = max(items, key=lambda x: x[2])
        _, _, delta, rect1, ia1, rect2, ia2 = max_item
        if delta < 5000 or len(items) >= 20 and delta < 1e9:
            break

        item1 = (rect1, ia1) + best_partition(target, ia1, *rect1)
        item2 = (rect2, ia2) + best_partition(target, ia2, *rect2)

        items.remove(max_item)
        items.append(item1)
        items.append(item2)

    print>>sys.stderr, len(items), 'items in partition'
    return [(item[0], item[1]) for item in items]


if __name__ == '__main__':
    import numpy
    import img_lib

    target_arr = img_lib.load_image('data/300px/4.png')
    target = img_lib.TargetImage(target_arr)

    result = numpy.zeros(target.arr.shape)
    for (x1, y1, x2, y2), ia in build_partition(target):
        result[y1:y2, x1:x2] = ia.instantiate(x2 - x1, y2 - y1)

    img_lib.save_image(result, 'tp.png')
