import math
import numpy
import cv2


MINUT_ORIENT_BLOCK_SIZE = 7
MIN_MINUTIAE_DIST = 5
MIN_RIDGE_LENGTH = 10
RIDGE_END_ANGLE_TOLER = numpy.pi / 8.0
MIN_MINUT_MASK_DIST = 20


def __draw_minutiae(fingerprint, ridge_endings, ridge_bifurcations, msg):
    mag = 5.0

    img = (fingerprint > 0).astype(numpy.uint8) * 255
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for ridge_ending in ridge_endings:
        p = (int(ridge_ending[0]), int(ridge_ending[1]))
        cv2.rectangle(img, (p[0] - 1, p[1] - 1), (p[0] + 2, p[1] + 2), (0, 0, 255), 1)

        delta_x = int(round(math.cos(ridge_ending[2]) * mag))
        delta_y = int(round(math.sin(ridge_ending[2]) * mag))
        cv2.line(img, (p[0], p[1]), (p[0] + delta_x, p[1] + delta_y), (0, 255, 255), 1)

    for bifurcation in ridge_bifurcations:
        p = (int(bifurcation[0]), int(bifurcation[1]))
        cv2.rectangle(img, (p[0] - 1, p[1] - 1), (p[0] + 2, p[1] + 2), (0, 255, 0), 1)

        delta_x = int(round(math.cos(bifurcation[2]) * mag))
        delta_y = int(round(math.sin(bifurcation[2]) * mag))
        cv2.line(img, (p[0], p[1]), (p[0] + delta_x, p[1] + delta_y), (0, 255, 255), 1)

    cv2.imshow(msg + ' minutiae, press any key.', img)
    cv2.waitKey(0)


def __compute_minutiae_angle(fingerprint, position, block_size, is_ridge_ending):
    h, w = fingerprint.shape

    block_step = int(block_size / 2)
    block = fingerprint[max(0, position[1] - block_step):min(position[1] + block_step + 1, h),
            max(0, position[0] - block_step):min(position[0] + block_step + 1, w)]
    block_h, block_w = block.shape
    block_center = (int(block_w / 2), int(block_h / 2))

    border_points = []
    for i in range(block_h):
        for j in range(block_w):
            if (i == 0 or i == block_h - 1 or j == 0 or j == block_w - 1) and block[i, j] > 0:
                border_points.append((j, i))

    if is_ridge_ending:
        if len(border_points) < 1:
            return None

        closest_point = None
        closest_distance = float('inf')
        for p in border_points:
            dist = math.sqrt((p[0] - block_center[0]) ** 2 + (p[1] - block_center[1]) ** 2)
            if dist < closest_distance:
                closest_point = p
                closest_distance = dist

        return math.atan2(closest_point[1] - block_center[1], closest_point[0] - block_center[0])

    else:
        if len(border_points) != 3:
            return None

        closest_points = None
        closest_distance = float('inf')
        for i in range(len(border_points) - 1):
            for j in range(i + 1, len(border_points)):
                dist = math.sqrt(
                    (border_points[i][0] - border_points[j][0]) ** 2 + (border_points[i][1] - border_points[j][1]) ** 2)
                if dist < closest_distance:
                    closest_points = [border_points[i], border_points[j]]
                    closest_distance = dist

        mid_point = numpy.mean(closest_points, axis=0)
        return math.atan2(mid_point[1] - block_center[1], mid_point[0] - block_center[0])


def _01_detect_minutiae(fingerprint, mask, block_size, view=False):
    ridge_endings = []
    ridge_bifurcations = []

    fingerprint = (fingerprint > 0).astype(numpy.uint8)

    h, w = fingerprint.shape

    for row in range(1, h - 1):
        for col in range(1, w - 1):
            if mask[row, col] > 0:
                if fingerprint[row, col] == 1:
                    block = fingerprint[row - 1: row + 2, col - 1: col + 2]

                    ridge_count = numpy.sum(block)

                    if ridge_count < 3:
                        angle = __compute_minutiae_angle(fingerprint, (col, row), block_size, is_ridge_ending=True)
                        if angle is not None:
                            ridge_endings.append((col, row, angle))

                    elif ridge_count > 3:
                        angle = __compute_minutiae_angle(fingerprint, (col, row), block_size, is_ridge_ending=False)
                        if angle is not None:
                            ridge_bifurcations.append((col, row, angle))

    if view:
        __draw_minutiae(fingerprint, ridge_endings, ridge_bifurcations, 'All')

    print('[INFO] Detected minutiae.')
    return ridge_endings, ridge_bifurcations


def _02_remove_false_positive_minutiae(fingerprint, mask, ridge_endings, ridge_bifurcations,
                                       min_minutiae_dist, min_ridge_length, ridge_ending_angle_tol, min_mask_dist,
                                       view=False):
    h, w = fingerprint.shape

    ridge_ending_count = len(ridge_endings)
    bifurcation_count = len(ridge_bifurcations)

    good_ridge_endings = [True] * ridge_ending_count
    good_bifurcations = [True] * bifurcation_count


    for i in range(0, ridge_ending_count - 1):
        for j in range(i + 1, ridge_ending_count):
            x0, y0, a0 = ridge_endings[i]
            x1, y1, a1 = ridge_endings[j]

            dist = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
            if dist < min_minutiae_dist:
                good_ridge_endings[i] = good_ridge_endings[j] = False

    for i in range(0, ridge_ending_count - 1):
        for j in range(i + 1, ridge_ending_count):
            x0, y0, a0 = ridge_endings[i]
            x1, y1, a1 = ridge_endings[j]

            if a0 < 0.0:
                a0 = 2.0 * numpy.pi + a0

            if a1 < 0.0:
                a1 = 2.0 * numpy.pi + a1

            dist = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
            if dist < min_ridge_length:
                a01 = math.atan2(y0 - y1, x0 - x1)
                if a01 < 0.0:
                    a01 = 2.0 * numpy.pi + a01

                if abs(a01 - a0) < ridge_ending_angle_tol or abs(a01 - a1) < ridge_ending_angle_tol:
                    if numpy.pi - ridge_ending_angle_tol < abs(a0 - a1) < numpy.pi + ridge_ending_angle_tol:
                        good_ridge_endings[i] = good_ridge_endings[j] = False

    for i in range(ridge_ending_count):
        for j in range(bifurcation_count):
            x0, y0, _ = ridge_endings[i]
            x1, y1, _ = ridge_bifurcations[j]

            dist = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
            if dist < min_minutiae_dist:
                good_ridge_endings[i] = good_bifurcations[j] = False

    for i in range(0, ridge_ending_count):
        if good_ridge_endings[i]:
            x0, y0, _ = ridge_endings[i]

            if x0 - min_mask_dist < 0 or y0 - min_mask_dist < 0 or \
                    x0 + min_mask_dist + 1 > w or y0 + min_mask_dist + 1 > h:
                good_ridge_endings[i] = False

            else:
                mask_block = mask[y0 - min_mask_dist:y0 + min_mask_dist + 1, x0 - min_mask_dist:x0 + min_mask_dist + 1]

                if numpy.min(mask_block) == 0:
                    good_ridge_endings[i] = False

    for i in range(0, bifurcation_count):
        if good_bifurcations[i]:
            x0, y0, _ = ridge_bifurcations[i]

            if x0 - min_mask_dist < 0 or y0 - min_mask_dist < 0 or \
                    x0 + min_mask_dist + 1 > w or y0 + min_mask_dist + 1 > h:
                good_ridge_endings[i] = False

            else:
                mask_block = mask[y0 - min_mask_dist:y0 + min_mask_dist + 1, x0 - min_mask_dist:x0 + min_mask_dist + 1]

                if numpy.min(mask_block) == 0:
                    good_ridge_endings[i] = False

                if numpy.min(mask_block) == 0:
                    good_bifurcations[i] = False

    ridge_endings = numpy.array(ridge_endings)[numpy.where(good_ridge_endings)]
    ridge_bifurcations = numpy.array(ridge_bifurcations)[numpy.where(good_bifurcations)]

    if view:
        __draw_minutiae(fingerprint, ridge_endings, ridge_bifurcations, 'Cleaned')

    print('[INFO] Removed bad-quality minutiae.')
    return ridge_endings, ridge_bifurcations


def describe(enhanced_fingerprint, mask, view=False):
    ridge_endings, bifurcations = _01_detect_minutiae(enhanced_fingerprint, mask, MINUT_ORIENT_BLOCK_SIZE, view=view)

    ridge_endings, bifurcations = _02_remove_false_positive_minutiae(enhanced_fingerprint, mask,
                                                                     ridge_endings, bifurcations,
                                                                     MIN_MINUTIAE_DIST, MIN_RIDGE_LENGTH,
                                                                     RIDGE_END_ANGLE_TOLER, MIN_MINUT_MASK_DIST,
                                                                     view=view)

    return ridge_endings, bifurcations
