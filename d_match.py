import math
import numpy
import cv2


HOUGH_SCALE_RANGE = [1.0]


HOUGH_ROTATION_STEP = numpy.pi / 8.0
HOUGH_ROTATION_RANGE = numpy.arange(-numpy.pi / 2.0, numpy.pi / 2.0, HOUGH_ROTATION_STEP)


HOUGH_TRANSLATION_OVERLAY_RATE = 0.5
HOUGH_TRANSLATION_STEP = 10


DIST_TRSH = 10
ANGLE_TRSH = numpy.pi / 8.0


def __compare(minutiae_1, minutiae_2, dist_threshold, angle_threshold):
    dist = math.sqrt((minutiae_1[0] - minutiae_2[0]) ** 2 + (minutiae_1[1] - minutiae_2[1]) ** 2)
    if dist > dist_threshold:
        return float('inf')

    a1 = minutiae_1[2]
    if a1 < 0.0:
        a1 = 2.0 * numpy.pi + a1

    a2 = minutiae_2[2]
    if a2 < 0.0:
        a2 = 2.0 * numpy.pi + a2

    angle_diff = abs(a1 - a2)
    if angle_diff > angle_threshold:
        return float('inf')

    return (dist / dist_threshold + angle_diff / angle_threshold) / 2.0


def _compute_matches(minutiae_1_points, minutiae_1_angles, minutiae_1_types,
                     minutiae_2_points, minutiae_2_angles, minutiae_2_types,
                     x_scale, y_scale, rotation, translation_overlay_rate, translation_step,
                     dist_threshold, angle_threshold):
    scale_matrix = numpy.zeros((3, 3), dtype=numpy.float32)
    scale_matrix[0, 0] = x_scale
    scale_matrix[1, 1] = y_scale
    scale_matrix[2, 2] = 1.0
    minutiae_2_points = cv2.perspectiveTransform(numpy.float32([minutiae_2_points]), scale_matrix)[0]

    if rotation < 0.0:
        rotation = 2.0 * numpy.pi + rotation

    sine = math.sin(rotation)
    cosine = math.cos(rotation)

    rotation_matrix = numpy.zeros((3, 3))
    rotation_matrix[0, 0] = cosine
    rotation_matrix[0, 1] = -sine
    rotation_matrix[1, 0] = sine
    rotation_matrix[1, 1] = cosine
    rotation_matrix[2, 2] = 1.0
    minutiae_2_points = cv2.perspectiveTransform(numpy.float32([minutiae_2_points]), rotation_matrix)[0]

    minutiae_2_angles = minutiae_2_angles.copy()
    for i in range(len(minutiae_2_angles)):
        angle = minutiae_2_angles[i]
        if angle < 0.0:
            angle = 2.0 * numpy.pi + angle

        new_angle = angle + rotation
        if new_angle > numpy.pi:
            new_angle = new_angle - 2.0 * numpy.pi

        minutiae_2_angles[i] = new_angle

    minutiae_1_points = minutiae_1_points - [numpy.min(minutiae_1_points[:, 0]), numpy.min(minutiae_1_points[:, 1])]
    minutiae_2_points = minutiae_2_points - [numpy.max(minutiae_2_points[:, 0]), numpy.max(minutiae_2_points[:, 1])]

    minutiae_1_corner_1 = numpy.array([numpy.min(minutiae_1_points[:, 0]), numpy.min(minutiae_1_points[:, 1])],
                                      dtype=int)
    minutiae_1_corner_2 = numpy.array([numpy.max(minutiae_1_points[:, 0]), numpy.max(minutiae_1_points[:, 1])],
                                      dtype=int)

    minutiae_1_w, minutiae_1_h = minutiae_1_corner_2 - minutiae_1_corner_1
    minutiae_1_x_offset = int(round((1.0 - translation_overlay_rate) * minutiae_1_w / 2.0))
    minutiae_1_y_offset = int(round((1.0 - translation_overlay_rate) * minutiae_1_h / 2.0))

    minutiae_2_corner_1 = numpy.array([numpy.min(minutiae_2_points[:, 0]), numpy.min(minutiae_2_points[:, 1])],
                                      dtype=int)
    minutiae_2_corner_2 = numpy.array([numpy.max(minutiae_2_points[:, 0]), numpy.max(minutiae_2_points[:, 1])],
                                      dtype=int)

    minutiae_2_w, minutiae_2_h = minutiae_2_corner_2 - minutiae_2_corner_1
    minutiae_2_x_offset = int(round((1.0 - translation_overlay_rate) * minutiae_2_w / 2.0))
    minutiae_2_y_offset = int(round((1.0 - translation_overlay_rate) * minutiae_2_h / 2.0))

    start_x = minutiae_1_x_offset + minutiae_2_x_offset
    stop_x = start_x + minutiae_1_w + minutiae_2_w - minutiae_1_x_offset - minutiae_2_x_offset
    start_y = minutiae_1_y_offset + minutiae_2_y_offset
    stop_y = start_y + minutiae_1_h + minutiae_2_h - minutiae_1_y_offset - minutiae_2_y_offset

    best_matches = []

    for x_translation in range(start_x, stop_x, translation_step):
        for y_translation in range(start_y, stop_y, translation_step):
            minutiae_2_points_transl = minutiae_2_points + [x_translation, y_translation]

            matches = []
            already_matched = []
            for i in range(minutiae_1_points.shape[0]):
                current_match = None
                current_match_dist = float('inf')

                for j in range(minutiae_2_points_transl.shape[0]):
                    if j not in already_matched and minutiae_1_types[i] == minutiae_2_types[j] and \
                            minutiae_2_points_transl[j][0] > 0.0 and minutiae_2_points_transl[j][1] > 0.0:
                        dist = __compare((minutiae_1_points[i][0], minutiae_1_points[i][1], minutiae_1_angles[i]),
                                         (minutiae_2_points_transl[j][0], minutiae_2_points_transl[j][1],
                                          minutiae_2_angles[j]), dist_threshold, angle_threshold)
                        if dist < current_match_dist:
                            current_match = j
                            current_match_dist = dist

                if current_match is not None:
                    matches.append((i, current_match))
                    already_matched.append(current_match)

            if len(best_matches) < len(matches):
                best_matches = matches

    return best_matches


def _draw_matches(fingerprint_1, fingerprint_2, matches,
                  ridge_endings_1, bifurcations_1,
                  ridge_endings_2, bifurcations_2):
    mag = 5.0

    h1, w1 = fingerprint_1.shape
    h2, w2 = fingerprint_2.shape

    output_image = numpy.zeros((max(h1, h2), w1 + w2), dtype=numpy.uint8)
    output_image[0:h1, 0:w1] = (fingerprint_1 > 0).astype(numpy.uint8) * 255
    output_image[0:h2, w1:w1 + w2] = (fingerprint_2 > 0).astype(numpy.uint8) * 255
    output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)

    for ridge_ending in ridge_endings_1:
        p = (int(ridge_ending[0]), int(ridge_ending[1]))
        cv2.rectangle(output_image, (p[0] - 1, p[1] - 1), (p[0] + 2, p[1] + 2), (0, 0, 255), 1)

        delta_x = int(round(math.cos(ridge_ending[2]) * mag))
        delta_y = int(round(math.sin(ridge_ending[2]) * mag))
        cv2.line(output_image, (p[0], p[1]), (p[0] + delta_x, p[1] + delta_y), (0, 255, 255), 1)

    for ridge_ending in ridge_endings_2:
        p = (int(ridge_ending[0] + w1), int(ridge_ending[1]))
        cv2.rectangle(output_image, (p[0] - 1, p[1] - 1), (p[0] + 2, p[1] + 2), (0, 0, 255), 1)

        delta_x = int(round(math.cos(ridge_ending[2]) * mag))
        delta_y = int(round(math.sin(ridge_ending[2]) * mag))
        cv2.line(output_image, (p[0] + w1, p[1]), (p[0] + delta_x + w1, p[1] + delta_y), (0, 255, 255), 1)

    for bifurcation in bifurcations_1:
        p = (int(bifurcation[0]), int(bifurcation[1]))
        cv2.rectangle(output_image, (p[0] - 1, p[1] - 1), (p[0] + 2, p[1] + 2), (0, 255, 0), 1)

        delta_x = int(round(math.cos(bifurcation[2]) * mag))
        delta_y = int(round(math.sin(bifurcation[2]) * mag))
        cv2.line(output_image, (p[0], p[1]), (p[0] + delta_x, p[1] + delta_y), (0, 255, 255), 1)

    for bifurcation in bifurcations_2:
        p = (int(bifurcation[0] + w1), int(bifurcation[1]))
        cv2.rectangle(output_image, (p[0] - 1, p[1] - 1), (p[0] + 2, p[1] + 2), (0, 255, 0), 1)

        delta_x = int(round(math.cos(bifurcation[2]) * mag))
        delta_y = int(round(math.sin(bifurcation[2]) * mag))
        cv2.line(output_image, (p[0] + w1, p[1]), (p[0] + delta_x + w1, p[1] + delta_y), (0, 255, 255), 1)


    for m in matches[0]:
        x0 = int(m[0][0])
        y0 = int(m[0][1])
        a0 = m[0][2]
        delta_x0 = int(round(math.sin(a0) * mag))
        delta_y0 = int(round(math.cos(a0) * mag))

        x1 = int(m[1][0])
        y1 = int(m[1][1])
        a1 = m[1][2]
        delta_x1 = int(round(math.sin(a1) * mag))
        delta_y1 = int(round(math.cos(a1) * mag))

        cv2.rectangle(output_image, (x0 - 1, y0 - 1), (x0 + 2, y0 + 2), (0, 0, 255), 1)
        cv2.rectangle(output_image, (x1 - 1 + w1, y1 - 1), (x1 + 2 + w1, y1 + 2), (0, 0, 255), 1)
        cv2.line(output_image, (x0, y0), (x0 + delta_x0, y0 + delta_y0), (0, 255, 255), 1)
        cv2.line(output_image, (x1 + w1, y1), (x1 + w1 + delta_x1, y1 + delta_y1), (0, 255, 255), 1)
        cv2.line(output_image, (x0, y0), (x1 + w1, y1), (0, 255, 255), 1)


    for m in matches[1]:
        x0 = int(m[0][0])
        y0 = int(m[0][1])
        a0 = m[0][2]
        delta_x0 = int(round(math.sin(a0) * mag))
        delta_y0 = int(round(math.cos(a0) * mag))

        x1 = int(m[1][0])
        y1 = int(m[1][1])
        a1 = m[1][2]
        delta_x1 = int(round(math.sin(a1) * mag))
        delta_y1 = int(round(math.cos(a1) * mag))

        cv2.rectangle(output_image, (x0 - 1, y0 - 1), (x0 + 2, y0 + 2), (0, 255, 0), 1)
        cv2.rectangle(output_image, (x1 - 1 + w1, y1 - 1), (x1 + 2 + w1, y1 + 2), (0, 255, 0), 1)
        cv2.line(output_image, (x0, y0), (x0 + delta_x0, y0 + delta_y0), (0, 255, 255), 1)
        cv2.line(output_image, (x1 + w1, y1), (x1 + w1 + delta_x1, y1 + delta_y1), (0, 255, 255), 1)
        cv2.line(output_image, (x0, y0), (x1 + w1, y1), (0, 255, 255), 1)

    cv2.imshow('Matches, press key.', output_image)
    cv2.waitKey(0)


def _01_hough_transform(ridge_endings_1, ridge_bifurcations_1, ridge_endings_2, ridge_bifurcations_2,
                        scale_range, rotation_range, translation_overlay_rate, translation_step,
                        dist_threshold, angle_threshold):

    minutiae_set_1 = numpy.concatenate((ridge_endings_1, ridge_bifurcations_1), axis=0)
    minutiae_set_2 = numpy.concatenate((ridge_endings_2, ridge_bifurcations_2), axis=0)
    if len(minutiae_set_1) == 0 or len(minutiae_set_2) == 0:
        return [], []

    minutiae_1_points = numpy.array([[minutiae_set_1[0][0], minutiae_set_1[0][1]]])
    minutiae_1_angles = [minutiae_set_1[0][2]]
    minutiae_1_types = [True] * len(ridge_endings_1) + [False] * len(ridge_bifurcations_1)
    for i in range(1, len(minutiae_set_1)):
        minutiae_1_points = numpy.append(minutiae_1_points, [[minutiae_set_1[i][0], minutiae_set_1[i][1]]], 0)
        minutiae_1_angles.append(minutiae_set_1[i][2])

    minutiae_2_points = numpy.array([[minutiae_set_2[0][0], minutiae_set_2[0][1]]])
    minutiae_2_angles = [minutiae_set_2[0][2]]
    minutiae_2_types = [True] * len(ridge_endings_2) + [False] * len(ridge_bifurcations_2)
    for i in range(1, len(minutiae_set_2)):
        minutiae_2_points = numpy.append(minutiae_2_points, [[minutiae_set_2[i][0], minutiae_set_2[i][1]]], 0)
        minutiae_2_angles.append(minutiae_set_2[i][2])

    best_matches = []
    best_config = None

    for x_scale in scale_range:
        for y_scale in scale_range:
            for rotation in rotation_range:
                matches = _compute_matches(minutiae_1_points, minutiae_1_angles, minutiae_1_types,
                                           minutiae_2_points, minutiae_2_angles, minutiae_2_types,
                                           x_scale, y_scale, rotation, translation_overlay_rate, translation_step,
                                           dist_threshold, angle_threshold)

                print('[INFO] Hough transform at', str([x_scale, y_scale, rotation]) + ':', len(matches), 'matches.')

                if len(best_matches) < len(matches):
                    best_matches = matches
                    best_config = [x_scale, y_scale, rotation]

    print('[INFO] Best Hough with:', len(best_matches), 'matches, at:', str(best_config) + '.')

    ridge_ending_matches = []
    bifurcation_matches = []
    for m in best_matches:
        if minutiae_1_types[m[0]]:
            ridge_ending_matches.append(((minutiae_set_1[m[0]][0], minutiae_set_1[m[0]][1], minutiae_1_angles[m[0]]),
                                         (minutiae_set_2[m[1]][0], minutiae_set_2[m[1]][1], minutiae_2_angles[m[1]])))

        else:
            bifurcation_matches.append(((minutiae_set_1[m[0]][0], minutiae_set_1[m[0]][1], minutiae_1_angles[m[0]]),
                                        (minutiae_set_2[m[1]][0], minutiae_set_2[m[1]][1], minutiae_2_angles[m[1]])))

    return ridge_ending_matches, bifurcation_matches


def match(fingerprint_1, ridge_endings_1, ridge_bifurcations_1, fingerprint_2, ridge_endings_2, ridge_bifurcations_2,
          view=False):

    matches = _01_hough_transform(ridge_endings_1, ridge_bifurcations_1, ridge_endings_2, ridge_bifurcations_2,
                                  HOUGH_SCALE_RANGE, HOUGH_ROTATION_RANGE,
                                  HOUGH_TRANSLATION_OVERLAY_RATE, HOUGH_TRANSLATION_STEP,
                                  DIST_TRSH, ANGLE_TRSH)

    if view:
        _draw_matches(fingerprint_1, fingerprint_2, matches, ridge_endings_1, ridge_bifurcations_1, ridge_endings_2,
                      ridge_bifurcations_2)

    return matches
