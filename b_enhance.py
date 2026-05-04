import math
import numpy
import cv2
import scipy.ndimage
import skimage.morphology
from skimage.filters import gabor_kernel


FINGERPRINT_HEIGHT = 352

FINGERPRINT_BLOCK = 16

FINGERPRINT_MASK_TRSH = 0.25

RIDGE_ORIENTATION_STEP = numpy.pi / 16
RIDGE_ORIENTATIONS = numpy.arange(-numpy.pi, numpy.pi + RIDGE_ORIENTATION_STEP, RIDGE_ORIENTATION_STEP)

WAVELENGTH_RATIO = 0.25

GABOR_OUTPUT_BIN_TRSH = -0.2

def __rotate_and_crop(image, rad_angle):
    h, w = image.shape

    degree_angle = 360.0 - (180.0 * rad_angle / numpy.pi)
    rotated = scipy.ndimage.rotate(image, degree_angle, reshape=False)

    crop_size = int(h / numpy.sqrt(2))
    crop_start = int((h - crop_size) / 2.0)

    rotated = rotated[crop_start: crop_start + crop_size, crop_start: crop_start + crop_size]
    return rotated

def _01_preprocess(fingerprint, output_height, dark_ridges=True, view=False):
    if len(fingerprint.shape) > 2 and fingerprint.shape[2] > 1:
        fingerprint = cv2.cvtColor(fingerprint, cv2.COLOR_BGR2GRAY)

    aspect_ratio = float(fingerprint.shape[0]) / fingerprint.shape[1]
    width = int(round(output_height / aspect_ratio))
    fingerprint = cv2.resize(fingerprint, (width, output_height))

    if not dark_ridges:
        fingerprint = abs(255 - fingerprint)

    fingerprint = cv2.equalizeHist(fingerprint, fingerprint)

    if view:
        cv2.imshow('Preprocessing, press any key.', fingerprint)
        cv2.waitKey(0)

    print('[INFO] Preprocessed fingerprint.')
    return fingerprint

def _02_segment(fingerprint, block_size, std_threshold, view=False):
    h, w = fingerprint.shape

    fingerprint = (fingerprint - numpy.mean(fingerprint)) / numpy.std(fingerprint)

    mask = numpy.zeros((h, w), numpy.uint8)

    block_step = int(block_size / 2.0)
    for row in range(h):
        for col in range(w):
            block = fingerprint[max(0, row - block_step):min(row + block_step + 1, h),
                    max(0, col - block_step):min(col + block_step + 1, w)]

            if numpy.std(block) > std_threshold:
                mask[row, col] = 255

    masked_values = fingerprint[mask > 0]
    fingerprint = (fingerprint - numpy.mean(masked_values)) / numpy.std(masked_values)
    fingerprint = cv2.bitwise_and(fingerprint, fingerprint, mask=mask)

    if view:
        img = fingerprint.copy()
        img = cv2.normalize(fingerprint, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow('Segmentation, press any key.', img)
        cv2.waitKey(0)

    print('[INFO] Segmented fingerprint.')
    return fingerprint, mask


def _03_compute_orientations(fingerprint, mask, block_size, view=False):
    h, w = fingerprint.shape

    y_gradient, x_gradient = numpy.gradient(fingerprint)

    orientations = numpy.arctan2(y_gradient, x_gradient)
    orientations = cv2.bitwise_and(orientations, orientations, mask=mask)

    magnitudes = numpy.sqrt(y_gradient ** 2 + x_gradient ** 2)
    magnitudes = cv2.bitwise_and(magnitudes, magnitudes, mask=mask)

    discret_orientations = numpy.zeros(orientations.shape, dtype=numpy.float32)
    block_step = int(block_size / 2.0)
    for row in range(h):
        for col in range(w):
            if mask[row, col] > 0:
                ori_block = orientations[max(0, row - block_step):min(row + block_step + 1, h),
                            max(0, col - block_step):min(col + block_step + 1, w)]
                mag_block = magnitudes[max(0, row - block_step):min(row + block_step + 1, h),
                            max(0, col - block_step):min(col + block_step + 1, w)]

                useful_magnitudes = numpy.where(mag_block > numpy.mean(mag_block))
                freqs, values = numpy.histogram(ori_block[useful_magnitudes], bins=RIDGE_ORIENTATIONS)

                best_value = numpy.mean(values[numpy.where(freqs == numpy.max(freqs))])
                orientation_index = int(round(best_value / RIDGE_ORIENTATION_STEP))
                discret_orientations[row, col] = RIDGE_ORIENTATIONS[orientation_index]

    discret_orientations = cv2.bitwise_and(discret_orientations, discret_orientations, mask=mask)

    if view:
        img = x_gradient.copy()
        img = cv2.normalize(x_gradient, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow('Orientation, x gradient, press any key.', img)
        cv2.waitKey(0)

        img = y_gradient.copy()
        img = cv2.normalize(y_gradient, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow('Orientation, y gradient, press any key.', img)
        cv2.waitKey(0)

        plot_step = 8

        img = fingerprint.copy()
        img = cv2.normalize(fingerprint, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        mag_enhance = 5.0
        start_pixel = int(plot_step / 2.0)
        for row in range(start_pixel, h, plot_step):
            for col in range(start_pixel, w, plot_step):
                angle = discret_orientations[row, col]
                magnitude = magnitudes[row, col] * mag_enhance

                if magnitude > 0:
                    delta_x = int(round(math.cos(angle) * magnitude))
                    delta_y = int(round(math.sin(angle) * magnitude))

                    cv2.line(img, (col, row), (col + delta_x, row + delta_y), (0, 255, 0), 1)

        cv2.imshow('Orientation, press any key.', img)
        cv2.waitKey(0)

    print('[INFO] Computed ridge orientations.')
    return discret_orientations, magnitudes


def _04_compute_ridge_frequency(fingerprint, mask, orientations, block_size, view=False):
    frequencies = []

    h, w = fingerprint.shape

    block_step = int(block_size / 2.0)
    for row in range(h):
        for col in range(w):
            if mask[row, col] > 0:
                block = fingerprint[max(0, row - block_step):min(row + block_step + 1, h),
                        max(0, col - block_step):min(col + block_step + 1, w)]

                rot_block = __rotate_and_crop(block, -orientations[row, col])

                if view:
                    img = rot_block.copy()
                    img = cv2.normalize(rot_block, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
                    cv2.imshow('block', img)
                    cv2.waitKey(120)

                ridge_proj = numpy.sum(rot_block, axis=0)
                ridge_peaks = numpy.zeros(ridge_proj.shape)
                ridge_peaks[numpy.where(ridge_proj > numpy.mean(ridge_proj))] = 1

                ridge_count = 0

                is_ridge = False
                for i in range(len(ridge_peaks)):
                    if ridge_peaks[i] == 1 and not is_ridge:
                        ridge_count = ridge_count + 1
                        is_ridge = True

                    elif ridge_peaks[i] == 0 and is_ridge:
                        ridge_count = ridge_count + 1
                        is_ridge = False

                frequencies.append(0.5 * ridge_count / len(ridge_peaks))

    print('[INFO] Computed ridge frequency.')
    if len(frequencies) > 0:
        return numpy.mean(frequencies)
    else:
        return 0


def _05_apply_gabor_filter(fingerprint, mask, orientations, ridge_frequency, std_wavelength_ratio, view=False):
    output = numpy.zeros(fingerprint.shape)

    h, w = fingerprint.shape

    fingerprint_filters = {}

    filter_std = std_wavelength_ratio * 1.0 / ridge_frequency
    for orientation in numpy.unique(orientations):
        kernel = numpy.real(gabor_kernel(ridge_frequency, orientation, sigma_x=filter_std, sigma_y=filter_std))
        fingerprint_filters[orientation] = scipy.ndimage.convolve(fingerprint, kernel)

    for row in range(h):
        for col in range(w):
            if mask[row, col] > 0:
                key_orientation = orientations[row, col]
                output[row, col] = fingerprint_filters[key_orientation][row, col]

    output = (output < GABOR_OUTPUT_BIN_TRSH).astype(numpy.uint8) * 255
    output = cv2.erode(output, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))

    if view:
        cv2.imshow('Filtering, press any key.', output)
        cv2.waitKey(0)

    print('[INFO] Applied Gabor filters.')
    return output


def _06_skeletonize(fingerprint, view=False):
    fingerprint = skimage.morphology.skeletonize(fingerprint / 255).astype(numpy.uint8) * 255

    if view:
        cv2.imshow('Skeletonization, press any key.', fingerprint)
        cv2.waitKey(0)

    print('[INFO] Skeletonized ridges.')
    return fingerprint


def enhance(fingerprint, dark_ridges=True, view=False):
    pp_fingerprint = _01_preprocess(fingerprint, FINGERPRINT_HEIGHT, dark_ridges, view=view)

    en_fingerprint, mask = _02_segment(pp_fingerprint, FINGERPRINT_BLOCK, FINGERPRINT_MASK_TRSH, view=view)

    orientations, magnitudes = _03_compute_orientations(en_fingerprint, mask, FINGERPRINT_BLOCK, view=view)

    ridge_freq = _04_compute_ridge_frequency(en_fingerprint, mask, orientations, FINGERPRINT_BLOCK)

    en_fingerprint = _05_apply_gabor_filter(en_fingerprint, mask, orientations, ridge_freq, WAVELENGTH_RATIO, view=view)

    en_fingerprint = _06_skeletonize(en_fingerprint, view=view)

    print('[INFO] Enhanced fingerprint.')
    return pp_fingerprint, en_fingerprint, mask
