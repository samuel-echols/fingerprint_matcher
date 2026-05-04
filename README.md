# Fingerprint Matcher

A fingerprint identification pipeline that acquires, enhances, describes, and matches fingerprint images using minutiae-based comparison.

## Pipeline

The project is split into four modules that run in sequence:

- **a_acquire** — loads a fingerprint image from a file
- **b_enhance** — preprocesses and enhances the image (grayscale conversion, histogram equalization, segmentation, orientation estimation, Gabor filtering, skeletonization)
- **c_describe** — extracts minutiae (ridge endings and bifurcations) from the enhanced image
- **d_match** — compares two fingerprints using a Hough transform over rotation and translation to find the best alignment, then counts matching minutiae pairs

## Usage

Add fingerprint image paths to `fingerprint_filepaths` in `main.py`, then run:

```bash
python main.py
```

The script compares every unique pair of fingerprints and prints the number of matching minutiae and total minutiae count for each.

## Matching

A match is counted when two minutiae of the same type (ridge ending or bifurcation) align within:
- **10 pixels** distance
- **22.5 degrees** angle difference

A higher match count indicates a stronger similarity. Scores around 8 are typical for non-matching fingers; scores of 30+ indicate a likely match.

## Requirements

- Python 3
- OpenCV (`cv2`)
- NumPy
- SciPy
- scikit-image
