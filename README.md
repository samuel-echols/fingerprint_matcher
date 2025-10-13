Fingerprint Matcher Project Summary
What This Project Does

The Fingerprint Matcher project provides a modular pipeline for fingerprint image processing and matching.
It consists of four main stages: acquisition, enhancement, description, and matching. Each stage is defined
in a separate module, making the pipeline flexible and easy to modify or extend.

Key functionalities include:
•	- Acquisition: Load and preprocess fingerprint images from disk.
•	- Enhancement: Improve image quality through contrast and filtering techniques.
•	- Description: Extract fingerprint features such as orientation, frequency, or keypoints.
•	- Matching: Compare feature sets to determine fingerprint similarity.
Requirements

This project is written in Python and depends on several widely used scientific and imaging libraries.
Python 3.10 or higher is recommended for best compatibility.

•	Required packages:
•	- numpy
•	- opencv-python
•	- scikit-image
•	- matplotlib (optional, for visualization)

To Run change image file paths in main.py
