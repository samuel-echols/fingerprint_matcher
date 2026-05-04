import cv2

def acquire_from_file(file_path, view=False):
    fingerprint = cv2.imread(file_path)

    if view:
        cv2.imshow('press any key', fingerprint)
        cv2.waitKey(0)

    print('[INFO] Acquired fingerprint from file.')
    return fingerprint
