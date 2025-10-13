import a_acquire
import b_enhance
import c_describe
import d_match

# List of file paths for the fingerprint images
fingerprint_filepaths = [
    './test-data/sams_leftthumb.jpeg',
    './test-data/Samuel_fingerPrint_lefthand_thumb.png'
]

# Acquire and enhance fingerprints, then describe ridge endings and bifurcations
fingerprint_data = []
for filepath in fingerprint_filepaths:
    fingerprint = a_acquire.acquire_from_file(filepath, view=False)
    pp_fingerprint, en_fingerprint, mask = b_enhance.enhance(fingerprint, dark_ridges=False, view=False)
    ridge_endings, bifurcations = c_describe.describe(en_fingerprint, mask, view=False)
    fingerprint_data.append((en_fingerprint, ridge_endings, bifurcations))

# Compare each pair of fingerprints
for i, (en_fingerprint_1, ridge_endings_1, bifurcations_1) in enumerate(fingerprint_data):
    for j, (en_fingerprint_2, ridge_endings_2, bifurcations_2) in enumerate(fingerprint_data):
        matches = d_match.match(en_fingerprint_1, ridge_endings_1, bifurcations_1, en_fingerprint_2, ridge_endings_2, bifurcations_2, view=True)
        all_features_1 = np.concatenate((ridge_endings_1, bifurcations_1), axis=0)
        all_features_2 = np.concatenate((ridge_endings_2, bifurcations_2), axis=0)
        print(f'Match between fingerprint {i + 1} and fingerprint {j + 1}: {len(matches)} matches,')
        print(f'Number of minutiae in fingerprint {i + 1}: {len(all_features_1)}')
        print(f'Number of minutiae in fingerprint {j + 1}: {len(all_features_2)}')


''' for j, (en_fingerprint_2, ridge_endings_2, bifurcations_2) in enumerate(fingerprint_data):
        matches = d_match.match(en_fingerprint_1, ridge_endings_1, bifurcations_1, en_fingerprint_2, ridge_endings_2, bifurcations_2, view=True)
        print(f'Match between fingerprint {i + 1} and fingerprint {j + 1}: {matches}')
'''
