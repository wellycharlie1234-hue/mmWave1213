import os
import h5py
import numpy as np

# Directory paths
DATA_DIR = r"C:\Users\user\Desktop\mmWave\Collect_RDI\Collect_RDI\Record\RDIPHD\6"
# DATA_DIR = r'C:\kai\Radar_Gesture_HandOff\data\0702_test'
PROCESSED_DATA_DIR = os.path.join('data', 'processed_data')
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_dataset.npz')

# Get gesture types from subfolder names and map each to a numeric label
gesture_types = sorted(
    [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
)
gesture_to_label = {gesture: idx for idx, gesture in enumerate(gesture_types)}
print(f"Gesture mapping: {gesture_to_label}")


def load_h5_file(file_path):
    """
    Load a .h5 file and extract 'DS1' and 'LABEL' arrays.
    """
    with h5py.File(file_path, 'r') as f:
        ds1 = np.array(f['DS1'], dtype=np.float32)
        label = np.array(f['LABEL']).astype(np.int32)
    return ds1, label


def generate_ground_truth(label, gesture_label, total_classes, max_length):
    """
    Generate a ground truth array based on the label.
    """
    ground_truth = np.zeros((max_length, total_classes))
    gesture_indices = np.where(label == 1)[0]

    if len(gesture_indices) == 0:
        ground_truth[:, 0] = 1  # Background only
        return ground_truth

    for segment in np.split(gesture_indices, np.where(np.diff(gesture_indices) > 1)[0] + 1):
        if segment.size == 0:
            continue
        start_idx, end_idx = segment[0], segment[-1]
        length = end_idx - start_idx + 1
        center = length // 2
        x = np.arange(length) - center
        sigma = length / 6
        gaussian_curve = np.exp(-0.5 * (x / sigma) ** 2)
        gaussian_curve /= gaussian_curve.max()
        ground_truth[start_idx:end_idx + 1, gesture_label] = gaussian_curve

    ground_truth[:, 0] = 1 - ground_truth[:, 1:].sum(axis=1)
    return ground_truth


def process_data():
    """
    Process all .h5 files from each gesture folder and save the processed data.
    """
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    all_features = []
    max_length = 0

    # Load files and determine maximum frame length
    for gesture in gesture_types:
        gesture_label = gesture_to_label[gesture]
        gesture_dir = os.path.join(DATA_DIR, gesture)
        for file_name in os.listdir(gesture_dir):
            if file_name.endswith('.h5'):
                file_path = os.path.join(gesture_dir, file_name)
                ds1, label = load_h5_file(file_path)
                max_length = max(max_length, ds1.shape[-1])
                all_features.append((ds1, gesture_label, label))

    features_padded = []
    labels_padded = []
    ground_truths_padded = []
    total_classes = len(gesture_types)

    for ds1, gesture_label, label in all_features:
        pad_width = ((0, 0), (0, 0), (0, 0), (0, max_length - ds1.shape[-1]))
        ds1_padded = np.pad(ds1, pad_width, mode='constant', constant_values=0)
        padded_label = np.full((max_length,), 0)
        padded_label[:len(label)] = label.squeeze()
        ground_truth = generate_ground_truth(padded_label, gesture_label, total_classes, max_length)

        features_padded.append(ds1_padded)
        labels_padded.append(padded_label)
        ground_truths_padded.append(ground_truth)

    features_padded = np.array(features_padded, dtype=np.float32)
    labels_padded = np.array(labels_padded, dtype=np.int32)
    ground_truths_padded = np.array(ground_truths_padded, dtype=np.float32)

    np.savez(
        PROCESSED_DATA_FILE,
        features=features_padded,
        labels=labels_padded,
        ground_truths=ground_truths_padded,
    )
    print(f"Saved processed data to {PROCESSED_DATA_FILE} (max_length={max_length})")


if __name__ == '__main__':
    process_data()
