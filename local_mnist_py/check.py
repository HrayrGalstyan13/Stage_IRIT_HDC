import numpy as np

def check_magic_number(file_path):
    """Check the magic number of an IDX file."""
    with open(file_path, 'rb') as f:
        magic_number = np.fromfile(f, dtype=np.uint32, count=1)[0]
        print(f"Magic number for {file_path}: {magic_number}")
        return magic_number

# Path to your files
train_images_path = 'train-images.idx3-ubyte'
train_labels_path = 'train-labels.idx1-ubyte'
test_images_path = 't10k-images.idx3-ubyte'
test_labels_path = 't10k-labels.idx1-ubyte'

# Check magic numbers for each file
check_magic_number(train_images_path)
check_magic_number(train_labels_path)
check_magic_number(test_images_path)
check_magic_number(test_labels_path)