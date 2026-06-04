import numpy as np
import struct
import random
from array import array
import time
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import threading
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed


# CONSTANTS
DIM = 10000
SEED = int(time.time())
PATCH_SIZE = 5   # z
S = 7          # split size

IMG_H, IMG_W = 28, 28

np.random.seed(SEED)


# MONITORING
cpu_usage = []
ram_usage = []
cpu_core_usage = [[] for _ in range(psutil.cpu_count(logical=False))]
monitoring = True

def monitor_resources():
    while monitoring:
        cpu_usage.append(psutil.cpu_percent(interval=0.2))
        ram_usage.append(psutil.virtual_memory().percent)
        per_core_usage = psutil.cpu_percent(percpu=True, interval=0.2)
        for i, usage in enumerate(per_core_usage):
            if i < len(cpu_core_usage):
                cpu_core_usage[i].append(usage)

monitor_thread = threading.Thread(target=monitor_resources)
monitor_thread.start()


# DATA LOADER 
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            image_data = array("B", file.read())

        images = np.array(image_data).reshape(size, rows, cols)
        return images, labels
    
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


# ITEM MEMORY
IM = {
    0: np.random.choice([0, 1], DIM),
    1: np.random.choice([0, 1], DIM)
}

# Patch position memory (
base_patch_x = np.random.choice([0, 1], DIM)
base_patch_y = np.random.choice([0, 1], DIM)

def random_hv(dim):
    return np.random.randint(0, 2, dim).astype(np.int32)

# Global grid position memory (S x S)
base_img_x = np.random.choice([0, 1], DIM)
base_img_y = np.random.choice([0, 1], DIM)

'''def majority_rule(v, n):
    half_n = n // 2
    if v == half_n:
        return random.getrandbits(1)
    return v > half_n'''  

# Optimized version of majority rule above
def majority_rule_vectorized(hv, n):
    half_n = n // 2
    rand = np.random.randint(0, 2, size=hv.shape, dtype=bool)
    return np.where(hv == half_n, rand, hv > half_n)    

# Flipping the vectors for local linear mapping
def generate_flip_sequence(base_vec, length, split):
    DIM = len(base_vec)
    flip_sequence = [base_vec.copy()]
    
    # Determine segment edges
    edges = [0] + [int(round((length-1) * (i+1)/split)) for i in range(split)]
    
    # Total bits to flip across one segment
    bits_per_segment = DIM // 2
    #segment_steps = [(edges[i+1] - edges[i]) for i in range(len(edges)-1)]
    
    for seg in range(len(edges)-1):
        start_idx = edges[seg]
        end_idx = edges[seg+1]
        steps_in_segment = end_idx - start_idx
        
        # Bits to flip per step in this segment
        bits_to_flip = int(bits_per_segment / steps_in_segment)
        
        # Track flipped indices per segment
        flipped_indices = set()
        
        for step in range(steps_in_segment):
            current_vec = flip_sequence[-1].copy()
            
            # Available indices for flipping (avoid repeats within the segment)
            available_indices = list(set(range(DIM)) - flipped_indices)
            
            # If fewer available than bits_to_flip, just take all
            if len(available_indices) < bits_to_flip:
                flip_indices = available_indices
            else:
                flip_indices = np.random.choice(available_indices, size=bits_to_flip, replace=False)
            
            # Flip the bits
            current_vec[flip_indices] = 1 - current_vec[flip_indices]
            
            # Update flipped indices
            flipped_indices.update(flip_indices)
            
            flip_sequence.append(current_vec)
    
    return np.array(flip_sequence)



# Shifting to have orthogonal vectors for patches
CIMx_patch = [np.roll(base_patch_x, i) for i in range(PATCH_SIZE)]
CIMy_patch = [np.roll(base_patch_y, i) for i in range(PATCH_SIZE)]

#Creating vectors with local linear mapping for S splits
CIMx_img = generate_flip_sequence(base_img_x,28,S)
CIMy_img = generate_flip_sequence(base_img_y,28,S)


# BINARIZATION
def binarize_image(image, Tbin=0):
    return np.where(image > Tbin, 1, 0)

# CREATING PATCHES
def select_pois_and_create_patches(image_bin):
    patches = []
    coords = []
    
    h, w = image_bin.shape
    r = PATCH_SIZE // 2

    for x in range(r, h - r):
        for y in range(r, w - r):
            if image_bin[x, y] == 1:
                patch = image_bin[x-r:x+r+1, y-r:y+r+1]
                patches.append(patch)
                coords.append((x, y))

    return patches, coords


# ENCODING PATCH
def encode_patch(patch):
    hv = np.zeros(DIM)
    count = 0
    for i in range(PATCH_SIZE):
        for j in range(PATCH_SIZE):
            val = patch[i, j]

            vx = CIMx_patch[i]
            vy = CIMy_patch[j]
            vp = IM[val]

            count+=1

            hv += vx ^ vy ^ vp       
    #a = np.array([majority_rule(val, count) for val in hv])
    a = majority_rule_vectorized(hv,count)
    return a

# ENCODING IMAGE
def encode_image(image):
    hv = np.zeros(DIM)
    count = 0
    image_bin = binarize_image(image)
    patches, coords = select_pois_and_create_patches(image_bin)
    for patch, (x, y) in zip(patches, coords):

        patch_hv = encode_patch(patch)

        vx = CIMx_img[x]
        vy = CIMy_img[y]
        count +=1
        hv += patch_hv ^ vx ^ vy
    # the version without patches
    ''''for x in range(IMG_H):
        for y in range(IMG_W):
            pixel = image_bin[x, y]

            # encode pixel instead of patch
            pixel_hv = IM[pixel]

            vx = CIMx_img[x]
            vy = CIMy_img[y]

            hv += pixel_hv ^ vx ^ vy
            count += 1'''

    #return np.array([majority_rule(val, count) for val in hv])
    return majority_rule_vectorized(hv,count)


# TRAIN / PREDICT


def train_hdc_model(X_train, y_train):
    class_hvs = np.zeros((10, DIM))
    class_counts = np.zeros(10, dtype=int)
    i = 0
    for image, label in zip(X_train, y_train):
        # Printing the progress
        i+=1
        if i % 5000 == 0:
            print(str(i) + ' images are done\n')
        hv = encode_image(image)
        class_hvs[label] += hv
        class_counts[label] += 1
        
    final_hvs = np.zeros_like(class_hvs, dtype=bool)

    for c in range(10):
        final_hvs[c] = majority_rule_vectorized(class_hvs[c], class_counts[c])

    return final_hvs

# HAMMING SIMILARITY
def hamming_similarity(v1, v2):
    hamming_dist = np.sum(v1 != v2) / v1.size
    return 1 - hamming_dist

# PREDICTION
def predict(class_hvs, image):
    hv = encode_image(image)
    sims = np.array([hamming_similarity(class_hv, hv) for class_hv in class_hvs])
    return np.argmax(sims)


# LOAD DATA

train_images_path = 'train-images.idx3-ubyte'
train_labels_path = 'train-labels.idx1-ubyte'
test_images_path = 't10k-images.idx3-ubyte'
test_labels_path = 't10k-labels.idx1-ubyte'

mnist_loader = MnistDataloader(train_images_path, train_labels_path,
                               test_images_path, test_labels_path)


(x_train, y_train), (x_test, y_test) = mnist_loader.load_data()


# TRAIN

start_train = time.time()
class_hvs = train_hdc_model(x_train, y_train)
train_time = time.time() - start_train

# TEST


start_test = time.time()
y_pred = [predict(class_hvs, img) for img in x_test]
test_time = time.time() - start_test

monitoring = False
monitor_thread.join()

# METRICS

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\nMetrics")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nTiming")
print(f"Training Time : {train_time:.2f}s")
print(f"Testing Time  : {test_time:.2f}s")

print("\nResource Usage")

if len(cpu_usage) > 0:
    print(f"Avg CPU Usage : {np.mean(cpu_usage):.2f}%")
    print(f"Max CPU Usage : {np.max(cpu_usage):.2f}%")

if len(ram_usage) > 0:
    print(f"Avg RAM Usage : {np.mean(ram_usage):.2f}%")
    print(f"Max RAM Usage : {np.max(ram_usage):.2f}%")

print("\nPer-Core CPU Usage")
for i, core in enumerate(cpu_core_usage):
    if len(core) > 0:
        print(f"Core {i}: Avg={np.mean(core):.2f}% | Max={np.max(core):.2f}%")


# PLOTS

plt.figure(figsize=(8, 5))
sns.barplot(x=["Accuracy", "Precision", "Recall", "F1"],
            y=[accuracy, precision, recall, f1])
plt.title("Metrics")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap="Blues")

plt.figure(figsize=(6, 4))
sns.barplot(x=["Train", "Test"], y=[train_time, test_time])

plt.show()