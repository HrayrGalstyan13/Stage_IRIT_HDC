import numpy as np
import struct
from array import array
import time
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import threading
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Constants
DIM = 50000  # Hyperdimensional vector size
SEED = 42     # Random seed

np.random.seed(SEED)

# Resource Monitoring Variables
cpu_usage = []
ram_usage = []
cpu_core_usage = [[] for _ in range(psutil.cpu_count(logical=False))]  # Separate list for each core
monitoring = True  # Turn False in case of errors

# Function to monitor resources (CPU and RAM)
def monitor_resources():
    while monitoring:
        cpu_usage.append(psutil.cpu_percent(interval=0.2))
        ram_usage.append(psutil.virtual_memory().percent)
        
        # Get per-core CPU usage
        per_core_usage = psutil.cpu_percent(percpu=True, interval=0.2)
        for i, usage in enumerate(per_core_usage):
            if i < len(cpu_core_usage):
                cpu_core_usage[i].append(usage)

monitor_thread = threading.Thread(target=monitor_resources)
monitor_thread.start()

class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        # Read labels
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
            labels = array("B", file.read())

        # Read images
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
            image_data = array("B", file.read())

        images = np.array(image_data).reshape(size, rows, cols)
        return images, labels
    
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


# Global character memory for HDC encoding
char_memory = {}

# Improved HDC Encoding for MNIST
def get_char_vector(char):
    if char not in char_memory:
        char_memory[char] = np.random.choice([-1, 1], DIM)
    return char_memory[char]

def encode_image(image):
    """
    Encode a single MNIST image into a hyperdimensional vector.
    - Flatten image to 1D
    - Bin pixel values (0-7) to reduce random vectors
    - Permute vectors based on pixel position
    """
    hv = np.zeros(DIM)
    flat_img = image.flatten()
    
    for i, pixel in enumerate(flat_img):
        binned_pixel = pixel // 32  # 0-7
        pixel_vec = get_char_vector(str(binned_pixel))
        permuted = np.roll(pixel_vec, i % DIM)
        hv += permuted
    
    return np.sign(hv)

# Training
def train_hdc_model(X_train, y_train):
    class_hvs = np.zeros((10, DIM))  # 10 classes for MNIST

    for image, label in zip(X_train, y_train):
        hv = encode_image(image)
        class_hvs[label] += hv

    class_hvs = np.sign(class_hvs)
    return class_hvs

# Prediction
def predict(class_hvs, image):
    hv = encode_image(image)
    sims = class_hvs @ hv / (np.linalg.norm(class_hvs, axis=1) * np.linalg.norm(hv) + 1e-8)
    return np.argmax(sims)

# Load MNIST Data
train_images_path = 'train-images.idx3-ubyte'
train_labels_path = 'train-labels.idx1-ubyte'
test_images_path = 't10k-images.idx3-ubyte'
test_labels_path = 't10k-labels.idx1-ubyte'

mnist_loader = MnistDataloader(train_images_path, train_labels_path,
                               test_images_path, test_labels_path)

(x_train, y_train), (x_test, y_test) = mnist_loader.load_data()

# Train HDC model
start_train = time.time()
class_hvs = train_hdc_model(x_train, y_train)
train_time = time.time() - start_train

# Testing Phase
start_test = time.time()
y_pred = [predict(class_hvs, img) for img in x_test]
test_time = time.time() - start_test

# Stop monitoring (no effect since monitoring=False)
monitoring = False
monitor_thread.join()

# Metrics calculations
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

# Print Metrics
print("\nGlobal Metrics")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# Timing
print("\nTiming")
print(f"Training Time : {train_time:.2f}s")
print(f"Testing Time  : {test_time:.2f}s")

# Resource usage
print("\nResource Usage")
if cpu_usage:
    print(f"Avg CPU Usage: {np.mean(cpu_usage):.2f}%")
    print(f"Max CPU Usage: {np.max(cpu_usage):.2f}%")
else:
    print("CPU usage data not collected.")

if ram_usage:
    print(f"Avg RAM Usage: {np.mean(ram_usage):.2f}%")
    print(f"Max RAM Usage: {np.max(ram_usage):.2f}%")
else:
    print("RAM usage data not collected.")

# Per-core CPU usage
for i, core_usage in enumerate(cpu_core_usage):
    if core_usage:
        print(f"Avg CPU Core {i} Usage: {np.mean(core_usage):.2f}%")

# Plotting
plt.ion()

plt.figure(figsize=(8, 5))
sns.barplot(x=["Accuracy", "Precision", "Recall", "F1"],
            y=[accuracy, precision, recall, f1])
plt.title("Global Metrics")
plt.ylim(0, 1)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap="Blues", xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.figure(figsize=(8, 5))
sns.barplot(x=["Train", "Test"], y=[train_time, test_time])
plt.title("Execution Time (seconds)")

plt.show(block=False)
plt.pause(30)