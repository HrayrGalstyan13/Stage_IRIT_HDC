import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import threading

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)


DIM = 10000
NGRAM = 3
SEED = 42

np.random.seed(SEED)


# Resource Monitoring

cpu_usage = []
ram_usage = []
monitoring = True

def monitor_resources():
    while monitoring:
        cpu_usage.append(psutil.cpu_percent(interval=0.2))
        #psutil.cpu_percent(percpu=True)
        ram_usage.append(psutil.virtual_memory().percent)

monitor_thread = threading.Thread(target=monitor_resources)
monitor_thread.start()

# Load Dataset
df = pd.read_csv("language_detection.csv")

le = LabelEncoder()
df["label"] = le.fit_transform(df["Language"])

X_train, X_test, y_train, y_test = train_test_split(
    df["Text"], df["label"], test_size=0.2, random_state=SEED
)

# Item Memory
char_memory = {}

def get_char_vector(char):
    if char not in char_memory:
        char_memory[char] = np.random.choice([-1, 1], DIM)
    return char_memory[char]


# Encoding
def encode_text(text):
    text = str(text).lower()
    hv = np.zeros(DIM)

    for i in range(len(text) - NGRAM + 1):
        ngram = text[i:i+NGRAM]
        ngram_vec = np.ones(DIM)

        for j, ch in enumerate(ngram):
            ch_vec = get_char_vector(ch)
            permuted = np.roll(ch_vec, j)
            ngram_vec *= permuted

        hv += ngram_vec

    return np.sign(hv)

# TRAINING
start_train = time.time()

num_classes = len(le.classes_)
class_hvs = np.zeros((num_classes, DIM))

encode_time = 0

for text, label in zip(X_train, y_train):
    t0 = time.time()
    hv = encode_text(text)
    encode_time += time.time() - t0

    class_hvs[label] += hv

class_hvs = np.sign(class_hvs)

train_time = time.time() - start_train


# PREDICTION
def predict(text):
    hv = encode_text(text)

    sims = class_hvs @ hv / (
        np.linalg.norm(class_hvs, axis=1) * np.linalg.norm(hv) + 1e-8
    )
    return np.argmax(sims)

start_test = time.time()

y_pred = []
inference_time = 0

for text in X_test:
    t0 = time.time()
    y_pred.append(predict(text))
    inference_time += time.time() - t0

test_time = time.time() - start_test

# Stop monitoring
monitoring = False
monitor_thread.join()

# METRICS
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\nGlobal Metrics")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# Per-class metrics
prec, rec, f1_per_class, _ = precision_recall_fscore_support(
    y_test, y_pred, labels=range(len(le.classes_))
)


# TIMING INFO
print("\nTiming")
print(f"Training Time : {train_time:.2f}s")
print(f"Encoding Time : {encode_time:.2f}s")
print(f"Testing Time  : {test_time:.2f}s")
print(f"Inference Time: {inference_time:.2f}s")


# RESOURCE USAGE
print("\nResource Usage")
print(f"Avg CPU Usage: {np.mean(cpu_usage):.2f}%")
print(f"Max CPU Usage: {np.max(cpu_usage):.2f}%")
print(f"Avg RAM Usage: {np.mean(ram_usage):.2f}%")
print(f"Max RAM Usage: {np.max(ram_usage):.2f}%")


# PLOTS (Non-blocking)

plt.ion()  # interactive mode ON

# 1. Global Metrics
plt.figure(figsize=(8, 5))
sns.barplot(x=["Accuracy", "Precision", "Recall", "F1"],
            y=[accuracy, precision, recall, f1])
plt.title("Global Metrics")
plt.ylim(0, 1)

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm,
            cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# 3. Per-language Metrics
x = np.arange(len(le.classes_))

plt.figure(figsize=(12, 6))
plt.bar(x - 0.25, prec, width=0.25, label="Precision")
plt.bar(x, rec, width=0.25, label="Recall")
plt.bar(x + 0.25, f1_per_class, width=0.25, label="F1")

plt.xticks(x, le.classes_, rotation=45)
plt.legend()
plt.title("Per-Language Metrics")

# 4. Timing Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=["Train", "Encode", "Test", "Infer"],
            y=[train_time, encode_time, test_time, inference_time])
plt.title("Execution Time (seconds)")

# 5. Resource Usage Plot
plt.figure(figsize=(10, 5))
plt.plot(cpu_usage, label="CPU %")
plt.plot(ram_usage, label="RAM %")
plt.legend()
plt.title("Resource Usage Over Time")


plt.show(block=False)

# Short pause so figures render, then script exits
plt.pause(30) 