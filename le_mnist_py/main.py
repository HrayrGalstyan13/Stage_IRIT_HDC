import numpy as np
import tensorflow as tf
from BNN import *
import struct
from array import array
import time
# Constants
DIM = 10000  # Hyperdimensional vector size
SEED = 42     # Random seed

np.random.seed(SEED)

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
num_feature = 784
num_value = 256
feature_memory = np.zeros((num_feature, DIM))
char_memory = np.zeros((num_value, DIM)) 

def create_char_vector():
    for k in range(num_value):
        char_memory[k] = np.random.choice([-1, 1], DIM)

def create_feature_vector():
    for k in range(num_feature):
        feature_memory[k] = np.random.choice([-1, 1], DIM)

create_char_vector()
create_feature_vector()    


def sign_plus(x):
    return 2 * (x >= 0) - 1

def encode_image(image):
    hv = np.zeros(DIM)
    flat_img = image.flatten()
    
    hv = (feature_memory * char_memory[flat_img]).sum(axis=0)
    
    return sign_plus(hv)

train_images_path = 'train-images.idx3-ubyte'
train_labels_path = 'train-labels.idx1-ubyte'
test_images_path = 't10k-images.idx3-ubyte'
test_labels_path = 't10k-labels.idx1-ubyte'

mnist_loader = MnistDataloader(train_images_path, train_labels_path,
                               test_images_path, test_labels_path)

(x_train, y_train), (x_test, y_test) = mnist_loader.load_data()

start_encoding = time.time()
print("Encoding training data...")
X_train_encoded = np.array([encode_image(img) for img in x_train])

end_encoding = time.time()

print(f"Training encoding time: "
      f"{end_encoding - start_encoding:.2f} seconds")


print("Encoding test data...")
start_encoding = time.time()
X_test_encoded  = np.array([encode_image(img) for img in x_test])

end_encoding = time.time()

print(f"Testing encoding time: "
      f"{end_encoding - start_encoding:.2f} seconds")




x_train = X_train_encoded.reshape((X_train_encoded.shape[0], -1)).astype(np.float32)
x_test  = X_test_encoded.reshape((X_test_encoded.shape[0], -1)).astype(np.float32)

# Labels
y_train = np.array(y_train, dtype=np.int64).flatten()
y_test  = np.array(y_test, dtype=np.int64).flatten()


# Hyperparameters


D = x_train.shape[1]
num_class = len(np.unique(y_train))

dropout_rate = 0.5
weight_decay = 5e-2
learning_rate = 1e-2

batch_size = 50
epochs = 100


# Create tf.data datasets


train_dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)
)

train_dataset = (
    train_dataset
    .shuffle(buffer_size=len(x_train))
    .batch(batch_size)
)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)
).batch(batch_size)


# Create model

model = BHDC(
    inshape=D,
    outshape=num_class,
    dropout_prob=dropout_rate
)


# Loss function

criterion = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True
)

# Optimizer

optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate
)


# Metrics logs


train_BNN_acc = []
test_BNN_acc = []

loss_accum = 0
loss_log = 0


# Training loop

start_training = time.time()
for epoch in range(epochs):

    #TRAIN
    for batch_train, batch_label in train_dataset:

        with tf.GradientTape() as tape:

            # Forward pass
            y_pred = model(batch_train, training=True)

            # Cross entropy loss
            ce_loss = criterion(batch_label, y_pred)

            # Weight decay (L2 regularization)
            l2_loss = tf.add_n([
                tf.nn.l2_loss(v)
                for v in model.trainable_variables
            ])

            loss = ce_loss + weight_decay * l2_loss

        # Compute gradients
        gradients = tape.gradient(
            loss,
            model.trainable_variables
        )

        # Update weights
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables)
        )

        # Match PyTorch random accumulation based on open source code
        if np.random.rand() < 0.2:
            loss_accum += float(loss.numpy())

    # EVALUATION

    print(epoch, loss_accum, end='\t')

    # Test predictions
    test_logits = model(x_test, training=False)
    test_pred = tf.argmax(test_logits, axis=1)

    test_acc = np.mean(
        test_pred.numpy() == y_test
    )

    print(test_acc)

    test_BNN_acc.append(test_acc)

    # Train predictions
    train_logits = model(x_train, training=False)
    train_pred = tf.argmax(train_logits, axis=1)

    train_acc = np.mean(
        train_pred.numpy() == y_train
    )

    train_BNN_acc.append(train_acc)


    # Adaptive learning rate

    if loss_accum > loss_log:

        current_lr = optimizer.learning_rate.numpy()

        optimizer.learning_rate.assign(
            current_lr * 0.5
        )

    loss_log = loss_accum
    loss_accum = 0

end_training = time.time()

total_training_time = (
    end_training - start_training
)

print("\n")
print(f"Total training time: "
      f"{total_training_time:.2f} seconds")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# Final Evaluation

start_testing = time.time()
# Get predictions
test_logits = model(x_test, training=False)
y_pred = tf.argmax(test_logits, axis=1).numpy()
end_testing = time.time()
total_testing_time = end_testing-start_testing
print("\n")
print(f"Total testing time: "
      f"{total_testing_time:.2f} seconds")
# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Precision
precision = precision_score(
    y_test,
    y_pred,
    average='weighted'
)

# Recall
recall = recall_score(
    y_test,
    y_pred,
    average='weighted'
)

# F1 Score
f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)


# Print Results


print("\nFINAL METRICS")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nCONFUSION MATRIX")
print(conf_matrix)

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        y_pred
    )
)