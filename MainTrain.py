import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Define the directory where the dataset is stored
image_directory='dataset/'


# Load image filenames for benign and malignant categories
bening_images=os.listdir(image_directory+'benign/')
malignant_images=os.listdir(image_directory+'malignant/')
dataset=[]
label = []
# print(bening_images)
INPUT_SIZE = 64

# Load and preprocess benign images
for image_name in bening_images:
    if image_name.lower().endswith('.png'):
        image_path = os.path.join(image_directory, 'benign', image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = Image.fromarray(image)
        image = image.resize((64, 64))
        dataset.append(np.array(image))
        label.append(0)


# Load and preprocess malignant images
for image_name in malignant_images:
    if image_name.lower().endswith('.png'):
        image_path = os.path.join(image_directory, 'malignant', image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((64, 64))
        dataset.append(np.array(image))
        label.append(1)


# Convert dataset and label lists to NumPy arrays
print=np.array(dataset)
print=np.array(label)



# Split the dataset into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(
    dataset, label, test_size=0.2, random_state=0)


# Normalize the pixel values to the range [0, 1]
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# Convert class labels to one-hot encoded format
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Build the CNN model
model = Sequential()

# First convolutional block
model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional block
model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third convolutional block
model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten and fully connected layers
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

# Compile the model with loss function, optimizer, and evaluation metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model using training data and validate on test data
history = model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10,
                     validation_data=(x_test, y_test), shuffle=False)



# Plot accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o') # type: ignore
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o') # type: ignore
plt.title('Model Accuracy over 10 Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o') # type: ignore
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o') # type: ignore
plt.title('Model Loss over 10 Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Predict the classes on the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Display the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get predicted probabilities for the positive class (malignant)
y_pred_prob = model.predict(x_test)[:, 1]  # Probability of class 1 (malignant)
y_true = np.argmax(y_test, axis=1)

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()



model.save('BreastCancer10Epochs.h5')
