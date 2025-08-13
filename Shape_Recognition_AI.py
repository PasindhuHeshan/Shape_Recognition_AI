#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Install kaggle API for dataset
get_ipython().system('pip install -q kaggle')


# In[2]:


#Upload kaggle API credentials
from google.colab import files
files.upload()


# In[3]:


#Configure kaggle environment
get_ipython().system('mkdir ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# In[4]:


# Download and extract dataset
get_ipython().system('kaggle datasets download -d singhnavjot2062001/geometric-shapes-circle-square-triangle')
get_ipython().system('unzip geometric-shapes-circle-square-triangle.zip -d shapes_dataset')

# Set dataset directory
dataset_dir = "./shapes_dataset"
temp_dir = "./shapes_dataset"  # For compatibility with later cells


# In[5]:


import os
import shutil
import random
from PIL import Image, ImageDraw

def analyze_directory_structure(root_path):
    """Print directory structure"""
    print("\nAnalyzing directory structure:")
    for root, dirs, files in os.walk(root_path):
        level = root.replace(root_path, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files[:5]:
            print(f"{subindent}{f}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files)-5} more files")


def organize_dataset(raw_path, target_path, test_ratio=0.2, val_ratio=0.2):
    """Organize raw dataset into train/test/val structure automatically"""
    print("\nOrganizing dataset...")

    # Detect classes from folder names in raw_path
    classes = [d for d in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, d))]
    print(f"Detected classes: {classes}")

    # Create directories
    for split in ['train', 'test', 'val']:
        for class_name in classes:
            os.makedirs(os.path.join(target_path, split, class_name), exist_ok=True)

    # Copy files into splits
    for class_name in classes:
        src_dir = os.path.join(raw_path, class_name)
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(files)

        test_count = int(len(files) * test_ratio)
        val_count = int(len(files) * val_ratio)

        test_files = files[:test_count]
        val_files = files[test_count:test_count + val_count]
        train_files = files[test_count + val_count:]

        for f in train_files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(target_path, 'train', class_name, f))
        for f in test_files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(target_path, 'test', class_name, f))
        for f in val_files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(target_path, 'val', class_name, f))

        print(f"{class_name}: {len(train_files)} train, {len(test_files)} test, {len(val_files)} val")


def verify_dataset_structure(train_path, test_path, val_path):
    """Verify dataset structure (safe)"""
    for path in [train_path, test_path, val_path]:
        os.makedirs(path, exist_ok=True)

    train_classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    print("\nVerification:")
    print(f"Train classes: {train_classes}")
    for split, path in [('train', train_path), ('test', test_path), ('val', val_path)]:
        print(f"\n{split.upper()} set:")
        for class_name in train_classes:
            class_dir = os.path.join(path, class_name)
            count = len(os.listdir(class_dir)) if os.path.exists(class_dir) else 0
            print(f"  {class_name}: {count} images")


def create_sample_dataset(path):
    """Create a small sample dataset with shapes if it doesn't exist"""
    print("\nCreating sample dataset...")
    os.makedirs(path, exist_ok=True)
    shapes = ["circle", "square", "triangle"]

    for shape in shapes:
        shape_dir = os.path.join(path, shape)
        os.makedirs(shape_dir, exist_ok=True)
        for i in range(10):  # 10 images per class
            img = Image.new('RGB', (64, 64), color='white')
            draw = ImageDraw.Draw(img)
            if shape == "circle":
                draw.ellipse([16, 16, 48, 48], outline="black", fill="red")
            elif shape == "square":
                draw.rectangle([16, 16, 48, 48], outline="black", fill="blue")
            elif shape == "triangle":
                draw.polygon([(32, 8), (8, 56), (56, 56)], outline="black", fill="green")
            img.save(os.path.join(shape_dir, f"{shape}_{i}.png"))
    print("Sample dataset created.")


# =====================
# MAIN EXECUTION (Colab local paths, no Drive)
# =====================

# Change this if your KaggleHub download path is different
raw_path = "/content/shapes_raw"        # Folder with class subfolders
target_path = "/content/shapes_split"   # Where train/test/val will go

# If raw dataset doesn't exist, create a sample one for testing
if not os.path.exists(raw_path) or not os.listdir(raw_path):
    create_sample_dataset(raw_path)

# Analyze original dataset
analyze_directory_structure(raw_path)

# Organize into train/test/val
organize_dataset(raw_path, target_path, test_ratio=0.2, val_ratio=0.2)

# Verify the split
train_path = os.path.join(target_path, "train")
test_path = os.path.join(target_path, "test")
val_path = os.path.join(target_path, "val")
verify_dataset_structure(train_path, test_path, val_path)


# In[7]:


import os
import random
import shutil
import tensorflow as tf

# Paths
dataset_dir = "/content/shapes_dataset"  # Change if different
train_path = os.path.join(dataset_dir, 'train')
val_path = os.path.join(dataset_dir, 'val')
test_path = os.path.join(dataset_dir, 'test')

def rename_folders_lowercase(base_path):
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(base_path, split)
        if os.path.exists(split_path):
            for folder in os.listdir(split_path):
                old_path = os.path.join(split_path, folder)
                new_path = os.path.join(split_path, folder.lower())
                if os.path.isdir(old_path) and old_path != new_path:
                    os.rename(old_path, new_path)

rename_folders_lowercase(dataset_dir)

# Ensure validation split exists
def create_validation_split(train_dir, val_dir, val_ratio=0.2):
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    class_names_local = [
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ]
    for class_name in class_names_local:
        src_dir = os.path.join(train_dir, class_name)
        dst_dir = os.path.join(val_dir, class_name)
        os.makedirs(dst_dir, exist_ok=True)

        files = [
            f for f in os.listdir(src_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        random.shuffle(files)
        val_count = max(1, int(len(files) * val_ratio))

        for f in files[:val_count]:
            shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))

# Run split creation if needed
create_validation_split(train_path, val_path, val_ratio=0.2)

# Class names (consistent with earlier fix)
class_names = ['circle', 'square', 'triangle']

# Dataset settings
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    class_names=class_names
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    class_names=class_names
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    class_names=class_names
)


# In[8]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()


# In[9]:


from tensorflow.keras import layers

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

augmented_train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = augmented_train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


# In[10]:


from tensorflow.keras import models

custom_cnn = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

custom_cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

custom_cnn.summary()


# In[11]:


epochs = 10
history_cnn = custom_cnn.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    ]
)

custom_cnn.save('geometric_shapes_cnn.h5')


# In[12]:


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

test_loss, test_acc = custom_cnn.evaluate(test_ds)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Confusion Matrix
y_true = []
y_pred = []
for images, labels in test_ds:
    y_true.extend(labels.numpy())
    y_pred.extend(custom_cnn.predict(images).argmax(axis=1))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_true, y_pred),
            annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.show()
plt.savefig('confusion_matrix_cnn.png')

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))


# In[13]:


from tensorflow.keras.applications import MobileNetV2

def preprocess_mobilenet(image, label):
    image = tf.image.resize(image, (224, 224))
    return image, label

mobilenet_train = train_ds.map(preprocess_mobilenet)
mobilenet_val = val_ds.map(preprocess_mobilenet)
mobilenet_test = test_ds.map(preprocess_mobilenet)

base_model = MobileNetV2(input_shape=(224, 224, 3),
                       include_top=False,
                       weights='imagenet',
                       pooling='avg')

base_model.trainable = False

mobilenet_model = models.Sequential([
    layers.Rescaling(1./127.5, offset=-1),
    base_model,
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

mobilenet_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_mobilenet = mobilenet_model.fit(
    mobilenet_train,
    validation_data=mobilenet_val,
    epochs=10,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)]
)
mobilenet_model.save('geometric_shapes_mobilenet.h5')


# In[14]:


# Evaluate MobileNetV2
test_loss_mobilenet, test_acc_mobilenet = mobilenet_model.evaluate(mobilenet_test)
print(f"\nMobileNetV2 Test Accuracy: {test_acc_mobilenet:.4f}")

# Confusion matrix
y_true_mobilenet = []
y_pred_mobilenet = []
for images, labels in mobilenet_test:
    y_true_mobilenet.extend(labels.numpy())
    y_pred_mobilenet.extend(mobilenet_model.predict(images, verbose=0).argmax(axis=1))

plt.figure(figsize=(8, 6))
cm_mobilenet = confusion_matrix(y_true_mobilenet, y_pred_mobilenet)
sns.heatmap(cm_mobilenet, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names,
            cmap='Blues')
plt.title('MobileNetV2 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
plt.savefig('confusion_matrix_mobilenet.png')

# Classification report
print("\nMobileNetV2 Classification Report:")
print(classification_report(y_true_mobilenet, y_pred_mobilenet,
                           target_names=class_names))

# Training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_mobilenet.history['accuracy'], label='Train Accuracy')
plt.plot(history_mobilenet.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy History')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_mobilenet.history['loss'], label='Train Loss')
plt.plot(history_mobilenet.history['val_loss'], label='Val Loss')
plt.title('Loss History')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('training_history_mobilenet.png')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_cnn.history['accuracy'], label='Train Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy History')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_cnn.history['loss'], label='Train Loss')
plt.plot(history_cnn.history['val_loss'], label='Val Loss')
plt.title('Loss History')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('training_history_cnn.png')


# In[15]:


# Performance Comparison
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

# First evaluate CNN model if not already done
if 'test_acc_cnn' not in locals():
    test_loss_cnn, test_acc_cnn = custom_cnn.evaluate(test_ds, verbose=0)
    print(f"Custom CNN Test Accuracy: {test_acc_cnn:.4f}")

# ROC Curve Function
def plot_roc_curve(model, dataset, model_name):
    y_true = []
    y_prob = []
    for images, labels in dataset:
        y_true.extend(labels.numpy())
        y_prob.extend(model.predict(images, verbose=0))

    y_true = label_binarize(y_true, classes=range(len(class_names)))
    y_prob = np.array(y_prob)

    plt.figure(figsize=(8, 6))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr,
                label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

# Generate ROC Curves
print("\nGenerating ROC Curves:")
plot_roc_curve(custom_cnn, test_ds, "Custom CNN")
plt.savefig('roc_curve_custom_cnn.png')
plot_roc_curve(mobilenet_model, mobilenet_test, "MobileNetV2")
plt.savefig('roc_curve_mobilenetv2.png')

# Model Comparison
plt.figure(figsize=(10, 5))
models = ['Custom CNN', 'MobileNetV2']
accuracies = [test_acc_cnn, test_acc_mobilenet]
colors = ['#1f77b4', '#ff7f0e']

bars = plt.bar(models, accuracies, color=colors)
plt.ylabel('Test Accuracy')
plt.title('Model Performance Comparison')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
plt.savefig('model_comparison.png')


# In[16]:


def plot_predictions(model, dataset, model_name, size=(224, 224)):
    plt.figure(figsize=(15, 12))
    for images, labels in dataset.take(1):
        if images.shape[1:3] != size:
            display_images = tf.image.resize(images, size)
        else:
            display_images = images

        preds = model.predict(images, verbose=0)
        for i in range(12):
            ax = plt.subplot(3, 4, i+1)
            plt.imshow(display_images[i].numpy().astype("uint8"))
            true_label = class_names[labels[i]]
            pred_label = class_names[np.argmax(preds[i])]
            confidence = np.max(preds[i])
            color = "green" if true_label == pred_label else "red"
            plt.title(f"True: {true_label}\nPred: {pred_label}\n({confidence:.2f})",
                     color=color, fontsize=9)
            plt.axis("off")
    plt.suptitle(f"{model_name} Predictions", fontsize=16)
    plt.tight_layout()
    plt.show()

print("\nSample Predictions:")
plot_predictions(custom_cnn, test_ds, "Custom CNN", size=(128, 128))
plt.savefig('sample_predictions_custom_cnn.png')
plot_predictions(mobilenet_model, mobilenet_test, "MobileNetV2")
plt.savefig('sample_predictions_mobilenet.png')


# In[17]:


import os
import zipfile
from google.colab import files

def package_results():
    """Package all results into a zip file for download"""
    try:
        # 1. Generate requirements.txt
        print("Generating requirements.txt...")
        get_ipython().system('pip freeze > requirements.txt')

        # 2. Find the notebook file
        notebook_files = [f for f in os.listdir() if f.endswith('.ipynb')]

        if not notebook_files:
            raise FileNotFoundError("No .ipynb notebook file found in current directory")

        notebook_name = notebook_files[0]  # Use first found notebook
        print(f"Found notebook: {notebook_name}")

        # 3. Convert notebook to python script
        print("Converting notebook to script...")
        get_ipython().system('jupyter nbconvert --to script "{notebook_name}"')
        py_script_name = notebook_name.replace('.ipynb', '.py')

        # 4. List all possible output files
        possible_files = [
            # Core files
            'requirements.txt',
            py_script_name,

            # Model files
            'geometric_shapes_custom_cnn.h5',
            'geometric_shapes_mobilenet.h5',

            # Visualization files
            'confusion_matrix_cnn.png',
            'confusion_matrix_mobilenet.png',
            'training_history_cnn.png',
            'training_history_mobilenet.png',
            'roc_curve_custom_cnn.png',
            'roc_curve_mobilenetv2.png',
            'model_comparison.png',
            'sample_predictions_custom_cnn.png',
            'sample_predictions_mobilenetv2.png'
        ]

        # 5. Create zip with only existing files
        print("\nCreating zip file...")
        with zipfile.ZipFile('shape_recognition_results.zip', 'w') as zipf:
            files_added = 0
            for file in possible_files:
                if os.path.exists(file):
                    zipf.write(file)
                    print(f"+ Added {file}")
                    files_added += 1
                else:
                    print(f"- Skipping {file} (not found)")

            if files_added == 0:
                raise ValueError("No files were added to the zip - nothing to download")

        # 6. Verify and download
        print("\nZip file created successfully. Contents:")
        with zipfile.ZipFile('shape_recognition_results.zip', 'r') as zipf:
            for f in zipf.namelist():
                print(f"  • {f}")

        print("\nInitiating download...")
        files.download('shape_recognition_results.zip')

    except Exception as e:
        print(f"\nError during packaging: {str(e)}")
        print("Make sure you've:")
        print("1. Run all cells that generate output files")
        print("2. Are in the correct directory")
        print("3. Have the notebook file present")

# Execute the packaging
package_results()

