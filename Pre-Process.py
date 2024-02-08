import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.preprocessing import LabelEncoder


# Function to check if an image file is valid
def is_valid_image_file(file_path):
    try:
        img = cv2.imread(file_path)
        if img is None or img.size == 0:
            return False
        return True
    except Exception as e:
        return False

# Function to apply advanced data augmentation
def apply_advanced_data_augmentation(image):
    # Random rotation
    angle = np.random.uniform(-30, 30)
    image = tf.keras.preprocessing.image.apply_affine_transform(image, theta=angle)

    # Random scaling
    scale_factor = np.random.uniform(0.8, 1.2)
    image = tf.keras.preprocessing.image.apply_affine_transform(image, zx=scale_factor, zy=scale_factor)

    # Random translation
    tx = np.random.uniform(-20, 20)
    ty = np.random.uniform(-20, 20)
    image = tf.keras.preprocessing.image.apply_affine_transform(image, tx=tx, ty=ty)

    # Random erasing
    if np.random.rand() > 0.5:
        p = 0.2
        s_l = 0.02
        s_h = 0.4
        r_1 = 0.3
        r_2 = 1 / r_1
        v_l = 0
        v_h = 255
        pixel_value = np.random.uniform(v_l, v_h)
        while True:
            s = np.random.uniform(s_l, s_h) * image.size
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, image.shape[1])
            top = np.random.randint(0, image.shape[0])
            if left + w <= image.shape[1] and top + h <= image.shape[0]:
                break
        image[top:top + h, left:left + w, :] = pixel_value

    # Random Gaussian Blur
    if np.random.rand() > 0.5:
        k_size = np.random.choice([3, 5, 7])
        image = cv2.GaussianBlur(image, (k_size, k_size), 0)

    # Random Brightness and Contrast Adjustment
    if np.random.rand() > 0.5:
        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.uniform(-10, 10)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Random channel shifts
    if np.random.rand() > 0.5:
        channel_shift_range = np.random.uniform(-10, 10)
        image = tf.keras.preprocessing.image.apply_channel_shift(image, channel_shift_range, channel_axis=2)

    # Random shear
    if np.random.rand() > 0.5:
        shear_range = np.random.uniform(-0.1, 0.1)
        image = tf.keras.preprocessing.image.apply_affine_transform(image, shear=shear_range)

    # Random zoom
    if np.random.rand() > 0.5:
        zoom_range = np.random.uniform(0.8, 1.2)
        image = tf.keras.preprocessing.image.apply_affine_transform(image, zx=zoom_range, zy=zoom_range)

    # Random horizontal flip
    if np.random.rand() > 0.5:
        image = np.fliplr(image)

    # Random vertical flip
    if np.random.rand() > 0.5:
        image = np.flipud(image)

    # Clip pixel values to the valid range [0, 255]
    image = np.clip(image, 0, 255)

    return image

# Function to visualize original and augmented images
def visualize_augmentation(original_images, augmented_images, original_labels, augmented_labels, num_samples=5):
# Select a random subset of images for visualization
    indices_to_visualize = np.random.randint(0, len(original_images), num_samples)

    # Plot the original and augmented images
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices_to_visualize):
        plt.subplot(2, num_samples, i + 1)

        # Convert the original image to RGB if it's in BGR format
        original_img = original_images[idx]
        if original_img.dtype != np.uint8:
            original_img = (original_img * 255).astype(np.uint8)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        plt.imshow(original_img)  # Plot the original image
        plt.title(f'Original - Label: {original_labels[idx]}')
        plt.axis('off')

        plt.subplot(2, num_samples, num_samples + i + 1)

        # Convert the augmented image to RGB if it's in BGR format
        augmented_img = augmented_images[idx]
        if augmented_img.dtype != np.uint8:
            augmented_img = (augmented_img * 255).astype(np.uint8)
        augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB)
        plt.imshow(augmented_img)  # Plot the augmented image
        plt.title(f'Augmented - Label: {augmented_labels[idx]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    
main_folder = 'ReSizedDataset/'  

def load_and_advanced_preprocess_images(main_folder, target_size=(160, 160), augmentation=True, max_images_per_folder=500):
    images = []  # Store processed images
    labels = []  # Store labels corresponding to each image
    valid_image_paths = []  # Store valid image paths for DataFrame

    label_encoder = LabelEncoder()
    clone_labels = ['TRI5001', 'TRI5002', 'TRI5003', 'TRI5004']
    encoded_labels = label_encoder.fit_transform(clone_labels)
    label_map = dict(zip(clone_labels, encoded_labels))

    for clone_label in clone_labels:
        subfolder_path = os.path.join(main_folder, clone_label)
        loaded_images = 0
        for filename in tqdm(os.listdir(subfolder_path), desc=f"Processing images from {clone_label}"):
            if loaded_images >= max_images_per_folder:
                break
            img_path = os.path.join(subfolder_path, filename)
            if not is_valid_image_file(img_path):
                continue
            img = cv2.imread(img_path)
            img = cv2.resize(img, target_size)
            images.append(img)
            labels.append(label_map[clone_label])
            valid_image_paths.append(img_path) 

            if augmentation:
                augmented_img = apply_advanced_data_augmentation(img)
                images.append(augmented_img)
                labels.append(label_map[clone_label])
                valid_image_paths.append(img_path)

            loaded_images += 1

    return np.array(images), np.array(labels), valid_image_paths

images, labels, image_paths = load_and_advanced_preprocess_images(main_folder, augmentation=True)

data_dict = {
    "Image_Path": image_paths,  
    "Label": labels
}
df = pd.DataFrame(data_dict)
df.to_csv('preprocessed-data/image_data.csv', index=False)


# Record the start time
start_time = time.time()

# Combine data and labels
X = np.array(images)
y = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df = pd.DataFrame(data_dict)
df.to_csv('preprocessed-data/image_data.csv', index=False)

# Print shapes of data and labels
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Convert BGR images to RGB before saving as NumPy arrays
X_train_rgb = [cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB) 
               if img.dtype != np.uint8 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in X_train]
X_test_rgb = [cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB) 
              if img.dtype != np.uint8 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in X_test]

# Save the preprocessed data as NumPy arrays
np.save('preprocessed-data/X_train_large_rgb.npy', np.array(X_train_rgb))
np.save('preprocessed-data/X_test_large_rgb.npy', np.array(X_test_rgb))
np.save('preprocessed-data/y_train_large.npy', y_train)
np.save('preprocessed-data/y_test_large.npy', y_test)

# Calculate and print the total count of the dataset
total_dataset_count = len(X_train) + len(X_test)
print("Total dataset count after preprocessing:", total_dataset_count)

# Record the end time
end_time = time.time()

# Calculate and print the total time taken
total_time = end_time - start_time
print("Total time taken:", total_time, "seconds")