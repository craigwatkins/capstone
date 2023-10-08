import numpy as np
from tensorflow import keras
import sqlite3
from sklearn.model_selection import train_test_split
from make_datasets import create_dataset_reds, create_dataset_blues, create_dataset_greens


def create_dataset_for_image_list(img_list, make_dataset_func):
    """
    Create a dataset for a list of images using a specified dataset creation function.

    Parameters:
    - img_list: List of images
    - make_dataset_func: Function to create dataset for a single image

    Returns:
    - Tuple of data and labels
    """
    data_list = []
    label_list = []

    for img in img_list:
        data, labels = make_dataset_func(img)
        data_list.append(data)
        label_list.append(labels)

    return np.vstack(data_list), np.hstack(label_list)


def train_model(training_images, val_images, model_save_path, training_func):
    """
    Train a neural network model for the given training and validation images.

    Parameters:
    - training_images: List of training images
    - val_images: List of validation images
    - model_save_path: Path to save the trained model
    - training_func: Function to create dataset for training

    Returns:
    - Trained model
    """
    # Create datasets for training and validation
    train_data, train_targets = create_dataset_for_image_list(training_images, training_func)
    val_data, val_targets = create_dataset_for_image_list(val_images, training_func)

    # Define the model architecture
    model = keras.Sequential([
        keras.layers.Dense(48, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(train_data, train_targets, validation_data=(val_data, val_targets), epochs=2, batch_size=800,
              shuffle=True)

    # Save the trained model
    model.save(model_save_path)

    return model


def get_images():
    """
    Fetch images from the database.

    Returns:
    - Tuple of training and validation images
    """
    start_id = 11000
    num_images = 80000

    # Fetch images
    cursor.execute('''
               SELECT image_id, red, green, blue 
               FROM satellite_images 
               WHERE image_id >= ? 
               LIMIT ?
           ''', (start_id, num_images))
    images = cursor.fetchall()
    train_height_width = (28, 28)
    image_data = []

    for img in images:
        img_id, r_data, g_data, b_data = img
        # Convert byte data to numpy arrays
        r = np.frombuffer(r_data, dtype=np.uint8).reshape(train_height_width)
        g = np.frombuffer(g_data, dtype=np.uint8).reshape(train_height_width)
        b = np.frombuffer(b_data, dtype=np.uint8).reshape(train_height_width)

        # Combine the three color channels
        image = [r, g, b]
        image_data.append(image)

    # Split the data into training and validation sets
    train_images, validation_images = train_test_split(image_data, test_size=0.2, random_state=42)

    return train_images, validation_images


def call_train_model():
    """
    Fetch images and train the model.
    """
    train_images, validation_images = get_images()
    model_save_path = "regression_model_reds3.keras"
    model = train_model(train_images, validation_images, model_save_path, create_dataset_reds)

    # Save the trained model
    model.save(model_save_path)


# Main execution
if __name__ == "__main__":
    # Connect to the database
    conn = sqlite3.connect('images_features.db')
    cursor = conn.cursor()

    # Call the training function
    call_train_model()

    # Close the database connection
    conn.close()

    # dependencies
    # pip install numpy tensorflow scikit-learn
