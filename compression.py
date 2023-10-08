import numpy as np
from PIL import Image
from keras.models import Model, load_model
import padded_binary as pb
from huffman_compression import HuffmanCoding
from make_datasets import create_dataset_reds, create_dataset_blues, create_dataset_greens
import os
import matplotlib.pyplot as plt


def save_as_binary(file_path, file_name, integers):
    """
    Save a list of integers as a binary file using Huffman coding.

    Args:
        file_path (str): Path to the directory where the file will be saved.
        file_name (str): Name of the file without extension.
        integers (list): List of integers to be saved.

    Returns:
        int: Size of the saved file in bytes.
    """
    integers = [int(x) + 256 for x in integers]
    huff = HuffmanCoding()
    save_info, encoded_list = huff.compress(integers)
    full_path = file_path + file_name + '.bin'
    pb.write_padded_bytes(save_info, full_path)
    return os.path.getsize(full_path)


def load_from_binary(file_name):
    """
    Load integers from a binary file using Huffman coding.

        Args:
            file_name (str): Path to the binary file.

        Returns:
            list: List of decompressed integers.
    """
    huff = HuffmanCoding()
    binaries = pb.read_padded_bytes(file_name)
    decompressed = huff.decompress_file(binaries)
    decompressed = [x - 256 for x in decompressed]
    return decompressed


def create_dataset_for_image_list(img_list, make_dataset_func):
    """
    Create a dataset for a list of images using a specified function.

    Args:
        img_list (list): List of images.
        make_dataset_func (function): Function to create dataset for an image.

    Returns:
        tuple: Tuple containing data and labels.
    """
    data_list = []
    label_list = []
    for img in img_list:
        data, labels = make_dataset_func(img)
        data_list.append(data)
        label_list.append(labels)
    return np.vstack(data_list), np.hstack(label_list)


def get_residuals(model, image, channel, func):
    """
    Calculate residuals for an image using a trained model.

    Args:
        model (Model): Trained Keras model.
        image (list): List containing RGB channels of the image.
        channel (int): Index of the channel (0 for red, 1 for green, 2 for blue).
        func (function): Function to create dataset for an image.

    Returns:
        ndarray: Residuals for the image.
    """
    height, width = image[0].shape
    data, targets = func(image)
    model_predictions = model.predict(data, verbose=0)
    # reshape data to be 2D
    model_predictions = np.reshape(model_predictions, (height - 1, width - 1))
    image[channel] = image[channel].astype(np.int16)  # change image data type to avoid overflow
    model_predictions = model_predictions * 255.0  # convert predictions to actual value
    model_predictions = np.round(model_predictions)
    # for each pixel in the top row of the image, get the difference between the pixel and the pixel to its left
    top_row = np.diff(image[channel][0, :])
    top_row = np.insert(top_row, 0, image[channel][0, 0])  # the first value is just the actual pixel value
    top_row = np.reshape(top_row, (1, width))  # reshape to make top_row a row vector
    left_column = np.diff(image[channel][:, 0])   # get the vertical diffs for the left column
    left_column = np.reshape(left_column, (height - 1, 1))  # reshape to make left_column a column vector
    actual_pixels = image[channel][1:, 1:]  # slice the image to remove edges
    residuals = actual_pixels - model_predictions  # get the difference between the model predictions and the actual value
    residuals = np.hstack((left_column, residuals))  # append the left column to the residuals
    residuals = np.vstack((top_row, residuals))  # append the top row to the residuals
    return residuals


def compress_image(img_path, output_dir, img_name, red_model, green_model, blue_model):
    """
    Compress an image using trained models for RGB channels.

        Args:
            img_path (str): Path to the directory containing the image.
            output_dir (str): Path to the directory where compressed file will be saved.
            img_name (str): Name of the image file.
            red_model (Model): Trained model for red channel.
            green_model (Model): Trained model for green channel.
            blue_model (Model): Trained model for blue channel.

        Returns:
            tuple: Tuple containing size of the compressed file and residuals.
    """
    reds, greens, blues = Image.open(img_path + img_name).split()
    image = [np.asarray(reds), np.asarray(greens), np.asarray(blues)]
    r_residuals = get_residuals(red_model, image, 0, create_dataset_reds)
    g_residuals = get_residuals(green_model, image, 1, create_dataset_greens)
    b_residuals = get_residuals(blue_model, image, 2, create_dataset_blues)
    residuals = r_residuals.flatten().tolist() + g_residuals.flatten().tolist() + b_residuals.flatten().tolist()
    file_size = save_as_binary(output_dir, img_name, residuals)
    print("image", img_name, "compressed.")
    return file_size, residuals


def build_reconstruction_arrays(residuals, height, width):
    """
    Build arrays for image reconstruction from residuals.

    Args:
        residuals (list): List of residuals.
        height (int): Height of the image.
        width (int): Width of the image.

    Returns:
        tuple: Tuple containing residuals, reconstruction, and normalized reconstruction.
    """
    residuals = np.reshape(residuals, (height, width))
    reconstruction = np.zeros((height, width))
    reconstruction[0, 0] = residuals[0, 0]
    # reconstruct the first row of the image
    for x in range(1, width):
        reconstruction[0, x] = reconstruction[0, x - 1] + residuals[0, x]
    # reconstruct the first column of the image
    for y in range(1, height):
        reconstruction[y, 0] = reconstruction[y - 1, 0] + residuals[y, 0]
    # normalize
    reconstruction_norm = reconstruction / 255.0
    return residuals, reconstruction, reconstruction_norm


def use_predictions(reconstruction, reconstruction_norm, predictions, residuals, x, y, index):
    """
    Use model predictions to update reconstruction arrays.

    Args:
        reconstruction (ndarray): Array for image reconstruction.
        reconstruction_norm (ndarray): Normalized reconstruction array.
        predictions (ndarray): Predictions from the model.
        residuals (ndarray): Residuals for the image.
        x (int): X-coordinate of the pixel.
        y (int): Y-coordinate of the pixel.
        index (int): Index of the prediction.

    Returns:
        tuple: Tuple containing updated reconstruction and normalized reconstruction.
    """
    residual = residuals[y, x]
    prediction_rounded = round(predictions[index, 0] * 255)
    reconstruction[y, x] = prediction_rounded + residual
    reconstruction_norm[y, x] = reconstruction[y, x] * 0.0039215686274509803921568627451
    return reconstruction, reconstruction_norm


def get_features(norms, x, y):
    """
    Extract pixel feature for normalized image values.

    Args:
        norms (ndarray): Normalized image values.
        x (int): X-coordinate of the pixel.
        y (int): Y-coordinate of the pixel.

    Returns:
        tuple: Tuple containing top, left, diagonal, a, and b features.
    """
    top = norms[y - 1, x]
    left = norms[y, x - 1]
    diagonal = norms[y - 1, x - 1]
    a = left - diagonal
    b = top - diagonal
    return top, left, diagonal, a, b


def get_full_features(norms, x, y):
    """
    Extract extended pixel features from normalized image values.

    Args:
        norms (ndarray): Normalized image values.
        x (int): X-coordinate of the pixel.
        y (int): Y-coordinate of the pixel.

    Returns:
        tuple: Tuple containing top, left, diagonal, a, b, c, and d features.
    """
    top, left, diagonal, a, b = get_features(norms, x, y)
    left_left = norms[y, x - 2]
    top_top = norms[y - 2, x]
    c = left - left_left
    d = top - top_top
    if x == 1:
        c = 0
    if y == 1:
        d = 0
    return top, left, diagonal, a, b, c, d


def reconstruct_reds(residuals, reconstruction, reconstruction_norm, model, height, width):
    """
    Reconstruct the red channel of an image using the model for the red channel.

    Args:
        residuals (ndarray): Residuals for the red channel.
        reconstruction (ndarray): Array for image reconstruction.
        reconstruction_norm (ndarray): Normalized reconstruction array.
        model (Model): Trained Keras model for the red channel.
        height (int): Height of the image.
        width (int): Width of the image.

    Returns:
        tuple: Tuple containing reconstructed and normalized red channel.
    """
    # select pixels on the diagonals to send to the model in batches
    for i in range(1, height):  # cover the top-left triangle
        diagonal_features = []
        xy = []
        for j in range(1, i + 1):
            x = i - (i - j)
            y = i - j + 1
            r_top, r_left, r_diagonal, r_a, r_b, r_c, r_d = get_full_features(reconstruction_norm, x, y)
            diagonal_features.append([r_a, r_b, r_c, r_d, r_top, r_left])
            xy.append((x, y))
        predictions = model.predict(diagonal_features, verbose=0)
        for k in range(len(predictions)):
            reconstruction, reconstruction_norm = use_predictions(reconstruction, reconstruction_norm, predictions,
                                                                  residuals, xy[k][0], xy[k][1], k)

    for i in range(2, width):  # cover the bottom-right triangle
        diagonal_features = []
        xy = []
        for j in range(width - 1, i - 1, -1):
            y = i + (height - j - 1)
            x = j
            r_top, r_left, r_diagonal, r_a, r_b, r_c, r_d = get_full_features(reconstruction_norm, x, y)
            diagonal_features.append([r_a, r_b, r_c, r_d, r_top, r_left])
            xy.append((x, y))
        predictions = model.predict(diagonal_features, verbose=0)
        for k in range(len(predictions)):
            reconstruction, reconstruction_norm = use_predictions(reconstruction, reconstruction_norm, predictions,
                                                                  residuals, xy[k][0], xy[k][1], k)

    return reconstruction, reconstruction_norm


def reconstruct_greens(residuals, reconstruction, reconstruction_norm, reconstruction_norm_red, model, height, width):
    """
    Reconstruct the green channel of an image using the model for the green channel and the reconstructed red channel.

    Args:
        residuals (ndarray): Residuals for the green channel.
        reconstruction (ndarray): Array for image reconstruction.
        reconstruction_norm (ndarray): Normalized reconstruction array.
        reconstruction_norm_red (ndarray): Normalized reconstruction of the red channel.
        model (Model): Trained Keras model for the green channel.
        height (int): Height of the image.
        width (int): Width of the image.

    Returns:
        tuple: Tuple containing reconstructed and normalized green channel.
    """
    # select pixels on the diagonals to send to the model in batches
    for i in range(1, height):  # cover the top-left triangle
        diagonal_features = []
        xy = []
        for j in range(1, i + 1):
            x = i - (i - j)
            y = i - j + 1
            r_top_red, r_left_red, r_diagonal_red, r_a_red, r_b_red = get_features(reconstruction_norm_red, x, y)
            r_top, r_left, r_diagonal, r_a, r_b, r_c, r_d = get_full_features(reconstruction_norm, x, y)
            diagonal_features.append([r_a_red, r_b_red, r_top_red, r_left_red, r_left, r_top,
                                      r_a, r_b, r_c, r_d, reconstruction_norm_red[y, x]])
            xy.append((x, y))
        predictions = model.predict(diagonal_features, verbose=0)
        for k in range(len(predictions)):
            reconstruction, reconstruction_norm = use_predictions(reconstruction, reconstruction_norm, predictions,
                                                                  residuals, xy[k][0], xy[k][1], k)

    for i in range(2, width):  # cover the bottom-right triangle
        diagonal_features = []
        xy = []
        for j in range(width - 1, i - 1, -1):
            y = i + (height - j - 1)
            x = j
            r_top_red, r_left_red, r_diagonal_red, r_a_red, r_b_red = get_features(reconstruction_norm_red, x, y)
            r_top, r_left, r_diagonal, r_a, r_b, r_c, r_d = get_full_features(reconstruction_norm, x, y)
            diagonal_features.append([r_a_red, r_b_red, r_top_red, r_left_red, r_left, r_top,
                                      r_a, r_b, r_c, r_d, reconstruction_norm_red[y, x]])
            xy.append((x, y))
        predictions = model.predict(diagonal_features, verbose=0)
        for k in range(len(predictions)):
            reconstruction, reconstruction_norm = use_predictions(reconstruction, reconstruction_norm, predictions,
                                                                  residuals, xy[k][0], xy[k][1], k)

    return reconstruction, reconstruction_norm


def reconstruct_blues(residuals, reconstruction, reconstruction_norm, reconstruction_norm_red,
                      reconstruction_norm_green, model, height, width):
    """
    Reconstruct the blue channel of an image using the model for the blue channel and the reconstructed red and green channels.

    Args:
        residuals (ndarray): Residuals for the blue channel.
        reconstruction (ndarray): Array for image reconstruction.
        reconstruction_norm (ndarray): Normalized reconstruction array.
        reconstruction_norm_red (ndarray): Normalized reconstruction of the red channel.
        reconstruction_norm_green (ndarray): Normalized reconstruction of the green channel.
        model (Model): Trained Keras model for the blue channel.
        height (int): Height of the image.
        width (int): Width of the image.

    Returns:
        tuple: Tuple containing reconstructed and normalized blue channel.
    """
    # select pixels on the diagonals to send to the model in batches
    for i in range(1, height):  # cover the top-left triangle
        diagonal_features = []
        xy = []
        for j in range(1, i + 1):
            x = i - (i - j)
            y = i - j + 1
            r_top_red, r_left_red, r_diagonal_red, r_a_red, r_b_red = get_features(reconstruction_norm_red, x, y)
            r_top_green, r_left_green, r_diagonal_green, r_a_green, r_b_green = get_features(reconstruction_norm_green,
                                                                                             x, y)
            r_top, r_left, r_diagonal, r_a, r_b, r_c, r_d = get_full_features(reconstruction_norm, x, y)
            diagonal_features.append([r_a_red, r_b_red, r_top_red, r_left_red, r_left_green, r_top_green,
                                      r_a_green, r_b_green, reconstruction_norm_red[y, x], r_a, r_b,
                                      r_c, r_d, r_left, r_top, reconstruction_norm_green[y, x]])
            xy.append((x, y))
        predictions = model.predict(diagonal_features, verbose=0)
        for k in range(len(predictions)):
            reconstruction, reconstruction_norm = use_predictions(reconstruction, reconstruction_norm, predictions,
                                                                  residuals, xy[k][0], xy[k][1], k)

    for i in range(2, width):  # cover the bottom-right triangle
        diagonal_features = []
        xy = []
        for j in range(width - 1, i - 1, -1):
            y = i + (height - j - 1)
            x = j
            r_top_red, r_left_red, r_diagonal_red, r_a_red, r_b_red = get_features(reconstruction_norm_red, x, y)
            r_top_green, r_left_green, r_diagonal_green, r_a_green, r_b_green = get_features(reconstruction_norm_green,
                                                                                             x, y)
            r_top, r_left, r_diagonal, r_a, r_b, r_c, r_d = get_full_features(reconstruction_norm, x, y)
            diagonal_features.append([r_a_red, r_b_red, r_top_red, r_left_red, r_left_green, r_top_green,
                                      r_a_green, r_b_green, reconstruction_norm_red[y, x], r_a, r_b,
                                      r_c, r_d, r_left, r_top, reconstruction_norm_green[y, x]])
            xy.append((x, y))
        predictions = model.predict(diagonal_features, verbose=0)
        for k in range(len(predictions)):
            reconstruction, reconstruction_norm = use_predictions(reconstruction, reconstruction_norm, predictions,
                                                                  residuals, xy[k][0], xy[k][1], k)

    return reconstruction, reconstruction_norm


def decompress_image(bin_path, height, width, save_path):
    """
    Decompress an image from a binary file using the models for the RGB channels.

    Args:
        bin_path (str): Path to the binary file.
        height (int): Height of the image.
        width (int): Width of the image.
        save_path (str): Path to save the decompressed image.

    Returns:
        tuple: Tuple containing reconstructed red, green, and blue channels.
    """
    red_model = load_model("regression_model_reds2.keras")
    green_model = load_model("regression_model_greens2.keras")
    blue_model = load_model("regression_model_blues2.keras")
    channels = load_from_binary(bin_path)
    # all channels are concatenated into one list when compressed
    # split the channels into their respective lists
    r_residuals = channels[:len(channels) // 3]
    g_residuals = channels[len(channels) // 3:2 * len(channels) // 3]
    b_residuals = channels[2 * len(channels) // 3:]

    r_residuals, reconstruction_r, reconstruction_r_norm = build_reconstruction_arrays(r_residuals, height, width)
    g_residuals, reconstruction_g, reconstruction_g_norm = build_reconstruction_arrays(g_residuals, height, width)
    b_residuals, reconstruction_b, reconstruction_b_norm = build_reconstruction_arrays(b_residuals, height, width)

    # reconstruct the rest of the image by using the models to predict the image values then subtract the residuals
    print("Reconstructing Red channel...")
    reconstruction_r, reconstruction_r_norm = reconstruct_reds(r_residuals, reconstruction_r,
                                                               reconstruction_r_norm, red_model, height, width)
    print("Reconstructing Green channel...")

    reconstruction_g, reconstruction_g_norm = reconstruct_greens(g_residuals, reconstruction_g, reconstruction_g_norm,
                                                                 reconstruction_r_norm, green_model, height, width)
    print("Reconstructing Blue channel...")
    reconstruction_b, reconstruction_b_norm = reconstruct_blues(b_residuals, reconstruction_b, reconstruction_b_norm,
                                                                reconstruction_r_norm, reconstruction_g_norm,
                                                                blue_model, height, width)

    reconstruction_r = reconstruction_r.astype(np.uint8)
    reconstruction_g = reconstruction_g.astype(np.uint8)
    reconstruction_b = reconstruction_b.astype(np.uint8)
    image = np.stack([reconstruction_r, reconstruction_g, reconstruction_b], axis=-1)
    Image.fromarray(image).save(save_path)
    return reconstruction_r, reconstruction_g, reconstruction_b


def batch_compress(input_dir, output_dir, red_model, green_model, blue_model):
    """
    Compress all images in a directory using trained models for respective RGB channels.

    Args:
        input_dir (str): Path to the directory containing images.
        output_dir (str): Path to the directory where compressed files will be saved.
        red_model (Model): Trained model for red channel.
        green_model (Model): Trained model for green channel.
        blue_model (Model): Trained model for blue channel.
    """
    input_files = os.listdir(input_dir)
    file_sizes = []
    png_sizes = []
    for file_name in input_files:
        image_path = os.path.join(input_dir, file_name)
        image = Image.open(image_path)
        if image is not None:
            file_size, residuals = compress_image(input_dir, output_dir, file_name, red_model, green_model, blue_model)
            file_sizes.append(file_size)
            png_sizes.append(os.path.getsize(input_dir + file_name))
        else:
            print(f"Error loading {file_name}")
    total_compressed = sum(file_sizes)
    total_png = sum(png_sizes)
    show_xy_plot(png_sizes, file_sizes, "PNG size", "Compressed size")
    print("Compressed size of all files is ", round(total_compressed / (total_png * 1.0), 5) * 100,
          "% of PNG file size.")
    print("Space saved is", total_png - total_compressed, "bytes.")


def validate_decompression(img_path, img_name, reds, greens, blues):
    """
    Validate the decompression of an image by comparing with the original.

    Args:
        img_path (str): Path to the directory containing the original image.
        img_name (str): Name of the image file.
        reds (ndarray): Reconstructed red channel.
        greens (ndarray): Reconstructed green channel.
        blues (ndarray): Reconstructed blue channel.
    """
    orig_reds = Image.open(img_path + img_name).split()[0]
    orig_reds = np.asarray(orig_reds)

    orig_greens = Image.open(img_path + img_name).split()[1]
    orig_greens = np.asarray(orig_greens)

    orig_blues = Image.open(img_path + img_name).split()[2]
    orig_blues = np.asarray(orig_blues)

    if np.array_equal(reds, orig_reds):
        print("red channel reconstructed correctly")
    else:
        print("red channel reconstructed incorrectly")
        print("original reds: \n", orig_reds)
        print("reconstructed reds: \n", reds)

    if np.array_equal(greens, orig_greens):
        print("green channel reconstructed correctly")
    else:
        print("green channel reconstructed incorrectly")
        print("original greens: \n", orig_greens)
        print("reconstructed greens: \n", greens)

    if np.array_equal(blues, orig_blues):
        print("blue channel reconstructed correctly")
    else:
        print("blue channel reconstructed incorrectly")
        print("original blues: \n", orig_blues)
        print("reconstructed blues: \n", blues)


def show_xy_plot(x, y, x_label, y_label, show_regression=0):
    """
    Plot two lists against each other on a 2D graph.

    Parameters:
    - x: List of x-values
    - y: List of y-values
    """
    plt.plot(x, y, 'o')
    if show_regression:
        # Compute the regression line
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, [m * xi + b for xi in x], color='red', label='Regression line')
    # Plot identity line
    plt.plot([min(x + y), max(x + y)], [min(x + y), max(x + y)], color='green', linestyle='--', label='Identity Line')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('Plot of ' + x_label + " vs. " + y_label)
    plt.grid(True)
    plt.show()


def test_functions():
    img_path = "images/"
    img_name = "kodakTest28.png"
    height = 28
    width = 28
    red_model = load_model("regression_model_reds2.keras")
    green_model = load_model("regression_model_greens2.keras")
    blue_model = load_model("regression_model_blues2.keras")
    input_dir = "images/test/"
    output_dir = "binaries/test/"

    batch_compress(input_dir, output_dir, red_model, green_model, blue_model)
    # decompress_image('binaries/test/satellite_0.png.bin', 28, 28, "images/decompressed.png")
    # compress_image(img_path, img_name, red_model, green_model, blue_model)
    # reds, greens, blues = decompress_image('binaries/'+img_name+'.bin', height, width)
    # validate_decompression(img_path, img_name, reds, greens, blues)


if __name__ == "__main__":
    # test_functions()
    pass
    # dependencies:
    # pip install numpy Pillow keras matplotlib


