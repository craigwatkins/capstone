# capstone
Capstone Project
This guide will walk you through the process of setting up and using the application to compress and decompress satellite images using our bespoke Feed Forward Neural Network (FFNN) model.

Prerequisites:
Python 3.8 or higher installed. If not, download and install it from (https://www.python.org/downloads/).

Downloading and Installing Necessary Software and Libraries:

Clone or download the Capstone project repository to your local machine.
Navigate to the project directory:

cd path_to_directory/Capstone

Install the required libraries:

pip install -r requirements.txt


Using the Application:

Open the terminal or command prompt.

Navigate to the application's directory:

cd path_to_directory/Capstone

To compress all images in a folder, use the batch_compress command:

python main.py batch_compress path_to_input_directory/ path_to_output_directory/

Example:

python main.py batch_compress images/test/ binaries/test/

To compress a single image, use the compress_image command:

python main.py compress_image path_to_image path_to_output_directory compressed_filename

Example:

python main.py compress_image images/test/satellite_1.png binaries/ satellite_1.png

To decompress a single image, use the decompress_image command:

python main.py decompress_image path_to_binary_file image_height image_width path_to_save_decompressed_image

Example:

python main.py decompress_image binaries/test/satellite_1.png.bin 28 28 images/decompressed.png

Viewing Compression Results:

After using the batch_compress command, the application will display the amount of disk space saved both in percentage and bytes compared to PNG compression. Additionally, a scatter plot graph will be generated showing each file’s PNG size and compressed size through the FFNN model. This visualization aids in determining the efficiency of the compression for each file.


Note: If you want to run the model training, the dataset can be found here:
https://www.kaggle.com/datasets/craigwat/partial-deepsat-sat-4-airborne-dataset-in-sqlite/data
