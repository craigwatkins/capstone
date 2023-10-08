
import argparse
from compression import batch_compress, compress_image, decompress_image
from keras.models import Model, load_model


def main():
    parser = argparse.ArgumentParser(description="Image Compression and Decompression CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Batch compress command
    batch_parser = subparsers.add_parser("batch_compress", help="Compress all images in a folder. Arguments: Input directory, Output directory")
    batch_parser.add_argument("input_dir", help="Input directory containing images")
    batch_parser.add_argument("output_dir", help="Output directory to save compressed files")

    # Compress image command
    compress_parser = subparsers.add_parser("compress_image", help="Compress a single image. Arguments: image path, output directory, compressed filename")
    compress_parser.add_argument("img_path", help="Path to the image")
    compress_parser.add_argument("output_dir", help="Output directory to save compressed file")
    compress_parser.add_argument("img_name", help="Image filename")

    # Decompress image command
    decompress_parser = subparsers.add_parser("decompress_image", help="Decompress a single image. Arguments: binary file path, image height, image width, file save path")
    decompress_parser.add_argument("bin_path", help="Path to the binary file")
    decompress_parser.add_argument("height", type=int, help="Height of the original image")
    decompress_parser.add_argument("width", type=int, help="Width of the original image")
    decompress_parser.add_argument("save_path", help="Path to save the image to")

    args = parser.parse_args()

    # Load models
    red_model = load_model("regression_model_reds2.keras")
    green_model = load_model("regression_model_greens2.keras")
    blue_model = load_model("regression_model_blues2.keras")

    if args.command == "batch_compress":
        batch_compress(args.input_dir, args.output_dir, red_model, green_model, blue_model)
    elif args.command == "compress_image":
        compress_image(args.img_path, args.output_dir, args.img_name, red_model, green_model, blue_model)
    elif args.command == "decompress_image":
        reds, greens, blues = decompress_image(args.bin_path, args.height, args.width, args.save_path)

    else:
        print("Invalid command. Use -h for help.")


if __name__ == "__main__":
    main()


# compress a directory of images
# python main.py batch_compress images/test/ binaries/test/

# decompress an image
# python main.py decompress_image binaries/test/satellite_1.png.bin 28 28 images/decompressed.png

# compress an image
# python main.py compress_image images/test/ binaries/ satellite_0.png
