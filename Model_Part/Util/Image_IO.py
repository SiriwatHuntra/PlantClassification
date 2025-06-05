import os
from pathlib import Path
from PIL import Image
import cv2

def file_converter(directory):
    """
    Convert image files in a directory to a specified format, handling various types
    and removing old files to prevent duplication.

    Parameters:
        directory (str): The path to the directory containing image files to convert.

    Supported formats:
        - Input: .png, .jpg, .jpeg, .bmp, .gif, .tiff
        - Output: .jpg
    """
    # Define supported file extensions
    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
    
    try:
        # Get a list of image paths with supported extensions
        pathlist = [p for p in Path(directory).glob("**/*") if p.suffix.lower() in supported_extensions]
        
        for path in pathlist:
            try:
                # Open the image file
                with Image.open(path) as im:
                    # Check if the file is not already in .jpg format
                    if path.suffix.lower() != ".jpg":
                        # Convert the image to RGB format
                        rgb_im = im.convert('RGB')
                        # Generate a new file path with .jpg extension
                        new_path = path.with_suffix('.jpg')
                        # Save the converted image as .jpg
                        rgb_im.save(new_path, quality=95)
                        print(f"Converted {path} to {new_path}")

                        # Remove the original file
                        path.unlink()
                        print(f"Deleted original file: {path}")
                    else:
                        # Skip conversion if already in .jpg format
                        print(f"File {path} is already in .jpg format. Skipping conversion.")

            except Exception as e:
                # Handle errors during file processing
                print(f"Error processing file {path}: {e}")

    except Exception as e:
        # Handle errors accessing the directory
        print(f"Failed to process directory {directory}: {e}")

def Get_Path_List(directory):
    """
    Get paths of all images in the specified folder.

    Parameters:
        directory (str): Destination path containing images.

    Returns:
        list: List of image paths in the specified folder.
    """
    # Collect paths of .jpg and .png files in the directory
    pathlist = list(Path(directory).glob('*.jpg')) + list(Path(directory).glob("*.png"))
    return pathlist

def get_labeled_image_paths(folder_path):
    """
    Get paths of all labeled images in subfolders of the specified folder.

    Parameters:
        folder_path (str): Path to the main folder containing subfolders of images.

    Returns:
        list: Nested list of image paths for each subfolder.
    """
    # Initialize an empty list to store image paths
    labeled_image_paths = [] 

    # Iterate through subfolders in the main folder
    for subfolder_name in sorted(os.listdir(folder_path)):
        # Construct the path to the subfolder
        subfolder_path = os.path.join(folder_path, subfolder_name)

        # Check if the path is a directory
        if os.path.isdir(subfolder_path):
            # Collect paths of image files in the subfolder
            image_paths = [
                os.path.join(subfolder_path, file_name)
                for file_name in os.listdir(subfolder_path)
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
            ]
            # Add the list of image paths to the result
            labeled_image_paths.append(image_paths)

    return labeled_image_paths

def ReadFile(imgPath, colorType):
    """
    Read an image file and convert it to the specified color type.

    Parameters:
        imgPath (str): Path to the image file.
        colorType (str): Desired color format (GREY, RGB, HSV).

    Returns:
        ndarray: Processed image.
    """
    # Read the image from the file path
    image = cv2.imread(imgPath)
    # Match the desired color format and convert the image
    match colorType:
        case 'GREY':
            # Convert the image to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        case 'RGB':
            # Convert the image to RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        case "HSV":
            # Convert the image to HSV format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        case _:
            # Handle invalid color type
            print("Invalid color type")
            pass

    return image

def SaveFile(image, output_dir, filename):
    """
    Save the given image to the specified directory with the provided filename.

    Parameters:
        image (ndarray): Image data to save.
        output_dir (str): Directory to save the image.
        filename (str): Name of the output file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Save the image using the specified filename
    cv2.imwrite(os.path.join(output_dir, filename), image)

def file_resizer(directory):
    """
    Resize all images in a directory to 512x512 pixels.

    Parameters:
        directory (str): Path to the directory containing images to resize.
    """
    # Get the list of image paths in the directory
    pathlist = Get_Path_List(directory)
    
    for path in pathlist:
        try:
            # Read the image from the file path
            image = cv2.imread(str(path))
            # Resize the image to 512x512 pixels
            image = cv2.resize(image, (512, 512))
            # Save the resized image back to the original directory
            filename = path.name
            cv2.imwrite(os.path.join(directory, filename), image)
            print(f"Resized {path} and saved as {filename}")
        except Exception as e:
            # Handle errors during resizing
            print(f"Error resizing file {path}: {e}")
