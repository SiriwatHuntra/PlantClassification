import cv2
import numpy as np
import os
from Util.Image_IO import ReadFile, SaveFile, get_labeled_image_paths

def adjust_brightness(image, brightness_range=(-70, 70), localized=False):
    """ Randomly adjusts image brightness within the given range. """
    if localized:
        mask = np.random.randint(brightness_range[0], brightness_range[1], size=image.shape[:2])
        for channel in range(image.shape[2]):
            image[:, :, channel] = np.clip(image[:, :, channel] + mask, 0, 255)
    else:
        brightness_offset = np.random.randint(brightness_range[0], brightness_range[1])
        image = np.clip(image.astype(np.int32) + brightness_offset, 0, 255).astype('uint8')
    return image

def apply_gaussian_blur(image, max_kernel_size=7, blur_type="gaussian"):
    """ 
    Applies blur to simulate focus loss or motion.
    
    Parameters:
        max_kernel_size (int): Max kernel size for blurring.
        blur_type (str): "gaussian" or "motion".
    """
    kernel_size = np.random.choice(range(1, max_kernel_size, 2))
    
    if blur_type == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif blur_type == "motion":
        # Simulate motion blur using a kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel /= kernel_size
        return cv2.filter2D(image, -1, kernel)
    else:
        raise ValueError("Invalid blur_type. Choose 'gaussian' or 'motion'.")

def perspective_warp(image):
    """ Applies a random perspective transformation to the image. """
    h, w = image.shape[:2]
    perturb = lambda: np.random.randint(-40, 40)
    src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_points = np.float32([[perturb(), perturb()], [w + perturb(), perturb()], 
                             [perturb(), h + perturb()], [w + perturb(), h + perturb()]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, matrix, (w, h))

def simulate_damage(image, damage_patches=5, patch_size=20):
    """ Simulates leaf damage by creating patches of missing sections. """
    h, w = image.shape[:2]
    for _ in range(damage_patches):
        x = np.random.randint(0, w - patch_size)
        y = np.random.randint(0, h - patch_size)
        image[y:y + patch_size, x:x + patch_size] = 0  # Black patch
    return image

def add_speckle_noise(image, noise_amount=0.02):
    """ Adds speckle noise (random noisy spots). """
    speckle = np.random.randn(*image.shape) * 255 * noise_amount
    noisy_image = np.clip(image + speckle, 0, 255).astype('uint8')
    return noisy_image


def apply_augmentation_and_save(folder_path, output_path, augmentation_func, augmentation_name, **kwargs):
    """ Applies the selected augmentation function to images and saves them with added debugging. """

    labeled_image_paths = get_labeled_image_paths(folder_path)

    for image_paths in labeled_image_paths:
        class_label = os.path.basename(os.path.dirname(image_paths[0]))
        class_output_dir = os.path.join(output_path, augmentation_name, class_label)
        os.makedirs(class_output_dir, exist_ok=True)

        for img_path in image_paths:
            try:
                # Debugging: Check if image can be read
                image = ReadFile(img_path, colorType='RGB')
                if image is None:
                    print(f"Error reading image: {img_path}")
                    continue

                # Apply the augmentation function
                augmented_image = augmentation_func(image, **kwargs)

                # Save the augmented image
                original_filename = os.path.basename(img_path)
                SaveFile(augmented_image, class_output_dir, original_filename)

                print(f"Augmented {img_path} with {augmentation_name} and saved to {class_output_dir}/{original_filename}")

            except Exception as e:
                print(f"Error processing {img_path} with {augmentation_name}: {e}")

if __name__ == "__main__":
    #  paths
    folder_path = "ClassificationModel/Image"
    output_path = "ClassificationModel/Noisy"

    # Dictionary of augmentations for easy testing
    augmentations = {
        "brightness": (adjust_brightness, {"brightness_range": (-70, 70)}),
        "blur": (apply_gaussian_blur, {"max_kernel_size": 7}),
        "perspective": (perspective_warp, {}),
        "speckle": (add_speckle_noise, {"noise_amount": 0.1}),
        "damage": (simulate_damage, {"damage_patches": 10, "patch_size": 25}),
    }

    # Loop through and apply each augmentation
    for aug_name, (aug_func, aug_kwargs) in augmentations.items():
        print(f"\nApplying {aug_name} augmentation...")
        apply_augmentation_and_save(
            folder_path, output_path, augmentation_func=aug_func, augmentation_name=aug_name, **aug_kwargs
        )
