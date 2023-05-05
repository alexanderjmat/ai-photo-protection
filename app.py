import numpy as np
import cv2
from PIL import Image

def context_aware_noise(image_array, noise_factor=0.005):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Compute the Laplacian of the grayscale image to emphasize edges
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)

    # Normalize the Laplacian
    laplacian_normalized = cv2.normalize(laplacian, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Generate Gaussian noise with the same shape as the image
    noise = np.random.normal(loc=0, scale=1, size=image_array.shape)

    # Add the noise to the image, scaling it by the noise_factor and the normalized Laplacian
    noisy_img_array = image_array + noise_factor * np.multiply(noise, np.repeat(laplacian_normalized[:, :, np.newaxis], 3, axis=2))

    return noisy_img_array

def apply_frequency_filter(image_array, filter_factor=0.5):
    def filter_channel(channel):
        # Convert the channel to the frequency domain using the Discrete Fourier Transform (DFT)
        dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shifted = np.fft.fftshift(dft)

        # Create a low-pass filter that preserves the center frequencies and attenuates the high frequencies
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - int(filter_factor * crow):crow + int(filter_factor * crow),
        ccol - int(filter_factor * ccol):ccol + int(filter_factor * ccol)] = 1

        # Apply the low-pass filter to the shifted DFT
        filtered_dft_shifted = dft_shifted * mask

        # Shift the DFT back and perform the inverse DFT to obtain the filtered channel
        filtered_dft = np.fft.ifftshift(filtered_dft_shifted)
        filtered_channel = cv2.idft(filtered_dft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        
        return filtered_channel

    # Convert the image array to float32
    image_array = np.float32(image_array)

    # Split the image into its color channels
    b_channel, g_channel, r_channel = cv2.split(image_array)

    # Apply the frequency filter to each color channel
    filtered_b_channel = filter_channel(b_channel)
    filtered_g_channel = filter_channel(g_channel)
    filtered_r_channel = filter_channel(r_channel)

    # Combine the filtered color channels
    filtered_img_array = cv2.merge([filtered_b_channel, filtered_g_channel, filtered_r_channel])

    return filtered_img_array



def protect_image(image_path, output_path, noise_factor=0.005, filter_factor=0.5):
    # Load the image and convert it to a NumPy array
    img = Image.open(image_path)
    img_array = np.array(img, dtype=np.float32)

    # Apply context-aware noise
    noisy_img_array = context_aware_noise(img_array, noise_factor)

    # Apply frequency filtering
    filtered_img_array = apply_frequency_filter(noisy_img_array, filter_factor)

    # Clip the processed image array to the valid range of values (0-255) and convert it back to uint8
    processed_img_array = np.clip(filtered_img_array, 0, 255).astype(np.uint8)

    # Convert the processed image array back to an image and save it
    processed_img = Image.fromarray(processed_img_array)
    processed_img.save(output_path)


