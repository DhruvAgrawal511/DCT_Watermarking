import cv2
import numpy as np
import random
import math

def psnr(img1, img2):
    """Computes the Peak Signal-to-Noise Ratio (PSNR)."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ncc(img1, img2):
    """Computes Normalized Cross-Correlation (NCC) between two images."""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return abs(np.mean((img1 - np.mean(img1)) * (img2 - np.mean(img2))) / (np.std(img1) * np.std(img2)))

def embed_watermark(image_path, watermark_path, output_path, alpha=0.02):
    """Embeds a watermark into an image using weighted addition."""
    image = cv2.imread(image_path)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)

    if image is None or watermark is None:
        print("Error: Could not read images.")
        return None

    watermark = cv2.resize(watermark, (image.shape[1], image.shape[0]))

    if len(watermark.shape) == 2 or watermark.shape[2] == 1:
        watermark = cv2.cvtColor(watermark, cv2.COLOR_GRAY2BGR)

    watermarked_image = cv2.addWeighted(image, 1, watermark, alpha, 0)
    cv2.imwrite(output_path, watermarked_image)
    return watermarked_image

def extract_watermark(watermarked_path, original_path, output_path, alpha=0.02):
    """Extracts the watermark by reversing the embedding process."""
    watermarked_image = cv2.imread(watermarked_path)
    original_image = cv2.imread(original_path)

    if watermarked_image is None or original_image is None:
        print("Error: Could not read watermarked or original images.")
        return None

    # Resize original image to match watermarked image (if necessary)
    original_image = cv2.resize(original_image, (watermarked_image.shape[1], watermarked_image.shape[0]))

    watermark = (watermarked_image - original_image * (1 - alpha)) / alpha
    watermark = np.clip(watermark, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, watermark)
    return watermark

def preprocess_for_ncc(original_wm, extracted_wm):
    """Ensures both images are grayscale and of the same size before NCC calculation."""
    extracted_wm = cv2.resize(extracted_wm, (original_wm.shape[1], original_wm.shape[0]))
    if len(extracted_wm.shape) == 3:
        extracted_wm = cv2.cvtColor(extracted_wm, cv2.COLOR_BGR2GRAY)
    return extracted_wm

# ATTACKS / DISTORTIONS

def scaling_half(img, original_size):
    return cv2.resize(img, (original_size[1], original_size[0]), fx=0.5, fy=0.5)

def scaling_bigger(img, original_size):
    return cv2.resize(img, (original_size[1], original_size[0]), fx=1.5, fy=1.5)

def cut_100_rows(img, original_size):
    cropped = img[50:-50, :]
    return cv2.resize(cropped, (original_size[1], original_size[0]))

def average_filter(img, original_size):
    kernel = np.ones((5,5), np.float32) / 25
    return cv2.filter2D(img, -1, kernel)

def median_filter(img, original_size):
    return cv2.medianBlur(img, 5)

def add_noise(img, original_size, noise_type):
    """Adds noise to an image and ensures pixel values remain in the valid range."""
    noisy_img = img.copy()
    if noise_type == "gauss":
        noise = np.random.normal(0, 25, img.shape).astype(np.int16)
        noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
    elif noise_type == "s&p":
        prob = 0.05
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    noisy_img[i][j] = 0
                elif rdn > (1 - prob):
                    noisy_img[i][j] = 255
    return noisy_img

def run_watermarking():
    """Runs the complete watermark embedding, extraction, and attack evaluation pipeline."""
    image_path = input("Enter the path of the main image: ")
    watermark_path = input("Enter the path of the watermark image: ")
    watermarked_output = "watermarked.jpg"
    extracted_watermark_output = "extracted_watermark.jpg"
    
    print("Embedding watermark...")
    watermarked_img = embed_watermark(image_path, watermark_path, watermarked_output, alpha=0.02)
    if watermarked_img is None:
        return
    print("Watermark embedded and saved as", watermarked_output)

    print("Extracting watermark...")
    extracted_wm = extract_watermark(watermarked_output, image_path, extracted_watermark_output, alpha=0.02)
    if extracted_wm is None:
        return
    print("Watermark extracted and saved as", extracted_watermark_output)

    original_wm = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if original_wm is None:
        print("Error: Could not read original watermark.")
        return

    original_image = cv2.imread(image_path)
    psnr_value = psnr(original_image, watermarked_img)
    print("PSNR between original and watermarked image:", psnr_value)

    extracted_wm = preprocess_for_ncc(original_wm, extracted_wm)
    print("NCC between original and extracted watermark:", ncc(original_wm, extracted_wm))

    original_size = original_image.shape[:2]

    # ATTACKS
    attacks = {
        "Scaling Half": scaling_half(watermarked_img, original_size),
        "Scaling Bigger": scaling_bigger(watermarked_img, original_size),
        "Cut 100 Rows": cut_100_rows(watermarked_img, original_size),
        "Average Filter": average_filter(watermarked_img, original_size),
        "Median Filter": median_filter(watermarked_img, original_size),
        "Gaussian Noise": add_noise(watermarked_img, original_size, "gauss"),
        "Salt & Pepper Noise": add_noise(watermarked_img, original_size, "s&p")
    }

    for attack_name, attacked_img in attacks.items():
        attacked_output = f"attacked_{attack_name.replace(' ', '_')}.jpg"
        cv2.imwrite(attacked_output, attacked_img)

        extracted_wm = extract_watermark(attacked_output, image_path, "extracted_" + attacked_output, alpha=0.02)
        if extracted_wm is None:
            print(f"Skipping NCC calculation for {attack_name} due to extraction failure.")
            continue

        extracted_wm = preprocess_for_ncc(original_wm, extracted_wm)
        print(f"NCC after {attack_name}: {ncc(original_wm, extracted_wm)}")

if __name__ == "__main__":
    run_watermarking()
