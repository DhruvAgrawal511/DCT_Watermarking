import cv2
import numpy as np
import random
import math

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ncc(img1, img2):
    """Computes the Normalized Cross-Correlation (NCC) between two images."""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return abs(np.mean(np.multiply((img1 - np.mean(img1)), (img2 - np.mean(img2)))) / (np.std(img1) * np.std(img2)))

def embed_watermark(image_path, watermark_path, output_path, alpha=0.5):
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

def extract_watermark(watermarked_path, original_path, output_path, alpha=0.5):
    watermarked_image = cv2.imread(watermarked_path)
    original_image = cv2.imread(original_path)

    if watermarked_image is None or original_image is None:
        print("Error: Could not read watermarked or original images.")
        return None

    watermark = (watermarked_image - original_image * (1 - alpha)) / alpha
    watermark = np.clip(watermark, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, watermark)
    return watermark

def preprocess_for_ncc(original_wm, extracted_wm):
    """Ensures both images have the same size and are grayscale."""
    extracted_wm = cv2.resize(extracted_wm, (original_wm.shape[1], original_wm.shape[0]))
    if len(extracted_wm.shape) == 3:
        extracted_wm = cv2.cvtColor(extracted_wm, cv2.COLOR_BGR2GRAY)
    return extracted_wm

def scaling_half(img):
    return cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

def scaling_bigger(img):
    return cv2.resize(img, (1100, 1100))

def cut_100_rows(img):
    return img[50:-50, :]

def average_filter(img):
    kernel = np.ones((5,5), np.float32) / 25
    return cv2.filter2D(img, -1, kernel)

def median_filter(img):
    return cv2.medianBlur(img, 5)

def add_noise(img, noise_type):
    if noise_type == "gauss":
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        return cv2.add(img, noise)
    elif noise_type == "s&p":
        prob = 0.05
        output = img.copy()
        thres = 1 - prob
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
        return output
    return img

def run_watermarking():
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

    psnr_value = psnr(cv2.imread(image_path), watermarked_img)
    print("PSNR between original and watermarked image:", psnr_value)

    extracted_wm = preprocess_for_ncc(original_wm, extracted_wm)
    print("NCC between original and extracted watermark:", ncc(original_wm, extracted_wm))

    attacks = {
        "Scaling Half": scaling_half(watermarked_img),
        "Scaling Bigger": scaling_bigger(watermarked_img),
        "Cut 100 Rows": cut_100_rows(watermarked_img),
        "Average Filter": average_filter(watermarked_img),
        "Median Filter": median_filter(watermarked_img),
        "Gaussian Noise": add_noise(watermarked_img, "gauss"),
        "Salt & Pepper Noise": add_noise(watermarked_img, "s&p")
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
