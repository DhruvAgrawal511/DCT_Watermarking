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
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return abs(np.mean(np.multiply((img1 - np.mean(img1)), (img2_resized - np.mean(img2_resized)))) / (np.std(img1) * np.std(img2_resized)))

def embed_dct_watermark(image_path, watermark_path, output_path, alpha=0.1):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    watermark = cv2.resize(watermark, (image.shape[1] // 4, image.shape[0] // 4))
    
    dct_image = cv2.dct(np.float32(image))
    x, y = watermark.shape
    dct_image[:x, :y] += alpha * watermark
    
    watermarked_image = cv2.idct(dct_image)
    watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
    
    cv2.imwrite(output_path, watermarked_image)
    return watermarked_image

def extract_dct_watermark(watermarked_path, original_path, output_path, alpha=0.1):
    watermarked_image = cv2.imread(watermarked_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    
    dct_watermarked = cv2.dct(np.float32(watermarked_image))
    dct_original = cv2.dct(np.float32(original_image))
    
    x, y = dct_original.shape[0] // 4, dct_original.shape[1] // 4
    extracted_watermark = (dct_watermarked[:x, :y] - dct_original[:x, :y]) / alpha
    extracted_watermark = np.clip(extracted_watermark, 0, 255).astype(np.uint8)
    
    cv2.imwrite(output_path, extracted_watermark)
    return extracted_watermark

def scaling_half(img):
    return cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

def scaling_bigger(img):
    return cv2.resize(img, (1100, 1100))

def cut_100_rows(img):
    return img[50:-50, :]

def average_filter(img):
    kernel = np.ones((5,5),np.float32)/25
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

def run_watermarking():
    image_path = input("Enter the path of the main image: ")
    watermark_path = input("Enter the path of the watermark image: ")
    watermarked_output = "watermarked.jpg"
    extracted_watermark_output = "extracted_watermark.jpg"
    
    print("Embedding watermark...")
    watermarked_img = embed_dct_watermark(image_path, watermark_path, watermarked_output, alpha=0.02)
    print("Watermark embedded and saved as", watermarked_output)
    
    print("Extracting watermark...")
    extracted_wm = extract_dct_watermark(watermarked_output, image_path, extracted_watermark_output, alpha=0.02)
    print("Watermark extracted and saved as", extracted_watermark_output)
    
    original_wm = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    print("PSNR between original and watermarked image:", psnr(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), watermarked_img))
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
        extracted_wm = extract_dct_watermark(attacked_output, image_path, "extracted_" + attacked_output, alpha=0.02)
        print(f"NCC after {attack_name}: {ncc(original_wm, extracted_wm)}")
    
if __name__ == "__main__":
    run_watermarking()
