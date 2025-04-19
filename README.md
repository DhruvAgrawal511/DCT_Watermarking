# StegoDCT

**StegoDCT** is a Python-based watermarking tool that embeds a digital watermark into an image using Discrete Cosine Transform (DCT). It also supports extracting the watermark and testing its resilience against various common image processing attacks.

This project is ideal for those exploring digital watermarking, image processing, or basic security techniques in multimedia.

---

## Features

- Embed a grayscale watermark into a grayscale image using DCT.
- Extract watermark from the watermarked image.
- Calculate:
  - PSNR (Peak Signal-to-Noise Ratio) to measure image quality.
  - NCC (Normalized Cross-Correlation) to measure similarity between watermarks.
- Test robustness of the watermark with:
  - Image Scaling (down/up)
  - Cropping (cut rows)
  - Filtering (average/median)
  - Noise (Gaussian and Salt & Pepper)

---

## Installation

Make sure Python is installed and run the following command to install dependencies:

```bash
pip install opencv-python numpy
```

---

## Usage

To run the program:

```bash
python StegoDCT.py
```

You will be prompted to enter:
- Path to the main (cover) image
- Path to the watermark image

Make sure both images are in grayscale (or will be converted).

The script will:
- Embed the watermark into the original image.
- Save the watermarked image.
- Extract the watermark and compare it with the original.
- Apply attacks to the watermarked image.
- Extract and evaluate the watermark after each attack.

---

## Example

Given:
- Original image: `file1.jpg`
- Watermark: `file2.jpg`

Run:

```bash
python watermarking.py
```

You’ll be prompted like this:

```
Enter the path of the main image: file1.jpg
Enter the path of the watermark image: file2.jpg
```

---

## Output

The following files will be generated:

- `watermarked.jpg` – Original image with embedded watermark
- `extracted_watermark.jpg` – Watermark extracted from `watermarked.jpg`
- `attacked_Scaling_Half.jpg` – Watermarked image after downscaling
- `attacked_Median_Filter.jpg` – Watermarked image after median filter
- `extracted_attacked_Scaling_Half.jpg` – Extracted watermark from attacked image
- (more such files for each type of attack)

Sample terminal output:

```
Embedding watermark...
Watermark embedded and saved as watermarked.jpg
Extracting watermark...
Watermark extracted and saved as extracted_watermark.jpg
PSNR between original and watermarked image: 39.45
NCC between original and extracted watermark: 0.87
NCC after Scaling Half: 0.65
NCC after Median Filter: 0.79
...
```

---

## Notes

- The watermark is embedded in the top-left corner of the DCT-transformed original image.
- The watermark image is resized to one-fourth the size of the original to ensure clean embedding.
- All operations are performed on grayscale images for simplicity.
- You can tune `alpha` (embedding strength) for better balance between watermark visibility and robustness.

---

## License

This project is licensed under the MIT License. Feel free to use or adapt it for learning, teaching, or building upon.

---
