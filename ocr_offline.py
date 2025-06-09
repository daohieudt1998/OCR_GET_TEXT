import cv2
import pytesseract
from PIL import Image
import numpy as np
import sys
import os

# 👉 Update this path to match your Tesseract installation
# link predict text: https://github.com/UB-Mannheim/tesseract/wiki
pytesseract.pytesseract.tesseract_cmd = r"D:\LapTrinh\OCR_TEXT\tesseract.exe"

def preprocess_image(image_path):
    """
    Preprocess the image for better OCR results.
    Steps: Resize, Grayscale, Threshold, Denoise.
    """
    img = cv2.imread(image_path)

    # Resize image (scale up)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize image using OTSU thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Denoise image using median blur
    denoised = cv2.medianBlur(thresh, 3)

    cv2.imwrite("image_predict.png", denoised)
    return denoised

def extract_text_from_image(image_path):
    """
    Perform OCR on the preprocessed image and return the extracted text.
    """
    preprocessed = preprocess_image(image_path)
    pil_img = Image.fromarray(preprocessed)

    # OCR: Change lang='vie' if using Vietnamese
    text = pytesseract.image_to_string(pil_img, lang='eng')
    return text

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ocr_extract.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.isfile(image_path):
        print(f"Error: File '{image_path}' not found.")
        sys.exit(1)

    result = extract_text_from_image(image_path)
    print("\n--- TEXT EXTRACTED FROM IMAGE ---\n")
    print(result)
