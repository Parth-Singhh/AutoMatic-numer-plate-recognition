import cv2
import pytesseract
import imutils

# -----------------------------------
# SET TESSERACT PATH (WINDOWS)
# -----------------------------------
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# -----------------------------------
# READ IMAGE
# -----------------------------------
image = cv2.imread('images/1.jpg')

# Check if image exists
if image is None:
    print("Error: Image not found!")
    print("Make sure '1.jpg' exists inside the images folder.")
    exit()

# Resize image
image = imutils.resize(image, width=600)

# -----------------------------------
# IMAGE PREPROCESSING
# -----------------------------------
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Reduce noise
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# Detect edges
edged = cv2.Canny(gray, 170, 200)

# -----------------------------------
# FIND CONTOURS
# -----------------------------------
cnts, _ = cv2.findContours(
    edged.copy(),
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE
)

# Sort contours by area
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

number_plate = None

# Find rectangular contour
for c in cnts:
    perimeter = cv2.arcLength(c, True)

    approx = cv2.approxPolyDP(
        c,
        0.02 * perimeter,
        True
    )

    # Number plate usually has 4 corners
    if len(approx) == 4:
        number_plate = approx
        break

# -----------------------------------
# CHECK IF PLATE DETECTED
# -----------------------------------
if number_plate is None:
    print("No number plate detected.")
    exit()

# -----------------------------------
# DRAW RECTANGLE AROUND PLATE
# -----------------------------------
cv2.drawContours(image, [number_plate], -1, (0, 255, 0), 3)

# -----------------------------------
# CREATE MASK
# -----------------------------------
mask = gray.copy()
mask[:] = 0

cv2.drawContours(mask, [number_plate], 0, 255, -1)

# -----------------------------------
# CROP NUMBER PLATE
# -----------------------------------
x, y, w, h = cv2.boundingRect(number_plate)

cropped = gray[y:y+h, x:x+w]

# -----------------------------------
# OCR RECOGNITION
# -----------------------------------
text = pytesseract.image_to_string(
    cropped,
    config='--psm 8'
)

# Clean text
text = text.strip()

# -----------------------------------
# OUTPUT
# -----------------------------------
print("\nDetected Number Plate:")
print(text)

# -----------------------------------
# DISPLAY RESULTS
# -----------------------------------
cv2.imshow("Original Image", image)
cv2.imshow("Detected Number Plate", cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()

