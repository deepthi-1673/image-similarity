import cv2
import pytesseract
from PIL import Image
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pytesseract.pytesseract.tesseract_cmd = r'H:\tesseract\tesseract.exe'
# Path to the teacher's and student's images
teacher_image_path = 'img1.png'
student_image_path = 'img2.jpg'

# Function to preprocess and extract text from an image using Tesseract OCR
def extract_text_from_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform thresholding to enhance the text
    _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Use pytesseract to extract text from the thresholded image
    extracted_text = pytesseract.image_to_string(threshold_image)

    return extracted_text

# Extract text from the teacher's image
teacher_text = extract_text_from_image(teacher_image_path)
print(teacher_text)


# Extract text from the student's image
student_text = extract_text_from_image(student_image_path)
print(student_text)

# Convert text to vector embeddings using TF-IDF
vectorizer = TfidfVectorizer()
text_embeddings = vectorizer.fit_transform([teacher_text, student_text])

# Sample image embeddings (random values for demonstration purposes)
teacher_image_embedding = np.random.rand(1, 512)  # Assuming 512-dimensional embeddings
student_image_embedding = np.random.rand(1, 512)  # Assuming 512-dimensional embeddings

# Calculate cosine similarity between text embeddings
text_similarity = cosine_similarity(text_embeddings)

# Calculate cosine similarity between image embeddings
image_similarity = cosine_similarity(teacher_image_embedding, student_image_embedding)

print("Cosine Similarity between Text Embeddings:")
print(text_similarity)
print("\nCosine Similarity between Image Embeddings:")
print(image_similarity)
# Define a threshold for cosine similarity
threshold = 0.8

# Compare cosine similarity scores with the threshold
if text_similarity[0, 1] >= threshold:
    print("The student's flowchart is similar to the teacher's flowchart.")
else:
    print("The student's flowchart is different from the teacher's flowchart.")
