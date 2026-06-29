"""
🗂️ PROJECT SANDBOX / EXPERIMENTAL UTILITY
Course Project: Bookshelf Catalog System

PURPOSE:
Early-stage experimental testing harness designed to verify the Roboflow Inference HTTP SDK 
integration, evaluate target object detection models, parse bounding box arrays, and check 
for polygon rotation attributes.

STATUS: 
Archived in Sandbox. Core business logic successfully migrated and integrated dynamically 
into the production Flask server framework (`book.py`).
"""

from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np

from dotenv import load_dotenv
load_dotenv()

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ.get("ROBOFLOW_API_KEY")
)

# Load the original image
image = cv2.imread("bookshelf/bookshelf15.5.jpg")

# Use the model that provides rotation information if available
result = CLIENT.infer(image, model_id="book_detect-tarrm-o8kgf/1")

book_count = 0

for prediction in result['predictions']:
    # Check if rotation information is available
    if 'rotation' in prediction:
        angle = prediction['rotation']
        # Get rotated rectangle coordinates
        # (This depends on how your API returns rotated boxes)
        # Example if it returns four corner points:
        points = np.array(prediction['points'], np.int32)
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    else:
        # Fall back to regular bounding box
        x = int(prediction['x'])
        y = int(prediction['y'])
        width = int(prediction['width'])
        height = int(prediction['height'])
        x1 = int(x - width/2)
        y1 = int(y - height/2)
        x2 = int(x + width/2)
        y2 = int(y + height/2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    book_count += 1

print(f"Number of books found: {book_count}")

cv2.imwrite("bookshelf_with_detections.jpg", image)
cv2.imshow("Results", image)
cv2.waitKey(0)
cv2.destroyAllWindows()