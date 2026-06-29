"""
🗂️ PROJECT SANDBOX / EXPERIMENTAL UTILITY
Course Project: Bookshelf Catalog System

PURPOSE:
Experimental implementation for automated shelf boundary segmentation using a classical 
digital signal processing heuristic (Horizontal Projection Profile of vertical Sobel edges 
coupled with SciPy peak/valley detection).

STATUS: 
Archived in Sandbox. This heuristic was deprecated in favor of direct end-to-end deep learning 
inference due to stability variations under real-world lighting and book geometry constraints.
"""

import cv2
import numpy as np
from scipy.signal import find_peaks

def detect_shelves(image_path, output_path, expected_shelves=4):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image")
        return
    
    # Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # Edge detection focusing on vertical lines (book spines)
    sobel_vertical = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_vertical = np.uint8(np.absolute(sobel_vertical))
    _, vertical_edges = cv2.threshold(sobel_vertical, 50, 255, cv2.THRESH_BINARY)
    
    # Calculate horizontal projection profile
    projection = np.sum(vertical_edges, axis=1)
    
    # Smooth the projection profile
    kernel_size = 10  # Increased kernel size for better shelf separation
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(projection, kernel, mode='same')
    
    # Find valleys (gaps between shelves)
    valleys, _ = find_peaks(-smoothed, distance=gray.shape[0]//expected_shelves, 
                          prominence=np.max(smoothed)*0.2)
    
    # If we found too many valleys, keep only the deepest ones
    if len(valleys) > expected_shelves-1:
        valley_depths = -smoothed[valleys]
        deepest_valleys = np.argsort(valley_depths)[-(expected_shelves-1):]
        valleys = valleys[np.sort(deepest_valleys)]
    
    # Determine shelf boundaries
    shelf_boundaries = []
    prev_boundary = 0
    for valley in sorted(valleys):
        shelf_boundaries.append((prev_boundary, valley))
        prev_boundary = valley
    shelf_boundaries.append((prev_boundary, gray.shape[0]-1))
    
    # Draw shelf dividers
    for i, (top, bottom) in enumerate(shelf_boundaries):
        # Draw top boundary (green)
        cv2.line(img, (0, top), (img.shape[1], top), (0, 255, 0), 2)
        
        # Label shelves
        cv2.putText(img, f"Shelf {i+1}", (10, top + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw bottom boundary of last shelf
    cv2.line(img, (0, shelf_boundaries[-1][1]), (img.shape[1], shelf_boundaries[-1][1]), (0, 255, 0), 2)
    
    # Save the result
    cv2.imwrite(output_path, img)
    print(f"Detected {len(shelf_boundaries)} shelves")
    print(f"Result saved to {output_path}")
    return shelf_boundaries

# Example usage
input_image = "bookshelf/bookshelf5.jpg"
output_image = "bookshelf_with_shelves.jpg"
shelf_boundaries = detect_shelves(input_image, output_image, expected_shelves=4)