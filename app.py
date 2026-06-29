# Import necessary modules
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
import logging
import hashlib
import pickle
import threading
from werkzeug.utils import secure_filename
from inference_sdk import InferenceHTTPClient
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure folders and limits
app.config['UPLOAD_FOLDER'] = 'static/uploads'           # Folder to store uploaded images
app.config['CACHE_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'cache')  # Folder for cached results
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024       # Limit uploads to 16MB

# Ensure cache folder exists
os.makedirs(app.config['CACHE_FOLDER'], exist_ok=True)

# Initialize Roboflow inference client (used for book/spine detection)
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ.get("ROBOFLOW_API_KEY")
)

# Lock to ensure thread-safe cache access
cache_lock = threading.Lock()

# Global error handler
@app.errorhandler(Exception)
def handle_exception(e):
    logger.exception("Unhandled exception occurred")
    return jsonify({'success': False, 'error': str(e)}), 500


# UTILITY FUNCTIONS

# Compute SHA256 hash of a file for consistent caching
def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

# Return the cache file path for a given hash
def cache_path_for_hash(hash_str):
    return os.path.join(app.config['CACHE_FOLDER'], f"{hash_str}.pkl")

# Load cached data if available
def load_cache(hash_str):
    path = cache_path_for_hash(hash_str)
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                logger.debug(f"Loaded cache for {hash_str}")
                return data
        except Exception as e:
            logger.warning(f"Failed to load cache file {path}: {e}")
    return None

# Save data to cache
def save_cache(hash_str, data):
    path = cache_path_for_hash(hash_str)
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f)
            logger.debug(f"Saved cache for {hash_str}")
    except Exception as e:
        logger.warning(f"Failed to save cache file {path}: {e}")



# IMAGE PROCESSING FUNCTIONS

# Adjust image brightness if too dark or bright
def adjust_brightness_if_needed(img, target_brightness=120, dark_threshold=90, bright_threshold=150):
    if img is None:
        return None, False
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
        l, a, b = cv2.split(lab)                    # Split channels
        avg_brightness = np.mean(l)

        if avg_brightness < dark_threshold or avg_brightness > bright_threshold:
            diff = target_brightness - avg_brightness
            l = np.clip(l + diff, 0, 255).astype(np.uint8)
            adjusted_lab = cv2.merge((l, a, b))
            adjusted_img = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
            return adjusted_img, True
        else:
            return img, False
    except Exception as e:
        logger.error(f"Brightness adjustment error: {str(e)}")
        return img, False

# Resize image to fit a max dimension (preserving aspect ratio)
def resize_image_for_detection(img, max_dim=800):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

# Detect books using Roboflow model
def detect_books_from_path(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Cannot read image: {image_path}")
            return []

        img, _ = adjust_brightness_if_needed(img)
        resized_img = resize_image_for_detection(img)

        # Save resized temporary image
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{os.path.basename(image_path)}")
        cv2.imwrite(temp_path, resized_img)

        # Try primary model, fallback to backup model
        try:
            result = CLIENT.infer(temp_path, model_id="book_detect-tarrm-o8kgf/1")
        except:
            result = CLIENT.infer(temp_path, model_id="book-spine-detection-2cci9/2")
        finally:
            os.remove(temp_path)

        # Scale bounding boxes back to original resolution
        scale_x = img.shape[1] / resized_img.shape[1]
        scale_y = img.shape[0] / resized_img.shape[0]

        books = []
        h, w = img.shape[:2]
        for i, pred in enumerate(result.get("predictions", [])):
            try:
                x, y = pred['x'] * scale_x, pred['y'] * scale_y
                width, height = pred['width'] * scale_x, pred['height'] * scale_y
                x1, y1 = max(0, int(x - width // 2)), max(0, int(y - height // 2))
                x2, y2 = min(w, int(x + width // 2)), min(h, int(y + height // 2))
                books.append({'id': i, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
            except KeyError:
                continue
        return books
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        return []


class BookMatcher:
    def __init__(self):
        # Initialize feature detectors
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.orb_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.sift = cv2.SIFT_create()
        self.sift_matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    def get_book_regions(self, image_path):
        return detect_books_from_path(image_path)

    def compute_histogram(self, img):
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    # Match a selected book region to a new image using ORB or SIFT
    def _match_with(self, desc1, new_gray, query_hist, new_img, book, method='orb'):
        try:
            roi = new_img[book['y1']:book['y2'], book['x1']:book['x2']]
            if roi.size == 0:
                return (None, 0)

            # Compare color histograms first (fast filter)
            hist = self.compute_histogram(roi)
            distance = cv2.compareHist(query_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if distance > 0.6:
                return (None, 0)

            # Create mask for keypoint matching
            mask = np.zeros(new_gray.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (book['x1'], book['y1']), (book['x2'], book['y2']), 255, -1)

            if method == 'orb':
                kp2, desc2 = self.orb.detectAndCompute(new_gray, mask)
                if desc2 is None or len(desc2) < 2:
                    return (None, 0)
                matches = self.orb_matcher.knnMatch(desc1, desc2, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            else:
                kp2, desc2 = self.sift.detectAndCompute(new_gray, mask)
                if desc2 is None or len(desc2) < 2:
                    return (None, 0)
                matches = self.sift_matcher.knnMatch(desc1, desc2, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

            return (book, len(good_matches))
        except:
            return (None, 0)

    # Match against all books using multithreading
    def _run_matching(self, desc1, gray_img, hist, img, new_books, method):
        best_match, max_score = None, 0
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._match_with, desc1, gray_img, hist, img, book, method) for book in new_books]
            for f in futures:
                book, score = f.result()
                if score > max_score:
                    best_match, max_score = book, score
        return best_match, max_score

    # Main interface for matching selected book between two images
    def find_matching_book(self, old_img_path, new_img_path, selected_coords, new_books=None):
        try:
            old_img = cv2.imread(old_img_path)
            new_img = cv2.imread(new_img_path)
            if old_img is None or new_img is None:
                return {'success': False, 'error': 'Images could not be loaded'}

            old_img, _ = adjust_brightness_if_needed(old_img)
            new_img, _ = adjust_brightness_if_needed(new_img)

            x1, y1, x2, y2 = selected_coords
            h, w = old_img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            book_img = old_img[y1:y2, x1:x2]
            if book_img.size == 0:
                return {'success': False, 'error': 'Selected book region is empty'}

            query_hist = self.compute_histogram(book_img)

            if new_books is None:
                new_books = self.get_book_regions(new_img_path)

            # First try ORB
            desc1_orb = self.orb.detectAndCompute(cv2.cvtColor(book_img, cv2.COLOR_BGR2GRAY), None)[1]
            if desc1_orb is not None and len(desc1_orb) >= 2:
                best_orb, score_orb = self._run_matching(desc1_orb, cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY), query_hist, new_img, new_books, 'orb')
            else:
                best_orb, score_orb = None, 0

            if best_orb and score_orb >= 10:
                return {
                    'success': True, 'total_books': len(new_books), 'best_match': best_orb,
                    'match_score': score_orb, 'method': 'orb',
                    'low_confidence': score_orb < 15,
                    'warning': 'Low confidence' if score_orb < 15 else ''
                }

            # Try SIFT as fallback
            desc1_sift = self.sift.detectAndCompute(cv2.cvtColor(book_img, cv2.COLOR_BGR2GRAY), None)[1]
            if desc1_sift is None or len(desc1_sift) < 2:
                return {'success': True, 'warning': 'Not enough features for SIFT'}

            best_sift, score_sift = self._run_matching(desc1_sift, cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY), query_hist, new_img, new_books, 'sift')
            return {
                'success': True, 'total_books': len(new_books), 'best_match': best_sift,
                'match_score': score_sift, 'method': 'sift',
                'low_confidence': score_sift < 10,
                'warning': 'Low confidence' if score_sift < 10 else 'ORB failed, SIFT used'
            }

        except Exception as e:
            logger.exception("Dual matching failed")
            return {'success': False, 'error': str(e)}


# FLASK ROUTES


@app.route('/')
def index():
    return render_template('index.html')

# Upload handler for old and new images
@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'old_image' not in request.files or 'new_image' not in request.files:
            return jsonify({'success': False, 'error': 'Missing files'}), 400

        old_file = request.files['old_image']
        new_file = request.files['new_image']
        if old_file.filename == '' or new_file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filenames'}), 400

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        old_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(old_file.filename))
        new_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(new_file.filename))

        old_file.save(old_path)
        new_file.save(new_path)

        # Brightness correction if needed
        old_img = cv2.imread(old_path)
        adjusted_old, changed_old = adjust_brightness_if_needed(old_img)
        if changed_old:
            cv2.imwrite(old_path, adjusted_old)

        new_img = cv2.imread(new_path)
        adjusted_new, changed_new = adjust_brightness_if_needed(new_img)
        if changed_new:
            cv2.imwrite(new_path, adjusted_new)

        # Load or detect book regions using caching
        old_hash = file_hash(old_path)
        new_hash = file_hash(new_path)
        matcher = BookMatcher()

        with cache_lock:
            old_books = load_cache(old_hash)
            if old_books is None:
                old_books = matcher.get_book_regions(old_path)
                save_cache(old_hash, old_books)

            new_books = load_cache(new_hash)
            if new_books is None:
                new_books = matcher.get_book_regions(new_path)
                save_cache(new_hash, new_books)

        return jsonify({
            'success': True,
            'old_image': old_path,
            'new_image': new_path,
            'books': old_books,
            'new_books': new_books,
            'old_book_count': len(old_books)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Match API route for a selected book region
@app.route('/find_match', methods=['POST'])
def find_match():
    try:
        data = request.get_json()
        for field in ['old_image', 'new_image', 'selected_book']:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing {field}'}), 400
        selected = data['selected_book']
        coords = (selected['x1'], selected['y1'], selected['x2'], selected['y2'])
        matcher = BookMatcher()
        result = matcher.find_matching_book(data['old_image'], data['new_image'], coords, new_books=data.get('new_books'))
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CACHE_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)
