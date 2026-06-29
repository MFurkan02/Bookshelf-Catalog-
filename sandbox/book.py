"""
🗂️ PROJECT SANDBOX / EXPERIMENTAL UTILITY
Course Project: Bookshelf Catalog System

PURPOSE:
Initial monolithic functional prototype showcasing the core Flask API, Roboflow inference layer, 
and cascading ORB/SIFT feature matchers.

STATUS: 
Archived in Sandbox. Replaced and upgraded by the production-ready `app.py`, which integrates 
advanced production patterns including SHA-256 deterministic caching, thread-safe memory gates, 
and input matrix downscaling.
"""

from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
import logging
from werkzeug.utils import secure_filename
from inference_sdk import InferenceHTTPClient
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ.get("ROBOFLOW_API_KEY")
)

def adjust_brightness_if_needed(img, target_brightness=120, dark_threshold=90, bright_threshold=140):
    if img is None:
        return None
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        avg_brightness = np.mean(l)
        logger.info(f"Original brightness: {avg_brightness}")

        if avg_brightness < dark_threshold or avg_brightness > bright_threshold:
            # Compute brightness shift value
            diff = target_brightness - avg_brightness
            l = np.clip(l + diff, 0, 255).astype(np.uint8)
            adjusted_lab = cv2.merge((l, a, b))
            adjusted_img = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
            logger.info(f"Adjusted brightness to target: {target_brightness}")
            return adjusted_img
        else:
            return img
    except Exception as e:
        logger.error(f"Brightness adjustment error: {str(e)}")
        return img


class BookMatcher:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.orb_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.sift = cv2.SIFT_create()
        self.sift_matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=50)
        )

    def get_book_regions(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Cannot read image: {image_path}")
                return []
            img = adjust_brightness_if_needed(img)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp.jpg")
            cv2.imwrite(temp_path, img)

            try:
                try:
                    result = CLIENT.infer(temp_path, model_id="book_detect-tarrm-o8kgf/1")
                except:
                    result = CLIENT.infer(temp_path, model_id="book-spine-detection-2cci9/2")
            finally:
                os.remove(temp_path)

            books = []
            h, w = img.shape[:2]
            for i, pred in enumerate(result.get("predictions", [])):
                try:
                    x, y = int(pred['x']), int(pred['y'])
                    width, height = int(pred['width']), int(pred['height'])
                    x1, y1 = max(0, x - width // 2), max(0, y - height // 2)
                    x2, y2 = min(w, x + width // 2), min(h, y + height // 2)
                    books.append({'id': i, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
                except KeyError:
                    continue
            return books
        except Exception as e:
            logger.error(f"Error in detection: {str(e)}")
            return []


    def compute_histogram(self, img):
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def _match_with(self, desc1, new_gray, query_hist, new_img, book, method='orb'):
        try:
            roi = new_img[book['y1']:book['y2'], book['x1']:book['x2']]
            if roi.size == 0:
                return (None, 0)

            hist = self.compute_histogram(roi)
            distance = cv2.compareHist(query_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if distance > 0.6:
                return (None, 0)

            mask = np.zeros(new_gray.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (book['x1'], book['y1']), (book['x2'], book['y2']), 255, -1)

            if method == 'orb':
                kp2, desc2 = self.orb.detectAndCompute(new_gray, mask)
                if desc2 is None or len(desc2) < 2:
                    return (None, 0)
                matches = self.orb_matcher.knnMatch(desc1, desc2, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            else:  # SIFT fallback
                kp2, desc2 = self.sift.detectAndCompute(new_gray, mask)
                if desc2 is None or len(desc2) < 2:
                    return (None, 0)
                matches = self.sift_matcher.knnMatch(desc1, desc2, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

            return (book, len(good_matches))
        except:
            return (None, 0)

    def _run_matching(self, desc1, gray_img, hist, img, new_books, method):
        best_match, max_score = None, 0
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._match_with, desc1, gray_img, hist, img, book, method
                ) for book in new_books
            ]
            for f in futures:
                book, score = f.result()
                if score > max_score:
                    best_match, max_score = book, score
        return best_match, max_score

    def find_matching_book(self, old_img_path, new_img_path, selected_coords, new_books=None):
        try:
            old_img = cv2.imread(old_img_path)
            new_img = cv2.imread(new_img_path)
            if old_img is None or new_img is None:
                return {'success': False, 'error': 'Images could not be loaded'}

            old_img = adjust_brightness_if_needed(old_img)
            new_img = adjust_brightness_if_needed(new_img)

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

            # === ORB TRY ===
            book_gray_orb = cv2.cvtColor(book_img, cv2.COLOR_BGR2GRAY)
            desc1_orb = self.orb.detectAndCompute(book_gray_orb, None)[1]
            if desc1_orb is not None and len(desc1_orb) >= 2:
                best_orb, score_orb = self._run_matching(desc1_orb, cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY), query_hist, new_img, new_books, method='orb')
            else:
                best_orb, score_orb = None, 0

            if best_orb and score_orb >= 10:
                return {
                    'success': True,
                    'total_books': len(new_books),
                    'best_match': best_orb,
                    'match_score': score_orb,
                    'method': 'orb',
                    'low_confidence': score_orb < 15,
                    'warning': 'Low confidence' if score_orb < 15 else ''
                }

            # === SIFT FALLBACK ===
            book_gray_sift = cv2.cvtColor(book_img, cv2.COLOR_BGR2GRAY)
            desc1_sift = self.sift.detectAndCompute(book_gray_sift, None)[1]
            if desc1_sift is None or len(desc1_sift) < 2:
                return {'success': True, 'warning': 'Not enough features for SIFT'}

            best_sift, score_sift = self._run_matching(desc1_sift, cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY), query_hist, new_img, new_books, method='sift')
            return {
                'success': True,
                'total_books': len(new_books),
                'best_match': best_sift,
                'match_score': score_sift,
                'method': 'sift',
                'low_confidence': score_sift < 10,
                'warning': 'Low confidence' if score_sift < 10 else 'ORB failed, SIFT used'
            }

        except Exception as e:
            logger.exception("Dual matching failed")
            return {'success': False, 'error': str(e)}


@app.route('/')
def index():
    return render_template('index.html')

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

        matcher = BookMatcher()
        old_books = matcher.get_book_regions(old_path)
        new_books = matcher.get_book_regions(new_path)

        return jsonify({
            'success': True,
            'old_image': old_path,
            'new_image': new_path,
            'books': old_books,
            'new_books': new_books,
            "old_book_count" : len(old_books)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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
        result = matcher.find_matching_book(
            data['old_image'],
            data['new_image'],
            coords,
            new_books=data.get('new_books')  # use precomputed if provided
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)
