import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Dict, Any
from PIL.ExifTags import TAGS
import exifread
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata
from fractions import Fraction
import easyocr
import re

class CarLicensePlateDetector:
    """
    A class to detect and recognize license plates on cars using the YOLO model and OCR.

    Attributes:
        model (YOLO): An instance of the YOLO object detection model.
        ocr (easyocr.Reader): EasyOCR reader for text extraction.
    """

    def __init__(self, weights_path: str):
        """
        Initializes the CarLicensePlateDetector with the given weights.

        Args:
            weights_path (str): The path to the weights file for the YOLO model.
        """
        self.model = YOLO(weights_path)
        print("Initializing EasyOCR for Mexico/USA plates...")
        self.ocr = easyocr.Reader(['en', 'es'], gpu=True)
        print("EasyOCR ready!")

    def recognize_license_plate(self, img_path: str) -> np.ndarray:
        """
        Recognizes the license plate in an image and draws a rectangle around it.

        Args:
            img_path (str): The path to the image file containing the car.

        Returns:
            np.ndarray: The image with the license plate region marked and annotated with the recognized text.
        """
        img = self.load_image(img_path)
        results = self.model.predict(img, save=False)
        boxes = results[0].boxes.xyxy
        recognized_text = None

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])

            # Extract license plate text from the ROI
            roi = img[y1:y2, x1:x2]
            license_plate = self.extract_license_plate_text(roi)

            # If license plate text is not empty, update recognized_text
            if license_plate:
                recognized_text = license_plate

            # Draw a rectangle around the license plate
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Print recognized text if available
        if recognized_text:
            print(f"License: {recognized_text}")
            img = self.draw_text(img, recognized_text, (x1, y1 - 20))

        # Prepare info
        image_info = self.get_image_info(img_path)
        info = {
            'DateTime': image_info.get('DateTime', None),
            'GPSLatitude': image_info.get('GPSLatitude', None),
            'GPSLongitude': image_info.get('GPSLongitude', None),
            'License': recognized_text
        }

        return info, img

    @staticmethod
    def draw_text(img: np.ndarray, text: str, xy: Tuple[int, int], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draws text on an image at a specified location.

        Args:
            img (np.ndarray): The image on which to draw text.
            text (str): The text to draw.
            xy (Tuple[int, int]): The (x, y) position where the text will be drawn on the image.
            color (Tuple[int, int, int], optional): The color for the text. Defaults to green.

        Returns:
            np.ndarray: The image with the text drawn on it.
        """
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        draw.text(xy, text, fill=color)
        return np.array(pil_img)

    @staticmethod
    def load_image(img_path: str) -> np.ndarray:
        """
        Loads an image from the specified path.

        Args:
            img_path (str): The path to the image file.

        Returns:
            np.ndarray: The loaded image.
        """
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img[:, :, ::-1].copy()  # Convert BGR to RGB

    @staticmethod
    def preprocess_roi(roi):
        """
        Improve OCR accuracy with multiple preprocessing techniques.
        Returns a list of processed images to try.
        """
        # Convert to BGR if needed
        if len(roi.shape) == 3 and roi.shape[2] == 3:
            roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        else:
            roi_bgr = roi
        
        # Resize for better OCR (make it larger)
        h, w = roi_bgr.shape[:2]
        scale = 3
        roi_large = cv2.resize(roi_bgr, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
        
        processed_imgs = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi_large, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Bilateral filter + Adaptive threshold
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        adaptive = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, 2)
        processed_imgs.append(adaptive)
        
        # Method 2: CLAHE + OTSU
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(gray)
        _, otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_imgs.append(otsu)
        
        # Method 3: Simple OTSU
        _, simple_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_imgs.append(simple_otsu)
        
        # Method 4: Inverted OTSU (for dark text on light background)
        _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        processed_imgs.append(otsu_inv)
        
        # Method 5: Sharpened
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        _, sharp_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_imgs.append(sharp_thresh)
        
        return processed_imgs

    def clean_plate_text(self, text: str) -> str:
        """
        Clean and format detected text for Mexico/USA plates.
        """
        if not text:
            return ""
        
        # Remove extra spaces and convert to uppercase
        text = text.upper().strip()
        text = re.sub(r'\s+', '', text)  # Remove all spaces
        
        # Common OCR corrections
        replacements = {
            'O': '0',  # Letter O to zero (context-dependent)
            'I': '1',  # Letter I to one (context-dependent)
            'Z': '2',  # Sometimes Z is misread as 2
            'S': '5',  # Sometimes S is misread as 5
            'B': '8',  # Sometimes B is misread as 8
            '|': 'I',
            ';': '',
            ':': '',
            '.': '',
            ',': '',
        }
        
        # Remove unwanted characters
        for char in [';', ':', '.', ',', '|', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')']:
            text = text.replace(char, '')
        
        # Keep only alphanumeric and hyphens
        text = re.sub(r'[^A-Z0-9-]', '', text)
        
        # Try to format based on patterns
        # Mexico common: ABC1234, ABC-123-4, ABC-12-34
        # USA common: ABC1234, ABC-1234, 123-ABC
        
        # If no hyphen exists, try to add one intelligently
        if '-' not in text and len(text) >= 6:
            # Check if it starts with letters
            match_letters_first = re.match(r'^([A-Z]{2,3})(\d{3,4})$', text)
            if match_letters_first:
                letters, numbers = match_letters_first.groups()
                return f"{letters}-{numbers}"
            
            # Check if it starts with numbers
            match_numbers_first = re.match(r'^(\d{3})([A-Z]{3})$', text)
            if match_numbers_first:
                numbers, letters = match_numbers_first.groups()
                return f"{numbers}-{letters}"
        
        return text

    def extract_license_plate_text(self, roi: np.ndarray) -> str:
        """
        Extracts text from license plate using EasyOCR with multiple preprocessing attempts.
        """
        try:
            # Get multiple preprocessed versions
            processed_imgs = self.preprocess_roi(roi)
            
            all_results = []
            
            # Try OCR on each preprocessed image
            for idx, proc_img in enumerate(processed_imgs):
                try:
                    # Use allowlist to restrict to valid characters
                    results = self.ocr.readtext(
                        proc_img,
                        detail=1,
                        paragraph=False,
                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
                    )
                    
                    for (bbox, text, confidence) in results:
                        if confidence > 0.2 and len(text) >= 4:  # Minimum confidence and length
                            cleaned = self.clean_plate_text(text)
                            if cleaned:
                                all_results.append((cleaned, confidence))
                                print(f"  Method {idx+1}: '{text}' -> '{cleaned}' (conf: {confidence:.2f})")
                
                except Exception as e:
                    continue
            
            if not all_results:
                return ""
            
            # Sort by confidence and return best result
            all_results.sort(key=lambda x: x[1], reverse=True)
            best_text = all_results[0][0]
            
            return best_text

        except Exception as e:
            print(f"OCR error: {e}")
            return ""

    def display_and_save(self, imgs: List[np.ndarray], save_path: str = "images/yolov8_car.jpg") -> None:
        """
        Displays and saves a list of images without altering their size.

        Args:
            imgs (List[np.ndarray]): A list of images to be displayed and saved.
            save_path (str): The file path where the image will be saved.
        """
        for i, img in enumerate(imgs):
            plt.subplot(1, len(imgs), i + 1)
            plt.axis("off")
            plt.imshow(img)
        plt.savefig(save_path, bbox_inches='tight')

    def process_video(self, video_path: str, output_path: str) -> None:
        """
        Processes a video file to detect and recognize license plates in each frame.

        Args:
            video_path (str): The path to the video file.
            output_path (str): The path where the output video will be saved.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Error opening video file")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0,
                             (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Process the frame
                _, annotated_frame = self.recognize_license_plate(frame)
                # Write the frame
                out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            else:
                break

        # Release everything when done
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def get_media_info(self, file_path: str) -> Union[str, Dict[str, Any]]:
        """
        Get media information from a file.

        Args:
            file_path (str): The path to the media file.

        Returns:
            Union[str, Dict[str, Any]]: A dictionary containing media information or an error message.

        Raises:
            Exception: If an error occurs while reading the file.
        """
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return self.get_image_info(file_path)
        elif file_path.lower().endswith(('.mp4', '.mov', '.avi')):
            return self.get_video_info(file_path)
        else:
            return "Unsupported file format"

    def get_image_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information from an image file.

        Args:
            file_path (str): The path to the image file.

        Returns:
            Dict[str, Any]: A dictionary containing image information.

        Raises:
            Exception: If an error occurs while reading the image data.
        """
        try:
            image = Image.open(file_path)
            raw_exif_data = image._getexif()

            if raw_exif_data is None:
                return {"Error": "No EXIF data found in the image."}

            exif_data = {
                TAGS[key]: value
                for key, value in raw_exif_data.items()
                if key in TAGS and value
            }
            datetime = exif_data.get('DateTime', 'Unknown')
            gps_info = self.extract_gps_data(file_path)
            return {'DateTime': datetime, **gps_info}
        except Exception as e:
            return f"Error reading image data: {e}"

    def extract_gps_data(self, file_path: str) -> Dict[str, Any]:
        """
        Extract GPS data from an image file.

        Args:
            file_path (str): The path to the image file.

        Returns:
            Dict[str, Any]: A dictionary containing GPS information.

        Raises:
            Exception: If an error occurs while extracting GPS data.
        """
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f)
        gps_info = {}
        for tag in tags.keys():
            if tag.startswith("GPS"):
                gps_info[tag] = tags[tag]
        return self.parse_gps_info(gps_info)

    def parse_gps_info(self, gps_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Parse GPS information from a dictionary.

        Args:
            gps_info (Dict[str, Any]): A dictionary containing GPS information.

        Returns:
            Dict[str, float]: A dictionary containing parsed GPS information.

        Raises:
            Exception: If an error occurs while parsing GPS data.
        """
        gps_data = {}
        if 'GPS GPSLatitude' in gps_info and 'GPS GPSLatitudeRef' in gps_info:
            gps_data['GPSLatitude'] = self.convert_to_degrees(gps_info['GPS GPSLatitude'].values)
            if gps_info['GPS GPSLatitudeRef'].printable != 'N':
                gps_data['GPSLatitude'] = -gps_data['GPSLatitude']
        if 'GPS GPSLongitude' in gps_info and 'GPS GPSLongitudeRef' in gps_info:
            gps_data['GPSLongitude'] = self.convert_to_degrees(gps_info['GPS GPSLongitude'].values)
            if gps_info['GPS GPSLongitudeRef'].printable != 'E':
                gps_data['GPSLongitude'] = -gps_data['GPSLongitude']
        return gps_data

    def convert_to_degrees(self, value: Tuple[int, int, int]) -> float:
        """
        Convert GPS coordinate values to degrees.

        Args:
            value (Tuple[int, int, int]): A tuple containing degrees, minutes, and seconds.

        Returns:
            float: The coordinate value in degrees.
        """
        d, m, s = value
        d = float(d.numerator) / float(d.denominator)
        m = float(m.numerator) / float(m.denominator)
        s = float(s.numerator) / float(s.denominator)
        return d + (m / 60.0) + (s / 3600.0)

    def get_video_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information from a video file.

        Args:
            file_path (str): The path to the video file.

        Returns:
            Dict[str, Any]: A dictionary containing video information or an error message.

        Raises:
            Exception: If an error occurs while reading the video data.
        """
        try:
            parser = createParser(file_path)
            if not parser:
                return "Unable to parse video file"
            with parser:
                metadata = extractMetadata(parser)
            return metadata.exportDictionary() if metadata else "No metadata found in video"
        except Exception as e:
            return f"Error reading video data: {e}"


if __name__ == '__main__':
    weights_path: str = 'models/best.pt'
    detector = CarLicensePlateDetector(weights_path)

    file_path = 'medias/polestar-1-2020-001-20220102102935-1280x925.jpg'
    info, _ = detector.recognize_license_plate(file_path)
    print(info)

    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Use the already loaded image
        _, recognized_img = detector.recognize_license_plate(file_path)
        image_output_path = './medias/yolov8_Scooter.jpg'
        cv2.imwrite(image_output_path, cv2.cvtColor(recognized_img, cv2.COLOR_RGB2BGR))
        print(f"Saved the image with the license plate to {image_output_path}")
    elif file_path.lower().endswith(('.mp4', '.mov', '.avi')):
        video_output_path = '/path/to/save/processed.mp4'
        detector.process_video(file_path, video_output_path)
        print(f"Saved the processed video to {video_output_path}")
    else:
        print("Unsupported media format")