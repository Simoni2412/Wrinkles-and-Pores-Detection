import sys
import cv2
import random
import numpy as np
import os
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout, 
                           QWidget, QMessageBox, QHBoxLayout, QFileDialog)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt

from tensorflow.keras import layers
from PIL import Image
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


# === Mediapipe Setup ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# === Custom Loss Functions for Wrinkle Model ===
def attention_block(x, g, inter_channels):
    theta_x = tf.keras.layers.Conv2D(inter_channels, (1,1), padding="same")(x)
    phi_g = tf.keras.layers.Conv2D(inter_channels, (1,1), padding="same")(g)
    add_xg = tf.keras.layers.Add()([theta_x, phi_g])
    act_xg = tf.keras.layers.Activation('relu')(add_xg)
    psi = tf.keras.layers.Conv2D(1, (1,1), padding="same", activation="sigmoid")(act_xg)
    return tf.keras.layers.Multiply()([x, psi])

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dsc = dice_loss(y_true, y_pred)
    return bce + dsc

def channel_avg(x):
    return tf.reduce_mean(x, axis=-1, keepdims=True)

def channel_max(x):
    return tf.reduce_max(x, axis=-1, keepdims=True)


def spatial_attention(x):
    avg_pool =  layers.Lambda(channel_avg)(x)
    max_pool = layers.Lambda(channel_max)(x)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attention = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    return layers.Multiply()([x, attention])

# === Load Skin Type Classifier Model ===
SKIN_TYPE_MODEL_PATH = "skin_type_classifier_best_June01.h5"

# === Load Pores Model ===
PORES_MODEL_PATH = "model_pores_batch3.h5"

# === Load Wrinkle Model ===
#wrinkle_model = None
MODEL_PATH = 'model_wrinkles_batch5.h5'

#Custom objects needed to load the model
custom_objects={
        'channel_avg': channel_avg,
        'channel_max': channel_max,
        'spatial_attention': spatial_attention,  # if applicable
        'dice_loss': dice_loss,
        'combined_loss': combined_loss,
        # add any other named functions
    }

try:
    if os.path.exists(SKIN_TYPE_MODEL_PATH):
        skin_type_model = load_model(SKIN_TYPE_MODEL_PATH)
    else:
        skin_type_model = None
except Exception as e:
    print(f"Error loading skin type model: {e}")
    skin_type_model = None

try:
    if os.path.exists(PORES_MODEL_PATH):
        pores_model = load_model(
            PORES_MODEL_PATH,
            custom_objects= custom_objects
        )
except Exception as e:
    print(f"Error loading pores model: {str(e)}")


try:
    if os.path.exists(MODEL_PATH):
        wrinkle_model = load_model(
            MODEL_PATH,
            custom_objects= custom_objects
        )
except Exception as e:
    print(f"Error loading wrinkle model: {str(e)}")

# IDark Circle Model
LEFT_EYE_IDXS = [
    124, 247, 7, 163, 144, 145, 153, 154, 243, 244, 245, 188, 114, 47, 100, 101, 50, 123, 116, 143]  # Left eye outer contour4

RIGHT_EYE_IDXS = [
    463, 464, 465, 412, 343, 277, 329, 330, 280, 352, 345, 372, 446, 249, 390, 373, 374, 380, 381, 398, 463 # Right eye outer contour
]

# Butterfly indices for pores model
BUTTERFLY_ZONE_INDICES = [111, 117, 119, 120, 121, 128, 122, 6, 351, 357, 350, 349, 348, 347, 346, 345, 352, 376, 433, 416, 434, 432, 410, 423, 278, 344, 440, 275, 4, 45, 220, 115, 48, 203, 186, 186, 212, 214, 192, 123, 116]

# U-zone indices (cheeks + jawline) for skin type
u_zone_indices = [452, 451, 450, 449, 448, 261, 265, 372, 345, 352, 376, 433, 288, 367, 397,
                  365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 135, 192, 123, 116,
                  143, 35, 31, 228, 229, 230, 231, 232, 233, 47, 142, 203, 92, 57, 43, 106,
                  182, 83, 18, 313, 406, 335, 273, 287, 410, 423, 371, 277, 453]

IMG_SIZE = (256, 256)
# === Face Detector ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

class GuardioUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.captured_image = None
        self.face_padding = 50
        
        if wrinkle_model is None:
            QMessageBox.warning(
                self,
                "Model Not Found",
                f"Wrinkle analysis model not found at {MODEL_PATH}. Only skin tone analysis will be available."
            )

        if pores_model is None:
            QMessageBox.warning(
                self,
                "Model Not Found",
                f"Pores analysis model not found at {PORES_MODEL_PATH}. Only skin tone analysis will be available."
            )

    def init_ui(self):
        self.setWindowTitle("Initial Skin Analysis")
        self.layout = QVBoxLayout()

        # Preview area
        self.video_label = QLabel(self)
        self.video_label.setMinimumSize(1000, 680)
        self.video_label.setMaximumSize(1000, 680)
        self.video_label.setStyleSheet("QLabel { background-color: black; }")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

        # Analysis results
        self.skin_label = QLabel("Skin Tone: Not analyzed", self)
        self.skin_label.setStyleSheet("QLabel{font-size: 16px}")
        self.layout.addWidget(self.skin_label)

        self.skin_type_label = QLabel("Skin Type: Not analyzed", self)
        self.skin_type_label.setStyleSheet("QLabel{font-size: 16px}")
        self.layout.addWidget(self.skin_type_label)

        self.dark_circle_label = QLabel("Dark Circle Score: Not analyzed", self)
        self.dark_circle_label.setStyleSheet("QLabel{font-size: 16px}")
        self.layout.addWidget(self.dark_circle_label)

        self.wrinkle_label = QLabel("Wrinkle Analysis: Not analyzed", self)
        self.wrinkle_label.setStyleSheet("QLabel{font-size:16px}")
        self.layout.addWidget(self.wrinkle_label)

        self.pores_label = QLabel("Pores Analysis: Not analyzed", self)
        self.pores_label.setStyleSheet("QLabel{font-size:16px}")
        self.layout.addWidget(self.pores_label)

        # Create horizontal layouts for button organization
        self.capture_layout = QHBoxLayout()
        self.analysis_layout = QHBoxLayout()

        button_style = """
        QPushButton {
            font-size: 16px;
            padding: 10px;
            margin: 5px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius:5px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        """

        # Capture options
        self.start_camera_button = QPushButton("Start Camera", self)
        self.start_camera_button.clicked.connect(self.start_camera)
        self.start_camera_button.setStyleSheet(button_style)
        self.capture_layout.addWidget(self.start_camera_button)

        self.capture_button = QPushButton("Capture Photo", self)
        self.capture_button.clicked.connect(self.capture_photo)
        self.capture_button.setEnabled(False)
        self.capture_button.setStyleSheet(button_style)
        self.capture_layout.addWidget(self.capture_button)

        self.upload_button = QPushButton("Upload Photo", self)
        self.upload_button.clicked.connect(self.upload_photo)
        self.upload_button.setStyleSheet(button_style)
        self.capture_layout.addWidget(self.upload_button)

        # Analysis button
        self.analyze_button = QPushButton("Analyze", self)
        self.analyze_button.clicked.connect(self.analyze_image)
        self.analyze_button.setEnabled(False)
        self.analyze_button.setStyleSheet(button_style)
        self.analysis_layout.addWidget(self.analyze_button)

        # Add layouts to main layout
        self.layout.addLayout(self.capture_layout)
        self.layout.addLayout(self.analysis_layout)

        self.setLayout(self.layout)
        
        # Set a fixed size for the window
        self.setFixedSize(1000, 680)

    def start_camera(self):
        """Toggle camera on/off"""
        print("Start Camera")
        if self.cap is None:
            print("Camera not started")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 680)
            self.timer.start(30)
            self.start_camera_button.setText("Stop Camera")
            self.capture_button.setEnabled(True)
            # Clear any previous analysis results
            self.skin_label.setText("Skin Tone: Not analyzed")
            self.skin_type_label.setText("Skin Type: Not analyzed")
            self.wrinkle_label.setText("Wrinkle Analysis: Not analyzed")
            self.pores_label.setText("Pores Analysis: Not analyzed")
        else:
            self.stop_camera()
            self.start_camera_button.setText("Start Camera")
            self.capture_button.setEnabled(False)

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.clear()

    def crop_to_face(self, frame, face):
        """Crop the image to face region with padding"""
        x, y, w, h = face
        height, width = frame.shape[:2]
        
        # Add padding around face
        x1 = max(0, x - self.face_padding)
        y1 = max(0, y - self.face_padding)
        x2 = min(width, x + w + self.face_padding)
        y2 = min(height, y + h + self.face_padding)
        
        # Ensure aspect ratio is maintained (1:1)
        w_crop = x2 - x1
        h_crop = y2 - y1
        if w_crop > h_crop:
            diff = w_crop - h_crop
            y1 = max(0, y1 - diff//2)
            y2 = min(height, y2 + diff//2)
        else:
            diff = h_crop - w_crop
            x1 = max(0, x1 - diff//2)
            x2 = min(width, x2 + diff//2)
        
        return frame[y1:y2, x1:x2]

    def capture_photo(self):
        """Capture a single photo when button is pressed"""
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (1000, 680))
            faces = self.detect_face(frame)
            
            if len(faces) > 0 and self.validate_conditions(frame, faces):
                # Crop to face region
                face_frame = self.crop_to_face(frame, faces[0])
                
                # Resize cropped image to standard size (maintaining aspect ratio)
                target_size = (512, 512)  # Standard size for analysis
                face_frame = cv2.resize(face_frame, target_size)
                
                # Save the cropped image
                filename = f"captured_face_{random.randint(1000, 9999)}.jpg"
                cv2.imwrite(filename, face_frame)
                
                self.captured_image = filename
                self.analyze_button.setEnabled(True)
                
                # Display the cropped image
                rgb_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qt_image))
                
                self.stop_camera()
                self.start_camera_button.setText("Start Camera")
                self.capture_button.setEnabled(False)

                QMessageBox.information(
                    self,
                    "Success",
                    "Face captured successfully! Click 'Analyze' to process the image."
                )
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Capture",
                    "Please ensure your face is clearly visible and properly positioned."
                )

    def analyze_image(self):
        """Analyze the captured photo"""
        if self.captured_image is None:
            return

        try:
            # First analyze skin tone
            self.detect_skin_tone(self.captured_image)
            # self.skin_label.setText(skin_tone_result)
            
            # Load the image for display
            image = cv2.imread(self.captured_image)
            if image is None:
                raise ValueError("Could not read the captured image for analysis")
            
            # Create a copy for overlay
            display_image = image.copy()
            
            # Then analyze wrinkles if model is available
            if wrinkle_model is not None:
                wrinkle_text, binary_mask = self.analyze_wrinkles(image)
                self.wrinkle_label.setText(wrinkle_text)

                if binary_mask is not None:
                    # Create overlay with wrinkle detection
                    display_image = self.create_overlay(image, binary_mask)

            #Pores analysis
            if pores_model is not None:
                pred_mask, cropped_img, bbox = self.analyze_pores(image)
                pred_score = self.calculate_pores_score(pred_mask)
                self.pores_label.setText(pred_score)

                if pred_mask is not None:
                    # Create overlay with Pores detection
                    display_pores_image = self.overlay_pores(image, pred_mask, bbox)

            # Add skin tone overlay
            # display_image = self.overlay_u_zone(display_image)[0]  # Get the overlay image
            # === Run skin type patch-based analysis ===
            skin_type_result = self.analyze_skin_type_patches(self.captured_image)
            self.skin_type_label.setText(skin_type_result)



            # === Run dark circle detection ===
            original_img, dark_circle_mask, score, dark_pixels_mask = self.detect_dark_circles_otsu(self.captured_image)
            display_dark_circle_image = self.dark_circle_overlay(original_img, dark_pixels_mask)
            self.dark_circle_label.setText(f"Dark Circle Score: {int(score)}")

            # # Resize for display while maintaining aspect ratio
            # display_size = (1000, 680)
            # h, w = display_image.shape[:2]
            # scale = min(display_size[0]/w, display_size[1]/h)
            # new_size = (int(w*scale), int(h*scale))
            #
            # # Resize image
            # display_image = cv2.resize(display_image, new_size)
            #
            # # Convert to RGB for Qt
            # display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            #
            # # Convert to QImage
            # h, w, ch = display_image_rgb.shape
            # bytes_per_line = ch * w
            # qt_image = QImage(display_image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            #
            # # Convert to pixmap and display
            # pixmap = QPixmap.fromImage(qt_image)
            #
            # self.video_label.clear()
            # self.video_label.setPixmap(pixmap)
            # self.video_label.setAlignment(Qt.AlignCenter)
            # self.video_label.repaint()
            # Resize and prepare wrinkle overlay
            display_size = (1000, 680)

            # --- Resize all overlays to same size ---
            def resize_to_display(img, target_size):
                h, w = img.shape[:2]
                scale = min(target_size[0] / w, target_size[1] / h)
                new_size = (int(w * scale), int(h * scale))
                return cv2.resize(img, new_size)

            display_image = resize_to_display(display_image, display_size)
            display_pores_image = resize_to_display(display_pores_image, display_size)
            display_dark_circle_image = resize_to_display(display_dark_circle_image, display_size)

            # --- Match sizes exactly (in case of rounding mismatches) ---
            h, w = display_image.shape[:2]
            display_pores_image = cv2.resize(display_pores_image, (w, h))
            display_dark_circle_image = cv2.resize(display_dark_circle_image, (w, h))

            # --- Blend all overlays ---
            combined = cv2.addWeighted(display_image, 0.6, display_pores_image, 0.3, 0)
            combined = cv2.addWeighted(combined, 1.0, display_dark_circle_image, 0.3, 0)

            # --- Convert to RGB for Qt ---
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

            # --- Convert to QImage and show ---
            h, w, ch = combined_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(combined_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.clear()
            self.video_label.setPixmap(pixmap)
            self.video_label.setAlignment(Qt.AlignCenter)
            self.video_label.repaint()



        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            QMessageBox.warning(
                self,
                "Analysis Error",
                "An error occurred during analysis. Please try again."
            )
            import traceback
            traceback.print_exc()

    def create_overlay(self, image, binary_mask):
        """Create an overlay showing the wrinkle detection"""
        # Resize mask to match image size
        resized_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))
        
        # Create colored overlay
        overlay = image.copy()
        colored_mask = np.zeros_like(overlay)
        colored_mask[resized_mask > 0] = [0, 255, 0]  # Green color for wrinkles
        
        # Add glow effect to make wrinkles more visible
        kernel = np.ones((3,3), np.uint8)
        dilated_mask = cv2.dilate(resized_mask, kernel, iterations=1)
        glow_mask = np.zeros_like(image)
        glow_mask[dilated_mask > 0] = [0, 128, 0]  # Darker green for glow
        
        # Blend the original image with the overlay
        result = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
        result = cv2.addWeighted(result, 0.9, glow_mask, 0.1, 0)
        
        return result

    def closeEvent(self, event):
        """Clean up when the application closes"""
        self.stop_camera()
        # Clean up temporary files
        if self.captured_image and os.path.exists(self.captured_image):
            os.remove(self.captured_image)
        # Clean up any uploaded files
        for filename in os.listdir('.'):
            if filename.startswith(('captured_face_', 'uploaded_face_')) and filename.endswith('.jpg'):
                try:
                    os.remove(filename)
                except Exception as e:
                    print(f"Error removing temporary file {filename}: {str(e)}")
        event.accept()

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces

    def validate_conditions(self, frame, faces):
        """Check lighting and face alignment conditions."""
        if len(faces) == 0:
            return False

        # Lighting condition
        brightness = frame.mean()
        if brightness < 80:  # Raised from 50 to 80 for better lighting
            return False

        # Face position and size validation (centered and large enough)
        h, w = frame.shape[:2]
        x, y, fw, fh = faces[0]  # first detected face

        face_center_x = x + fw // 2
        face_center_y = y + fh // 2

        frame_center_x = w // 2
        frame_center_y = h // 2

        # Check if face is centered within a small range
        offset_x = abs(face_center_x - frame_center_x)
        offset_y = abs(face_center_y - frame_center_y)

        if offset_x > w * 0.1 or offset_y > h * 0.1:
            return False

        # Check if face is large enough (indicates closeness)
        if fw < w * 0.2 or fh < h * 0.2:
            return False

        return True


    def overlay_u_zone(self, frame):
        h, w = frame.shape[:2]
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return frame, None  # No overlay or mask

        landmarks = results.multi_face_landmarks[0]
        u_points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in u_zone_indices]

        # Mask for color sampling
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(u_points, dtype=np.int32)], 255)

        # Overlay for visualization
        overlay = frame.copy()
        cv2.polylines(overlay, [np.array(u_points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        return overlay, mask

    def detect_skin_tone(self, image):
        image = cv2.imread(image)
        h, w = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            self.skin_label.setText("No face detected.")
            return

        landmarks = results.multi_face_landmarks[0]
        u_points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in u_zone_indices]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(u_points, dtype=np.int32)], 255)

        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        mean_color = cv2.mean(lab_image, mask=mask)
        L, A, B = mean_color[:3]

        if L > 200:
            skin_tone_label = "Fair"
        elif L > 180:
            skin_tone_label = "Light"
        elif L > 150:
            skin_tone_label = "Medium"
        elif L > 120:
            skin_tone_label = "Tan"
        elif L > 90:
            skin_tone_label = "Deep"
        else:
            skin_tone_label = "Dark"

        self.skin_label.setText(f"Skin Tone: {skin_tone_label}")

    def dark_circle_overlay(self, image, dark_pixels_mask):
        # Create an overlay image (same size as original)
        overlay = np.zeros_like(image, dtype=np.uint8)

        # # Create a boolean mask for dark pixels within the eye region
        # dark_pixels_mask = (combined_dark_circle_mask_full_size == 0) & (total_eye_region_mask > 0)

        # Set red color [B, G, R] = [0, 0, 255] where dark_pixels_mask is True
        overlay[dark_pixels_mask] = [0, 0, 255]

        # Blend the overlay with the original image
        alpha = 0.7  # Transparency factor
        overlaid_image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

        return overlaid_image

    def preprocess_for_wrinkle(self, image):
        # Convert to RGB and normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        normalized = enhanced.astype(np.float32) / 255.0
        image = np.stack([normalized]*3, axis=-1)
        image = cv2.resize(image, (256, 256))
        return np.expand_dims(image, axis=0)

    def wrinkle_score(self, mask):
        score = np.sum(mask) / mask.size
        return np.clip(score * 100, 0, 100)

    def wrinkle_score_to_age(self, score, min_age=20, max_age=80):
        age = min_age + (max_age - min_age) * score
        return min(int(age), 70)

    def analyze_wrinkles(self, image):
        output_dir = 'output/predicted_masks'
        os.makedirs(output_dir, exist_ok=True)
        if wrinkle_model is None:
            return "Wrinkle Analysis: Model not available"
            
        try:
            preprocessed = self.preprocess_for_wrinkle(image)
            predicted_mask = wrinkle_model.predict(preprocessed)
            if predicted_mask.shape[-1] == 1:
                predicted_mask = predicted_mask[0, ..., 0]
            else:
                predicted_mask = predicted_mask[0]
            
            binary_mask = (predicted_mask > 0.5).astype(np.uint8) 

            # Determine original filename (e.g., uploaded_face_1234.jpg)
             # Convert mask to uint8
            if predicted_mask.dtype != np.uint8:
                predicted_mask = (predicted_mask * 255).astype(np.uint8)

            # Saving the predicted mask
            original_filename = os.path.basename(self.captured_image)
            filename_wo_ext = os.path.splitext(original_filename)[0]
            mask_filename = f"{filename_wo_ext}_mask.png"
            mask_path = os.path.join(output_dir, mask_filename)
            cv2.imwrite(mask_path, predicted_mask)
            print(f"Saved wrinkle mask to: {mask_path}")

            
            score = self.wrinkle_score(binary_mask)
            age = self.wrinkle_score_to_age(score)
            # print(cv2.imshow("Predicted Mask", predicted_mask))
            return f"Wrinkle Score: {score:.1f}, Estimated Skin Age: {age}", binary_mask
        except Exception as e:
            print(f"Error in wrinkle analysis: {str(e)}")
            return

    def detect_landmarks(self, image):
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0]
        return None

    def crop_to_butterfly_zone(self, image, landmarks, indices):
        h, w = image.shape[:2]
        butterfly_pts = self.get_landmark_coords(image, landmarks, indices)
        if butterfly_pts.size == 0:
            return None, None
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [butterfly_pts], 255)
        ys, xs = np.where(mask > 0)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        cropped_image = image[y_min:y_max, x_min:x_max]
        bbox = (x_min, y_min, x_max, y_max)
        return cropped_image, bbox

    def analyze_pores(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmark = self.detect_landmarks(image)
        cropped_img, bbox = self.crop_to_butterfly_zone(image, landmark.landmark, BUTTERFLY_ZONE_INDICES)
        img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        normalized = enhanced.astype(np.float32) / 255.0

        # Convert to 3 channels
        image = np.stack([normalized] * 3, axis=-1)

        # img = np.expand_dims(img, axis=0)

        # Ensure IMG_SIZE is a tuple of integers
        image = cv2.resize(image, IMG_SIZE)
        img = np.expand_dims(image, axis=0)

        #cv2.imshow((image * 255).astype(np.uint8))
        pred = pores_model.predict(img)
        # If model output is (1, 256, 256, 1), squeeze to (256, 256)
        if pred.shape[-1] == 1:
            pred = pred[0, ..., 0]
        else:
            pred = pred[0]
        return pred, cropped_img, bbox

    def calculate_pores_score(self, pred_mask, threshold=0.2):
        """
        Calculates a pores score out of 100, where higher is better (fewer pores).
        """
        pores_mask = (pred_mask > threshold).astype(np.uint8)
        pore_pixel_count = np.sum(pores_mask)
        total_pixel_count = pores_mask.size
        pore_fraction = pore_pixel_count / total_pixel_count
        pores_score = (1 - pore_fraction) * 100
        return f"Wrinkle Score: {pores_score:.1f}"

    def overlay_pores(self, image, mask, bbox=None, alpha=0.7, mask_color=(255, 0, 255)):
        """
        Safely overlays a predicted mask onto the original image using a bounding box.

        Args:
            original_image: The full, original image.
            pred_mask: The prediction mask from the model (for the cropped region).
            bbox: The (x_min, y_min, x_max, y_max) bounding box for the crop.
            threshold: The value to binarize the mask (0-1).
            alpha: The transparency of the overlay.
            mask_color: The (B, G, R) color for the mask.

        Returns:
            The image with the mask overlaid.
        """
        # Create a copy of the original image to draw on
        overlay_img = image.copy()

        # 1. Binarize the predicted mask
        mask_bin = (mask > 0.1).astype(np.uint8) * 255

        # 2. Handle the "no pores detected" edge case
        if np.sum(mask_bin) == 0:
            return image  # Nothing to overlay, return the original

        # 3. Get coordinates and dimensions from the bounding box
        x_min, y_min, x_max, y_max = bbox
        h_bbox, w_bbox = y_max - y_min, x_max - x_min

        # 4. Handle invalid bounding box
        if h_bbox <= 0 or w_bbox <= 0:
            return image

        # 5. Resize the mask to the size of the bounding box
        mask_resized = cv2.resize(mask_bin, (w_bbox, h_bbox), interpolation=cv2.INTER_NEAREST)

        # 6. Create a solid color mask for blending
        color_mask = np.zeros((h_bbox, w_bbox, 3), dtype=np.uint8)
        color_mask[mask_resized > 0] = mask_color

        # 7. Extract the Region of Interest (ROI) from the image
        roi = overlay_img[y_min:y_max, x_min:x_max]

        # 8. Blend the color mask with the ROI
        blended_roi = cv2.addWeighted(roi, 1.0, color_mask, alpha, 0)

        # 9. Place the blended ROI back into the main image
        overlay_img[y_min:y_max, x_min:x_max] = blended_roi

        return overlay_img

    def analyze_skin_type_patches(self, image_path):
        if skin_type_model is None:
            return "Skin Type Model: Not available"

        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((520, 520))
            image_array = np.array(image)

            patch_size = (260, 260)
            stride = 130
            h, w = image_array.shape[:2]
            patch_predictions = []

            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    patch = image_array[y:y+patch_size[0], x:x+patch_size[1]]
                    if patch.shape[0] != patch_size[0] or patch.shape[1] != patch_size[1]:
                        continue
                    patch_preprocessed = tf.keras.applications.efficientnet_v2.preprocess_input(patch)
                    patch_preprocessed = np.expand_dims(patch_preprocessed, axis=0)
                    pred = skin_type_model.predict(patch_preprocessed, verbose=0)
                    pred_class = np.argmax(pred)
                    patch_predictions.append(pred_class)

            if not patch_predictions:
                return "Skin Type: Unable to predict (no valid patches)"

            unique, counts = np.unique(patch_predictions, return_counts=True)
            vote_result = dict(zip(unique, counts))
            total = sum(counts)
            class_labels = {0: 'Dry', 1: 'Normal', 2: 'Oily'}
            majority_class = max(vote_result, key=vote_result.get)
            dominant_type = class_labels[majority_class]

            # Detect combination skin type
            threshold_percent = 25
            combined_types = [class_labels[cls] for cls, cnt in vote_result.items() if (cnt / total) * 100 >= threshold_percent]

            if len(combined_types) > 1:
                return f"Skin Type: Combination ({', '.join(combined_types)})"
            else:
                return f"Skin Type: {dominant_type}"

        except Exception as e:
            print(f"Error during patch-based skin type analysis: {e}")
            return "Skin Type: Analysis error"

    def draw_guide(self, frame, validations_met):
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        axis_x = width // 9
        axis_y = height // 4
        color = (0, 255, 0) if validations_met else (0, 0, 255)
        cv2.ellipse(frame, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, color, 2)
        return frame

    def update_frame(self):
        """Update the video preview"""
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (1000, 680))
            faces = self.detect_face(frame)
            validations_met = self.validate_conditions(frame, faces)

            # Draw guide for face positioning
            display_frame = self.draw_guide(frame, validations_met)
            
            # Draw rectangle around detected face
            if len(faces) > 0:
                x, y, w, h = faces[0]
                # Draw crop region with padding
                x1 = max(0, x - self.face_padding)
                y1 = max(0, y - self.face_padding)
                x2 = min(frame.shape[1], x + w + self.face_padding)
                y2 = min(frame.shape[0], y + h + self.face_padding)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Convert to Qt format and display
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def get_landmark_coords(self,image, landmarks, indexes):
        """Extracts pixel coordinates for given landmark indexes."""
        h, w = image.shape[:2]
        # Ensure indexes are within bounds
        valid_indexes = [i for i in indexes if i is not None and 0 <= i < len(landmarks)]
        return np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in valid_indexes], np.int32)

    
    def dark_circle_segments(self,image, landmarks):
        """ Generate segmented facial region masks and segments for left/right eyes """
        h, w = image.shape[:2]

        # Convert landmark coordinates to pixel positions
        left_eye_pts = self.get_landmark_coords(image, landmarks.landmark, LEFT_EYE_IDXS)
        right_eye_pts = self.get_landmark_coords(image, landmarks.landmark, RIGHT_EYE_IDXS)

        # Initialize blank masks for each region (size of original image)
        left_eye_mask_full = np.zeros((h, w), dtype=np.uint8)
        right_eye_mask_full = np.zeros((h, w), dtype=np.uint8)

        # Fill masks with corresponding regions (using 255 for filled area)
        if left_eye_pts.size > 0:
            cv2.fillPoly(left_eye_mask_full, [np.array(left_eye_pts, dtype=np.int32)], 255)  # Left eye region
        if right_eye_pts.size > 0:
            cv2.fillPoly(right_eye_mask_full, [np.array(right_eye_pts, dtype=np.int32)], 255)  # Right eye region

        # Extract segmented images using individual masks (size of original image)
        left_eye_segment = cv2.bitwise_and(image, image, mask=left_eye_mask_full)
        right_eye_segment = cv2.bitwise_and(image, image, mask=right_eye_mask_full)

        # Also return the filled masks for later use in scoring/combining
        return left_eye_segment, right_eye_segment, left_eye_mask_full, right_eye_mask_full
    
    
    def detect_dark_circles_otsu(self,image):
        """
        Detects dark circles in the left and right eye regions separately
        using landmark-based segmentation and Otsu's thresholding.

        Args:
            image_path (str): Path to the input image file.

        Returns:
            Tuple: (original_image, combined_dark_circle_mask_full_size, dark_circle_score).
                Returns (None, None, None) if face or eye region detection fails.
        """

        # Read the image
        original_image = cv2.imread(image)

        if original_image is None:
            print(f"Error: Could not read image file at path: {image}")
            return None, None, None

        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = original_image.shape[:2]

        # --- Initialize and use FaceMesh directly within this function ---

        landmarks = self.detect_landmarks(rgb_image)

        # Get segmented eye regions (same size as original image) and the masks
        left_segment, right_segment, left_eye_mask_full, right_eye_mask_full = self.dark_circle_segments(original_image, landmarks)

        # Initialize masks for detected dark circles (full size)
        left_dark_circle_mask_full_size = np.zeros((h_orig, w_orig), dtype=np.uint8)
        right_dark_circle_mask_full_size = np.zeros((h_orig, w_orig), dtype=np.uint8)


        # Process Left Eye Segment
        if left_segment.shape[0] > 0 and left_segment.shape[1] > 0 and np.max(left_segment) > 0: # Check if segment is valid and not all black
            gray_left_eye = cv2.cvtColor(left_segment, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur (check kernel size)
            ksize = (7, 7) # Must be odd
            if gray_left_eye.shape[0] >= ksize[0] and gray_left_eye.shape[1] >= ksize[1]:
                blurred_left_eye = cv2.GaussianBlur(gray_left_eye, ksize, 0)
            else:
                print(f"Left eye segment size too small for blur kernel {ksize} in {image}.")
                blurred_left_eye = gray_left_eye # Skip blur if too small

            # Apply Otsu's thresholding to the left eye segment
            try:
                # THRESH_BINARY_INV might be better if dark circles are lower intensity
                ret_left, thresh_left = cv2.threshold(blurred_left_eye, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Removed + cv2.THRESH_OTSU
                # You might need to invert if dark circles appear as 0
                # ret_left, thresh_left = cv2.threshold(blurred_left_eye, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                left_dark_circle_mask_full_size = thresh_left # This mask is already full size
            except cv2.error as e:
                print(f"Error during left eye thresholding for {image}: {e}")


        # Process Right Eye Segment
        if right_segment.shape[0] > 0 and right_segment.shape[1] > 0 and np.max(right_segment) > 0: # Check if segment is valid and not all black
            gray_right_eye = cv2.cvtColor(right_segment, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur (check kernel size)
            ksize = (7, 7) # Must be odd
            if gray_right_eye.shape[0] >= ksize[0] and gray_right_eye.shape[1] >= ksize[1]:
                blurred_right_eye = cv2.GaussianBlur(gray_right_eye, ksize, 0)
            else:
                print(f"Right eye segment size too small for blur kernel {ksize} in {image}.")
                blurred_right_eye = gray_right_eye # Skip blur if too small

            # Apply Otsu's thresholding to the right eye segment
            try:
                # THRESH_BINARY_INV might be better if dark circles are lower intensity
                ret_right, thresh_right = cv2.threshold(blurred_right_eye, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Removed + cv2.THRESH_OTSU
                # You might need to invert if dark circles appear as 0
                # ret_right, thresh_right = cv2.threshold(blurred_right_eye, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                right_dark_circle_mask_full_size = thresh_right # This mask is already full size
            except cv2.error as e:
                print(f"Error during right eye thresholding for {image}: {e}")


        # Combine the left and right dark circle masks (full size)
        # Use bitwise_or to combine the binary masks
        combined_dark_circle_mask_full_size = cv2.bitwise_or(left_dark_circle_mask_full_size, right_dark_circle_mask_full_size)

        # --- Optional: Refine the combined mask ---
        # Apply morphological operations to clean up the mask (e.g., remove small noise)
        kernel = np.ones((3, 3), np.uint8)
        combined_dark_circle_mask_full_size = cv2.morphologyEx(combined_dark_circle_mask_full_size, cv2.MORPH_OPEN, kernel) # Opening to remove small objects

        # --- Calculate Dark Circle Score ---
        # A simple score can be based on the proportion of detected dark circle pixels
        # within the total eye region area.
        total_eye_region_mask = cv2.bitwise_or(left_eye_mask_full, right_eye_mask_full) # Combine the original eye region masks
        total_eye_pixel_count = np.sum(total_eye_region_mask > 0) # Count non-zero pixels in the eye region mask

        # Create a boolean mask for dark pixels within the eye region
        dark_pixels_mask = (combined_dark_circle_mask_full_size == 0) & (total_eye_region_mask > 0)

        dark_circle_pixel_count = np.sum(combined_dark_circle_mask_full_size > 0) # Count non-zero pixels in the dark circle mask

        dark_circle_score = 0.0
        if total_eye_pixel_count > 0:
            dark_circle_score = (dark_circle_pixel_count / total_eye_pixel_count) * 100 # Score as percentage of eye area

        # You could refine the score calculation, e.g., consider intensity within the dark circle mask on the original image.
        # E.g., mean_intensity_in_dark_circles = cv2.mean(original_image, mask=combined_dark_circle_mask_full_size)
        # A lower intensity might indicate more severe dark circles. You could incorporate this.

        return original_image, combined_dark_circle_mask_full_size, dark_circle_score, dark_pixels_mask
    
    
    def upload_photo(self):
        """Handle photo upload from file system"""
        print("Starting photo upload...")
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Upload Photo",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        
        if file_name:
            print(f"Selected file: {file_name}")
            try:
                # Read the uploaded image
                image = cv2.imread(file_name)
                if image is None:
                    print("Failed to read image file")
                    raise ValueError("Could not read the image file")
                
                print(f"Image loaded successfully. Shape: {image.shape}")
                
                # Convert BGR to RGB first
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Calculate scaling to fit in display area while maintaining aspect ratio
                display_size = (640, 480)
                h, w = image_rgb.shape[:2]
                print(f"Original image size: {w}x{h}")
                
                # Calculate scale factor
                scale = min(display_size[0]/w, display_size[1]/h)
                new_size = (int(w*scale), int(h*scale))
                print(f"New size after scaling: {new_size}")
                
                # Resize image
                display_image = cv2.resize(image_rgb, new_size)
                # Convert directly to QImage
                h, w, ch = display_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(display_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # Convert to pixmap and set it
                pixmap = QPixmap.fromImage(qt_image)
                print(f"Created pixmap of size: {pixmap.width()}x{pixmap.height()}")
                
                # Stop camera if it's running
                self.stop_camera()
                self.start_camera_button.setText("Start Camera")
                self.capture_button.setEnabled(False)
                
                # Update the label
                self.video_label.clear()  # Clear any existing content
                self.video_label.setPixmap(pixmap)
                self.video_label.setAlignment(Qt.AlignCenter)
                self.video_label.repaint()  # Force a repaint
                
                print("Image displayed. Processing for face detection...")
                
                # Now process for face detection
                faces = self.detect_face(image)
                if len(faces) == 0:
                    print("No faces detected in the image")
                    QMessageBox.warning(
                        self,
                        "No Face Detected",
                        "Could not detect a face in the uploaded image. Please try another photo."
                    )
                    return
                
                print(f"Found {len(faces)} faces")
                
                # Crop and save for analysis
                face_frame = self.crop_to_face(image, faces[0])
                filename = f"uploaded_face_{random.randint(1000, 9999)}.jpg"
                cv2.imwrite(filename, face_frame)
                self.captured_image = filename
                self.analyze_button.setEnabled(True)
                
                
                print("Upload process completed successfully")
                
            except Exception as e:
                print(f"Error in upload_photo: {str(e)}")
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Error processing uploaded image: {str(e)}"
                )
                import traceback
                traceback.print_exc()  # Print detailed error information


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GuardioUI()
    window.show()
    sys.exit(app.exec_())
