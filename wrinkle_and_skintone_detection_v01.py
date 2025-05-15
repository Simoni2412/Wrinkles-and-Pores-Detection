import sys
import cv2
import random
import numpy as np
import os
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout, 
                           QWidget, QMessageBox, QHBoxLayout, QFileDialog)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt

import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model


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

# === Load Wrinkle Model ===
#wrinkle_model = None
MODEL_PATH = 'model_yesspatial_yesaugment_batch5.h5'

try:
    if os.path.exists(MODEL_PATH):
        wrinkle_model = load_model(
            MODEL_PATH,
            custom_objects={
                'attention_block': attention_block,
                'dice_loss': dice_loss,
                'combined_loss': combined_loss
            }
        )
except Exception as e:
    print(f"Error loading wrinkle model: {str(e)}")

# U-zone indices (cheeks + jawline)
u_zone_indices = [452, 451, 450, 449, 448, 261, 265, 372, 345, 352, 376, 433, 288, 367, 397,
                  365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 135, 192, 123, 116,
                  143, 35, 31, 228, 229, 230, 231, 232, 233, 47, 142, 203, 92, 57, 43, 106,
                  182, 83, 18, 313, 406, 335, 273, 287, 410, 423, 371, 277, 453]

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

    def init_ui(self):
        self.setWindowTitle("Wrinkle & Skintone Analysis")
        self.layout = QVBoxLayout()

        # Preview area
        self.video_label = QLabel(self)
        self.video_label.setMinimumSize(800, 680)
        self.video_label.setMaximumSize(800, 680)
        self.video_label.setStyleSheet("QLabel { background-color: black; }")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

        # Analysis results
        self.skin_label = QLabel("Skin Tone: Not analyzed", self)
        self.skin_label.setStyleSheet("QLabel{font-size: 16px}")
        self.layout.addWidget(self.skin_label)

        self.wrinkle_label = QLabel("Wrinkle Analysis: Not analyzed", self)
        self.wrinkle_label.setStyleSheet("QLabel{font-size:16px}")
        self.layout.addWidget(self.wrinkle_label)

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
        self.setFixedSize(800, 680)

    def start_camera(self):
        """Toggle camera on/off"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
            self.start_camera_button.setText("Stop Camera")
            self.capture_button.setEnabled(True)
            # Clear any previous analysis results
            self.skin_label.setText("Skin Tone: Not analyzed")
            self.wrinkle_label.setText("Wrinkle Analysis: Not analyzed")
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
            frame = cv2.resize(frame, (640, 480))
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
            
            # Add skin tone overlay
            # display_image = self.overlay_u_zone(display_image)[0]  # Get the overlay image
            
            # Resize for display while maintaining aspect ratio
            display_size = (640, 480)
            h, w = display_image.shape[:2]
            scale = min(display_size[0]/w, display_size[1]/h)
            new_size = (int(w*scale), int(h*scale))
            
            # Resize image
            display_image = cv2.resize(display_image, new_size)
            
            # Convert to RGB for Qt
            display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage
            h, w, ch = display_image_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(display_image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Convert to pixmap and display
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
        colored_mask = np.zeros_like(image)
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
            
            score = self.wrinkle_score(binary_mask)
            age = self.wrinkle_score_to_age(score)
            # print(cv2.imshow("Predicted Mask", predicted_mask))
            return f"Wrinkle Score: {score:.1f}, Estimated Age: {age}", binary_mask
        except Exception as e:
            print(f"Error in wrinkle analysis: {str(e)}")
            return "Wrinkle Analysis: Error during processing"

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
            frame = cv2.resize(frame, (640, 480))
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
