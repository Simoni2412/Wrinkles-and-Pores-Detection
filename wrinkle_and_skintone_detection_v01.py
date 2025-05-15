import sys
import cv2
import random
import numpy as np
import os
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

import mediapipe as mp

# === Mediapipe Setup ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

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
        self.timer.timeout.connect(self.capture_frame)
        self.captured_frames = []
        self.max_frames_to_save = 4

    def init_ui(self):
        self.setWindowTitle("Guardio Face Capture & Skin Tone Detection")
        self.layout = QVBoxLayout()

        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        self.skin_label = QLabel("Skin Tone: Not analyzed", self)
        self.layout.addWidget(self.skin_label)

        self.start_button = QPushButton("Start Capture", self)
        self.start_button.clicked.connect(self.start_capture)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Capture & Analyze", self)
        self.stop_button.clicked.connect(self.stop_capture)
        self.layout.addWidget(self.stop_button)

        self.setLayout(self.layout)

    def start_capture(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(50)

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

    def detect_skin_tone(self, image_path):
        image = cv2.imread(image_path)
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

        if L > 80:
            skin_tone_label = "Fair"
        elif L > 65:
            skin_tone_label = "Light"
        elif L > 50:
            skin_tone_label = "Medium"
        elif L > 40:
            skin_tone_label = "Tan"
        elif L > 30:
            skin_tone_label = "Deep"
        else:
            skin_tone_label = "Dark"

        self.skin_label.setText(f"Skin Tone: {skin_tone_label}")

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.resize(frame, (640, 480))
        faces = self.detect_face(frame)
        validations_met = self.validate_conditions(frame, faces)

        # Overlay zone on frame and get mask
        overlayed_frame, _ = self.overlay_u_zone(frame)
        display_frame = self.draw_guide(overlayed_frame, validations_met)

        if validations_met and len(self.captured_frames) < self.max_frames_to_save:
            filename = f"frame_{random.randint(1000, 9999)}.jpg"
            high_res_filename = f"high_res_{filename}"

            # Save the high-resolution frame only when the validation is satisfied
            high_res_frame = cv2.resize(frame, (1280, 960))  # Increase resolution for high-quality capture
            cv2.imwrite(high_res_filename, high_res_frame)
            self.captured_frames.append(high_res_filename)
            print(f"Captured High-Res: {high_res_filename}")

        # Convert for Qt display
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        qimg = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def draw_guide(self, frame, validations_met):
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        axis_x = width // 9
        axis_y = height // 4
        color = (0, 255, 0) if validations_met else (0, 0, 255)
        cv2.ellipse(frame, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, color, 2)
        return frame

    def stop_capture(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

        if self.captured_frames:
            selected_frame = self.captured_frames[-1]
            print(f"Analyzing skin tone for: {selected_frame}")
            self.detect_skin_tone(selected_frame)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GuardioUI()
    window.show()
    sys.exit(app.exec_())
