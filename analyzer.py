from PIL import Image
import numpy as np
import cv2
import random
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import sys

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

# === Load Models ===
WRINKLE_MODEL_PATH = 'model_yesspatial_yesaugment_batch5_v01.h5'
SKIN_TYPE_MODEL_PATH = "skin_type_classifier_best_June01.h5"

def load_wrinkle_model():
    if os.path.exists(WRINKLE_MODEL_PATH):
        try:
            return load_model(
                WRINKLE_MODEL_PATH,
                custom_objects={
                    'dice_loss': dice_loss,
                    'combined_loss': combined_loss,
                    'attention_block': attention_block,
                }
            )
        except Exception as e:
            print(f"Error loading wrinkle model: {str(e)}")
    return None

def load_skin_type_model():
    if os.path.exists(SKIN_TYPE_MODEL_PATH):
        try:
            return load_model(SKIN_TYPE_MODEL_PATH)
        except Exception as e:
            print(f"Error loading skin type model: {e}")
    return None


wrinkle_model = load_wrinkle_model()
skin_type_model = load_skin_type_model()

# === Mediapipe Setup ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# === Face Detector ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# === Dark Circle Model ===
LEFT_EYE_IDXS = [124, 247, 7, 163, 144, 145, 153, 154, 243, 244, 245, 188, 114, 47, 100, 101, 50, 123, 116, 143]  # Left eye outer contour4
RIGHT_EYE_IDXS = [463, 464, 465, 412, 343, 277, 329, 330, 280, 352, 345, 372, 446, 249, 390, 373, 374, 380, 381, 398, 463] # Right eye outer contour
u_zone_indices = [452, 451, 450, 449, 448, 261, 265, 372, 345, 352, 376, 433, 288, 367, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 135, 192, 123, 116,143, 35, 31, 228, 229, 230, 231, 232, 233, 47, 142, 203, 92, 57, 43, 106,182, 83, 18, 313, 406, 335, 273, 287, 410, 423, 371, 277, 453]


print("Succesfully entered the analyzer file")

    
def detect_face(frame):
        print("Succesfully entered the detect face function")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        print("Leaving the detect face function")
        return faces

def crop_to_face(frame, face):
        """Crop the image to face region with padding"""
        print("Succesfully entered the crop face function")
        x, y, w, h = face
        height, width = frame.shape[:2]
        
        # Add padding around face
        x1 = max(0, x - face_padding)
        y1 = max(0, y - face_padding)
        x2 = min(width, x + w + face_padding)
        y2 = min(height, y + h + face_padding)
        
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
        print("Leaving the crop face function")
        return frame[y1:y2, x1:x2]

def get_landmark_coords(image, landmarks, indexes):
        """Extracts pixel coordinates for given landmark indexes."""
        
        print("Entering the get landmarks coordinate function")
        
        h, w = image.shape[:2]
        # Ensure indexes are within bounds
        valid_indexes = [i for i in indexes if i is not None and 0 <= i < len(landmarks)]
        
        print("Leaving the get landmark coordination function")
        return np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in valid_indexes], np.int32)

def draw_guide(frame, validations_met):

        print("Entering the draw guide function")

        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        axis_x = width // 9
        axis_y = height // 4
        color = (0, 255, 0) if validations_met else (0, 0, 255)
        cv2.ellipse(frame, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, color, 2)
        
        print("Leaving the draw guide function")
        return frame

def update_frame(image):
        """Update the video preview"""
        print("Entering the update frame function")
        if cap is None:
            return

        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (1000, 680))
            faces = detect_face(frame)
            validations_met = validate_conditions(frame, faces)

            # Draw guide for face positioning
            display_frame = draw_guide(frame, validations_met)
            
            # Draw rectangle around detected face
            if len(faces) > 0:
                x, y, w, h = faces[0]
                # Draw crop region with padding
                x1 = max(0, x - face_padding)
                y1 = max(0, y - face_padding)
                x2 = min(frame.shape[1], x + w + face_padding)
                y2 = min(frame.shape[0], y + h + face_padding)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print("Leaving the update frame function")
            # # Convert to Qt format and display
            # rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            # h, w, ch = rgb_frame.shape
            # bytes_per_line = ch * w
            # qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # video_label.setPixmap(QPixmap.fromImage(qt_image))

def different_zones(image, landmarks):
        """ Generate segmented facial region masks and segments for left/right eyes """
        
        print("Entering the different zones function")
        
        h, w = image.shape[:2]

        # Convert landmark coordinates to pixel positions
        left_eye_pts = get_landmark_coords(image, landmarks.landmark, LEFT_EYE_IDXS)
        right_eye_pts = get_landmark_coords(image, landmarks.landmark, RIGHT_EYE_IDXS)

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

        print("Leaving the different zones function")
        # Also return the filled masks for later use in scoring/combining
        return left_eye_segment, right_eye_segment, left_eye_mask_full, right_eye_mask_full
    
def preprocess_for_wrinkle(image):
        # Convert to RGB and normalize

        print("Entering the preprocess for wrinkles function")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        normalized = enhanced.astype(np.float32) / 255.0
        image = np.stack([normalized]*3, axis=-1)
        image = cv2.resize(image, (256, 256))
        print("Leaving the preprocess for wrinkles function")
        return np.expand_dims(image, axis=0)

def wrinkle_score(mask):
        print("Entering the wrinkle score function")
        score = np.sum(mask) / mask.size
        print("Leaving the wrinkle score function")
        return np.clip(score * 100, 0, 100)

def wrinkle_score_to_age(score, min_age=20, max_age=80):
        print("Entering the wrinkle score to age function")
        age = min_age + (max_age - min_age) * score
        print("Leaving the wrinkle score to age function")
        return min(int(age), 70)

def analyze_wrinkles(image):
        print("Entering the analyze wrinkle function")
        output_dir = 'output/predicted_masks'
        os.makedirs(output_dir, exist_ok=True)
        if wrinkle_model is None:
            return {"wrinkle_score": None, "estimated_age": None, "message": "Wrinkle model not available"}
            
        try:
            preprocessed = preprocess_for_wrinkle(image)
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
            # original_filename = os.path.basename(image)
            # filename_wo_ext = os.path.splitext(original_filename)[0]
            # Save mask
            filename_wo_ext = "default"
            if isinstance(image, str):  # If image is a path
                filename_wo_ext = os.path.splitext(os.path.basename(image))[0]

            mask_filename = f"{filename_wo_ext}_mask.png"
            mask_path = os.path.join(output_dir, mask_filename)
            cv2.imwrite(mask_path, predicted_mask)
            print(f"Saved wrinkle mask to: {mask_path}")

            
            score = wrinkle_score(binary_mask)
            age = wrinkle_score_to_age(score)
            # print(cv2.imshow("Predicted Mask", predicted_mask))
            print("Leaving the wrinkle analyze to age function")
            return score, age
        
        except Exception as e:
            print(f"Error in wrinkle analysis: {str(e)}")
            return None, None

def detect_skin_tone(image):
        print("Entering the detect skin tone function")
        # image = cv2.imread(image)
        h, w = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            skin_label.setText("No face detected.")
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
        print("Exiting the skin tone function")
        return skin_tone_label

def analyze_skin_type_patches(image):
        print("Entering the skin type patches function")
        if skin_type_model is None:
            return "Skin Type Model: Not available"

        from PIL import Image
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        try:
            # image = Image.open(image_path).convert("RGB")
            # If image is a NumPy array, convert to PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype('uint8')).convert("RGB")
            else:
                image = Image.open(image).convert("RGB")  # fallback for file path

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

            print("exiting the skin type function")
            
            if len(combined_types) > 1:
                return f"Combination ({', '.join(combined_types)})"
            else:
                return dominant_type

        except Exception as e:
            print(f"Error during patch-based skin type analysis: {e}")
            return "Skin Type: Analysis error"

def detect_dark_circles_otsu(image):
        print("Entering the dark circle function")
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
        original_image = image

        if original_image is None:
            print(f"Error: Could not read image file at path: {image}")
            return None, None, None

        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = original_image.shape[:2]

        # --- Initialize and use FaceMesh directly within this function ---
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            print(f"No face detected in {image}.")
            return original_image, np.zeros((h_orig, w_orig), dtype=np.uint8), 0.0 # Return original image and empty mask

        landmarks = results.multi_face_landmarks[0]

        # Get segmented eye regions (same size as original image) and the masks
        left_segment, right_segment, left_eye_mask_full, right_eye_mask_full = different_zones(original_image, landmarks)

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
                print(f"Left eye segment size too small for blur kernel {ksize} in {image_path}.")
                blurred_left_eye = gray_left_eye # Skip blur if too small

            # Apply Otsu's thresholding to the left eye segment
            try:
                # THRESH_BINARY_INV might be better if dark circles are lower intensity
                ret_left, thresh_left = cv2.threshold(blurred_left_eye, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Removed + cv2.THRESH_OTSU
                # You might need to invert if dark circles appear as 0
                # ret_left, thresh_left = cv2.threshold(blurred_left_eye, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                left_dark_circle_mask_full_size = thresh_left # This mask is already full size
            except cv2.error as e:
                print(f"Error during left eye thresholding for {image_path}: {e}")


        # Process Right Eye Segment
        if right_segment.shape[0] > 0 and right_segment.shape[1] > 0 and np.max(right_segment) > 0: # Check if segment is valid and not all black
            gray_right_eye = cv2.cvtColor(right_segment, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur (check kernel size)
            ksize = (7, 7) # Must be odd
            if gray_right_eye.shape[0] >= ksize[0] and gray_right_eye.shape[1] >= ksize[1]:
                blurred_right_eye = cv2.GaussianBlur(gray_right_eye, ksize, 0)
            else:
                print(f"Right eye segment size too small for blur kernel {ksize} in {image_path}.")
                blurred_right_eye = gray_right_eye # Skip blur if too small

            # Apply Otsu's thresholding to the right eye segment
            try:
                # THRESH_BINARY_INV might be better if dark circles are lower intensity
                ret_right, thresh_right = cv2.threshold(blurred_right_eye, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Removed + cv2.THRESH_OTSU
                # You might need to invert if dark circles appear as 0
                # ret_right, thresh_right = cv2.threshold(blurred_right_eye, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                right_dark_circle_mask_full_size = thresh_right # This mask is already full size
            except cv2.error as e:
                print(f"Error during right eye thresholding for {image_path}: {e}")


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

        dark_circle_pixel_count = np.sum(combined_dark_circle_mask_full_size > 0) # Count non-zero pixels in the dark circle mask

        dark_circle_score = 0.0
        if total_eye_pixel_count > 0:
            dark_circle_score = (dark_circle_pixel_count / total_eye_pixel_count) * 100 # Score as percentage of eye area

        # You could refine the score calculation, e.g., consider intensity within the dark circle mask on the original image.
        # E.g., mean_intensity_in_dark_circles = cv2.mean(original_image, mask=combined_dark_circle_mask_full_size)
        # A lower intensity might indicate more severe dark circles. You could incorporate this.
        print("exiting the dark circle function")
        return original_image, combined_dark_circle_mask_full_size, dark_circle_score
    