import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Literal

class FacialLandmarkAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def calculate_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def calculate_ear(self, landmarks: List[Tuple[float, float]]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for both eyes.

        Eye landmarks indices (MediaPipe):
        Left eye: [362, 385, 387, 263, 373, 380]
        Right eye: [33, 160, 158, 133, 153, 144]
        """
        LEFT_EYE = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        def calculate_eye_ear(eye_indices: List[int]) -> float:
            # Vertical distances
            vertical1 = self.calculate_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
            vertical2 = self.calculate_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
            # Horizontal distance
            horizontal = self.calculate_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])

            return (vertical1 + vertical2) / (2.0 * horizontal)

        left_ear = calculate_eye_ear(LEFT_EYE)
        right_ear = calculate_eye_ear(RIGHT_EYE)

        return (left_ear + right_ear) / 2.0

    def calculate_mar(self, landmarks: List[Tuple[float, float]]) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR).

        Mouth landmarks indices (MediaPipe):
        [61, 291, 39, 181, 0, 17, 269, 405]
        """
        MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]

        # Vertical distances
        vertical1 = self.calculate_distance(landmarks[MOUTH[1]], landmarks[MOUTH[7]])
        vertical2 = self.calculate_distance(landmarks[MOUTH[2]], landmarks[MOUTH[6]])
        vertical3 = self.calculate_distance(landmarks[MOUTH[3]], landmarks[MOUTH[5]])
        # Horizontal distance
        horizontal = self.calculate_distance(landmarks[MOUTH[0]], landmarks[MOUTH[4]])

        return (vertical1 + vertical2 + vertical3) / (3.0 * horizontal)

    def get_drowsiness_level(self, ear: float, mar: float) -> Literal['safe', 'warning', 'danger']:
        """
        Determine drowsiness level based on EAR and MAR thresholds.
        """
        EAR_THRESHOLD_DANGER = 0.15
        EAR_THRESHOLD_WARNING = 0.20
        MAR_THRESHOLD = 0.65

        if ear < EAR_THRESHOLD_DANGER or mar > MAR_THRESHOLD:
            return 'danger'
        elif ear < EAR_THRESHOLD_WARNING:
            return 'warning'

        return 'safe'

    def process_image(self, image: np.ndarray) -> Optional[dict]:
        """
        Process a single image and return facial metrics.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Dictionary with 'ear', 'mar', 'drowsiness_level' or None if no face detected
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None

        # Get landmarks for the first face
        face_landmarks = results.multi_face_landmarks[0]

        # Convert landmarks to list of (x, y) tuples
        landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]

        # Calculate metrics
        ear = self.calculate_ear(landmarks)
        mar = self.calculate_mar(landmarks)
        drowsiness_level = self.get_drowsiness_level(ear, mar)

        return {
            'ear': ear,
            'mar': mar,
            'drowsiness_level': drowsiness_level
        }

    def process_video_frame(self, frame: np.ndarray) -> Optional[dict]:
        """
        Process a video frame (alias for process_image for clarity).
        """
        return self.process_image(frame)

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
