import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from facial_analysis import FacialLandmarkAnalyzer


class TestFacialLandmarkAnalyzer:
    """Test suite for FacialLandmarkAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a FacialLandmarkAnalyzer instance for testing."""
        return FacialLandmarkAnalyzer()

    def test_calculate_distance(self, analyzer):
        """Test distance calculation between two points."""
        p1 = (0.0, 0.0)
        p2 = (3.0, 4.0)
        distance = analyzer.calculate_distance(p1, p2)
        assert distance == 5.0  # 3-4-5 triangle

    def test_calculate_ear_with_mock_landmarks(self, analyzer):
        """Test EAR calculation with mock eye landmarks."""
        # Create mock landmarks for a face with open eyes
        # This is a simplified test - in practice, we'd use real landmark data
        mock_landmarks = [(0.0, 0.0)] * 468  # MediaPipe has 468 landmarks

        # Set up eye landmarks for "open eyes" with realistic proportions
        # Left eye indices: [362, 385, 387, 263, 373, 380]
        mock_landmarks[362] = (0.3, 0.3)  # left corner
        mock_landmarks[385] = (0.32, 0.28)  # top left
        mock_landmarks[387] = (0.32, 0.32)  # bottom left
        mock_landmarks[263] = (0.35, 0.3)  # right corner
        mock_landmarks[373] = (0.335, 0.28)  # top right
        mock_landmarks[380] = (0.335, 0.32)  # bottom right

        # Right eye indices: [33, 160, 158, 133, 153, 144]
        mock_landmarks[33] = (0.6, 0.3)  # left corner
        mock_landmarks[160] = (0.62, 0.28)  # top left
        mock_landmarks[158] = (0.62, 0.32)  # bottom left
        mock_landmarks[133] = (0.65, 0.3)  # right corner
        mock_landmarks[153] = (0.635, 0.28)  # top right
        mock_landmarks[144] = (0.635, 0.32)  # bottom right

        ear = analyzer.calculate_ear(mock_landmarks)
        assert isinstance(ear, float)
        assert 0.0 <= ear <= 1.0  # EAR should be between 0 and 1

    def test_calculate_mar_with_mock_landmarks(self, analyzer):
        """Test MAR calculation with mock mouth landmarks."""
        mock_landmarks = [(0.0, 0.0)] * 468

        # Mouth indices: [61, 291, 39, 181, 0, 17, 269, 405]
        mock_landmarks[61] = (0.4, 0.6)   # left corner
        mock_landmarks[291] = (0.45, 0.55)  # top left
        mock_landmarks[39] = (0.45, 0.65)   # bottom left
        mock_landmarks[181] = (0.5, 0.6)   # center top
        mock_landmarks[0] = (0.5, 0.7)    # center bottom
        mock_landmarks[17] = (0.55, 0.6)   # center right top
        mock_landmarks[269] = (0.55, 0.65)  # bottom right
        mock_landmarks[405] = (0.6, 0.6)   # right corner

        mar = analyzer.calculate_mar(mock_landmarks)
        assert isinstance(mar, float)
        assert mar >= 0.0

    def test_get_drowsiness_level_safe(self, analyzer):
        """Test drowsiness level classification for safe state."""
        level = analyzer.get_drowsiness_level(0.25, 0.5)
        assert level == 'safe'

    def test_get_drowsiness_level_warning(self, analyzer):
        """Test drowsiness level classification for warning state."""
        level = analyzer.get_drowsiness_level(0.18, 0.5)
        assert level == 'warning'

    def test_get_drowsiness_level_danger_ear(self, analyzer):
        """Test drowsiness level classification for danger state (low EAR)."""
        level = analyzer.get_drowsiness_level(0.12, 0.5)
        assert level == 'danger'

    def test_get_drowsiness_level_danger_mar(self, analyzer):
        """Test drowsiness level classification for danger state (high MAR)."""
        level = analyzer.get_drowsiness_level(0.25, 0.7)
        assert level == 'danger'

    @patch('facial_analysis.mp.solutions.face_mesh.FaceMesh')
    def test_process_image_no_face(self, mock_face_mesh_class, analyzer):
        """Test processing an image with no face detected."""
        # Mock the FaceMesh instance
        mock_face_mesh_instance = MagicMock()
        mock_face_mesh_instance.process.return_value = MagicMock(multi_face_landmarks=None)
        mock_face_mesh_class.return_value = mock_face_mesh_instance

        # Create a blank image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = analyzer.process_image(image)
        assert result is None

    def test_process_image_with_face(self):
        """Test processing an image with a face (using mocked face detection)."""
        with patch('facial_analysis.mp.solutions.face_mesh.FaceMesh') as mock_face_mesh_class:
            analyzer = FacialLandmarkAnalyzer()

            # Create mock face landmarks using MagicMock
            mock_face_landmarks = MagicMock()
            mock_face_landmarks.landmark = []

            # Create 468 landmarks with default values
            for i in range(468):
                mock_lm = MagicMock()
                mock_lm.x = 0.5
                mock_lm.y = 0.5
                mock_face_landmarks.landmark.append(mock_lm)

            # Override specific landmarks for eyes and mouth
            # Left eye indices: [362, 385, 387, 263, 373, 380]
            mock_face_landmarks.landmark[362].x = 0.3
            mock_face_landmarks.landmark[362].y = 0.3
            mock_face_landmarks.landmark[385].x = 0.35
            mock_face_landmarks.landmark[385].y = 0.25
            mock_face_landmarks.landmark[387].x = 0.35
            mock_face_landmarks.landmark[387].y = 0.35
            mock_face_landmarks.landmark[263].x = 0.4
            mock_face_landmarks.landmark[263].y = 0.3
            mock_face_landmarks.landmark[373].x = 0.375
            mock_face_landmarks.landmark[373].y = 0.25
            mock_face_landmarks.landmark[380].x = 0.375
            mock_face_landmarks.landmark[380].y = 0.35

            # Right eye indices: [33, 160, 158, 133, 153, 144]
            mock_face_landmarks.landmark[33].x = 0.6
            mock_face_landmarks.landmark[33].y = 0.3
            mock_face_landmarks.landmark[160].x = 0.65
            mock_face_landmarks.landmark[160].y = 0.25
            mock_face_landmarks.landmark[158].x = 0.65
            mock_face_landmarks.landmark[158].y = 0.35
            mock_face_landmarks.landmark[133].x = 0.7
            mock_face_landmarks.landmark[133].y = 0.3
            mock_face_landmarks.landmark[153].x = 0.675
            mock_face_landmarks.landmark[153].y = 0.25
            mock_face_landmarks.landmark[144].x = 0.675
            mock_face_landmarks.landmark[144].y = 0.35

            # Mouth indices: [61, 291, 39, 181, 0, 17, 269, 405]
            mock_face_landmarks.landmark[61].x = 0.4
            mock_face_landmarks.landmark[61].y = 0.6
            mock_face_landmarks.landmark[291].x = 0.45
            mock_face_landmarks.landmark[291].y = 0.55
            mock_face_landmarks.landmark[39].x = 0.45
            mock_face_landmarks.landmark[39].y = 0.65
            mock_face_landmarks.landmark[181].x = 0.5
            mock_face_landmarks.landmark[181].y = 0.6
            mock_face_landmarks.landmark[0].x = 0.5
            mock_face_landmarks.landmark[0].y = 0.7
            mock_face_landmarks.landmark[17].x = 0.55
            mock_face_landmarks.landmark[17].y = 0.6
            mock_face_landmarks.landmark[269].x = 0.55
            mock_face_landmarks.landmark[269].y = 0.65
            mock_face_landmarks.landmark[405].x = 0.6
            mock_face_landmarks.landmark[405].y = 0.6

            mock_result = MagicMock()
            mock_result.multi_face_landmarks = [mock_face_landmarks]
            mock_face_mesh_instance = MagicMock()
            mock_face_mesh_instance.process.return_value = mock_result
            mock_face_mesh_class.return_value = mock_face_mesh_instance

            # Create an image
            image = np.full((480, 640, 3), (128, 128, 128), dtype=np.uint8)
            result = analyzer.process_image(image)

            assert result is not None
            assert 'ear' in result
            assert 'mar' in result
            assert 'drowsiness_level' in result
            assert result['drowsiness_level'] in ['safe', 'warning', 'danger']

    def test_thresholds(self, analyzer):
        """Test that drowsiness thresholds work as expected."""
        # Test EAR thresholds
        assert analyzer.get_drowsiness_level(0.16, 0.5) == 'warning'  # Between 0.15 and 0.20
        assert analyzer.get_drowsiness_level(0.14, 0.5) == 'danger'  # Below 0.15

        # Test MAR threshold
        assert analyzer.get_drowsiness_level(0.25, 0.64) == 'safe'   # Below 0.65
        assert analyzer.get_drowsiness_level(0.25, 0.66) == 'danger'  # Above 0.65

    @pytest.mark.parametrize("ear,mar,expected", [
        (0.25, 0.5, 'safe'),
        (0.18, 0.5, 'warning'),
        (0.12, 0.5, 'danger'),
        (0.25, 0.7, 'danger'),
        (0.14, 0.6, 'danger'),  # Both conditions
    ])
    def test_drowsiness_levels_parametrized(self, analyzer, ear, mar, expected):
        """Parametrized test for drowsiness level classification."""
        level = analyzer.get_drowsiness_level(ear, mar)
        assert level == expected


class TestIntegration:
    """Integration tests for the facial analysis system."""

    @pytest.fixture
    def analyzer(self):
        return FacialLandmarkAnalyzer()

    @patch('facial_analysis.mp.solutions.face_mesh.FaceMesh')
    def test_full_pipeline(self, mock_face_mesh_class, analyzer):
        """Test the complete analysis pipeline with mocked face detection."""
        # Mock the FaceMesh instance
        mock_face_mesh_instance = MagicMock()
        mock_face_mesh_instance.process.return_value = MagicMock(multi_face_landmarks=None)
        mock_face_mesh_class.return_value = mock_face_mesh_instance

        # Test with a blank image (no face)
        result = analyzer.process_image(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result is None  # No face detected


if __name__ == "__main__":
    pytest.main([__file__])
