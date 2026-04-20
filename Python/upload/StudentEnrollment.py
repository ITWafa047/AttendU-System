"""
StudentEnrollment.py - Corrected Version
FIXED: Critical bugs in detect_faces(), brightness validation, type hints
"""

import numpy as np
import cv2
from fastapi import UploadFile, HTTPException
from typing import Optional, Union, Tuple, List, Dict
import insightface
import onnxruntime as ort
import threading
import logging
import os
from pathlib import Path
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageValidator:
    """
    ImageValidator: Comprehensive utility for validating and processing face recognition images.

    Validation Pipeline (fail-fast):
        1. Format validation (MIME type, extension)
        2. Load image (bytes → RGB array)
        3. Size validation (minimum resolution)
        4. Face detection
        5. Single face validation
        6. Face quality checks (size, ratio)
        7. Background validation (white, uniform)
        8. Face alignment (eye-based rotation)
        9. Blur validation (sharpness)
        10. Brightness validation
        11. Embedding generation (with augmentations)
    """

    def __init__(self):
        """Initialize validator with configuration and face detector."""

        # File format validation
        self.ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
        self.ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
        self.MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit

        # Image dimension constraints
        self.MIN_WIDTH = 500
        self.MIN_HEIGHT = 656
        self.CONFIDENCE_THRESHOLD = 0.9

        # Face size constraints
        self.MIN_FACE_WIDTH = 80
        self.MIN_FACE_HEIGHT = 80
        self.MIN_FACE_RATIO = 0.02  # 2% of image area
        self.MIN_CONFIDENCE = 0.9

        # Lighting constraints
        self.MIN_BRIGHTNESS = 80
        self.MAX_BRIGHTNESS = 200
        self.MAX_VARIANCE = 30
        self.BLUR_THRESHOLD = 100

        # Initialize face detector
        logger.info("Initializing insightface FaceAnalysis...")
        self.face_detector = insightface.app.FaceAnalysis()
        self.face_detector.prepare(
            ctx_id=0, det_size=(640, 640)
        )  # GPU if available, else CPU
        logger.info("Face detector initialized successfully")

    async def load_image(self, file: Optional[UploadFile] = None) -> np.ndarray:
        """
        Convert upload file to RGB numpy array.

        Args:
            file: FastAPI UploadFile

        Returns:
            np.ndarray: Image in RGB format (H, W, 3)

        Raises:
            ValueError: If file is invalid or corrupted
            HTTPException: If file is too large
        """
        # Step 1: Validate file exists
        if file is None or file.filename == "":
            raise ValueError("No file uploaded")

        # Step 2: Validate file size
        if file.size and file.size > self.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {self.MAX_FILE_SIZE / 1024 / 1024:.0f}MB)",
            )

        # Step 3: Read file bytes
        raw_bytes = await file.read()
        if not raw_bytes:
            raise ValueError("File is empty")

        # Step 4: Convert bytes to numpy array
        try:
            byte_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        except Exception as e:
            logger.error(f"Failed to convert bytes to array: {e}")
            raise ValueError("Corrupted image file")

        # Step 5: Decode with OpenCV
        image_bgr = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Invalid or corrupted image file")

        # Step 6: Convert BGR → RGB
        try:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Failed to convert BGR to RGB: {e}")
            raise ValueError("Failed to process image format")

        logger.info(f"Image loaded: {image_rgb.shape}")
        return image_rgb

    async def validate_format(self, file: UploadFile) -> bool:
        """Validate file format (extension and MIME type)."""

        if not file.filename or file.filename.strip() == "":
            raise HTTPException(status_code=400, detail="Invalid filename")

        # Extract and validate extension
        if "." not in file.filename:
            raise HTTPException(status_code=400, detail="File has no extension")

        extension = file.filename.rsplit(".", 1)[-1].lower()
        if extension not in self.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}",
            )

        # Validate MIME type
        if file.content_type not in self.ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400, detail=f"Invalid content type: {file.content_type}"
            )

        logger.info(f"Format validation passed: {extension}")
        return True

    def validate_size(self, image: np.ndarray) -> bool:
        """Validate image meets minimum resolution."""

        height, width = image.shape[:2]

        if width < self.MIN_WIDTH:
            raise HTTPException(
                status_code=400,
                detail=f"Image too narrow: {width}px (minimum {self.MIN_WIDTH}px)",
            )

        if height < self.MIN_HEIGHT:
            raise HTTPException(
                status_code=400,
                detail=f"Image too short: {height}px (minimum {self.MIN_HEIGHT}px)",
            )

        logger.info(f"Size validation passed: {width}x{height}")
        return True

    def detect_faces(self, image: np.ndarray) -> Dict[str, Union[int, List[Dict]]]:
        """
        Detect faces using insightface.FaceAnalysis.

        FIXED: Now correctly handles insightface return type (list of Face objects).

        Args:
            image: RGB numpy array

        Returns:
            dict with:
                - 'faces_count': Number of confident detections
                - 'faces': List of face dicts with 'bbox', 'score', 'landmarks'

        Raises:
            HTTPException: On detection failure or no faces found
        """
        try:
            detections = self.face_detector.get(image)
            logger.info(
                f"Face detection returned {len(detections) if detections else 0} result(s)"
            )
        except Exception as e:
            logger.error(f"Face detection failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Face detection failed: {str(e)}"
            )

        # Handle empty detection
        if not detections or len(detections) == 0:
            raise HTTPException(status_code=400, detail="No faces detected in image")

        # Convert insightface Face objects to dicts
        # insightface returns list of Face objects with attributes:
        # - bbox: [x1, y1, x2, y2]
        # - det_score: detection confidence
        # - kps: keypoints array (5 points: 2 eyes, nose, 2 mouth corners)
        faces = []
        for det in detections:
            try:
                # Normalize bbox to [x1, y1, x2, y2] format
                bbox = det.bbox.astype(int).tolist()

                # Extract landmarks from keypoints (kps)
                # kps is shape (5, 2): [[x0, y0], [x1, y1], ..., [x4, y4]]
                kps = det.kps.astype(int)
                landmarks = {
                    "left_eye": tuple(kps[0]),  # Point 0
                    "right_eye": tuple(kps[1]),  # Point 1
                    "nose": tuple(kps[2]),  # Point 2
                    "left_mouth": tuple(kps[3]),  # Point 3
                    "right_mouth": tuple(kps[4]),  # Point 4
                }

                face_dict = {
                    "bbox": bbox,
                    "bbox_format": "xyxy",  # Explicit format marker [x1, y1, x2, y2]
                    "score": float(det.det_score),
                    "landmarks": landmarks,
                }
                faces.append(face_dict)

            except (AttributeError, IndexError, TypeError) as e:
                logger.warning(f"Failed to process detection: {e}")
                continue

        if not faces:
            raise HTTPException(status_code=400, detail="Failed to process detections")

        # Filter by confidence threshold
        confident_faces = [f for f in faces if f["score"] >= self.CONFIDENCE_THRESHOLD]

        if not confident_faces:
            best_score = max(f["score"] for f in faces)
            raise HTTPException(
                status_code=400,
                detail=f"No confident faces found. Best score: {best_score:.2f}",
            )

        logger.info(
            f"Face detection: {len(confident_faces)} confident face(s) out of {len(faces)}"
        )

        return {"faces_count": len(confident_faces), "faces": confident_faces}

    def validate_single_face(self, faces: List[Dict]) -> Dict:
        """
        Validate exactly one face is detected.

        Args:
            faces: List of face dictionaries from detect_faces()

        Returns:
            Single face dictionary

        Raises:
            ValueError: If not exactly one face
        """
        if len(faces) == 0:
            raise ValueError("No faces detected")

        if len(faces) > 1:
            raise ValueError(
                f"Multiple faces detected ({len(faces)}). Expected exactly 1."
            )

        logger.info("Single face validation passed")
        return faces[0]

    def validate_face_quality(self, image: np.ndarray, face: Dict) -> np.ndarray:
        """
        Validate face region quality and extract crop.

        Args:
            image: Full RGB image array
            face: Face dict with 'bbox' and optional 'score'

        Returns:
            Cropped face region (numpy array)

        Raises:
            ValueError: On any quality check failure
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image input")

        if not isinstance(face, dict):
            raise ValueError("Invalid face input")

        image_height, image_width = image.shape[:2]
        image_area = image_width * image_height

        # Extract and validate bounding box
        bbox = face.get("bbox")
        if bbox is None:
            raise ValueError("Bounding box missing from face data")

        try:
            # bbox is [x1, y1, x2, y2] from insightface
            x1, y1, x2, y2 = [int(v) for v in bbox]
            width = x2 - x1
            height = y2 - y1
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid bounding box format: {bbox}")

        # Validate bbox is within image bounds
        if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
            raise ValueError(
                f"Bounding box [{x1}, {y1}, {x2}, {y2}] exceeds image bounds [{image_width}, {image_height}]"
            )

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid bbox dimensions: {width}x{height}")

        # Validate face size
        if width < self.MIN_FACE_WIDTH or height < self.MIN_FACE_HEIGHT:
            raise ValueError(
                f"Face too small: {width}x{height}px (minimum {self.MIN_FACE_WIDTH}x{self.MIN_FACE_HEIGHT}px)"
            )

        # Validate face-to-image area ratio
        face_area = width * height
        ratio = face_area / image_area

        if ratio < self.MIN_FACE_RATIO:
            raise ValueError(
                f"Face occupies only {ratio*100:.2f}% of image (minimum {self.MIN_FACE_RATIO*100:.1f}%)"
            )

        # Crop face region
        face_region = image[y1:y2, x1:x2]

        if face_region.size == 0:
            raise ValueError("Cropped face region is empty")

        # Validate detection confidence
        confidence = face.get("score")
        if confidence is not None:
            if float(confidence) < self.MIN_CONFIDENCE:
                raise ValueError(
                    f"Low confidence face: {confidence:.2f} (minimum {self.MIN_CONFIDENCE})"
                )

        logger.info(
            f"Face quality validation passed: {width}x{height}px, confidence {confidence:.2f}"
        )
        return face_region

    def validate_background(self, image: np.ndarray, face: Dict) -> Tuple[bool, str]:
        """
        Validate background is white and uniform.

        Args:
            image: Full RGB image
            face: Face dict with 'bbox'

        Returns:
            (True, "message") if valid, (False, "reason") if invalid
        """
        if image is None or image.size == 0:
            return False, "Invalid image provided"

        image_height, image_width = image.shape[:2]

        # Extract bbox and create background mask
        bbox = face.get("bbox")
        if bbox is None:
            return False, "Bounding box not found"

        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except (TypeError, ValueError):
            return False, "Invalid bounding box"

        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)

        # Create mask of background pixels (everything outside face bbox)
        mask = np.ones((image_height, image_width), dtype=bool)
        mask[y1:y2, x1:x2] = False

        if not mask.any():
            return False, "No background pixels outside face region"

        # Convert to grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Extract background pixels
        background_pixels = gray[mask].astype(np.float32)

        # FIXED: Validate brightness is in valid range (MIN to MAX)
        mean_intensity = float(np.mean(background_pixels))

        if mean_intensity < self.MIN_BRIGHTNESS:
            return (
                False,
                f"Background too dark: {mean_intensity:.1f} (minimum {self.MIN_BRIGHTNESS})",
            )

        if mean_intensity > self.MAX_BRIGHTNESS:
            return (
                False,
                f"Background too bright: {mean_intensity:.1f} (maximum {self.MAX_BRIGHTNESS})",
            )

        # Check uniformity
        std_dev = float(np.std(background_pixels))

        if std_dev > self.MAX_VARIANCE:
            return (
                False,
                f"Background not uniform: std={std_dev:.1f} (maximum {self.MAX_VARIANCE})",
            )

        logger.info(
            f"Background validation passed: brightness={mean_intensity:.1f}, uniformity={std_dev:.1f}"
        )
        return True, "Background is valid"

    def validate_blur(self, face_region: np.ndarray) -> bool:
        """
        Validate face is sharp enough for embedding extraction.

        Args:
            face_region: Cropped face image (RGB format)

        Returns:
            True if sharp enough

        Raises:
            ValueError: If too blurry
        """
        if face_region is None or face_region.size == 0:
            raise ValueError("Invalid face region: empty image")

        # Convert to grayscale
        if len(face_region.shape) == 3:
            gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_region.copy()

        # Compute Laplacian variance (sharpness metric)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        if variance < self.BLUR_THRESHOLD:
            raise ValueError(
                f"Image too blurry (variance={variance:.2f}, threshold={self.BLUR_THRESHOLD}). "
                f"Please use a sharper image."
            )

        logger.info(f"Blur validation passed: variance={variance:.2f}")
        return True

    def validate_brightness(self, face_region: np.ndarray) -> bool:
        """
        Validate face has proper lighting.

        Args:
            face_region: Cropped face image (RGB format)

        Returns:
            True if brightness valid

        Raises:
            ValueError: If too dark or too bright
        """
        if face_region is None or face_region.size == 0:
            raise ValueError("Invalid face region: empty image")

        # Convert to grayscale
        if len(face_region.shape) == 3:
            gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_region.copy()

        # Measure brightness
        mean_intensity = float(gray.mean())

        if mean_intensity < self.MIN_BRIGHTNESS:
            raise ValueError(
                f"Face too dark: {mean_intensity:.1f} (minimum {self.MIN_BRIGHTNESS})"
            )

        if mean_intensity > self.MAX_BRIGHTNESS:
            raise ValueError(
                f"Face too bright: {mean_intensity:.1f} (maximum {self.MAX_BRIGHTNESS})"
            )

        logger.info(f"Brightness validation passed: {mean_intensity:.1f}")
        return True

    def align_face(self, image: np.ndarray, face: Dict) -> np.ndarray:
        """
        Align face using eye-based rotation and normalize to 112x112.

        Args:
            image: Full RGB image
            face: Face dict with 'bbox' and 'landmarks'

        Returns:
            Aligned face image (112, 112, 3) in RGB format

        Raises:
            ValueError: On invalid landmarks or coordinates
        """
        # Extract and validate landmarks
        landmarks = face.get("landmarks")
        if not landmarks:
            raise ValueError("Face landmarks not found")

        left_eye = landmarks.get("left_eye")
        right_eye = landmarks.get("right_eye")

        if left_eye is None or right_eye is None:
            raise ValueError("Left or right eye landmark missing")

        try:
            left_eye_center = (float(left_eye[0]), float(left_eye[1]))
            right_eye_center = (float(right_eye[0]), float(right_eye[1]))
        except (TypeError, IndexError, ValueError):
            raise ValueError("Invalid eye landmark format")

        # FIXED: Validate landmarks are within image bounds
        image_height, image_width = image.shape[:2]

        for point, name in [
            (left_eye_center, "left_eye"),
            (right_eye_center, "right_eye"),
        ]:
            if not (0 <= point[0] < image_width and 0 <= point[1] < image_height):
                raise ValueError(
                    f"Invalid {name} landmark: {point} outside image [{image_width}x{image_height}]"
                )

        # Compute rotation angle from eye positions
        delta_x = right_eye_center[0] - left_eye_center[0]
        delta_y = right_eye_center[1] - left_eye_center[1]
        angle = float(np.degrees(np.arctan2(delta_y, delta_x)))

        # Rotation center = midpoint between eyes
        center = (
            (left_eye_center[0] + right_eye_center[0]) / 2.0,
            (left_eye_center[1] + right_eye_center[1]) / 2.0,
        )

        # Build rotation matrix and rotate image
        h, w = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        aligned_image = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Extract and transform bbox
        bbox = face.get("bbox")
        if bbox is None:
            raise ValueError("Bounding box not found")

        try:
            # bbox is [x1, y1, x2, y2] from insightface
            x1_orig, y1_orig, x2_orig, y2_orig = [int(v) for v in bbox]
        except (TypeError, ValueError):
            raise ValueError(f"Invalid bounding box: {bbox}")

        # Transform bbox corners through rotation matrix
        corners = np.array(
            [
                [x1_orig, y1_orig, 1],
                [x2_orig, y1_orig, 1],
                [x1_orig, y2_orig, 1],
                [x2_orig, y2_orig, 1],
            ]
        ).T  # Shape: (3, 4)

        rotated_corners = rotation_matrix @ corners

        # Get new bbox from rotated corners
        x1 = int(np.floor(np.min(rotated_corners[0, :])))
        y1 = int(np.floor(np.min(rotated_corners[1, :])))
        x2 = int(np.ceil(np.max(rotated_corners[0, :])))
        y2 = int(np.ceil(np.max(rotated_corners[1, :])))

        # Clamp to image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid transformed bbox: [{x1}, {y1}, {x2}, {y2}]")

        # Extract aligned face region
        aligned_face = aligned_image[y1:y2, x1:x2]

        # Resize to ArcFace standard (112x112)
        aligned_face = cv2.resize(
            aligned_face, (112, 112), interpolation=cv2.INTER_LINEAR
        )

        logger.info(f"Face alignment completed: angle={angle:.1f}°, size=112x112")
        return aligned_face

    async def final_process(self, file: UploadFile) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run full validation pipeline.

        Steps (fail-fast):
            1. Format validation
            2. Load image
            3. Size validation
            4. Face detection
            5. Single face validation
            6. Face quality check
            7. Background validation
            8. Face alignment
            9. Blur validation
            10. Brightness validation
            11. Embedding generation

        Args:
            file: Uploaded image file

        Returns:
            - mean_embedding: (512,) normalized mean embedding
            - embeddings_augmentations: (6, 512) augmented embeddings

        Raises:
            ValueError: On any validation failure
            HTTPException: On file/format issues
        """
        try:
            logger.info(f"Starting validation pipeline for {file.filename}")

            # Step 1: Format
            await self.validate_format(file)

            # Step 2: Load
            image = await self.load_image(file)

            # Step 3: Size
            self.validate_size(image)

            # Step 4 & 5: Detect + Single face
            detections = self.detect_faces(image)
            face = self.validate_single_face(detections["faces"])

            # Step 6: Quality
            face_region = self.validate_face_quality(image, face)

            # Step 7: Background
            is_valid, bg_msg = self.validate_background(image, face)
            if not is_valid:
                raise ValueError(f"Background validation failed: {bg_msg}")

            # Step 8: Alignment
            face_region = self.align_face(image, face)

            # Step 9: Blur
            self.validate_blur(face_region)

            # Step 10: Brightness
            self.validate_brightness(face_region)

            # Step 11: Embedding
            mean_embedding, embeddings_augmentations = FaceProcessor.generate_embedding(
                face_region
            )

            logger.info("Validation pipeline completed successfully")
            return mean_embedding, embeddings_augmentations

        except Exception as e:
            logger.error(f"Validation pipeline failed: {e}", exc_info=True)
            raise
        finally:
            # Cleanup
            gc.collect()


class FaceProcessor:
    """
    Process faces and extract embeddings using ArcFace model.

    Features:
        - Robust embedding extraction with augmentations
        - ONNX Runtime with GPU acceleration
        - Thread-safe model caching
        - Numerical stability
    """

    _session = None
    _current_model_path = None
    _session_lock = threading.Lock()
    providers = ["CUDAExecutionProvider", "GPUExecutionProvider"]

    @staticmethod
    def preprocess(aligned_face: np.ndarray) -> np.ndarray:
        """Preprocess face for ArcFace model."""

        if aligned_face.shape != (112, 112, 3):
            raise ValueError(f"Expected (112, 112, 3), got {aligned_face.shape}")

        if aligned_face.dtype != np.uint8:
            raise ValueError(f"Expected uint8, got {aligned_face.dtype}")

        image = aligned_face.astype(np.float32)
        image = image / 127.5 - 1.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        return image

    @staticmethod
    def extract_embedding(
        preprocessed_image: np.ndarray, model_path: str = "arcface.onnx"
    ) -> np.ndarray:
        """
        Extract embedding using ArcFace ONNX model.

        FIXED: Validates model path exists and uses numerical stability.
        """

        # FIXED: Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ArcFace model not found: {model_path}\n"
                f"Download from: https://github.com/deepinsight/insightface/releases\n"
                f"Place in project root or specify correct path."
            )

        # Thread-safe model loading and inference
        with FaceProcessor._session_lock:
            if (
                FaceProcessor._session is None
                or FaceProcessor._current_model_path != model_path
            ):
                logger.info(f"Loading ArcFace model: {model_path}")
                FaceProcessor._session = ort.InferenceSession(
                    model_path, providers=FaceProcessor.providers
                )
                FaceProcessor._current_model_path = model_path
                logger.info(
                    f"Model loaded with providers: {FaceProcessor._session.get_providers()}"
                )

            input_name = FaceProcessor._session.get_inputs()[0].name
            output_name = FaceProcessor._session.get_outputs()[0].name

            output = FaceProcessor._session.run(
                [output_name], {input_name: preprocessed_image}
            )

        embedding = output[0][0]

        # FIXED: Numerical stability in normalization
        norm = np.linalg.norm(embedding)
        embedding = embedding / (norm + 1e-8)

        return embedding

    @staticmethod
    def generate_embedding(
        face: np.ndarray, model_path: str = "arcface.onnx"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate robust embedding with augmentations.

        Augmentations:
            1. Original
            2. Horizontally flipped
            3. Brightness +40
            4. Brightness -40
            5. Rotated +10°
            6. Rotated -10°

        Args:
            face: Aligned face (112, 112, 3) uint8 RGB
            model_path: Path to ArcFace ONNX model

        Returns:
            - mean_embedding: (512,) normalized average
            - embeddings_stack: (6, 512) all embeddings
        """

        # Validate input
        if face.shape != (112, 112, 3):
            raise ValueError(f"Expected shape (112, 112, 3), got {face.shape}")

        if face.dtype != np.uint8:
            raise ValueError(f"Expected uint8, got {face.dtype}")

        logger.info("Generating embeddings with augmentations...")

        # Create augmentations
        augmentations = []

        # 1. Original
        augmentations.append(face)

        # 2. Flip
        augmentations.append(cv2.flip(face, 1))

        # 3. Brightness +40
        bright_plus = np.clip(face.astype(np.int16) + 40, 0, 255).astype(np.uint8)
        augmentations.append(bright_plus)

        # 4. Brightness -40
        bright_minus = np.clip(face.astype(np.int16) - 40, 0, 255).astype(np.uint8)
        augmentations.append(bright_minus)

        # 5. Rotate +10°
        center = (56, 56)
        rot_matrix_pos = cv2.getRotationMatrix2D(center, 10, 1.0)
        rotated_pos = cv2.warpAffine(
            face,
            rot_matrix_pos,
            (112, 112),
            borderMode=cv2.BORDER_REFLECT_101,
            flags=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        augmentations.append(rotated_pos)

        # 6. Rotate -10°
        rot_matrix_neg = cv2.getRotationMatrix2D(center, -10, 1.0)
        rotated_neg = cv2.warpAffine(
            face,
            rot_matrix_neg,
            (112, 112),
            borderMode=cv2.BORDER_REFLECT_101,
            flags=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        augmentations.append(rotated_neg)

        # Extract embeddings
        embeddings = []
        for i, aug_image in enumerate(augmentations):
            preprocessed = FaceProcessor.preprocess(aug_image)
            embedding = FaceProcessor.extract_embedding(preprocessed, model_path)
            embeddings.append(embedding)
            logger.info(f"Extracted embedding {i+1}/6")

        # Stack and aggregate
        embeddings_stack = np.stack(embeddings, axis=0)  # (6, 512)
        mean_embedding = np.mean(embeddings_stack, axis=0)  # (512,)

        # Normalize
        norm = np.linalg.norm(mean_embedding)
        normalized_embedding = mean_embedding / (norm + 1e-8)

        logger.info("Embedding generation completed")
        return normalized_embedding, embeddings_stack
