import numpy as np
import cv2
from fastapi import UploadFile, HTTPException
from typing import Optional, Union, Tuple
from retinaface import RetinaFace
import onnxruntime as ort


class ImageValidator:
    """
    ImageValidator is a comprehensive utility class designed to validate and process uploaded images for face recognition applications. It performs a series of checks to ensure that the input image meets the necessary criteria for accurate face embedding extraction using ArcFace.
    """

    def __init__(self):

        # file format
        self.ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
        self.ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}

        # Image Dimensions
        self.MIN_WIDTH = 500
        self.MIN_HEIGHT = 656

        self.CONFIDENCE_THRESHOLD = 0.9

        self.MIN_FACE_WIDTH = 80
        self.MIN_FACE_HEIGHT = 80
        self.MIN_FACE_RATIO = 0.02  # 2% of image area
        self.MIN_CONFIDENCE = 0.9

        self.MIN_BRIGHTNESS = 80
        self.MAX_BRIGHTNESS = 200
        self.MAX_VARIANCE = 30
        self.BLUR_THRESHOLD = 100

    async def load_image(self, file: Optional[UploadFile] = None) -> np.ndarray:
        """
        Convert an upload file into a Numpy array (RGB image).

        Args:
            file: The uploaded file from FastAPI

        Returns:
            np.ndarray: The decoded image in RGB format

        Raises:
            ValueError: If no file is uploaded, image is invalid, or image is corrupted
        """

        # Step 1: Validate file existence
        if file is None or file.filename == "":
            raise ValueError("No file uploaded")

        # Step 2: Read file content as bytes
        raw_bytes = await file.read()

        if not raw_bytes:
            raise ValueError("No file uploaded")

        # Step 3: Convert bytes to NumPy array
        try:
            byte_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        except Exception:
            raise ValueError("Corrupted image")

        # Step 4: Decode image using OpenCV
        image_bgr = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)

        # Step 5: Validate decoded image
        if image_bgr is None:
            raise ValueError("Invalid image file")

        # Step 6: Convert BGR -> RGB (required for ArcFace compatibility)
        try:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            raise ValueError("Corrupted image")

        # Step 7: Return processed image
        return image_rgb

    async def validate_format(self, file: UploadFile) -> bool:
        # Step 1: Validate filename exists
        if not file.filename or file.filename.strip() == "":
            raise HTTPException(status_code=400, detail="Invalid file name")

        # Step 2 & 3: Extract and normalize extension
        filename = file.filename
        if "." not in filename:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        extension = filename.rsplit(".", 1)[-1].lower()  # image.png -> png

        # Step 4 & 5: Validate extension against allowed formats
        if extension not in self.ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Step 6: Validate MIME type (optional but recommended)
        if file.content_type not in self.ALLOWED_MIME_TYPES:
            raise HTTPException(status_code=400, detail="Invalid content type")

        # Step 7: All checks passed
        return True

    def validate_size(self, image: np.ndarray) -> bool:
        # Step 1: Extract dimensions from image shape (width, height, channels)
        height, width = image.shape[:2]

        # Step 2: Validate Width
        if width < self.MIN_WIDTH:
            raise HTTPException(
                status_code=400,
                detail=f"Image width is too small, Minimum required: {self.MIN_WIDTH}px, got: {width}px",
            )

        # Step 3: Validate Height
        if height < self.MIN_HEIGHT:
            raise HTTPException(
                status_code=400,
                detail=f"Image height is too small. Minimum required: {self.MIN_HEIGHT}px, got: {height}px",
            )

        # Step 4: All checks passed
        return True

    def detect_faces(self, image: np.ndarray) -> dict:
        # Step 1: Run face detection model
        try:
            detections = RetinaFace.detect_faces(image)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Face detection failed: {str(e)}"
            )

        # Step 2: Extract faces list — RetinaFace returns a dict or 0 if no faces
        if not isinstance(detections, dict):
            raise HTTPException(status_code=400, detail="No face detected")

        # Step 3: Build normalized faces list from RetinaFace output
        faces = [
            {
                "bbox": data["facial_area"],
                "score": data["score"],
                "landmarks": data.get("landmarks"),
            }
            for data in detections.values()
        ]

        # Step 4: Count detected faces
        faces_count = len(faces)

        # Step 5: Validate at least one face exists
        if faces_count == 0:
            raise HTTPException(status_code=400, detail="No face detected")

        # Step 6: Filter by confidence threshold
        confident_faces = [f for f in faces if f["score"] >= self.CONFIDENCE_THRESHOLD]

        if len(confident_faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected")

        # Step 7: Return results
        return {"faces_count": len(confident_faces), "faces": confident_faces}

    def validate_single_face(self, faces: list) -> dict:
        """
        Validates that exactly one face is detected in the image.

        Args:
            faces: List of detected faces from detect_faces()

        Returns:
            dict: Single face data (bounding box + confidence score)

        Raises:
            ValueError: If no face or multiple faces are detected
        """

        face_count = len(faces)

        if face_count == 0:
            raise ValueError("No face detected")

        if face_count > 1:
            raise ValueError("Multiple faces detected")

        return faces[0]

    def validate_face_quality(self, image: np.ndarray, face: dict) -> np.ndarray:
        """
        Validate the quality of a detected face before embedding extraction.

        Args:
            image: Input image as numpy array (H, W, C)
            face: Dictionary containing face data with keys:
                - 'bbox': [x, y, width, height]
                - 'confidence': (optional) detection confidence score

        Returns:
            face_region: Cropped face numpy array if all checks pass

        Raises:
            ValueError: With descriptive message if any quality check fails
        """

        # Step 1: Validate inputs
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid image input")

        if face is None or not isinstance(face, dict):
            raise ValueError("Invalid face input")

        image_height, image_width = image.shape[:2]
        image_area = image_width * image_height

        # Step 2: Extract bounding box
        bbox = face.get("bbox")
        if bbox is None:
            raise ValueError("Invalid face region: missing bounding box")

        try:
            x, y, width, height = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        except (TypeError, IndexError, ValueError):
            raise ValueError("Invalid face region: malformed bounding box")

        # Step 3: Validate bounding box boundaries
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError(
                "Invalid face region: negative or zero bounding box values"
            )

        if (x + width) > image_width or (y + height) > image_height:
            raise ValueError(
                "Invalid face region: bounding box exceeds image boundaries"
            )

        # Step 4: Validate face size
        if width < self.MIN_FACE_WIDTH or height < self.MIN_FACE_HEIGHT:
            raise ValueError(
                f"Face is too small: detected {width}px , {height}px, "
                f"minimum required is {self.MIN_FACE_WIDTH}x{self.MIN_FACE_HEIGHT}px"
            )

        # Step 5: Validate face-to-image area ratio
        face_area = width * height
        ratio = face_area / image_area

        if ratio < self.MIN_FACE_RATIO:
            raise ValueError(
                f"Face is too small relative to image: "
                f"ratio={ratio:.4f} ({ratio*100:.2f}%), "
                f"minimum required is {self.MIN_FACE_RATIO*100:.1f}%"
            )

        # Step 6: Crop face region
        face_region = image[y : y + height, x : x + width]

        if face_region.size == 0:
            raise ValueError("Invalid face region: cropped region is empty")

        # Step 7: (Optional) Validate confidence score
        confidence = face.get("score")
        if confidence is not None:
            if not isinstance(confidence, (int, float)):
                raise ValueError("Invalid confidence score type")
            if float(confidence) < self.MIN_CONFIDENCE:
                raise ValueError(
                    f"Low confidence face detection: "
                    f"score={confidence:.4f}, "
                    f"minimum required is {self.MIN_CONFIDENCE}"
                )

        # Step 8: Return valid face region
        return face_region

    def validate_background(self, image: np.ndarray, face: dict) -> Tuple[bool, str]:
        """
        Validate that the image has a white (or near-white), uniform background.

        Args:
            image:          RGB numpy array (converted from BGR in load_image).
            face:           Face dict containing 'bbox': (x, y, w, h).
        Returns:
            (True,  "Background is valid")              — all checks passed.
            (False, "<reason>")                         — first failing check.
        """

        # Step 1: Validate inputs
        if image is None or image.size == 0:
            return False, "Invalid image provided"

        if len(image.shape) < 2:
            return False, "Image must be at least 2-dimensional"

        # Step 2: Extract image dimensions
        image_height, image_width = image.shape[:2]

        # Step 3: Extract bounding box from face dict and create background mask
        bbox = face.get("bbox")
        if bbox is None:
            return False, "Face bounding box not found"

        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_width, x + w)
        y2 = min(image_height, y + h)

        mask = np.ones((image_height, image_width), dtype=bool)

        mask[y1:y2, x1:x2] = False

        if not mask.any():
            return False, "No background pixels found outside the face region"

        # Step 4: Convert to grayscale for brightness analysis
        # Note: image is in RGB format (converted at line 77)
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Step 5: Extract background pixels
        background_pixels = gray[mask].astype(np.float32)

        # Step 6: Measure brightness
        mean_intensity = float(np.mean(background_pixels))

        # Step 7: Validate brightness threshold
        if mean_intensity < self.MAX_BRIGHTNESS:
            return (
                False,
                f"Background is not white enough "
                f"(mean brightness: {mean_intensity:.1f}, required: >= {self.MAX_BRIGHTNESS})",
            )

        # Step 8: (Optional) Check colour variance — uniform background
        std_dev = float(np.std(background_pixels))

        if std_dev > self.MAX_VARIANCE:
            return (
                False,
                f"Background is not uniform "
                f"(std deviation: {std_dev:.1f}, allowed: <= {self.MAX_VARIANCE})",
            )

        # Step 9: All checks passed
        return True, "Background is valid"

    def validate_blur(self, face_region: np.ndarray) -> bool:
        """
        Validate that a face image is sharp enough for embedding extraction.

        Args:
            face_region: numpy array of the cropped face region (RGB format)

        Returns:
            True if image is sharp enough

        Raises:
            ValueError: if the image is too blurry or invalid
        """

        # Step 1: Input Validation
        if face_region is None or face_region.size == 0:
            raise ValueError("Invalid face region: empty or None image provided")

        if len(face_region.shape) < 2:
            raise ValueError("Invalid face region: image must be at least 2D")

        # Step 2: Convert to grayscale
        if len(face_region.shape) == 3:
            gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_region.copy()

        # Step 3: Apply Laplacian operator to measure sharpness via edges
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Step 4: Compute variance of Laplacian (key sharpness metric)
        variance = laplacian.var()

        # Step 5: Validate sharpness against threshold
        if variance < self.BLUR_THRESHOLD:
            raise ValueError(
                f"Image is too blurry (variance={variance:.2f}, threshold={self.BLUR_THRESHOLD}). "
                f"Please provide a sharper image for accurate face recognition."
            )

        return True

    def validate_brightness(self, face_region: np.ndarray) -> bool:
        """
        Validates that a face image has proper lighting.

        Args:
            face_region: numpy array — cropped face image (RGB format)

        Returns:
            True if brightness is valid

        Raises:
            ValueError: if the image is too dark, too bright, or invalid
        """

        # Step 1: Input validation
        if face_region is None or face_region.size == 0:
            raise ValueError("Invalid face region: empty or None image provided")

        # Step 2: Convert to Grayscale
        if len(face_region.shape) == 3:
            gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_region.copy()

        # Step 3: Compute Average Brightness
        mean_intensity = float(gray.mean())

        # Step 4: Validate Brightness
        if mean_intensity < self.MIN_BRIGHTNESS:
            raise ValueError(
                f"Image is too dark (mean intensity: {mean_intensity:.1f}, "
                f"minimum required: {self.MIN_BRIGHTNESS})"
            )
        elif mean_intensity > self.MAX_BRIGHTNESS:
            raise ValueError(
                f"Image is too bright (mean intensity: {mean_intensity:.1f}, "
                f"maximum allowed: {self.MAX_BRIGHTNESS})"
            )

        # Step 5: All checks passed
        return True

    def align_face(self, image: np.ndarray, face: dict) -> np.ndarray:
        """
        Align a face for ArcFace model input.

        Performs:
            - Eye-based rotation to horizontally align the eyes
            - Transforms bounding box coordinates through rotation matrix
            - Cropping using the rotated bounding box
            - Resizing to 112x112 (ArcFace standard)

        Args:
            image (np.ndarray): Full input image as a NumPy array (RGB format).
            face (dict): Face detection result containing:
                - 'bbox'       : [x1, y1, x2, y2]  (or [x, y, w, h] — handled below)
                - 'landmarks'  : dict with keys:
                                    'left_eye'  : (x, y)
                                    'right_eye' : (x, y)
                                    'nose'      : (x, y)  [optional]

        Returns:
            np.ndarray: Aligned and resized face image of shape (112, 112, 3).

        Raises:
            ValueError: If landmarks are missing or coordinates are invalid.
        """

        # ------------------------------------------------------------------ #
        # 1. Validate & extract landmarks
        # ------------------------------------------------------------------ #
        landmarks = face.get("landmarks")
        if not landmarks:
            raise ValueError(
                "Face landmarks not found. "
                "Use a detector that returns landmarks (e.g. RetinaFace, MTCNN)."
            )

        left_eye = landmarks.get("left_eye")
        right_eye = landmarks.get("right_eye")

        if left_eye is None or right_eye is None:
            raise ValueError(
                "Invalid face landmarks: 'left_eye' and 'right_eye' are required."
            )

        try:
            left_eye_center = (float(left_eye[0]), float(left_eye[1]))
            right_eye_center = (float(right_eye[0]), float(right_eye[1]))
        except (TypeError, IndexError):
            raise ValueError(
                "Invalid face landmarks: eye coordinates must be (x, y) tuples."
            )

        # ------------------------------------------------------------------ #
        # 2. Compute the rotation angle from eye positions
        # ------------------------------------------------------------------ #
        delta_x = right_eye_center[0] - left_eye_center[0]
        delta_y = right_eye_center[1] - left_eye_center[1]

        # angle in degrees — positive = counter-clockwise correction needed
        angle = float(np.degrees(np.arctan2(delta_y, delta_x)))

        # ------------------------------------------------------------------ #
        # 3. Compute rotation center = midpoint between the two eyes
        # ------------------------------------------------------------------ #
        center = (
            (left_eye_center[0] + right_eye_center[0]) / 2.0,
            (left_eye_center[1] + right_eye_center[1]) / 2.0,
        )

        # ------------------------------------------------------------------ #
        # 4. Build rotation matrix & rotate the full image
        # ------------------------------------------------------------------ #
        h, w = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        aligned_image = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # ------------------------------------------------------------------ #
        # 5. Extract bounding box and transform coordinates through rotation
        # ------------------------------------------------------------------ #
        bbox = face.get("bbox")
        if bbox is None:
            raise ValueError("Face bounding box ('bbox') not found in face dict.")

        try:
            bbox = [float(v) for v in bbox]
        except (TypeError, ValueError):
            raise ValueError("Invalid bounding box: values must be numeric.")

        # Support both [x1,y1,x2,y2] and [x,y,w,h] formats
        if len(bbox) != 4:
            raise ValueError("Invalid bounding box: expected 4 values.")

        # Improved format detection: check if values appear to be width/height or coordinates
        x, y, v3, v4 = bbox

        # If v3/v4 are small relative to x/y, they're likely width/height; otherwise coordinates
        if v3 < 500 and v4 < 656:  # Reasonable size bounds for face width/height
            x1_orig = int(round(x))
            y1_orig = int(round(y))
            x2_orig = x1_orig + int(round(v3))
            y2_orig = y1_orig + int(round(v4))
        else:
            x1_orig, y1_orig, x2_orig, y2_orig = [int(round(v)) for v in bbox]

        # Transform bbox corners through rotation matrix
        # The rotation matrix is 2x3: [[cos, -sin, tx], [sin, cos, ty]]
        # Apply transformation: point' = rotation_matrix @ [x, y, 1]
        corners = np.array(
            [
                [x1_orig, y1_orig, 1],  # top-left
                [x2_orig, y1_orig, 1],  # top-right
                [x1_orig, y2_orig, 1],  # bottom-left
                [x2_orig, y2_orig, 1],  # bottom-right
            ]
        ).T  # (3, 4) for matrix multiplication

        # Transform all corners through rotation matrix: (2, 3) @ (3, 4) -> (2, 4)
        rotated_corners = rotation_matrix @ corners

        # Get new bounding box from transformed corners
        x1 = int(np.floor(np.min(rotated_corners[0, :])))
        y1 = int(np.floor(np.min(rotated_corners[1, :])))
        x2 = int(np.ceil(np.max(rotated_corners[0, :])))
        y2 = int(np.ceil(np.max(rotated_corners[1, :])))

        # Clamp to rotated image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid transformed bounding box: [{x1},{y1},{x2},{y2}]")

        aligned_face = aligned_image[y1:y2, x1:x2]

        # ------------------------------------------------------------------ #
        # 6. Resize to ArcFace standard input size (112 × 112)
        # ------------------------------------------------------------------ #
        aligned_face = cv2.resize(
            aligned_face, (112, 112), interpolation=cv2.INTER_LINEAR
        )

        return aligned_face

    async def final_process(self, file: UploadFile) -> np.ndarray:
        """
        Run the full image validation pipeline and return a face embedding.

        Steps (fail-fast — first failure raises immediately):
            1. validate_format   — allowed MIME type & extension
            2. load_image        — decode bytes → RGB numpy array
            3. validate_size     — minimum resolution check
            4. detect_faces      — face detection with RetinaFace
            5. validate_single_face — exactly one face required
            6. validate_face_quality — size ratio + bbox validation
            7. validate_background — uniform white background check
            8. align_face        — eye-based rotation & 112×112 resize
            9. validate_blur     — Laplacian sharpness check
            10. validate_brightness — mean intensity check
            11. generate_embedding — robust embedding with augmentations

        Parameters
        ----------
        file : UploadFile
            The image file received from the API endpoint.

        Returns
        -------
        np.ndarray
            A (512,) normalized face embedding ready for comparison and storage.

        Raises
        ------
        ValueError
            On any format, size, face count, quality, background,
            blur, or brightness failure.
        HTTPException
            On file format validation failures or face detection errors.
        RuntimeError
            If the face detector itself cannot be initialised.
        """
        # Step 1: format
        await self.validate_format(file)

        # Step 2: load
        image: np.ndarray = await self.load_image(file)

        # Step 3: size
        self.validate_size(image)

        # Step 4 & 5: detect + single-face check
        faces = self.detect_faces(image)
        face = self.validate_single_face(faces["faces"])

        # Step 6: face quality + crop
        face_region: np.ndarray = self.validate_face_quality(image, face)

        # Step 7: background (validate before alignment on original image)
        is_valid, bg_msg = self.validate_background(image, face)
        if not is_valid:
            raise ValueError(f"Background validation failed: {bg_msg}")

        # Step 8: align face for ArcFace input
        face_region = self.align_face(image, face)

        # Step 9: blur (validate on aligned face)
        self.validate_blur(face_region)

        # Step 10: brightness (validate on aligned face)
        self.validate_brightness(face_region)

        faces_embedding = FaceProcessor.generate_embedding(face_region)

        return faces_embedding


class FaceProcessor:
    """
    FaceProcessor is responsible for processing face images, including preprocessing, embedding extraction, normalization, augmentation, and aggregation.
    """

    # Class-level session cache to avoid reloading model on every call
    _session = None
    _current_model_path = None
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    @staticmethod
    def preprocess(aligned_face: np.ndarray) -> np.ndarray:
        """Preprocess aligned face for ArcFace model input."""
        if aligned_face.shape != (112, 112, 3):
            raise ValueError("Input image must be 112x112 with 3 channels")

        image = aligned_face.astype(np.float32)
        image = image / 127.5 - 1
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image

    @staticmethod
    def extract_embedding(
        preprocessed_image: np.ndarray, model_path="arcface.onnx"
    ) -> np.ndarray:
        """Extract face embedding using ArcFace model with GPU acceleration."""
        # Lazy-load and cache the model session to avoid reloading on every call
        if (
            FaceProcessor._session is None
            or FaceProcessor._current_model_path != model_path
        ):
            FaceProcessor._session = ort.InferenceSession(
                model_path, providers=FaceProcessor.providers
            )
            FaceProcessor._current_model_path = model_path

        input_name = FaceProcessor._session.get_inputs()[0].name
        output_name = FaceProcessor._session.get_outputs()[0].name

        output = FaceProcessor._session.run(
            [output_name], {input_name: preprocessed_image}
        )

        embedding = output[0][0]
        norm = np.linalg.norm(embedding)
        embedding = embedding / norm if norm > 0 else embedding

        return embedding

    @staticmethod
    def generate_embedding(face: np.ndarray, model_path="arcface.onnx") -> np.ndarray:
        """
        Generate robust face embedding using augmented samples.

        Creates multiple augmented versions of the input face (original, flipped,
        brightness variations, rotated ±10°), extracts embeddings for each, computes
        the mean embedding, and returns the normalized result.

        Augmentation Strategy:
            - Original: Baseline unmodified face
            - Flipped: Horizontal flip to capture symmetric features
            - Brightness ±40: Additive brightness shifts (preserves contrast)
            - Rotated ±10°: Both positive and negative rotations (robustness to head tilt)

        Args:
            face (np.ndarray): Aligned face image of shape (112, 112, 3) in RGB format (uint8).
            model_path (str): Path to the ArcFace ONNX model. Defaults to "arcface.onnx".

        Returns:
            np.ndarray: (512,) normalized mean embedding across augmentations.

        Raises:
            ValueError: If input shape is invalid or dtype is not uint8.
        """
        # Step 1: Validate input
        if face.shape != (112, 112, 3):
            raise ValueError(f"Expected face shape (112, 112, 3), got {face.shape}")
        if face.dtype != np.uint8:
            raise ValueError(f"Expected uint8 input, got {face.dtype}")

        # Step 2: Create augmentations (all kept as uint8 for consistency)
        augmentations = []

        # 2.1: Original — baseline unmodified face
        augmentations.append(face)

        # 2.2: Horizontally flipped — captures symmetric features
        flipped = cv2.flip(face, 1)
        augmentations.append(flipped)

        # 2.3: Brightness +40 (additive shift preserves contrast better than multiplicative)
        # Note: int16 prevents underflow/overflow during addition
        bright_plus = np.clip(face.astype(np.int16) + 40, 0, 255).astype(np.uint8)
        augmentations.append(bright_plus)

        # 2.4: Brightness -40 (additive shift preserves contrast better than multiplicative)
        # Note: int16 prevents underflow during subtraction
        bright_minus = np.clip(face.astype(np.int16) - 40, 0, 255).astype(np.uint8)
        augmentations.append(bright_minus)

        # 2.5: Rotated +10 degrees (clockwise tilt)
        height, width = 112, 112
        center = (width / 2, height / 2)
        rotation_matrix_pos = cv2.getRotationMatrix2D(center, 10, 1.0)
        rotated_pos = cv2.warpAffine(
            face,
            rotation_matrix_pos,
            (width, height),
            borderMode=cv2.BORDER_REFLECT_101,
            flags=cv2.INTER_LINEAR,
        ).astype(
            np.uint8
        )  # Ensure uint8 dtype
        augmentations.append(rotated_pos)

        # 2.6: Rotated -10 degrees (counter-clockwise tilt) — bidirectional augmentation
        rotation_matrix_neg = cv2.getRotationMatrix2D(center, -10, 1.0)
        rotated_neg = cv2.warpAffine(
            face,
            rotation_matrix_neg,
            (width, height),
            borderMode=cv2.BORDER_REFLECT_101,
            flags=cv2.INTER_LINEAR,
        ).astype(
            np.uint8
        )  # Ensure uint8 dtype
        augmentations.append(rotated_neg)

        # Step 3: Extract embeddings for each augmentation
        embeddings = []
        for aug_image in augmentations:
            # Preprocess the augmented image
            preprocessed = FaceProcessor.preprocess(aug_image)
            # Extract embedding
            embedding = FaceProcessor.extract_embedding(preprocessed, model_path)
            embeddings.append(embedding)

        # Step 4: Stack embeddings into array (6_augmentations, 512)
        embeddings_stack = np.stack(embeddings, axis=0)

        # Step 5: Compute mean embedding across augmentations
        mean_embedding = np.mean(embeddings_stack, axis=0)

        # Step 6: Normalize final embedding (L2 normalization)
        norm = np.linalg.norm(mean_embedding)
        normalized_embedding = mean_embedding / norm if norm > 0 else mean_embedding

        # Step 7: Return normalized mean embedding
        return normalized_embedding
