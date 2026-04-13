import numpy as np
import cv2
from fastapi import UploadFile, HTTPException
from typing import Optional, Union, Tuple
from retinaface import RetinaFace


class ImageValidator:

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
            raise ValueError("No file uplaoded")

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

    def validate_background(
        self, image: np.ndarray, face: Tuple[int, int, int, int]
    ) -> Tuple[bool, str]:
        """
        Validate that the image has a white (or near-white), uniform background.

        Args:
            image:          BGR numpy array (as returned by cv2.imread).
            face:           Bounding box of the detected face as (x, y, w, h).
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

        # Step 3: Create background mask (exclude face region)
        x, y, w, h = face

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_width, x + w)
        y2 = min(image_height, y + h)

        mask = np.ones((image_height, image_width), dtype=bool)

        mask[y1:y2, x1:x2] = False

        if not mask.any():
            return False, "No background pixels found outside the face region"

        # Step 4: Convert to grayscale for brightness analysis
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
            face_region: numpy array of the cropped face region (BGR format)

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
            face_region: numpy array — cropped face image (BGR format)

        Returns:
            True if brightness is valid, or a string error message if not.
        """

        # Step 1: Convert to Grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)

        # Step 2: Compute Average Brightness
        mean_intensity = gray.mean()

        # Step 3: Validate Brightness
        if mean_intensity < self.MIN_BRIGHTNESS:
            return "Image is too dark"
        elif mean_intensity > self.MAX_BRIGHTNESS:
            return "Image is too bright"

        # Step 4: Return Success
        return True

    def process(self, file: UploadFile) -> np.ndarray:
        """
        Run the full image validation pipeline and return a clean face crop.

        Steps (fail-fast — first failure raises immediately):
            1. validate_format   — allowed MIME type & extension
            2. load_image        — decode bytes → BGR numpy array
            3. validate_size     — minimum resolution check
            4. detect_faces      — Haar cascade detection
            5. validate_single_face — exactly one face required
            6. validate_face_quality — size ratio + 112×112 crop
            7. validate_background — uniform background check
            8. validate_blur     — Laplacian sharpness check
            9. validate_brightness — mean grey-level check

        Parameters
        ----------
        file : UploadFile
            The image file received from the API endpoint.

        Returns
        -------
        np.ndarray
            A validated, 112×112 BGR face region ready for ArcFace embedding.

        Raises
        ------
        ValueError
            On any format, size, face count, quality, background,
            blur, or brightness failure.
        RuntimeError
            If the face detector itself cannot be initialised.
        """
        # Step 1: format
        self.validate_format(file)

        # Step 2: load
        image: np.ndarray = self.load_image(file)

        # Step 3: size
        self.validate_size(image)

        # Step 4 & 5: detect + single-face check
        faces = self.detect_faces(image)
        face = self.validate_single_face(faces)

        # Step 6: face quality + crop
        face_region: np.ndarray = self.validate_face_quality(image, face)

        # Step 7: background
        self.validate_background(image, face)

        # Step 8: blur
        self.validate_blur(face_region)

        # Step 9: brightness
        self.validate_brightness(face_region)

        # Step 10: All checks passed
        return face_region
