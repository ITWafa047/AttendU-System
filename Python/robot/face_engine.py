import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceEngine:
    def __init__(self, det_size=(640, 640), gpu_id=0):
        """
        Initialize ArcFace model with GPU support.
        """
        self.app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])

        self.app.prepare(ctx_id=gpu_id, det_size=det_size)

    # -----------------------------
    # Face Detection
    # -----------------------------
    def detect_faces(self, image):
        """
        Detect faces in an image.
        Returns list of detected faces.
        """
        return self.app.get(image)

    # -----------------------------
    # Extract Single Face Embedding
    # -----------------------------
    def extract_embedding(self, image, min_det_score=0.8):
        """
        Extract normalized embedding from image.
        Returns (embedding, bbox) or None if quality fails.
        bbox format: (x1, y1, width, height)
        """

        faces = self.detect_faces(image)

        # Must detect exactly one face
        if len(faces) != 1:
            return None

        face = faces[0]

        if face.det_score < min_det_score:
            return None

        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)

        # Extract bounding box: bbox format is [x1, y1, x2, y2]
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        return embedding, (x1, y1, w, h)

    # -----------------------------
    # Extract From Image Path
    # -----------------------------
    def extract_from_path(self, image_path, min_det_score=0.8):
        img = cv2.imread(image_path)

        if img is None:
            return None

        return self.extract_embedding(img, min_det_score)

    def extract_from_frame(self, frame):

        faces = self.app.get(frame)

        if len(faces) == 0:
            return None

        results = []

        for face in faces:
            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)

            landmarks = None

            if hasattr(face, "landmark_3d_68") and face.landmark_3d_68 is not None:
                landmarks = face.landmark_3d_68
            elif hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
                landmarks = face.landmark_2d_106
            else:
                print("⚠️ No landmarks found for this face")

            results.append({
                "bbox": face.bbox, 
                "embedding": embedding, 
                "landmarks": landmarks,
                "yaw": face.pose[1]  # Add this line
            })

        return results