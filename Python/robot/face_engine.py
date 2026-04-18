import numpy as np
from insightface.app import FaceAnalysis


class FaceEngine:

    def __init__(self, det_size=(640, 640), gpu_id=0):
        """
        Initialize ArcFace model with GPU support.
        """
        self.app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])

        self.app.prepare(ctx_id=gpu_id, det_size=det_size)


    def extract_embedding(self, frame):
        """
        Extract embedding from the first face detected in the frame.
        Returns a dictionary with embedding, bbox, and confidence score.
        Returns None if no faces are detected or an error occurs.
        """
        try:
            faces = self.app.get(frame)

            if len(faces) == 0:
                print("⚠️ No faces detected in frame")
                return None

            # Get the first (main) face
            face = faces[0]
            embedding = face.embedding
            
            # Validate and normalize embedding
            norm = np.linalg.norm(embedding)
            if norm == 0:
                print("⚠️ Zero norm detected, cannot normalize embedding")
                return None
            
            embedding = embedding / norm

            return {
                "embedding": embedding,
                "bbox": face.bbox,
                "confidence": face.det_score
            }
        
        except Exception as e:
            print(f"❌ Error extracting embedding: {str(e)}")
            return None