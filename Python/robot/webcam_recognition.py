import cv2
import pickle
import numpy as np
from face_engine import FaceEngine
import time
import os
from anti_spoofing.anti_spoof_manager import AntiSpoofManager



# Like API From Backend
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMBEDDINGS_PATH = os.path.join(BASE_PATH, r"D:\\Graduation Project 2026\\Project Implementation\\AttendU System\\Python\\robot\\team_embeddings.pkl")   


class WebcamRecognition:

    def __init__(self, embeddings_path=EMBEDDINGS_PATH):

        self.face_engine = FaceEngine()
        self.last_update_time = {}
        self.UPDATE_INTERVAL = 600  # seconds

        self.THRESHOLD = 0.68
        self.UPDATE_THRESHOLD = 0.82
        self.anti_spoof = AntiSpoofManager()

        with open(embeddings_path, "rb") as f:
            self.database = pickle.load(f)
        print(f"Loaded Students: {len(self.database)}")

    @staticmethod
    def cosine_similarity(emb1, emb2):
        """Calculate cosine similarity between two embeddings (already normalized)."""
        return float(np.dot(emb1, emb2))

    def search_database(self, query_embedding):

        best_match = None
        best_similarity = -1

        for student_id, data in self.database.items():
            db_embedding = data["embedding"]
            sim = self.cosine_similarity(query_embedding, db_embedding)

            if sim > best_similarity:
                best_similarity = sim
                best_match = student_id

        return best_match, best_similarity

    def self_update(self, student_id, new_embedding):

        data = self.database[student_id]
        old_embedding = data["embedding"]
        count = data["count"]

        new_mean = (old_embedding * count + new_embedding) / (count + 1)
        new_mean = new_mean / np.linalg.norm(new_mean)

        self.database[student_id]["embedding"] = new_mean
        self.database[student_id]["count"] = count + 1

        print(f"🔄 Updated embedding for {student_id} (count = {count + 1})")

        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(self.database, f)

    def run(self):
        """Run the webcam recognition loop."""
        cap = cv2.VideoCapture(0)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame from webcam")
                    break

                # Extract embedding from the first face
                face_data = self.face_engine.extract_embedding(frame)

                if face_data is not None:
                    bbox = face_data["bbox"]
                    embedding = face_data["embedding"]

                    try:
                        best_match, similarity = self.search_database(embedding)

                        # Anti-spoofing verification first
                        if best_match is not None and similarity >= self.THRESHOLD:
                            is_live, message = self.anti_spoof.verify(best_match, face_data)

                            if is_live:
                                color = (0, 255, 0)  # Green - Verified
                                label = f"{best_match} (Verified)"
                                
                                # Update embedding if high confidence
                                if similarity >= self.UPDATE_THRESHOLD:
                                    current_time = time.time()
                                    last_time = self.last_update_time.get(best_match, 0)
                                    if current_time - last_time > self.UPDATE_INTERVAL:
                                        self.self_update(best_match, embedding)
                                        self.last_update_time[best_match] = current_time
                            else:
                                color = (0, 0, 255)  # Red - Spoof detected
                                label = f"{best_match} ({message})"
                        else:
                            color = (0, 0, 255)  # Red - Unknown
                            label = "Unknown"

                        x1, y1, x2, y2 = map(int, bbox)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.6,
                            color,
                            2,
                        )
                    except Exception as e:
                        print(f"❌ Error during face recognition: {str(e)}")
                        cv2.putText(
                            frame,
                            "Error",
                            (50, 50),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )

                cv2.imshow("Face Recognition", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("👋 Exiting face recognition")
                    break

        except Exception as e:
            print(f"❌ Critical error in main loop: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()