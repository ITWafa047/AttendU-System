
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, ConnectionFailure
from datetime import datetime
from typing import Optional, Dict, List


class MongoDBService:
    """
    MongoDBService is responsible for managing interactions with the MongoDB database, including connecting to the database, performing CRUD operations, and handling database-related errors.
    """

    def __init__(self, connection_string: str = "mongodb://localhost:27017/", db_name: str = "attendu"):
        """
        Initialize MongoDB service.
        
        Args:
            connection_string: MongoDB connection URI (default: localhost)
            db_name: Database name (default: attendu)
        """
        self.connection_string = connection_string
        self.db_name = db_name
        self.client = None
        self.db = None
        self.students_collection = None
    
    def connect(self):
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.students_collection = self.db['students']
            
            # Create compound unique index on (group_id, student_id)
            self.students_collection.create_index([("group_id", 1), ("student_id", 1)], unique=True)
            print(f"✓ Connected to MongoDB database: {self.db_name}")
        except ConnectionFailure as e:
            raise Exception(f"Failed to connect to MongoDB: {str(e)}")
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            print("✓ Disconnected from MongoDB")
    
    def insert_student(self, student_data: Dict) -> bool:
        """
        Insert a new student document.
        
        Args:
            student_data: Dictionary containing student information
                - group_id: str
                - student_id: str
                - embeddings: dict with mean_embedding and augmented_embeddings
                - courses: List[str]
                - attendance: List[dict]
        
        Returns:
            True if insertion successful, False otherwise
        """
        try:
            # Add timestamps
            student_data["created_at"] = datetime.utcnow()
            student_data["updated_at"] = datetime.utcnow()
            
            result = self.students_collection.insert_one(student_data)
            return result.inserted_id is not None
        except DuplicateKeyError:
            raise Exception(f"Student {student_data['student_id']} already exists in group {student_data['group_id']}")
        except Exception as e:
            raise Exception(f"Failed to insert student: {str(e)}")
    
    def find_student_by_id(self, group_id: str, student_id: str) -> Optional[Dict]:
        """
        Find a student by ID within a specific group.
        
        Args:
            group_id: Group identifier
            student_id: Student identifier
        
        Returns:
            Student document if found, None otherwise
        """
        try:
            student = self.students_collection.find_one({"group_id": group_id, "student_id": student_id})
            return student
        except Exception as e:
            raise Exception(f"Failed to find student: {str(e)}")
    
    def find_all_students_by_group(self, group_id: str) -> List[Dict]:
        """
        Find all students in a group.
        
        Args:
            group_id: Group identifier
        
        Returns:
            List of student documents in the group
        """
        try:
            students = list(self.students_collection.find({"group_id": group_id}))
            return students
        except Exception as e:
            raise Exception(f"Failed to retrieve students by group: {str(e)}")
    
    def update_student_attendance(self, group_id: str, student_id: str, attendance_record: Dict) -> bool:
        """
        Add attendance record for a student in a group.
        
        Args:
            group_id: Group identifier
            student_id: Student identifier
            attendance_record: Attendance record to add
        
        Returns:
            True if update successful, False if student not found
        """
        try:
            result = self.students_collection.update_one(
                {"group_id": group_id, "student_id": student_id},
                {
                    "$push": {"attendance": attendance_record},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            return result.matched_count > 0
        except Exception as e:
            raise Exception(f"Failed to update attendance: {str(e)}")
    
    def update_student_courses(self, group_id: str, student_id: str, courses: List[str]) -> bool:
        """
        Update student's courses in a group.
        
        Args:
            group_id: Group identifier
            student_id: Student identifier
            courses: List of course IDs
        
        Returns:
            True if update successful, False if student not found
        """
        try:
            result = self.students_collection.update_one(
                {"group_id": group_id, "student_id": student_id},
                {
                    "$set": {
                        "courses": courses,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            return result.matched_count > 0
        except Exception as e:
            raise Exception(f"Failed to update courses: {str(e)}")
    
    def delete_student(self, group_id: str, student_id: str) -> bool:
        """
        Delete a student document from a group.
        
        Args:
            group_id: Group identifier
            student_id: Student identifier
        
        Returns:
            True if deletion successful, False if student not found
        """
        try:
            result = self.students_collection.delete_one({"group_id": group_id, "student_id": student_id})
            return result.deleted_count > 0
        except Exception as e:
            raise Exception(f"Failed to delete student: {str(e)}")
    
    def find_all_students(self) -> List[Dict]:
        """
        Find all students across all groups.
        
        Returns:
            List of all student documents
        """
        try:
            students = list(self.students_collection.find({}))
            return students
        except Exception as e:
            raise Exception(f"Failed to retrieve students: {str(e)}")
