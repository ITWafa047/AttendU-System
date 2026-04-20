from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to sys.path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from StudentEnrollment import ImageValidator, FaceProcessor
from database.mongodb_service import MongoDBService
from sessions_schedules.create_schedules import (
    ScheduleCreateRequest, 
    ScheduleResponse, 
    ScheduleListResponse,
    create_schedules,
    validate_schedule_data
)

# Initialize FastAPI app
app = FastAPI(
    title="AttendU Student Registration API",
    description="API for registering new students with facial embeddings",
    version="1.0.0"
)

# Initialize database service
db_service = MongoDBService()

# Initialize image validator
image_validator = ImageValidator()


# ============================================================================
# Pydantic Models
# ============================================================================

class StudentRegistrationRequest(BaseModel):
    """Request model for student registration."""
    group_id: str = Field(..., description="Group/Class identifier (e.g., 'CS101-2024-Spring')")
    student_id: str = Field(..., description="Unique student identifier within the group")
    courses: List[str] = Field(default_factory=list, description="List of course IDs")
    attendance: List[dict] = Field(default_factory=list, description="Initial attendance records")


class StudentRegistrationResponse(BaseModel):
    """Response model for student registration."""
    success: bool
    message: str
    group_id: str
    student_id: str
    embedding_status: str
    timestamp: str


# ============================================================================
# Utility Functions
# ============================================================================

def embeddings_to_dict(mean_embedding: np.ndarray, embeddings_stack: np.ndarray) -> dict:
    """
    Convert embedding arrays to serializable dictionary format.
    
    Args:
        mean_embedding: (512,) normalized mean embedding
        embeddings_stack: (6, 512) stack of augmented embeddings
    
    Returns:
        Dictionary with serializable embedding data
    """
    return {
        "mean_embedding": mean_embedding.tolist(),
        "augmented_embeddings": embeddings_stack.tolist(),
        "embedding_dim": 512,
        "num_augmentations": 6
    }


def dict_to_embeddings(embeddings_dict: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert serialized embedding dictionary back to numpy arrays.
    
    Args:
        embeddings_dict: Dictionary with embedding data
    
    Returns:
        Tuple of (mean_embedding, embeddings_stack) as numpy arrays
    """
    mean_embedding = np.array(embeddings_dict["mean_embedding"], dtype=np.float32)
    augmented_embeddings = np.array(embeddings_dict["augmented_embeddings"], dtype=np.float32)
    return mean_embedding, augmented_embeddings


# ============================================================================
# Endpoints
# ============================================================================

@app.post("/api/v1/students/register", response_model=StudentRegistrationResponse)
async def register_student(
    group_id: str = Form(..., description="Group/Class identifier (e.g., 'CS101-2024-Spring')"),
    student_id: str = Form(..., description="Unique student identifier within the group"),
    image: UploadFile = File(..., description="Student facial image"),
    courses: str = Form(default="[]", description="JSON string of course IDs list"),
    attendance: str = Form(default="[]", description="JSON string of initial attendance records")
):
    """
    Register a new student with facial embedding.
    
    This endpoint:
    1. Validates the uploaded image
    2. Extracts facial embeddings (mean + augmented)
    3. Stores student data in MongoDB
    
    Request Parameters:
    - group_id: Group/Class identifier (e.g., 'CS101-2024-Spring')
    - student_id: Unique student identifier within the group (string)
    - image: Student face image file (JPG/PNG)
    - courses: JSON list of course IDs (e.g., '["COMP101", "MATH201"]')
    - attendance: JSON list of attendance records (e.g., '[]')
    
    Returns:
    - 200 OK: Student successfully registered with embeddings
    - 400 Bad Request: Invalid image or validation failure
    - 409 Conflict: Student already exists in this group
    - 500 Internal Server Error: Database or processing error
    """
    try:
        # Step 1: Parse JSON inputs
        try:
            courses_list = json.loads(courses) if isinstance(courses, str) else courses
            attendance_list = json.loads(attendance) if isinstance(attendance, str) else attendance
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON format: {str(e)}"
            )
        
        # Step 2: Validate courses and attendance are lists
        if not isinstance(courses_list, list):
            raise HTTPException(
                status_code=400,
                detail="Courses must be a JSON array"
            )
        if not isinstance(attendance_list, list):
            raise HTTPException(
                status_code=400,
                detail="Attendance must be a JSON array"
            )
        
        # Step 3: Check if student already exists in this group
        existing_student = db_service.find_student_by_id(group_id, student_id)
        if existing_student:
            raise HTTPException(
                status_code=409,
                detail=f"Student {student_id} already registered in group {group_id}"
            )
        
        # Step 4: Validate and process image
        # This runs the full ImageValidator pipeline:
        # - Format validation
        # - Size validation
        # - Face detection
        # - Single face validation
        # - Face quality checks
        # - Background validation
        # - Face alignment
        # - Blur validation
        # - Brightness validation
        mean_embedding, embeddings_augmentations = await image_validator.final_process(image)
        
        # Step 5: Convert embeddings to serializable format
        embeddings_data = embeddings_to_dict(mean_embedding, embeddings_augmentations)
        
        # Step 6: Prepare student document
        student_data = {
            "group_id": group_id,
            "student_id": student_id,
            "embeddings": embeddings_data,
            "courses": courses_list,
            "attendance": attendance_list,
            "status": "active",
            "created_at": None,  # Will be set by database service
            "updated_at": None   # Will be set by database service
        }
        
        # Step 7: Insert student into database
        result = db_service.insert_student(student_data)
        
        if not result:
            raise HTTPException(
                status_code=500,
                detail="Failed to store student in database"
            )
        
        # Step 8: Return success response
        return StudentRegistrationResponse(
            success=True,
            message=f"Student {student_id} successfully registered in group {group_id}",
            group_id=group_id,
            student_id=student_id,
            embedding_status="Generated (1 mean + 6 augmentations)",
            timestamp=None
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Image validation failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Registration failed: {str(e)}"
        )


@app.get("/api/v1/groups/{group_id}/students/{student_id}")
async def get_student_embeddings(group_id: str, student_id: str):
    """
    Retrieve student embeddings and data from a specific group.
    
    Returns:
    - 200 OK: Student found with embeddings
    - 404 Not Found: Student not found in this group
    - 500 Internal Server Error: Database error
    """
    try:
        student = db_service.find_student_by_id(group_id, student_id)
        
        if not student:
            raise HTTPException(
                status_code=404,
                detail=f"Student {student_id} not found in group {group_id}"
            )
        
        return {
            "group_id": student["group_id"],
            "student_id": student["student_id"],
            "embeddings": student["embeddings"],
            "courses": student["courses"],
            "attendance": student["attendance"],
            "status": student["status"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve student: {str(e)}"
        )


@app.put("/api/v1/groups/{group_id}/students/{student_id}/attendance")
async def update_attendance(
    group_id: str,
    student_id: str,
    attendance_record: dict
):
    """
    Update student attendance record in a specific group.
    
    Returns:
    - 200 OK: Attendance updated
    - 404 Not Found: Student not found in this group
    - 500 Internal Server Error: Database error
    """
    try:
        result = db_service.update_student_attendance(group_id, student_id, attendance_record)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Student {student_id} not found in group {group_id}"
            )
        
        return {
            "success": True,
            "message": f"Attendance updated for student {student_id} in group {group_id}",
            "group_id": group_id,
            "student_id": student_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update attendance: {str(e)}"
        )


@app.get("/api/v1/groups/{group_id}/students")
async def get_group_students(group_id: str):
    """
    Retrieve all students in a specific group.
    
    Returns:
    - 200 OK: List of students in the group
    - 500 Internal Server Error: Database error
    """
    try:
        students = db_service.find_all_students_by_group(group_id)
        return {
            "success": True,
            "group_id": group_id,
            "student_count": len(students),
            "students": students
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve students: {str(e)}"
        )


@app.get("/api/v1/students")
async def get_all_students():
    """
    Retrieve all students across all groups.
    
    Returns:
    - 200 OK: List of all students
    - 500 Internal Server Error: Database error
    """
    try:
        students = db_service.find_all_students()
        return {
            "success": True,
            "total_students": len(students),
            "students": students
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve all students: {str(e)}"
        )


@app.put("/api/v1/groups/{group_id}/students/{student_id}/courses")
async def update_student_courses(
    group_id: str,
    student_id: str,
    courses: List[str]
):
    """
    Update student's courses in a specific group.
    
    Request body: {"courses": ["COMP101", "MATH201"]}
    
    Returns:
    - 200 OK: Courses updated
    - 404 Not Found: Student not found in this group
    - 500 Internal Server Error: Database error
    """
    try:
        result = db_service.update_student_courses(group_id, student_id, courses)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Student {student_id} not found in group {group_id}"
            )
        
        return {
            "success": True,
            "message": f"Courses updated for student {student_id} in group {group_id}",
            "group_id": group_id,
            "student_id": student_id,
            "courses": courses
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update courses: {str(e)}"
        )


@app.delete("/api/v1/groups/{group_id}/students/{student_id}")
async def delete_student(group_id: str, student_id: str):
    """
    Delete a student from a specific group.
    
    Returns:
    - 200 OK: Student deleted successfully
    - 404 Not Found: Student not found in this group
    - 500 Internal Server Error: Database error
    """
    try:
        result = db_service.delete_student(group_id, student_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Student {student_id} not found in group {group_id}"
            )
        
        return {
            "success": True,
            "message": f"Student {student_id} deleted from group {group_id}",
            "group_id": group_id,
            "student_id": student_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete student: {str(e)}"
        )


class GenericDataModel(BaseModel):
    """Generic data model for storing arbitrary data."""
    collection_name: str = Field(..., description="MongoDB collection name")
    data: dict = Field(..., description="Data to store")


@app.post("/api/v1/data/store")
async def store_generic_data(payload: GenericDataModel):
    """
    Store generic data in MongoDB in a specified collection.
    
    Request body:
    {
        "collection_name": "sessions",
        "data": {
            "session_id": "s123",
            "date": "2024-01-15",
            "course": "CS101"
        }
    }
    
    Returns:
    - 200 OK: Data stored successfully
    - 400 Bad Request: Invalid collection or data
    - 500 Internal Server Error: Database error
    """
    try:
        if not payload.collection_name or not payload.data:
            raise HTTPException(
                status_code=400,
                detail="collection_name and data fields are required"
            )
        
        # Get or create collection
        collection = db_service.db[payload.collection_name]
        
        # Add timestamp
        payload.data["stored_at"] = datetime.utcnow()
        
        # Insert document
        result = collection.insert_one(payload.data)
        
        if not result.inserted_id:
            raise HTTPException(
                status_code=500,
                detail="Failed to store data in database"
            )
        
        return {
            "success": True,
            "message": f"Data stored in collection '{payload.collection_name}'",
            "collection_name": payload.collection_name,
            "document_id": str(result.inserted_id),
            "database": "attendu"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store data: {str(e)}"
        )


@app.get("/api/v1/data/retrieve/{collection_name}")
async def retrieve_generic_data(collection_name: str, limit: int = 100):
    """
    Retrieve data from a MongoDB collection.
    
    Query parameters:
    - limit: Maximum number of documents to retrieve (default: 100)
    
    Returns:
    - 200 OK: List of documents from the collection
    - 500 Internal Server Error: Database error
    """
    try:
        if not collection_name:
            raise HTTPException(
                status_code=400,
                detail="collection_name is required"
            )
        
        collection = db_service.db[collection_name]
        documents = list(collection.find({}).limit(limit))
        
        # Convert ObjectId to string for JSON serialization
        for doc in documents:
            doc['_id'] = str(doc['_id'])
        
        return {
            "success": True,
            "collection_name": collection_name,
            "document_count": len(documents),
            "documents": documents,
            "database": "attendu"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve data: {str(e)}"
        )


@app.get("/api/v1/database/info")
async def database_info():
    """
    Get information about the MongoDB database.
    
    Returns:
    - 200 OK: Database info including collections
    """
    try:
        collections = db_service.db.list_collection_names()
        
        collection_stats = []
        for collection_name in collections:
            collection = db_service.db[collection_name]
            doc_count = collection.count_documents({})
            collection_stats.append({
                "name": collection_name,
                "document_count": doc_count
            })
        
        return {
            "success": True,
            "database_name": "attendu",
            "total_collections": len(collections),
            "collections": collection_stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get database info: {str(e)}"
        )


# ============================================================================
# Schedule Endpoints
# ============================================================================

@app.post("/api/v1/schedules/create")
async def create_schedule_endpoint(schedule: ScheduleCreateRequest) -> ScheduleResponse:
    """
    Create and store a new schedule in MongoDB.
    
    Request body:
    {
        "course_name": "Data Structures",
        "session_type": "lecture",
        "day": "saturday",
        "start_time": "10:00",
        "end_time": "12:00",
        "location": "Room 101",
        "instructor_name": "Dr. Smith",
        "group_name": "Group A",
        "session_date": "2024-01-15"
    }
    
    Returns:
    - 201 Created: Schedule created successfully
    - 400 Bad Request: Invalid schedule data
    - 500 Internal Server Error: Database error
    """
    try:
        # Convert Pydantic model to dict
        schedule_data = schedule.dict()
        
        # Validate schedule data
        is_valid, error_msg = validate_schedule_data(schedule_data)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )
        
        # Create schedule object
        schedule_obj = create_schedules(
            course_name=schedule_data["course_name"],
            session_type=schedule_data["session_type"],
            day=schedule_data["day"],
            start_time=schedule_data["start_time"],
            end_time=schedule_data["end_time"],
            location=schedule_data["location"],
            instructor_name=schedule_data["instructor_name"],
            group_name=schedule_data["group_name"],
            session_date=schedule_data.get("session_date")
        )
        
        # Store in MongoDB
        schedules_collection = db_service.db["schedules"]
        result = schedules_collection.insert_one(schedule_obj)
        
        if not result.inserted_id:
            raise HTTPException(
                status_code=500,
                detail="Failed to store schedule in database"
            )
        
        return ScheduleResponse(
            success=True,
            message=f"Schedule for {schedule_data['course_name']} created successfully",
            schedule_id=str(result.inserted_id),
            schedule=schedule_obj
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create schedule: {str(e)}"
        )


@app.get("/api/v1/schedules")
async def get_all_schedules(limit: int = 100) -> ScheduleListResponse:
    """
    Retrieve all schedules from the database.
    
    Query parameters:
    - limit: Maximum number of schedules to retrieve (default: 100)
    
    Returns:
    - 200 OK: List of all schedules
    - 500 Internal Server Error: Database error
    """
    try:
        schedules_collection = db_service.db["schedules"]
        schedules = list(schedules_collection.find({}).limit(limit))
        
        # Convert ObjectId to string for JSON serialization
        for schedule in schedules:
            schedule['_id'] = str(schedule['_id'])
            if isinstance(schedule.get('created_at'), datetime):
                schedule['created_at'] = schedule['created_at'].isoformat()
            if isinstance(schedule.get('updated_at'), datetime):
                schedule['updated_at'] = schedule['updated_at'].isoformat()
        
        return ScheduleListResponse(
            success=True,
            total_schedules=len(schedules),
            schedules=schedules
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve schedules: {str(e)}"
        )


@app.get("/api/v1/schedules/{schedule_id}")
async def get_schedule_by_id(schedule_id: str):
    """
    Retrieve a specific schedule by ID.
    
    Returns:
    - 200 OK: Schedule found
    - 404 Not Found: Schedule not found
    - 500 Internal Server Error: Database error
    """
    try:
        from bson import ObjectId
        
        schedules_collection = db_service.db["schedules"]
        schedule = schedules_collection.find_one({"_id": ObjectId(schedule_id)})
        
        if not schedule:
            raise HTTPException(
                status_code=404,
                detail=f"Schedule with ID {schedule_id} not found"
            )
        
        # Convert ObjectId and datetime to string for JSON serialization
        schedule['_id'] = str(schedule['_id'])
        if isinstance(schedule.get('created_at'), datetime):
            schedule['created_at'] = schedule['created_at'].isoformat()
        if isinstance(schedule.get('updated_at'), datetime):
            schedule['updated_at'] = schedule['updated_at'].isoformat()
        
        return {
            "success": True,
            "schedule": schedule
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve schedule: {str(e)}"
        )


@app.get("/api/v1/schedules/by-course/{course_name}")
async def get_schedules_by_course(course_name: str, limit: int = 100):
    """
    Retrieve all schedules for a specific course.
    
    Returns:
    - 200 OK: List of schedules for the course
    - 500 Internal Server Error: Database error
    """
    try:
        schedules_collection = db_service.db["schedules"]
        schedules = list(schedules_collection.find({"course_name": course_name}).limit(limit))
        
        # Convert ObjectId and datetime to string for JSON serialization
        for schedule in schedules:
            schedule['_id'] = str(schedule['_id'])
            if isinstance(schedule.get('created_at'), datetime):
                schedule['created_at'] = schedule['created_at'].isoformat()
            if isinstance(schedule.get('updated_at'), datetime):
                schedule['updated_at'] = schedule['updated_at'].isoformat()
        
        return {
            "success": True,
            "course_name": course_name,
            "total_schedules": len(schedules),
            "schedules": schedules
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve course schedules: {str(e)}"
        )


@app.get("/api/v1/schedules/by-group/{group_name}")
async def get_schedules_by_group(group_name: str, limit: int = 100):
    """
    Retrieve all schedules for a specific group.
    
    Returns:
    - 200 OK: List of schedules for the group
    - 500 Internal Server Error: Database error
    """
    try:
        schedules_collection = db_service.db["schedules"]
        schedules = list(schedules_collection.find({"group_name": group_name}).limit(limit))
        
        # Convert ObjectId and datetime to string for JSON serialization
        for schedule in schedules:
            schedule['_id'] = str(schedule['_id'])
            if isinstance(schedule.get('created_at'), datetime):
                schedule['created_at'] = schedule['created_at'].isoformat()
            if isinstance(schedule.get('updated_at'), datetime):
                schedule['updated_at'] = schedule['updated_at'].isoformat()
        
        return {
            "success": True,
            "group_name": group_name,
            "total_schedules": len(schedules),
            "schedules": schedules
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve group schedules: {str(e)}"
        )


@app.get("/api/v1/schedules/by-day/{day}")
async def get_schedules_by_day(day: str, limit: int = 100):
    """
    Retrieve all schedules for a specific day.
    
    Valid days: saturday, sunday, monday, tuesday, wednesday, thursday, friday
    
    Returns:
    - 200 OK: List of schedules for the day
    - 400 Bad Request: Invalid day
    - 500 Internal Server Error: Database error
    """
    try:
        valid_days = ["saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday"]
        if day.lower() not in valid_days:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid day. Must be one of: {', '.join(valid_days)}"
            )
        
        schedules_collection = db_service.db["schedules"]
        schedules = list(schedules_collection.find({"day": day.lower()}).limit(limit))
        
        # Convert ObjectId and datetime to string for JSON serialization
        for schedule in schedules:
            schedule['_id'] = str(schedule['_id'])
            if isinstance(schedule.get('created_at'), datetime):
                schedule['created_at'] = schedule['created_at'].isoformat()
            if isinstance(schedule.get('updated_at'), datetime):
                schedule['updated_at'] = schedule['updated_at'].isoformat()
        
        return {
            "success": True,
            "day": day.lower(),
            "total_schedules": len(schedules),
            "schedules": schedules
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve day schedules: {str(e)}"
        )


@app.put("/api/v1/schedules/{schedule_id}")
async def update_schedule(schedule_id: str, schedule: ScheduleCreateRequest):
    """
    Update an existing schedule.
    
    Returns:
    - 200 OK: Schedule updated successfully
    - 404 Not Found: Schedule not found
    - 400 Bad Request: Invalid schedule data
    - 500 Internal Server Error: Database error
    """
    try:
        from bson import ObjectId
        
        # Validate schedule data
        schedule_data = schedule.dict()
        is_valid, error_msg = validate_schedule_data(schedule_data)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )
        
        # Add updated timestamp
        schedule_data["updated_at"] = datetime.utcnow()
        
        # Update in database
        schedules_collection = db_service.db["schedules"]
        result = schedules_collection.update_one(
            {"_id": ObjectId(schedule_id)},
            {"$set": schedule_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Schedule with ID {schedule_id} not found"
            )
        
        return {
            "success": True,
            "message": f"Schedule {schedule_id} updated successfully",
            "schedule_id": schedule_id,
            "modified_count": result.modified_count
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update schedule: {str(e)}"
        )


@app.delete("/api/v1/schedules/{schedule_id}")
async def delete_schedule(schedule_id: str):
    """
    Delete a schedule.
    
    Returns:
    - 200 OK: Schedule deleted successfully
    - 404 Not Found: Schedule not found
    - 500 Internal Server Error: Database error
    """
    try:
        from bson import ObjectId
        
        schedules_collection = db_service.db["schedules"]
        result = schedules_collection.delete_one({"_id": ObjectId(schedule_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Schedule with ID {schedule_id} not found"
            )
        
        return {
            "success": True,
            "message": f"Schedule {schedule_id} deleted successfully",
            "schedule_id": schedule_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete schedule: {str(e)}"
        )


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "AttendU Student Registration API",
        "version": "1.0.0"
    }


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail
        }
    )


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup."""
    try:
        db_service.connect()
        print("✓ Database connection established")
    except Exception as e:
        print(f"✗ Failed to connect to database: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown."""
    try:
        db_service.close()
        print("✓ Database connection closed")
    except Exception as e:
        print(f"✗ Failed to close database: {str(e)}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for GPU compatibility
        log_level="info"
    )
