from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
import numpy as np
import json
import sys
from pathlib import Path

# Add parent directory to sys.path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from StudentEnrollment import ImageValidator, FaceProcessor
from database.mongodb_service import MongoDBService

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
