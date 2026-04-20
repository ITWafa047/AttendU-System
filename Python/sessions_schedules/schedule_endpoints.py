"""
Schedule Endpoints Module
=========================

This module contains all FastAPI endpoints for managing schedules.
It handles CRUD operations for schedules stored in MongoDB.

Endpoints:
- POST   /api/v1/schedules/create - Create a new schedule
- GET    /api/v1/schedules - Retrieve all schedules
- GET    /api/v1/schedules/{schedule_id} - Get schedule by ID
- GET    /api/v1/schedules/by-course/{course_name} - Get schedules by course
- GET    /api/v1/schedules/by-group/{group_name} - Get schedules by group
- GET    /api/v1/schedules/by-day/{day} - Get schedules by day
- PUT    /api/v1/schedules/{schedule_id} - Update a schedule
- DELETE /api/v1/schedules/{schedule_id} - Delete a schedule
"""

from fastapi import FastAPI, HTTPException
from datetime import datetime
from bson import ObjectId

import sys
from pathlib import Path

# Add parent directory to sys.path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from database.mongodb_service import MongoDBService


from create_schedules import (
    ScheduleCreateRequest,
    ScheduleResponse,
    ScheduleListResponse,
    create_schedules,
    validate_schedule_data,
)


def register_schedule_endpoints(app: FastAPI, db_service):
    """
    Register all schedule endpoints with the FastAPI application.

    Args:
        app: FastAPI application instance
        db_service: MongoDB service instance for database operations
    """

    @app.post("/api/v1/schedules/create")
    async def create_schedule_endpoint(
        schedule: ScheduleCreateRequest,
    ) -> ScheduleResponse:
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
                raise HTTPException(status_code=400, detail=error_msg)

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
                session_date=schedule_data.get("session_date"),
            )

            # Store in MongoDB
            schedules_collection = db_service.db["schedules"]
            result = schedules_collection.insert_one(schedule_obj)

            if not result.inserted_id:
                raise HTTPException(
                    status_code=500, detail="Failed to store schedule in database"
                )

            return ScheduleResponse(
                success=True,
                message=f"Schedule for {schedule_data['course_name']} created successfully",
                schedule_id=str(result.inserted_id),
                schedule=schedule_obj,
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to create schedule: {str(e)}"
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
                schedule["_id"] = str(schedule["_id"])
                if isinstance(schedule.get("created_at"), datetime):
                    schedule["created_at"] = schedule["created_at"].isoformat()
                if isinstance(schedule.get("updated_at"), datetime):
                    schedule["updated_at"] = schedule["updated_at"].isoformat()

            return ScheduleListResponse(
                success=True, total_schedules=len(schedules), schedules=schedules
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to retrieve schedules: {str(e)}"
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
            schedules_collection = db_service.db["schedules"]
            schedule = schedules_collection.find_one({"_id": ObjectId(schedule_id)})

            if not schedule:
                raise HTTPException(
                    status_code=404, detail=f"Schedule with ID {schedule_id} not found"
                )

            # Convert ObjectId and datetime to string for JSON serialization
            schedule["_id"] = str(schedule["_id"])
            if isinstance(schedule.get("created_at"), datetime):
                schedule["created_at"] = schedule["created_at"].isoformat()
            if isinstance(schedule.get("updated_at"), datetime):
                schedule["updated_at"] = schedule["updated_at"].isoformat()

            return {"success": True, "schedule": schedule}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to retrieve schedule: {str(e)}"
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
            schedules = list(
                schedules_collection.find({"course_name": course_name}).limit(limit)
            )

            # Convert ObjectId and datetime to string for JSON serialization
            for schedule in schedules:
                schedule["_id"] = str(schedule["_id"])
                if isinstance(schedule.get("created_at"), datetime):
                    schedule["created_at"] = schedule["created_at"].isoformat()
                if isinstance(schedule.get("updated_at"), datetime):
                    schedule["updated_at"] = schedule["updated_at"].isoformat()

            return {
                "success": True,
                "course_name": course_name,
                "total_schedules": len(schedules),
                "schedules": schedules,
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to retrieve course schedules: {str(e)}"
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
            schedules = list(
                schedules_collection.find({"group_name": group_name}).limit(limit)
            )

            # Convert ObjectId and datetime to string for JSON serialization
            for schedule in schedules:
                schedule["_id"] = str(schedule["_id"])
                if isinstance(schedule.get("created_at"), datetime):
                    schedule["created_at"] = schedule["created_at"].isoformat()
                if isinstance(schedule.get("updated_at"), datetime):
                    schedule["updated_at"] = schedule["updated_at"].isoformat()

            return {
                "success": True,
                "group_name": group_name,
                "total_schedules": len(schedules),
                "schedules": schedules,
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to retrieve group schedules: {str(e)}"
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
            valid_days = [
                "saturday",
                "sunday",
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
            ]
            if day.lower() not in valid_days:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid day. Must be one of: {', '.join(valid_days)}",
                )

            schedules_collection = db_service.db["schedules"]
            schedules = list(
                schedules_collection.find({"day": day.lower()}).limit(limit)
            )

            # Convert ObjectId and datetime to string for JSON serialization
            for schedule in schedules:
                schedule["_id"] = str(schedule["_id"])
                if isinstance(schedule.get("created_at"), datetime):
                    schedule["created_at"] = schedule["created_at"].isoformat()
                if isinstance(schedule.get("updated_at"), datetime):
                    schedule["updated_at"] = schedule["updated_at"].isoformat()

            return {
                "success": True,
                "day": day.lower(),
                "total_schedules": len(schedules),
                "schedules": schedules,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to retrieve day schedules: {str(e)}"
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
            # Validate schedule data
            schedule_data = schedule.dict()
            is_valid, error_msg = validate_schedule_data(schedule_data)
            if not is_valid:
                raise HTTPException(status_code=400, detail=error_msg)

            # Add updated timestamp
            schedule_data["updated_at"] = datetime.utcnow()

            # Update in database
            schedules_collection = db_service.db["schedules"]
            result = schedules_collection.update_one(
                {"_id": ObjectId(schedule_id)}, {"$set": schedule_data}
            )

            if result.matched_count == 0:
                raise HTTPException(
                    status_code=404, detail=f"Schedule with ID {schedule_id} not found"
                )

            return {
                "success": True,
                "message": f"Schedule {schedule_id} updated successfully",
                "schedule_id": schedule_id,
                "modified_count": result.modified_count,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to update schedule: {str(e)}"
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
            schedules_collection = db_service.db["schedules"]
            result = schedules_collection.delete_one({"_id": ObjectId(schedule_id)})

            if result.deleted_count == 0:
                raise HTTPException(
                    status_code=404, detail=f"Schedule with ID {schedule_id} not found"
                )

            return {
                "success": True,
                "message": f"Schedule {schedule_id} deleted successfully",
                "schedule_id": schedule_id,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to delete schedule: {str(e)}"
            )


# ============================================================================
# Main Entry Point
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="Schedule API",
    description="API for managing schedules",
    version="1.0.0",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for GPU compatibility
        log_level="info",
    )

# Initialize database service
db_service = MongoDBService()