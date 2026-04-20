import json
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# ============================================================================
# Pydantic Models for Schedule
# ============================================================================

class ScheduleCreateRequest(BaseModel):
    """Request model for creating a schedule."""
    course_name: str = Field(..., description="Name of the course")
    session_type: str = Field(..., description="Type of session (lecture, section, lab, practical, seminar, tutorial)")
    day: str = Field(..., description="Day of the session (saturday, sunday, monday, tuesday, wednesday, thursday, friday)")
    start_time: str = Field(..., description="Start time in HH:MM format")
    end_time: str = Field(..., description="End time in HH:MM format")
    location: str = Field(..., description="Location of the session")
    instructor_name: str = Field(..., description="Name of the instructor")
    group_name: str = Field(..., description="Name of the group/class")
    session_date: Optional[str] = Field(default=None, description="Session date in YYYY-MM-DD format")


class ScheduleResponse(BaseModel):
    """Response model for schedule operations."""
    success: bool
    message: str
    schedule_id: Optional[str] = None
    schedule: Optional[dict] = None


class ScheduleListResponse(BaseModel):
    """Response model for schedule list operations."""
    success: bool
    total_schedules: int
    schedules: List[dict] = []


# ============================================================================
# Utility Functions
# ============================================================================

def create_schedules(
    course_name: str,
    session_type: str,
    day: str,
    start_time: str,
    end_time: str,
    location: str,
    instructor_name: str,
    group_name: str,
    session_date: Optional[str] = None
) -> dict:
    """
    Create a schedule object for a session.
    
    Args:
        - course_name: name of the course
        - session_type: type of the session (lecture, section, lab)
        - day: day of the session (saturday, sunday, monday, tuesday, wednesday, thursday, friday)
        - start_time: start time of the session in HH:MM format
        - end_time: end time of the session in HH:MM format
        - location: location of the session
        - instructor_name: name of the instructor
        - group_name: name of the group
        - session_date: optional session date in YYYY-MM-DD format

    Returns:
        schedule object with all the session details

    Example:
        - course_name: "Data Structures"
        - session_type: "lecture"
        - day: "saturday"
        - start_time: "10:00"
        - end_time: "12:00"
        - location: "Room 101"
        - instructor_name: "Dr. Smith"
        - group_name: "Group A"
    """
    schedule = {
        "course_name": course_name,
        "session_type": session_type,
        "day": day,
        "start_time": start_time,
        "end_time": end_time,
        "location": location,
        "instructor_name": instructor_name,
        "group_name": group_name,
        "session_date": session_date,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }

    return schedule


def validate_schedule_data(schedule_data: dict) -> tuple:
    """
    Validate schedule data.
    
    Args:
        schedule_data: Dictionary containing schedule information
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    import re
    
    # Validate day
    valid_days = ["saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday"]
    if schedule_data.get("day", "").lower() not in valid_days:
        return False, f"Invalid day. Must be one of: {', '.join(valid_days)}"
    
    # Validate session type
    valid_types = ["lecture", "section", "lab", "practical", "seminar", "tutorial"]
    if schedule_data.get("session_type", "").lower() not in valid_types:
        return False, f"Invalid session type. Must be one of: {', '.join(valid_types)}"
    
    # Validate time format (HH:MM)
    time_pattern = r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$"
    if not re.match(time_pattern, schedule_data.get("start_time", "")):
        return False, "Invalid start_time format. Use HH:MM"
    if not re.match(time_pattern, schedule_data.get("end_time", "")):
        return False, "Invalid end_time format. Use HH:MM"
    
    # Validate that end_time is after start_time
    start_h, start_m = map(int, schedule_data.get("start_time", "").split(":"))
    end_h, end_m = map(int, schedule_data.get("end_time", "").split(":"))
    start_total = start_h * 60 + start_m
    end_total = end_h * 60 + end_m
    
    if end_total <= start_total:
        return False, "End time must be after start time"
    
    # Check required fields
    required_fields = ["course_name", "session_type", "day", "start_time", "end_time", 
                      "location", "instructor_name", "group_name"]
    missing_fields = [field for field in required_fields if not schedule_data.get(field)]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    return True, ""
