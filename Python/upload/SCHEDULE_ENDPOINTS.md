# Schedule Endpoints Documentation

## Overview
The AttendU API provides comprehensive endpoints for managing course schedules. All schedule data is stored in the `attendu` database under the `schedules` collection in MongoDB.

---

## Schedule Data Structure

Each schedule document contains the following fields:

```json
{
  "_id": "ObjectId (MongoDB ID)",
  "course_name": "string - Name of the course",
  "session_type": "string - Type of session (lecture, section, lab, practical, seminar, tutorial)",
  "day": "string - Day of the week (saturday, sunday, monday, tuesday, wednesday, thursday, friday)",
  "start_time": "string - Start time in HH:MM format (24-hour)",
  "end_time": "string - End time in HH:MM format (24-hour)",
  "location": "string - Location/room of the session",
  "instructor_name": "string - Name of the instructor",
  "group_name": "string - Name of the group/class",
  "session_date": "string (optional) - Specific session date in YYYY-MM-DD format",
  "created_at": "ISO 8601 - Timestamp when created",
  "updated_at": "ISO 8601 - Timestamp when last updated"
}
```

---

## Endpoints

### 1. **Create Schedule**

#### `POST /api/v1/schedules/create`
Create and store a new schedule in MongoDB.

**Request Body:**
```json
{
  "course_name": "Data Structures",
  "session_type": "lecture",
  "day": "saturday",
  "start_time": "10:00",
  "end_time": "12:00",
  "location": "Room 101",
  "instructor_name": "Dr. Ahmed Smith",
  "group_name": "CS101-Group-A",
  "session_date": "2024-01-15"
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/schedules/create" \
  -H "Content-Type: application/json" \
  -d '{
    "course_name": "Data Structures",
    "session_type": "lecture",
    "day": "saturday",
    "start_time": "10:00",
    "end_time": "12:00",
    "location": "Room 101",
    "instructor_name": "Dr. Smith",
    "group_name": "Group A"
  }'
```

**Response (201 Created):**
```json
{
  "success": true,
  "message": "Schedule for Data Structures created successfully",
  "schedule_id": "507f1f77bcf86cd799439011",
  "schedule": {
    "course_name": "Data Structures",
    "session_type": "lecture",
    "day": "saturday",
    "start_time": "10:00",
    "end_time": "12:00",
    "location": "Room 101",
    "instructor_name": "Dr. Smith",
    "group_name": "Group A",
    "session_date": "2024-01-15",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z"
  }
}
```

---

### 2. **Get All Schedules**

#### `GET /api/v1/schedules`
Retrieve all schedules from the database.

**Query Parameters:**
- `limit` (integer, optional): Maximum number of schedules to retrieve (default: 100, max: 1000)

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/schedules?limit=50"
```

**Response (200 OK):**
```json
{
  "success": true,
  "total_schedules": 2,
  "schedules": [
    {
      "_id": "507f1f77bcf86cd799439011",
      "course_name": "Data Structures",
      "session_type": "lecture",
      "day": "saturday",
      "start_time": "10:00",
      "end_time": "12:00",
      "location": "Room 101",
      "instructor_name": "Dr. Smith",
      "group_name": "Group A",
      "session_date": "2024-01-15",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:30:00Z"
    },
    {
      "_id": "507f1f77bcf86cd799439012",
      "course_name": "Algorithms",
      "session_type": "lab",
      "day": "sunday",
      "start_time": "14:00",
      "end_time": "16:00",
      "location": "Lab 202",
      "instructor_name": "Dr. Johnson",
      "group_name": "Group B",
      "session_date": "2024-01-16",
      "created_at": "2024-01-15T10:35:00Z",
      "updated_at": "2024-01-15T10:35:00Z"
    }
  ]
}
```

---

### 3. **Get Schedule by ID**

#### `GET /api/v1/schedules/{schedule_id}`
Retrieve a specific schedule by its MongoDB ID.

**Path Parameters:**
- `schedule_id` (string, required): MongoDB ObjectId of the schedule

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/schedules/507f1f77bcf86cd799439011"
```

**Response (200 OK):**
```json
{
  "success": true,
  "schedule": {
    "_id": "507f1f77bcf86cd799439011",
    "course_name": "Data Structures",
    "session_type": "lecture",
    "day": "saturday",
    "start_time": "10:00",
    "end_time": "12:00",
    "location": "Room 101",
    "instructor_name": "Dr. Smith",
    "group_name": "Group A",
    "session_date": "2024-01-15",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z"
  }
}
```

---

### 4. **Get Schedules by Course**

#### `GET /api/v1/schedules/by-course/{course_name}`
Retrieve all schedules for a specific course.

**Path Parameters:**
- `course_name` (string, required): Name of the course

**Query Parameters:**
- `limit` (integer, optional): Maximum number of schedules (default: 100)

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/schedules/by-course/Data%20Structures?limit=50"
```

**Response (200 OK):**
```json
{
  "success": true,
  "course_name": "Data Structures",
  "total_schedules": 3,
  "schedules": [...]
}
```

---

### 5. **Get Schedules by Group**

#### `GET /api/v1/schedules/by-group/{group_name}`
Retrieve all schedules for a specific group/class.

**Path Parameters:**
- `group_name` (string, required): Name of the group

**Query Parameters:**
- `limit` (integer, optional): Maximum number of schedules (default: 100)

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/schedules/by-group/Group%20A?limit=50"
```

**Response (200 OK):**
```json
{
  "success": true,
  "group_name": "Group A",
  "total_schedules": 2,
  "schedules": [...]
}
```

---

### 6. **Get Schedules by Day**

#### `GET /api/v1/schedules/by-day/{day}`
Retrieve all schedules for a specific day of the week.

**Path Parameters:**
- `day` (string, required): Day of the week (saturday, sunday, monday, tuesday, wednesday, thursday, friday)

**Query Parameters:**
- `limit` (integer, optional): Maximum number of schedules (default: 100)

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/schedules/by-day/saturday?limit=100"
```

**Response (200 OK):**
```json
{
  "success": true,
  "day": "saturday",
  "total_schedules": 5,
  "schedules": [...]
}
```

---

### 7. **Update Schedule**

#### `PUT /api/v1/schedules/{schedule_id}`
Update an existing schedule.

**Path Parameters:**
- `schedule_id` (string, required): MongoDB ObjectId of the schedule

**Request Body:**
```json
{
  "course_name": "Data Structures (Updated)",
  "session_type": "lecture",
  "day": "monday",
  "start_time": "11:00",
  "end_time": "13:00",
  "location": "Room 105",
  "instructor_name": "Dr. Smith",
  "group_name": "Group A"
}
```

**Example Request:**
```bash
curl -X PUT "http://localhost:8000/api/v1/schedules/507f1f77bcf86cd799439011" \
  -H "Content-Type: application/json" \
  -d '{
    "course_name": "Data Structures",
    "session_type": "lecture",
    "day": "monday",
    "start_time": "11:00",
    "end_time": "13:00",
    "location": "Room 105",
    "instructor_name": "Dr. Smith",
    "group_name": "Group A"
  }'
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Schedule 507f1f77bcf86cd799439011 updated successfully",
  "schedule_id": "507f1f77bcf86cd799439011",
  "modified_count": 1
}
```

---

### 8. **Delete Schedule**

#### `DELETE /api/v1/schedules/{schedule_id}`
Delete a schedule.

**Path Parameters:**
- `schedule_id` (string, required): MongoDB ObjectId of the schedule

**Example Request:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/schedules/507f1f77bcf86cd799439011"
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Schedule 507f1f77bcf86cd799439011 deleted successfully",
  "schedule_id": "507f1f77bcf86cd799439011"
}
```

---

## Validation Rules

### Session Type
Valid values: `lecture`, `section`, `lab`, `practical`, `seminar`, `tutorial`

### Day
Valid values: `saturday`, `sunday`, `monday`, `tuesday`, `wednesday`, `thursday`, `friday`

### Time Format
- Must be in HH:MM format (24-hour)
- Valid range: 00:00 to 23:59
- End time must be after start time
- Example: "10:00", "14:30", "23:59"

### Required Fields
All of these fields are mandatory:
- `course_name`
- `session_type`
- `day`
- `start_time`
- `end_time`
- `location`
- `instructor_name`
- `group_name`

### Optional Fields
- `session_date` (format: YYYY-MM-DD)

---

## Error Responses

### 400 Bad Request
```json
{
  "error": true,
  "status_code": 400,
  "detail": "Invalid day. Must be one of: saturday, sunday, monday, tuesday, wednesday, thursday, friday"
}
```

### 404 Not Found
```json
{
  "error": true,
  "status_code": 404,
  "detail": "Schedule with ID 507f1f77bcf86cd799439011 not found"
}
```

### 500 Internal Server Error
```json
{
  "error": true,
  "status_code": 500,
  "detail": "Failed to create schedule: Connection timeout"
}
```

---

## MongoDB Collection

**Collection Name:** `schedules`
**Database:** `attendu`
**Indexes:** Created automatically on first insertion

### Example MongoDB Query
```javascript
// Get all schedules for a specific course
db.schedules.find({ "course_name": "Data Structures" })

// Get all Saturday lectures
db.schedules.find({ "day": "saturday", "session_type": "lecture" })

// Get schedules by group
db.schedules.find({ "group_name": "Group A" })
```

---

## Integration with Other Endpoints

Schedule endpoints work seamlessly with:
- **Student Management**: Link students to schedules via `group_name`
- **Generic Data Storage**: Schedules can also be queried via the `/api/v1/data/retrieve/schedules` endpoint
- **Database Info**: View schedule collection stats via `/api/v1/database/info`

---

## Best Practices

1. **Unique Identifiers**: Use MongoDB ObjectId as returned in `_id` field for all schedule references
2. **Time Validation**: Always ensure end_time > start_time before making requests
3. **Day Naming**: Use lowercase day names for consistency
4. **Time Format**: Always use 24-hour format (HH:MM)
5. **Batch Operations**: Use `limit` parameter to paginate large result sets
6. **Course Naming**: Keep course names consistent across requests for better filtering
7. **Group Naming**: Use standardized group naming conventions (e.g., "CS101-Group-A")

---

## Rate Limits

No rate limits currently implemented. For production deployment, consider:
- Limiting list queries to max 1000 results
- Implementing pagination for large datasets
- Adding caching for frequently accessed schedules

