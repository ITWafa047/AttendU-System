# AttendU FastAPI - MongoDB Integration Documentation

## Overview
The AttendU Student Registration API is built with **FastAPI** and uses **MongoDB** database (named: `attendu`) for persistent data storage. This document provides comprehensive information about all available endpoints and how to use them.

---

## Database Configuration

- **Database Name**: `attendu`
- **Connection String**: `mongodb://localhost:27017/`
- **Default Collections**:
  - `students` - Stores student records with facial embeddings
  - Additional collections can be created dynamically via the generic data store endpoints

---

## API Endpoints

### 1. **Student Registration**

#### `POST /api/v1/students/register`
Register a new student with facial embedding.

**Request Type**: `multipart/form-data`

**Parameters:**
- `group_id` (string, required): Group/Class identifier (e.g., 'CS101-2024-Spring')
- `student_id` (string, required): Unique student identifier within the group
- `image` (file, required): Student facial image (JPG/PNG)
- `courses` (string, default: "[]"): JSON string of course IDs
- `attendance` (string, default: "[]"): JSON string of initial attendance records

**Example Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/students/register" \
  -H "accept: application/json" \
  -F "group_id=CS101-2024-Spring" \
  -F "student_id=STU001" \
  -F "image=@student_photo.jpg" \
  -F "courses=[\"COMP101\", \"MATH201\"]" \
  -F "attendance=[]"
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Student STU001 successfully registered in group CS101-2024-Spring",
  "group_id": "CS101-2024-Spring",
  "student_id": "STU001",
  "embedding_status": "Generated (1 mean + 6 augmentations)",
  "timestamp": null
}
```

---

### 2. **Get Student by ID**

#### `GET /api/v1/groups/{group_id}/students/{student_id}`
Retrieve student embeddings and data from a specific group.

**Parameters:**
- `group_id` (path, required): Group identifier
- `student_id` (path, required): Student identifier

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/groups/CS101-2024-Spring/students/STU001"
```

**Response (200 OK):**
```json
{
  "group_id": "CS101-2024-Spring",
  "student_id": "STU001",
  "embeddings": {
    "mean_embedding": [...],
    "augmented_embeddings": [...],
    "embedding_dim": 512,
    "num_augmentations": 6
  },
  "courses": ["COMP101", "MATH201"],
  "attendance": [],
  "status": "active"
}
```

---

### 3. **Get All Students in a Group**

#### `GET /api/v1/groups/{group_id}/students`
Retrieve all students in a specific group.

**Parameters:**
- `group_id` (path, required): Group identifier

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/groups/CS101-2024-Spring/students"
```

**Response (200 OK):**
```json
{
  "success": true,
  "group_id": "CS101-2024-Spring",
  "student_count": 2,
  "students": [
    {
      "group_id": "CS101-2024-Spring",
      "student_id": "STU001",
      "embeddings": {...},
      "courses": ["COMP101", "MATH201"],
      "attendance": []
    },
    {
      "group_id": "CS101-2024-Spring",
      "student_id": "STU002",
      "embeddings": {...},
      "courses": ["COMP101"],
      "attendance": []
    }
  ]
}
```

---

### 4. **Get All Students**

#### `GET /api/v1/students`
Retrieve all students across all groups.

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/students"
```

**Response (200 OK):**
```json
{
  "success": true,
  "total_students": 5,
  "students": [...]
}
```

---

### 5. **Update Student Attendance**

#### `PUT /api/v1/groups/{group_id}/students/{student_id}/attendance`
Add or update attendance record for a student.

**Parameters:**
- `group_id` (path, required): Group identifier
- `student_id` (path, required): Student identifier
- `attendance_record` (body, required): Attendance record object

**Request Body Example:**
```json
{
  "date": "2024-01-15",
  "session_id": "sess_123",
  "status": "present",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Example Request:**
```bash
curl -X PUT "http://localhost:8000/api/v1/groups/CS101-2024-Spring/students/STU001/attendance" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-01-15",
    "status": "present"
  }'
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Attendance updated for student STU001 in group CS101-2024-Spring",
  "group_id": "CS101-2024-Spring",
  "student_id": "STU001"
}
```

---

### 6. **Update Student Courses**

#### `PUT /api/v1/groups/{group_id}/students/{student_id}/courses`
Update student's courses in a specific group.

**Parameters:**
- `group_id` (path, required): Group identifier
- `student_id` (path, required): Student identifier
- `courses` (body, required): List of course IDs

**Request Body Example:**
```json
{
  "courses": ["COMP101", "MATH201", "ENG101"]
}
```

**Example Request:**
```bash
curl -X PUT "http://localhost:8000/api/v1/groups/CS101-2024-Spring/students/STU001/courses" \
  -H "Content-Type: application/json" \
  -d '{"courses": ["COMP101", "MATH201", "ENG101"]}'
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Courses updated for student STU001 in group CS101-2024-Spring",
  "group_id": "CS101-2024-Spring",
  "student_id": "STU001",
  "courses": ["COMP101", "MATH201", "ENG101"]
}
```

---

### 7. **Delete Student**

#### `DELETE /api/v1/groups/{group_id}/students/{student_id}`
Delete a student from a specific group.

**Parameters:**
- `group_id` (path, required): Group identifier
- `student_id` (path, required): Student identifier

**Example Request:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/groups/CS101-2024-Spring/students/STU001"
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Student STU001 deleted from group CS101-2024-Spring",
  "group_id": "CS101-2024-Spring",
  "student_id": "STU001"
}
```

---

### 8. **Store Generic Data**

#### `POST /api/v1/data/store`
Store generic data in MongoDB in any specified collection.

**Request Body:**
```json
{
  "collection_name": "sessions",
  "data": {
    "session_id": "s123",
    "date": "2024-01-15",
    "course": "CS101",
    "duration_minutes": 60
  }
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/data/store" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "sessions",
    "data": {
      "session_id": "s123",
      "date": "2024-01-15",
      "course": "CS101"
    }
  }'
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Data stored in collection 'sessions'",
  "collection_name": "sessions",
  "document_id": "507f1f77bcf86cd799439011",
  "database": "attendu"
}
```

---

### 9. **Retrieve Generic Data**

#### `GET /api/v1/data/retrieve/{collection_name}`
Retrieve data from any MongoDB collection.

**Parameters:**
- `collection_name` (path, required): Collection name
- `limit` (query, optional): Maximum number of documents to retrieve (default: 100)

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/data/retrieve/sessions?limit=50"
```

**Response (200 OK):**
```json
{
  "success": true,
  "collection_name": "sessions",
  "document_count": 5,
  "documents": [
    {
      "_id": "507f1f77bcf86cd799439011",
      "session_id": "s123",
      "date": "2024-01-15",
      "course": "CS101",
      "stored_at": "2024-01-15T10:30:00Z"
    }
  ],
  "database": "attendu"
}
```

---

### 10. **Get Database Information**

#### `GET /api/v1/database/info`
Get information about the MongoDB database including all collections.

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/database/info"
```

**Response (200 OK):**
```json
{
  "success": true,
  "database_name": "attendu",
  "total_collections": 3,
  "collections": [
    {
      "name": "students",
      "document_count": 25
    },
    {
      "name": "sessions",
      "document_count": 150
    },
    {
      "name": "courses",
      "document_count": 10
    }
  ]
}
```

---

### 11. **Health Check**

#### `GET /api/v1/health`
Health check endpoint to verify API is running.

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "AttendU Student Registration API",
  "version": "1.0.0"
}
```

---

## Error Responses

### 400 Bad Request
```json
{
  "error": true,
  "status_code": 400,
  "detail": "Invalid JSON format: Expecting value"
}
```

### 404 Not Found
```json
{
  "error": true,
  "status_code": 404,
  "detail": "Student STU001 not found in group CS101-2024-Spring"
}
```

### 409 Conflict
```json
{
  "error": true,
  "status_code": 409,
  "detail": "Student STU001 already registered in group CS101-2024-Spring"
}
```

### 500 Internal Server Error
```json
{
  "error": true,
  "status_code": 500,
  "detail": "Failed to retrieve student: Connection timeout"
}
```

---

## Starting the Server

```bash
# Navigate to the upload directory
cd Python/upload

# Start the FastAPI server
python upload_server.py

# The API will be available at: http://localhost:8000
# Interactive API documentation: http://localhost:8000/docs
# Alternative API documentation: http://localhost:8000/redoc
```

---

## MongoDB Collections

### `students` Collection
Stores student records with facial embeddings.

**Document Structure:**
```json
{
  "_id": "ObjectId",
  "group_id": "CS101-2024-Spring",
  "student_id": "STU001",
  "embeddings": {
    "mean_embedding": [...],
    "augmented_embeddings": [...],
    "embedding_dim": 512,
    "num_augmentations": 6
  },
  "courses": ["COMP101", "MATH201"],
  "attendance": [...],
  "status": "active",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

### Custom Collections
Any collection can be created dynamically using the generic data store endpoint.

---

## Key Features

✅ **FastAPI Integration** - Modern, high-performance Python web framework
✅ **MongoDB Database** - Document-based storage with "attendu" database
✅ **Student Management** - Register, retrieve, update, and delete students
✅ **Facial Embeddings** - Store and retrieve face embeddings for recognition
✅ **Attendance Tracking** - Record and manage student attendance
✅ **Course Management** - Manage student courses
✅ **Generic Data Storage** - Store any data in MongoDB collections
✅ **Auto Documentation** - Interactive API docs via Swagger UI and ReDoc
✅ **Error Handling** - Comprehensive error messages and status codes
✅ **Database Info** - View database statistics and collection information

---

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- PyMongo
- MongoDB Server (running on localhost:27017)

---

## Notes

- All timestamps use UTC timezone
- Database name is fixed as `attendu`
- Student uniqueness is enforced by compound index on (group_id, student_id)
- Embedding dimensions are fixed at 512
- Number of augmentations per student is 6
