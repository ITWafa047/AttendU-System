from sqlalchemy import (
    Column,
    Enum,
    Integer,
    String,
    ForeignKey,
    DateTime,
    Boolean,
    Time,
    Date,
    JSON,
)
from database import Base
from sqlalchemy.orm import relationship


class Group(Base):
    __tablename__ = "groups"

    group_id = Column(Integer, primary_key=True, index=True)
    group_name = Column(String(100), unique=True)

    students = relationship("Student", back_populates="group")
    courses = relationship("Course", back_populates="group")


class Student(Base):
    __tablename__ = "students"

    student_id = Column(Integer, primary_key=True, index=True)
    student_name = Column(String(100))
    email = Column(String(100), unique=True)
    phone_number = Column(String(20))
    gender = Column(String(10))
    national_id = Column(String(20), unique=True)

    group_id = Column(Integer, ForeignKey("groups.group_id"))

    student_image = Column(String(255))
    registration_date = Column(DateTime)

    mean_embeddings = Column(JSON, nullable=True)  # New field for mean embeddings
    stack_embeddings = Column(JSON, nullable=True)  # New field for stack embeddings

    group = relationship("Group", back_populates="students")


class Course(Base):
    __tablename__ = "courses"

    course_id = Column(Integer, primary_key=True)
    course_name = Column(String(100))
    course_code = Column(String(50), unique=True)
    course_description = Column(String(255))
    start_date = Column(DateTime)
    end_date = Column(DateTime)

    group_id = Column(Integer, ForeignKey("groups.group_id"))

    group = relationship("Group", back_populates="courses")


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True)
    user_name = Column(String(100))
    email = Column(String(100), unique=True)
    phone_number = Column(String(20))
    gender = Column(String(10))
    role = Column(Enum("admin", "instructor"))
    password_hash = Column(String(255))


class InstructorCourse(Base):
    __tablename__ = "instructor_courses"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    course_id = Column(Integer, ForeignKey("courses.course_id"))


class SessionSchedule(Base):
    __tablename__ = "session_schedules"

    session_id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.course_id"))

    session_type = Column(String(20))
    day = Column(String(20))

    start_time = Column(Time)
    end_time = Column(Time)

    location = Column(String(100))
    session_date = Column(Date)
    session_status = Column(Boolean)
