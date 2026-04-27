"""
Script to insert courses into the database related to groups
Run this script from the database directory
Make sure to run insert_groups.py first to create groups
"""

from settings import engine, SessionLocal, Base
from models import Course, Group
from datetime import datetime, timedelta

# Create all tables if they don't exist
Base.metadata.create_all(bind=engine)

# Get a database session
db = SessionLocal()

try:
    print("=" * 50)
    print("Inserting Courses into the Database")
    print("=" * 50)
    
    # Course data organized by GROUP - 5 courses per group
    # Dictionary structure: group_index -> list of courses
    courses_by_group = {
        # Group A - Computer Science (5 courses)
        0: [
            {"name": "Data Structures", "code": "CS101", "description": "Fundamentals of data structures and algorithms"},
            {"name": "Web Development", "code": "CS102", "description": "Front-end and back-end web development"},
            {"name": "Object-Oriented Programming", "code": "CS103", "description": "OOP concepts and design principles"},
            {"name": "Database Fundamentals", "code": "CS104", "description": "Database design and SQL basics"},
            {"name": "Computer Networks", "code": "CS105", "description": "Network architecture and protocols"},
        ],
        # Group B - Information Technology (5 courses)
        1: [
            {"name": "Database Management", "code": "IT201", "description": "SQL and database design principles"},
            {"name": "Network Administration", "code": "IT202", "description": "Network setup and management"},
            {"name": "System Administration", "code": "IT203", "description": "Linux and Windows server administration"},
            {"name": "IT Infrastructure", "code": "IT204", "description": "IT infrastructure planning and deployment"},
            {"name": "Help Desk Support", "code": "IT205", "description": "User support and troubleshooting"},
        ],
        # Group C - Software Engineering (5 courses)
        2: [
            {"name": "Software Design Patterns", "code": "SE301", "description": "Design patterns and best practices"},
            {"name": "Project Management", "code": "SE302", "description": "Software project management methodologies"},
            {"name": "Software Testing", "code": "SE303", "description": "Testing strategies and quality assurance"},
            {"name": "Software Architecture", "code": "SE304", "description": "Designing scalable software systems"},
            {"name": "Agile Development", "code": "SE305", "description": "Agile methodologies and Scrum framework"},
        ],
        # Group D - Data Science (5 courses)
        3: [
            {"name": "Machine Learning Basics", "code": "DS401", "description": "Introduction to machine learning"},
            {"name": "Data Analysis", "code": "DS402", "description": "Statistical analysis and data visualization"},
            {"name": "Python for Data Science", "code": "DS403", "description": "Python programming for data science"},
            {"name": "Big Data Processing", "code": "DS404", "description": "Big data technologies and Hadoop"},
            {"name": "Deep Learning", "code": "DS405", "description": "Neural networks and deep learning models"},
        ],
        # Group E - Cybersecurity (5 courses)
        4: [
            {"name": "Network Security", "code": "CS501", "description": "Advanced network security concepts"},
            {"name": "Cryptography", "code": "CS502", "description": "Encryption and cryptographic algorithms"},
            {"name": "Ethical Hacking", "code": "CS503", "description": "Penetration testing and vulnerability assessment"},
            {"name": "Security Protocols", "code": "CS504", "description": "SSL/TLS and secure communications"},
            {"name": "Incident Response", "code": "CS505", "description": "Cybersecurity incident handling and response"},
        ],
        # Group F - Cloud Computing (5 courses)
        5: [
            {"name": "Cloud Fundamentals", "code": "CC601", "description": "Introduction to cloud computing concepts"},
            {"name": "AWS Services", "code": "CC602", "description": "Amazon Web Services platform and services"},
            {"name": "Azure Cloud", "code": "CC603", "description": "Microsoft Azure cloud services"},
            {"name": "Cloud Architecture", "code": "CC604", "description": "Designing cloud-based applications"},
            {"name": "DevOps on Cloud", "code": "CC605", "description": "DevOps practices in cloud environments"},
        ],
        # Group G - Web Development (5 courses)
        6: [
            {"name": "HTML & CSS", "code": "WD701", "description": "Web page structure and styling"},
            {"name": "JavaScript Basics", "code": "WD702", "description": "Client-side scripting with JavaScript"},
            {"name": "React.js", "code": "WD703", "description": "Building interactive UIs with React"},
            {"name": "Backend Development", "code": "WD704", "description": "Server-side development with Node.js"},
            {"name": "Web Security", "code": "WD705", "description": "Web application security and best practices"},
        ],
        # Group H - Mobile Development (5 courses)
        7: [
            {"name": "Android Development", "code": "MD801", "description": "Native Android app development"},
            {"name": "iOS Development", "code": "MD802", "description": "Native iOS app development with Swift"},
            {"name": "Flutter", "code": "MD803", "description": "Cross-platform mobile development with Flutter"},
            {"name": "React Native", "code": "MD804", "description": "Cross-platform apps with React Native"},
            {"name": "Mobile UI/UX", "code": "MD805", "description": "Mobile app design and user experience"},
        ],
        # Group I - Artificial Intelligence (5 courses)
        8: [
            {"name": "AI Fundamentals", "code": "AI901", "description": "Introduction to artificial intelligence concepts"},
            {"name": "Machine Learning", "code": "AI902", "description": "Advanced machine learning algorithms"},
            {"name": "Natural Language Processing", "code": "AI903", "description": "NLP and text processing techniques"},
            {"name": "Computer Vision", "code": "AI904", "description": "Image recognition and computer vision"},
            {"name": "Robotics AI", "code": "AI905", "description": "AI applications in robotics and automation"},
        ],
        # Group J - DevOps & Infrastructure (5 courses)
        9: [
            {"name": "Docker Containers", "code": "DO1001", "description": "Containerization with Docker"},
            {"name": "Kubernetes Orchestration", "code": "DO1002", "description": "Container orchestration with Kubernetes"},
            {"name": "CI/CD Pipelines", "code": "DO1003", "description": "Continuous integration and deployment"},
            {"name": "Infrastructure as Code", "code": "DO1004", "description": "IaC with Terraform and Ansible"},
            {"name": "Monitoring & Logging", "code": "DO1005", "description": "System monitoring, logging, and alerting"},
        ],
    }
    
    # Get all groups from database
    groups = db.query(Group).all()
    
    if not groups:
        print("✗ No groups found in database!")
        print("Please run insert_groups.py first to create groups.")
    else:
        print(f"✓ Found {len(groups)} groups in database")
        print("\n" + "=" * 50)
        print("Creating Courses for Each Group")
        print("=" * 50)
        
        # Check if courses already exist and delete them first
        existing_count = db.query(Course).count()
        if existing_count > 0:
            print(f"Found {existing_count} existing courses. Deleting...")
            db.query(Course).delete()
            db.commit()
            print("✓ Deleted existing courses\n")
        
        # Set base dates for courses
        base_start_date = datetime(2026, 9, 1)  # Start in September 2026
        base_end_date = datetime(2026, 12, 31)  # End in December 2026
        
        # Insert courses - 5 courses per group
        total_courses = 0
        for group_index, group in enumerate(groups):
            print(f"\nGroup {group_index + 1}: {group.group_name}")
            print(f"  Adding 5 courses for group_id = {group.group_id}")
            
            # Get courses for this specific group
            if group_index in courses_by_group:
                group_courses = courses_by_group[group_index]
                
                for course_num, course_info in enumerate(group_courses, 1):
                    # Calculate dates for each course (offset by weeks)
                    weeks_offset = (course_num - 1) * 2  # 2-week offset between courses
                    start_date = base_start_date + timedelta(weeks=weeks_offset)
                    end_date = base_end_date
                    
                    db_course = Course(
                        course_name=course_info["name"],
                        course_code=course_info["code"],
                        course_description=course_info["description"],
                        start_date=start_date,
                        end_date=end_date,
                        group_id=group.group_id  # ← Associate course with this group_id
                    )
                    db.add(db_course)
                    print(f"    ✓ Course {course_num}: {course_info['code']} - {course_info['name']}")
                    total_courses += 1
        
        db.commit()  # Commit all courses at once
        print(f"\n✓ Successfully committed {total_courses} courses to database!\n")
        
        # Display all courses with their groups
        print("=" * 100)
        print("All Courses in Database (Organized by Group):")
        print("=" * 100)
        
        for group in groups:
            print(f"\n📚 {group.group_name} (Group ID: {group.group_id})")
            print("-" * 100)
            
            # Get all courses for this specific group
            group_courses = db.query(Course).filter(Course.group_id == group.group_id).all()
            
            if group_courses:
                for course in group_courses:
                    print(f"  Course ID: {course.course_id} | Code: {course.course_code} | Name: {course.course_name}")
                    print(f"    Description: {course.course_description}")
                    print(f"    Start Date: {course.start_date.strftime('%Y-%m-%d')} | End Date: {course.end_date.strftime('%Y-%m-%d')}")
            else:
                print("  No courses found for this group")
        
        # Final summary
        all_courses = db.query(Course).all()
        print("\n" + "=" * 100)
        print(f"📊 SUMMARY: Total {len(all_courses)} courses across {len(groups)} groups")
        print(f"   Average: {len(all_courses) // len(groups)} courses per group")
        print("=" * 100)

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    db.rollback()
finally:
    db.close()
