"""
Script to insert 10 sample groups into the database
Run this script from the database directory
"""

from settings import engine, SessionLocal, Base
from models import Group

# Create all tables if they don't exist
Base.metadata.create_all(bind=engine)

# Get a database session
db = SessionLocal()

try:
    print("=" * 50)
    print("Inserting 10 Groups into the Database")
    print("=" * 50)
    
    groups_data = [
        "Group A - Computer Science",
        "Group B - Information Technology",
        "Group C - Software Engineering",
        "Group D - Data Science",
        "Group E - Cybersecurity",
        "Group F - Cloud Computing",
        "Group G - Web Development",
        "Group H - Mobile Development",
        "Group I - Artificial Intelligence",
        "Group J - DevOps & Infrastructure",
    ]
    
    # Check if groups already exist and delete them first
    existing_count = db.query(Group).count()
    if existing_count > 0:
        print(f"Found {existing_count} existing groups. Deleting...")
        db.query(Group).delete()
        db.commit()
        print("✓ Deleted existing groups")
    
    # Insert new groups
    for group_name in groups_data:
        db_group = Group(group_name=group_name)
        db.add(db_group)
    
    db.commit()  # Commit all groups at once
    print("✓ Successfully committed 10 groups to database!")
    
    # Display all groups
    print("\n" + "=" * 50)
    print("All Groups in Database:")
    print("=" * 50)
    groups = db.query(Group).all()
    for group in groups:
        print(f"ID: {group.group_id} | Name: {group.group_name}")
    
    print("\n" + "=" * 50)
    print(f"Total groups: {len(groups)}")
    print("=" * 50)

except Exception as e:
    print(f"✗ Error: {e}")
    db.rollback()
finally:
    db.close()
