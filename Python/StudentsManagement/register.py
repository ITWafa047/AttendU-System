from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from upload.imageValidator import ImageValidator
from upload.faceProcessor import FaceProcessor
from database.models import Student
from database.settings import SessionLocal

router = APIRouter()
validator = ImageValidator()
processor = FaceProcessor()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/")
async def register_student(
    student_code: str = Form(...),
    full_name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    gender: str = Form(...),
    national_id: str = Form(...),
    group_id: int = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    exists = (
        db.query(Student)
        .filter(
            (Student.student_code == student_code)
            | (Student.national_id == national_id)
            | (Student.email == email)
        )
        .first()
    )
    if exists:
        raise HTTPException(status_code=400, detail="Student already registered.")

    await validator.validate_format(image)
    image_rgb = await validator.load_image(image)
    validator.size_validation(image_rgb)

    faces_info = validator.faces_detection(image_rgb)
    single_face = validator.single_face_validation(faces_info["faces"])

    validator.background_validation(image_rgb, single_face)

    aligned_face = validator.face_alignment(image_rgb, single_face)

    validator.blur_validation(aligned_face)
    validator.brightness_validation(aligned_face)


    mean_embedddings, stack_embeddings = processor.generate_embedding(aligned_face)


    new_student = Student(
        student_code=student_code,
        full_name=full_name,
        email=email,
        phone=phone,
        gender=gender,
        national_id=national_id,
        group_id=group_id,
        mean_embeddings=mean_embedddings.tolist(),  
        stack_embeddings=stack_embeddings.tolist(),  
    )

    db.add(new_student)
    db.commit()
    db.refresh(new_student)


    return {
        "status": "Success",
        "message": f"Student {full_name} registered successfully.",
        "student_id": new_student.id,
    }
