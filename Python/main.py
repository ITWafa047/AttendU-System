# this is a main server file for the project, it will be used to run the server and handle requests


from StudentsManagement.register import router as register_router
from fastapi import FastAPI

app = FastAPI()
app.include_router(register_router, prefix="/register-student")
__main__ = "main"
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)