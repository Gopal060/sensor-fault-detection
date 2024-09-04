from fastapi import FastAPI

app = FastAPI()

@app.get("/")   
def print_info():
    return "Hey I'm FastAPI framework"