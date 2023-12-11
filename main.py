from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/{name}{apellido}")
async def root(name: str,apellido: str):
    return {"message": f"Hello {name} {apellido}"}