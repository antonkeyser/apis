from fastapi import FastAPI, HTTPException, status

app = FastAPI()

app.include_router(router)
 