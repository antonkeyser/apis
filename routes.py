# routes.py

from fastapi import APIRouter
from task_controller import router as tasks_router
from ml_controller import router as ml_router



router = APIRouter()

#Incluir el path de los routes

router.include_router(tasks_router)
router.include_router(ml_router)

