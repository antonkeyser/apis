from fastapi import APIRouter, HTTPException, status
from models import Task


router = APIRouter()

tasks_list = []


@router.get("/hola")
def saludo():
    return "Hola a todos"

@router.get('/tasks')
def get_tasks():
    return tasks_list 

@router.post('/tasks', status_code=status.HTTP_201_CREATED)
def create_tasks(task_object: Task):
    tasks_list.append(task_object)
    return task_object


@router.put('/tasks/{task_id}', status_code=203)
def update_tasks(task_id : int , task_object: Task):

    for index, item in enumerate(tasks_list):
        if item.id == task_id:
            tasks_list[index] = task_object
            return task_object
    else:
        HTTPException(status_code=404,detail="Task not found")


@router.delete('/tasks/{task_id}')
def delete_tasks(task_id:int):
    for index, item in enumerate(tasks_list):
        if item.id == task_id:
            tasks_list.pop(index)
            return {"message": "Task deleted"}
    else:
        HTTPException(status_code=404,detail="Task not found")



