from pydantic import BaseModel

class Task(BaseModel):
    id: int
    task: str
    completed: bool


class Prediction_Input(BaseModel):
    id: int
    powerhorse: float

class Prediction_Output(BaseModel):
    id: int
    powerhorse: float
    mpg: float