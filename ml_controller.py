from fastapi import APIRouter, HTTPException, status
from models import Prediction_Input
from models import Prediction_Output
import tensorflow as tf


router = APIRouter()


@router.get('/ml',status_code=status.HTTP_201_CREATED, response_model=Prediction_Output)
def get_preds(p_id:int,p_powerhorse:float):
    # Cargar el modelo de la red neuronal
    dnn_model = tf.keras.models.load_model("modelo.keras")

    # Datos de entrada 
    new_data = [[p_powerhorse]]  

    # Convertir los datos de entrada a tensores de TensorFlow
    new_data_tensor = tf.constant(new_data, dtype=tf.float32)

    # Realizar la predicción utilizando el modelo de la red neuronal
    prediction = dnn_model.predict(new_data_tensor)
    prediction_dict = Prediction_Output(id=p_id,powerhorse=p_powerhorse,mpg=float(prediction[0,0]))

    return prediction_dict
