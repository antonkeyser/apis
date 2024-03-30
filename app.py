import tensorflow as tf
#import pymongo
#from pymongo import MongoClient
import pandas as pd
import numpy as np
#import ssl

"""def get_connect_mongo():
    CONNECTION_STRING = "mongodb+srv://atlas:T6.HYX68T8Wr6nT@cluster0.enioytp.mongodb.net/?retryWrites=true&w=majority"
    # Establecer la conexión a MongoDB con SSL
    client = pymongo.MongoClient(CONNECTION_STRING, ssl=True)
    return client
"""

"""def get_connect_mongo():
    CONNECTION_STRING ="mongodb+srv://atlas:T6.HYX68T8Wr6nT@cluster0.enioytp.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(CONNECTION_STRING)
    return client"""

# Leer datos de drive a predecir
df = pd.read_csv(r"C:\Users\renzo\Documents\Datapath\apis\Pulsar.csv")

#Cargar el modelo de la red neuronal

dnn_model = tf.keras.models.load_model("modelo.keras")


#Lista anidada para representar el dato
#new_data = [[140.562500,55.683782,-0.234571,-0.699648,3.199833,19.110426,7.975532,74.242225]] 
new_data = np.array(df.iloc[:,:8])

#Convertir los datos de entrada a tensores de tensorflow
#new_data_tensor = tf.constant(new_data, dtype=tf.float32)

#Realizar la prediccion utilizando el modelo de la red neuronal
prediction = dnn_model.predict(new_data)

#Enviar predicciones
df_pred = pd.DataFrame(prediction, columns = ['CLASS_PRED'])

result_df = df.join(df_pred)

print("Datos de entrada: ", new_data)
print("Prediccion de Class: ", prediction)

"""
#Cargar a MongoDB
df_to_dict = result_df.to_dict("records")
connection = get_connect_mongo()
dbname = connection['retail_db']
dbname["pred_RACT"].drop()
dbname["pred_RACT"].insert_many(df_to_dict)
connection.close()
"""