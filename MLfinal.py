import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

datos = pd.read_csv(r"C:\Users\renzo\Documents\Datapath\apis\Pulsar.csv")

# Calcular la cantidad de valores nulos por variable
missing_values_count = datos.isnull().sum()

# Filtrar las variables que tienen valores nulos
missing_values_count = missing_values_count[missing_values_count > 0]

# Verificar si hay valores nulos antes de visualizar
if not missing_values_count.empty:
    # Visualizar la cantidad de valores nulos por variable utilizando un gráfico de barras
    plt.figure(figsize=(10, 6))
    missing_values_count.plot(kind='bar', color='skyblue')
    plt.title('Cantidad de valores nulos por variable')
    plt.xlabel('Variables')
    plt.ylabel('Cantidad de valores nulos')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("No se encontraron valores nulos en el conjunto de datos.")


# Dividir el conjunto de datos en train_set y test_set
train_set, test_set = train_test_split(datos, test_size=0.2, random_state=42)

# Dividir el conjunto de entrenamiento en train_set y cross_validation_set
train_set, cross_validation_set = train_test_split(train_set, test_size=0.2, random_state=42)

# Calcular la correlación de Pearson entre todas las variables y la variable objetivo
correlation = datos.corr()['Class'].abs().sort_values(ascending=False)

# Excluir la variable objetivo ("Class")
correlation = correlation.drop('Class')

# Calcular la matriz de correlación entre todas las variables
correlation_matrix = datos.corr()

# Calcular la correlación de Pearson entre todas las variables y la variable objetivo
correlation_with_target = correlation_matrix['Class'].abs().sort_values(ascending=False)

# Excluir la variable objetivo ("Class")
correlation_with_target = correlation_with_target.drop('Class')

print("Correlación de las variables con la variable objetivo (Class):")
print(correlation_with_target)

# Calcular el factor de inflación de la varianza (VIF) para evaluar la multicolinealidad
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_data

# Calcular el VIF para todas las variables excepto la variable objetivo
variables_excluyendo_target = datos.drop('Class', axis=1)
vif_scores = calculate_vif(variables_excluyendo_target)

print("\nFactor de inflación de la varianza (VIF) para cada variable:")
print(vif_scores)

# Separar las características (X) y la variable objetivo (y)
X = datos.drop('Class', axis=1)
y = datos['Class']

# Estandarizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calcular el VIF para los componentes principales
vif_scores_pca = calculate_vif(pd.DataFrame(X_pca))

# Combinar los componentes principales con la variable objetivo
pca_df = pd.DataFrame(X_pca, columns=[f"Componente_{i+1}" for i in range(X_pca.shape[1])])
pca_df['Class'] = y

# Calcular la matriz de correlación
pca_correlation_matrix = pca_df.corr()

# Calcular la matriz de correlación entre los componentes principales y la variable objetivo
pca_target_correlation = pca_df.corr()['Class'].abs().sort_values(ascending=False)

# Excluir la variable objetivo ("Class")
pca_target_correlation = pca_target_correlation.drop('Class')


# Seleccionar las tres variables principales con mayor correlación con la variable objetivo
top_3_variables = pca_target_correlation.head(3).index.tolist()

# Inicializar el objeto MinMaxScaler
scaler = MinMaxScaler()

# Seleccionar las tres variables principales para normalizar
X_selected = pca_df[['Componente_1', 'Componente_2', 'Componente_7']]

# Normalizar las variables seleccionadas
X_normalized = scaler.fit_transform(X_selected)



# Separar las características (X) y la variable objetivo (y)
X_train = train_set.drop('Class', axis=1)
y_train = train_set['Class']
X_test = test_set.drop('Class', axis=1)
y_test = test_set['Class']




# Seleccionar las tres variables principales para normalizar
X_selected_normalized = X_selected.copy()  # Hacer una copia de las características seleccionadas
scaler = MinMaxScaler()

# Normalizar las características seleccionadas
X_selected_normalized[['Componente_1', 'Componente_2', 'Componente_7']] = scaler.fit_transform(X_selected_normalized[['Componente_1', 'Componente_2', 'Componente_7']])

# Convertir las series de pandas a matrices numpy
y_train = np.array(y_train)
y_test = np.array(y_test)

# Asegurarnos de que las etiquetas tengan la forma correcta
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Inicializar el objeto MinMaxScaler
scaler = MinMaxScaler()

# Normalizar las características seleccionadas
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Definir el modelo de red neuronal
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_normalized.shape[1],)),
    Dropout(0.5),  # Regularización por abandono para reducir el sobreajuste
    Dense(64, activation='relu'),
    Dropout(0.5),  # Regularización por abandono
    Dense(1, activation='sigmoid')  # Capa de salida para la clasificación binaria
])

# Compilar el modelo
model.compile(optimizer='adam',  # Método de optimización
              loss='binary_crossentropy',  # Función de pérdida para problemas de clasificación binaria
              metrics=['accuracy'])  # Métrica para evaluar el rendimiento del modelo

# Resumen del modelo
model.summary()

# Entrenar el modelo con los datos de entrenamiento
history = model.fit(X_train_normalized, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluar el modelo en los datos de prueba
test_loss, test_acc = model.evaluate(X_test_normalized, y_test)
print('Precisión del modelo en el conjunto de prueba:', test_acc)

# Obtener la pérdida y la precisión del historial de entrenamiento
loss = history.history['loss']
accuracy = history.history['accuracy']

# Obtener la pérdida y la precisión de la validación cruzada del historial de entrenamiento (si está disponible)
if 'val_loss' in history.history:
    val_loss = history.history['val_loss']
if 'val_accuracy' in history.history:
    val_accuracy = history.history['val_accuracy']

# Guardar el modelo
#model.save("modelo_tf.h5")
# Cargar el modelo
#model_loaded = tf.keras.models.load_model("modelo_tf.h5")

model.save("modelo.keras")
#model_loaded2 = tf.keras.models.load_model("modelo_tf.keras")