# Importar bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Cargar datos de transacciones de transporte
transport_data = pd.read_csv('transport_data.csv')

# Cargar datos meteorológicos
weather_data = pd.read_csv('weather_data.csv')

# Fusionar datos en función de la fecha o la ubicación
merged_data = pd.merge(transport_data, weather_data, on='fecha_comun')

# Preprocesamiento de datos
# (puedes agregar más pasos dependiendo de la naturaleza de tus datos)
X = merged_data[['hora', 'estacion_origen', 'temperatura', 'lluvia']]
y = merged_data['demanda_pasajeros']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar características (opcional, dependiendo del modelo)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Realizar predicciones en el conjunto de prueba
predictions = model.predict(X_test_scaled)

# Evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
