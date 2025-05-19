import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('Real estate.csv')

print(df.columns)

print(df['Y house price of unit area'].median())

#crear columna price class
df.loc[df['Y house price of unit area'] > 38.45, 'price_class'] = 1
df.loc[df['Y house price of unit area'] <= 38.45, 'price_class'] = 0

print(df)

#Separar Características y Objetivo
#features: Todas las columnas excepto Y house price of unit area y price_class.
#target: Columna price_class.
features = df.drop(['Y house price of unit area', 'price_class'], axis=1)
target = df['price_class']

#Crear y Entrenar el Modelo
#Crear una instancia del DecisionTreeClassifier con random_state=12345.
#Entrenar el modelo usando features y target.

model = DecisionTreeClassifier(random_state=12345)
model.fit(features, target)

#Cargar el Dataset de Prueba
test_df = pd.read_csv('Real estate.csv')[:3]

#crear columna price class
test_df.loc[test_df['Y house price of unit area'] > 38.45, 'price_class'] = 1
test_df.loc[test_df['Y house price of unit area'] <= 38.45, 'price_class'] = 0

#Separar Características y Objetivo del Conjunto de Prueba
test_features = test_df.drop(['Y house price of unit area', 'price_class'], axis=1)
test_target = test_df['price_class']

#Realizar Predicciones
test_predictions = model.predict(test_features)

#Comparar Predicciones con Valores Reales

print('Predicciones:', test_predictions)
print('Respuestas correctas:', test_target.values)

def error_count(correct_answers, predictions):
    """
    Calcula el número de discrepancias entre las respuestas correctas y las predicciones del modelo.

    Parameters:
        correct_answers (array-like): Las respuestas correctas (valores reales del target).
        predictions (array-like): Las predicciones realizadas por el modelo.

    Returns:
        int: Número de discrepancias entre correct_answers y predictions.
    """
    discrepancies = (correct_answers != predictions).sum()  # Sumar las discrepancias
    return discrepancies

# Usar la función con las respuestas correctas y las predicciones
errors = error_count(test_target.values, test_predictions)

# Mostrar el resultado
print(f"Número de discrepancias: {errors}")

def accuracy(correct_answers, predictions):
    """
    Calcula la exactitud del modelo comparando las respuestas correctas con las predicciones.

    Parameters:
        correct_answers (array-like): Las respuestas correctas (valores reales del target).
        predictions (array-like): Las predicciones realizadas por el modelo.

    Returns:
        float: Puntuación de exactitud (entre 0 y 1).
    """
    correct_count = (correct_answers == predictions).sum()  # Número de respuestas correctas
    total_count = len(predictions)  # Total de predicciones realizadas
    return correct_count / total_count  # Exactitud

# Usar la función con las respuestas correctas y las predicciones
accuracy_score_before= accuracy(test_target.values, test_predictions)

# Mostrar el resultado
print(f"Exactitud: {accuracy_score_before:.2f}")

train_predictions = model.predict(features)
test_predictions = model.predict(test_features)

#Mostrar exactictud
print('Exactitud')
print('Training set:', accuracy_score(target, train_predictions))
print('Test set:', accuracy_score(test_target, test_predictions))


#Puedes establecer la profundidad de un árbol en sklearn con el parámetro max_depth:

# especificar la profundidad (ilimitado por defecto)
#model = DecisionTreeClassifier(random_state=12345, max_depth=3)

#model.fit(features, target)