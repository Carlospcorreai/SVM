import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import time
import platform

# Cargar los datos desde una URL
df = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')

# Imprimir las primeras filas para revisar las columnas disponibles
print(df.head())

# Definir las columnas requeridas para el análisis
required_columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# Filtrar las columnas que están disponibles en el dataset
available_columns = [col for col in required_columns if col in df.columns]

# Seleccionar las columnas que existen en el dataset
data = df[available_columns]

# Manejar valores faltantes en las columnas 'Age' y 'Embarked'
if 'Age' in data.columns:
    data['Age'].fillna(data['Age'].median(), inplace=True)
if 'Embarked' in data.columns:
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Convertir variables categóricas en variables dummy (numéricas)
if 'Sex' in data.columns and 'Embarked' in data.columns:
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
elif 'Sex' in data.columns:
    data = pd.get_dummies(data, columns=['Sex'], drop_first=True)

# Separar los datos en conjuntos de entrenamiento y prueba
X = data.drop('Survived', axis=1)  # Las variables independientes
y = data['Survived']  # La variable dependiente (si sobrevivió o no)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------------
# ¿Qué es la validación cruzada K-Fold?
# -----------------------------------------------
# La validación cruzada K-Fold es una técnica de evaluación que se utiliza para estimar
# el rendimiento de un modelo de machine learning de manera más confiable. 
# En lugar de entrenar y evaluar el modelo solo una vez con una partición fija de 
# datos de entrenamiento y prueba, K-Fold divide los datos en "K" subconjuntos (llamados "folds").
#
# ¿Cómo funciona?
# - Los datos se dividen en K grupos (o folds) de tamaño similar.
# - En cada iteración, el modelo se entrena usando K-1 folds y se evalúa usando el fold restante.
# - Este proceso se repite K veces, y en cada iteración se usa un fold diferente como conjunto de prueba.
# - Finalmente, los resultados de las K evaluaciones se promedian para obtener una estimación más robusta
#   del rendimiento del modelo.
#
# Ventajas de la validación K-Fold:
# 1. Utiliza todos los datos tanto para entrenamiento como para prueba. Cada dato aparece
#    en un conjunto de prueba exactamente una vez y en un conjunto de entrenamiento K-1 veces.
# 2. Reduce el riesgo de sobreajuste (overfitting) o subajuste (underfitting), ya que el modelo
#    se evalúa en diferentes particiones de los datos.
# 3. Proporciona una estimación más confiable del rendimiento del modelo en comparación
#    con una simple partición de entrenamiento y prueba.
#
# En este código, estamos usando K=5, lo que significa que los datos se dividen en 5 grupos
# y el proceso de entrenamiento y prueba se repite 5 veces, con cada fold actuando como
# conjunto de prueba una vez.
# -----------------------------------------------

# Definir KFold Cross-Validation

# KFold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# -----------------------------------------------
# ¿Qué es platform.processor()?
# -----------------------------------------------
# La función platform.processor() es una función de la librería estándar de Python,
# que forma parte del módulo `platform`. Esta función devuelve una cadena (string)
# que describe el procesador de la máquina donde se está ejecutando el código.
#
# ¿Por qué usamos platform.processor()?
# - Es útil para obtener información sobre la arquitectura del hardware en el que 
#   se está ejecutando el programa.
# - A veces, el tipo de procesador puede influir en el tiempo de entrenamiento 
#   de los modelos, ya que diferentes procesadores tienen diferentes capacidades
#   de cálculo.
#
# ¿Qué devuelve?
# - Devuelve una cadena con una descripción del procesador. Por ejemplo:
#   - En sistemas Intel, puede devolver algo como 'Intel64 Family 6 Model 158 Stepping 10'.
#   - En sistemas basados en ARM, podría devolver 'armv7l'.
#
# En este código, usamos platform.processor() para registrar el tipo de procesador 
# utilizado en los resultados de la validación cruzada y poder evaluar cómo
# el rendimiento del modelo podría variar dependiendo del hardware.
# -----------------------------------------------

# Obtener información del procesador de la máquina
processor = platform.processor()

# -----------------------------------------------
# ¿Qué es un Pipeline en Scikit-learn?
# -----------------------------------------------
# El Pipeline es una herramienta poderosa de Scikit-learn que permite encadenar
# varios pasos de procesamiento y modelado en un solo flujo de trabajo.
#
# ¿Por qué usar un Pipeline?
# - Facilita la integración de diferentes pasos como el preprocesamiento de datos
#   (por ejemplo, escalado de características) y el entrenamiento del modelo en un solo paso.
# - Automatiza el proceso y garantiza que el mismo preprocesamiento se aplique tanto
#   a los datos de entrenamiento como a los datos de prueba.
# - Ayuda a evitar fugas de datos (data leakage), asegurando que el preprocesamiento
#   se ajuste solo a los datos de entrenamiento y no incluya información de los datos de prueba.
#
# Componentes principales de un Pipeline:
# 1. **Scaler (Escalado de datos)**: Algunas técnicas de machine learning, como
#    SVM o redes neuronales, requieren que las características estén escaladas
#    a una misma escala para un rendimiento óptimo.
#    La clase `StandardScaler` se utiliza para escalar los datos de forma que tengan
#    media 0 y desviación estándar 1.
# 2. **Modelo (Estimator)**: Después del preprocesamiento, se entrena un modelo de machine learning
#    en los datos escalados. Este modelo puede ser una regresión logística, un SVM, 
#    un árbol de decisión, etc.
#
# ¿Cómo funciona un Pipeline?
# - El Pipeline ejecuta todos los pasos secuencialmente: primero escala los datos y luego entrena el modelo.
# - Durante el entrenamiento, el escalador se ajusta a los datos de entrenamiento y el modelo aprende
#   a partir de esos datos.
# - Durante la validación (o en el uso en producción), los datos de entrada se escalan automáticamente
#   utilizando los parámetros calculados durante el entrenamiento (media y desviación estándar del scaler).
#
# Ejemplo:
# Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])
# En este ejemplo, los datos primero se escalan y luego se pasan al modelo de regresión logística.
#
# En este código estamos utilizando Pipeline para asegurarnos de que nuestros datos
# se escalan correctamente antes de pasar al modelo de machine learning.
# -----------------------------------------------

# Ejemplo de creación de un Pipeline:
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),  # Paso 1: Escalar los datos
#     ('model', LogisticRegression())  # Paso 2: Entrenar el modelo de regresión logística
# ])

# Modelo 1: Regresión Logística
logreg_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('model', LogisticRegression(max_iter=1000))
])
start_time = time.time()

# -----------------------------------------------
# ¿Qué es cross_val_score?
# -----------------------------------------------
# La función cross_val_score es una herramienta que proporciona Scikit-learn
# para realizar la validación cruzada en un modelo de machine learning.
#
# ¿Qué hace cross_val_score?
# - Esta función evalúa el modelo de forma automática utilizando K-Fold cross-validation.
# - Divide el conjunto de datos de entrenamiento en "K" pliegues (folds).
# - El modelo se entrena en K-1 pliegues y se evalúa en el pliegue restante (fold).
# - Este proceso se repite K veces (una vez por cada pliegue), asegurando que cada fold
#   se utilice como conjunto de prueba una vez.
#
# Parámetros importantes de cross_val_score:
# 1. **estimator**: El modelo o pipeline que estamos evaluando. Puede ser un modelo simple
#    (como una regresión logística o SVM) o un pipeline que incluya preprocesamiento (escalado).
# 2. **X, y**: Los datos que queremos usar para entrenar y probar el modelo (X = características, y = etiquetas).
# 3. **cv**: El número de pliegues (folds) en la validación cruzada (por ejemplo, K=5).
# 
# ¿Qué devuelve?
# - cross_val_score devuelve una lista con los puntajes (scores) obtenidos en cada iteración de la validación cruzada.
# - Cada elemento en esta lista es la puntuación del modelo en un fold específico (como la precisión, si es un problema de clasificación).
# - Podemos calcular el promedio de estas puntuaciones para obtener un estimado más robusto del rendimiento del modelo.
#
# Ejemplo:
# Si usamos K=5, cross_val_score entrenará y evaluará el modelo 5 veces, 
# y devolverá 5 puntajes (uno por cada pliegue). 
# Después, podemos promediar estos puntajes para obtener la precisión media del modelo.
# -----------------------------------------------

# Ejemplo de uso de cross_val_score:
# cv_scores = cross_val_score(pipeline, X_train, y_train, cv=kfold)

logreg_cv_scores = cross_val_score(logreg_pipeline, X_train, y_train, cv=kfold)
logreg_time = time.time() - start_time
logreg_results = {
    'Algoritmo': 'Regresión Logística',
    'Precisión Media (Validación Cruzada)': np.mean(logreg_cv_scores),
    'Tiempo de Entrenamiento (s)': round(logreg_time, 4),
    'Procesador': processor
}

# Modelo 2: Árbol de Decisión
dtree_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('model', DecisionTreeClassifier())
])
start_time = time.time()
dtree_cv_scores = cross_val_score(dtree_pipeline, X_train, y_train, cv=kfold)
dtree_time = time.time() - start_time
dtree_results = {
    'Algoritmo': 'Árbol de Decisión',
    'Precisión Media (Validación Cruzada)': np.mean(dtree_cv_scores),
    'Tiempo de Entrenamiento (s)': round(dtree_time, 4),
    'Procesador': processor
}

# Modelo 3: SVM
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('model', SVC(probability=True))
])
start_time = time.time()
svm_cv_scores = cross_val_score(svm_pipeline, X_train, y_train, cv=kfold)
svm_time = time.time() - start_time
svm_results = {
    'Algoritmo': 'SVM',
    'Precisión Media (Validación Cruzada)': np.mean(svm_cv_scores),
    'Tiempo de Entrenamiento (s)': round(svm_time, 4),
    'Procesador': processor
}

# Modelo 4: Red Neuronal
mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('model', MLPClassifier(max_iter=1000))
])
start_time = time.time()
mlp_cv_scores = cross_val_score(mlp_pipeline, X_train, y_train, cv=kfold)
mlp_time = time.time() - start_time
mlp_results = {
    'Algoritmo': 'Red Neuronal',
    'Precisión Media (Validación Cruzada)': np.mean(mlp_cv_scores),
    'Tiempo de Entrenamiento (s)': round(mlp_time, 4),
    'Procesador': processor
}

# Crear una tabla con los resultados de todos los modelos
results_df = pd.DataFrame([logreg_results, dtree_results, svm_results, mlp_results])
print("\nTabla de Desempeño de los Modelos:")
print(results_df)

# Entrenar y graficar el mejor modelo de árbol de decisión
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
plt.figure(figsize=(15,10))
plot_tree(dtree, feature_names=X.columns, class_names=['No Sobrevive', 'Sobrevive'], filled=True)
plt.show()

# Entrenar la red neuronal y evaluar su desempeño
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
print(classification_report(y_test, y_pred))

# Comparación de modelos en términos de precisión en datos de prueba
performance = {}
logreg_pipeline.fit(X_train, y_train)
performance['Regresión Logística'] = accuracy_score(y_test, logreg_pipeline.predict(X_test))

dtree_pipeline.fit(X_train, y_train)
performance['Árbol de Decisión'] = accuracy_score(y_test, dtree_pipeline.predict(X_test))

svm_pipeline.fit(X_train, y_train)
performance['SVM'] = accuracy_score(y_test, svm_pipeline.predict(X_test))

mlp_pipeline.fit(X_train, y_train)
performance['Red Neuronal'] = accuracy_score(y_test, mlp_pipeline.predict(X_test))

# Crear un DataFrame con la precisión en datos de prueba para cada modelo
performance_df = pd.DataFrame(list(performance.items()), columns=['Modelo', 'Precisión en Datos de Prueba'])
print("\nTabla de Precisión en Datos de Prueba:")
print(performance_df)

# Graficar la precisión de los diferentes modelos
plt.bar(performance.keys(), performance.values())
plt.title('Precisión de los Algoritmos')
plt.ylabel('Precisión')
plt.xticks(rotation=45)
plt.show()

# Curva ROC para el modelo de regresión logística
logreg_pipeline.fit(X_train, y_train)
logreg_probs = logreg_pipeline.predict_proba(X_test)[:,1]  # Probabilidades para la clase "Sobrevivió"
fpr, tpr, thresholds = roc_curve(y_test, logreg_probs)  # Calcular la curva ROC
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (área = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Regresión Logística')
plt.legend(loc="lower right")
plt.show()