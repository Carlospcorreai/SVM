import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

# Cargar el conjunto de datos Iris
X, y = datasets.load_iris(return_X_y=True)

# Ver la forma de los datos
X.shape, y.shape

# Dividir los datos en conjuntos de entrenamiento y prueba, 
# reservando el 40% de los datos para la prueba (test_size=0.4).
# El parámetro random_state asegura que la división sea reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)

# Mostrar la forma (dimensiones) del conjunto de entrenamiento (X e y)
X_train.shape, y_train.shape

# Mostrar la forma (dimensiones) del conjunto de prueba (X e y)
X_test.shape, y_test.shape

# Entrenar un clasificador de máquina de soporte vectorial (SVM) con un kernel lineal.
# El parámetro C=1 regula el margen del clasificador.
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

# Calcular la puntuación de precisión del modelo en el conjunto de prueba.
clf.score(X_test, y_test)


# Importar la función cross_val_score para realizar validación cruzada
from sklearn.model_selection import cross_val_score

# Crear un clasificador SVM con un kernel lineal y un estado aleatorio fijo para reproducibilidad
clf = svm.SVC(kernel='linear', C=1, random_state=42)

# Realizar validación cruzada con 5 particiones (cv=5)
scores = cross_val_score(clf, X, y, cv=5)

# Mostrar las puntuaciones de precisión obtenidas en cada partición
scores

# Imprimir la precisión promedio y la desviación estándar de las puntuaciones obtenidas
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# Importar la función cross_validate para realizar validación cruzada con múltiples métricas
from sklearn.model_selection import cross_validate

# Importar la métrica de evaluación recall_score
from sklearn.metrics import recall_score

# Definir las métricas de evaluación que se utilizarán en la validación cruzada
scoring = ['precision_macro', 'recall_macro']

# Crear un clasificador SVM con un kernel lineal y un estado aleatorio fijo para reproducibilidad
clf = svm.SVC(kernel='linear', C=1, random_state=0)

# Realizar validación cruzada con las métricas definidas
scores = cross_validate(clf, X, y, scoring=scoring)

# Mostrar las claves del diccionario de resultados ordenadas
sorted(scores.keys())

# Mostrar las puntuaciones de recall macro obtenidas en cada partición
scores['test_recall_macro']

## Ejemplo de K-Fold con K=5

from sklearn.model_selection import KFold
import numpy as np

# Crear un conjunto de datos de ejemplo
X = np.arange(20)  # Datos de ejemplo

# Crear el objeto KFold con 5 particiones, sin barajar los datos y sin estado aleatorio fijo
kf = KFold(n_splits=5, shuffle=False, random_state=None)

# Iterar sobre cada partición (fold)
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}")
    print(f"Entrenamiento: {train_index}, Prueba: {test_index}\n")


import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np

# Crear un conjunto de datos de ejemplo
X = np.arange(20)  # Datos de ejemplo

# Crear el objeto KFold con 5 particiones, barajando los datos y con un estado aleatorio fijo
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Crear una figura y un eje para la visualización
fig, ax = plt.subplots()

# Iterar sobre cada partición (fold)
for i, (train_index, test_index) in enumerate(kf.split(X)):
    # Crear listas de índices para los conjuntos de entrenamiento y prueba
    y_train = [i + 0.5] * len(train_index)
    y_test = [i + 0.5] * len(test_index)
    
    # Graficar los índices de entrenamiento y prueba
    ax.scatter(train_index, y_train, c='blue', marker='s', s=100, label='Entrenamiento' if i == 0 else "")
    ax.scatter(test_index, y_test, c='red', marker='o', s=100, label='Prueba' if i == 0 else "")

# Configurar etiquetas y título del gráfico
ax.set_xlabel('Índice de muestra')
ax.set_yticks([i + 0.5 for i in range(5)])
ax.set_yticklabels([f'Fold {i+1}' for i in range(5)])
ax.set_title('Visualización de K-Fold Cross-Validation')
ax.legend()


# Crear un conjunto de datos de ejemplo
X = np.arange(10)  # Datos de ejemplo
y = np.random.randint(0, 2, size=10)  # Etiquetas de ejemplo (0 o 1)
groups = np.array([1, 1, 1, 1, 1, 1, 2, 3, 4, 5])  # Grupos de ejemplo

# Crear el objeto GroupKFold con 5 particiones
gkf = GroupKFold(n_splits=5)

# Iterar sobre cada partición (fold)
for train_index, test_index in gkf.split(X, y, groups):
    print(f"Entrenamiento: {train_index}, Prueba: {test_index}")



























## Ejemplo de K-Fold con K=5

from sklearn.model_selection import KFold
import numpy as np

# Crear un conjunto de datos de ejemplo
X = np.arange(20)  # Datos de ejemplo

# Crear el objeto KFold con 5 particiones, sin barajar los datos y sin estado aleatorio fijo
kf = KFold(n_splits=5, shuffle=True, random_state=1)

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np

# Crear un conjunto de datos de ejemplo
X = np.arange(20)  # Datos de ejemplo

# Crear el objeto KFold con 5 particiones, barajando los datos y con un estado aleatorio fijo
kf = KFold(n_splits=5, shuffle=False, random_state=None)

# Crear una figura y un eje para la visualización
fig, ax = plt.subplots()

# Iterar sobre cada partición (fold)
for i, (train_index, test_index) in enumerate(kf.split(X)):
    # Crear listas de índices para los conjuntos de entrenamiento y prueba
    y_train = [i + 0.5] * len(train_index)
    y_test = [i + 0.5] * len(test_index)
    
    # Graficar los índices de entrenamiento y prueba
    ax.scatter(train_index, y_train, c='blue', marker='s', s=100, label='Entrenamiento' if i == 0 else "")
    ax.scatter(test_index, y_test, c='red', marker='o', s=100, label='Prueba' if i == 0 else "")

# Configurar etiquetas y título del gráfico
ax.set_xlabel('Índice de muestra')
ax.set_yticks([i + 0.5 for i in range(5)])
ax.set_yticklabels([f'Fold {i+1}' for i in range(5)])
ax.set_title('Visualización de K-Fold Cross-Validation')
ax.legend()

# Crear un conjunto de datos de ejemplo
X = np.arange(10)  # Datos de ejemplo
y = np.random.randint(0, 2, size=10)  # Etiquetas de ejemplo (0 o 1)
groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])  # Grupos de ejemplo

# Crear el objeto GroupKFold con 5 particiones
gkf = GroupKFold(n_splits=5)

# Iterar sobre cada partición (fold)
for train_index, test_index in gkf.split(X, y, groups):
    print(f"Entrenamiento: {train_index}, Prueba: {test_index}")

from sklearn.model_selection import GroupKFold
import numpy as np

# Crear un conjunto de datos de ejemplo
X = np.arange(10)  # Datos de ejemplo
y = np.random.randint(0, 2, size=10)  # Etiquetas de ejemplo (0 o 1)
groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])  # Grupos de ejemplo

# Crear el objeto GroupKFold con 5 particiones
gkf = GroupKFold(n_splits=5)

# Iterar sobre cada partición (fold)
for train_index, test_index in gkf.split(X, y, groups):
    print(f"Entrenamiento: {train_index}, Prueba: {test_index}")

    import matplotlib.pyplot as plt
    from sklearn.model_selection import GroupKFold
    import numpy as np

    # Crear un conjunto de datos de ejemplo
    X = np.arange(10)  # Datos de ejemplo
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])  # Grupos de ejemplo

    # Crear el objeto GroupKFold con 5 particiones
    gkf = GroupKFold(n_splits=5)

    # Crear una figura y un eje para la visualización
    fig, ax = plt.subplots()

    # Iterar sobre cada partición (fold)
    for i, (train_index, test_index) in enumerate(gkf.split(X, groups=groups)):
        # Crear listas de índices para los conjuntos de entrenamiento y prueba
        y_train = [i + 0.5] * len(train_index)
        y_test = [i + 0.5] * len(test_index)
        
        # Graficar los índices de entrenamiento y prueba
        ax.scatter(train_index, y_train, c='blue', marker='s', s=100, label='Entrenamiento' if i == 0 else "")
        ax.scatter(test_index, y_test, c='red', marker='o', s=100, label='Prueba' if i == 0 else "")

    # Configurar etiquetas y título del gráfico
    ax.set_xlabel('Índice de muestra')
    ax.set_yticks([i + 0.5 for i in range(5)])
    ax.set_yticklabels([f'Fold {i+1}' for i in range(5)])
    ax.set_title('Visualización de GroupKFold')
    ax.legend()

    # Mostrar el gráfico
    plt.show()