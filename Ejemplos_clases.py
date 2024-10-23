from sklearn.neural_network import MLPClassifier

# Conjunto de datos de ejemplo
X = [[0., 0.], [1., 1.]]
y = [0, 1]

# Configuración del MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

# Entrenamiento del modelo
clf.fit(X, y)

# Realizar predicciones con nuevos datos
clf.predict([[2., 2.], [-1., -2.],[3., 3] ])

# Ver las formas de las matrices de pesos
[coef.shape for coef in clf.coefs_]

# Predecir probabilidades para nuevas muestras
clf.predict_proba([[2., 2.], [1., 2.]])