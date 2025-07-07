from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

# Datos sintéticos: [long_titulo, long_descripcion, precio]
X = np.random.rand(100, 3) * [100, 500, 10000]
y = np.random.randint(0, 2, 100)  # Clases: 0 = normal, 1 = sospechosa

# Entrenar modelo
clf = RandomForestClassifier()
clf.fit(X, y)

# Guardar modelo en disco
joblib.dump(clf, "modelo_fraude.pkl")

print("✅ Modelo generado y guardado como 'modelo_fraude.pkl'")
