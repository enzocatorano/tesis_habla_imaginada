"""
model.py
========
Modelos ESMB_BR (Ensemble of Statistical Models of Binary Classifiers) según
Bolaños y Rufiner.

Contiene dos modelos intrínsecamente diferentes:

  ESMB_BR_Binary
    - N clasificadores binarios especializados, operando en paralelo
    - Cada uno predice 0/1 para su clase específica ("¿es mi clase?")
    - Todos reciben los MISMOS datos (todas las clases mezcladas)
    - Fusión: lógica "uno de N" — solo acepta si exactamente 1 dice "sí",
      descarta si 0 o >1 dicen "sí"
    - Retorna: 0..n_classes-1 o -1 (descartado)

  ESMB_BR_Multiclass
    - Un único modelo entrenado con labels multiclase
    - Sin clasificadores paralelos, sin votación, sin fusión
    - Retorna: 0..n_classes-1 directamente
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


class ESMB_BR_Binary:
    """
    Ensamble de N clasificadores binarios especializados + fusión "uno de N".

    Parámetros
    ----------
    n_classes : int
        Número de clases (5 para vocales, 6 para comandos).
    learning_cycles : int
        Número de estimadores (ciclos de LogitBoost).
    learning_rate : float
        Learning rate para LogitBoost.
    max_depth : int
        Profundidad máxima de los árboles.
    semilla : int
        Semilla aleatoria.

    Notas
    -----
    Basado en fitcensemble de MATLAB con LogitBoost.
    Cada especialista es un GradientBoostingClassifier(loss='log_loss').
    """

    def __init__(
        self,
        n_classes: int = 5,
        learning_cycles: int = 11,
        learning_rate: float = 0.12,
        max_depth: int = 10,
        semilla: int = 42,
    ):
        self.n_classes = n_classes
        self.learning_cycles = learning_cycles
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.semilla = semilla
        self.modules = []

        for i in range(n_classes):
            model = GradientBoostingClassifier(
                loss="log_loss",
                n_estimators=learning_cycles,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=semilla + i,
            )
            self.modules.append(model)

    def fit(self, X, y):
        """
        Entrena los N clasificadores binarios especializados.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
            Labels 0-indexed (0..n_classes-1).
        """
        y = np.asarray(y)
        for c in range(self.n_classes):
            y_binary = (y == c).astype(int)
            self.modules[c].fit(X, y_binary)
        return self

    def predict(self, X) -> np.ndarray:
        """
        Predice usando la fusión "uno de N".

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Clase predicha (0..n_classes-1) o -1 si fue descartado.
            -1 = ningún clasificador o más de uno dijo "sí"
        """
        X = np.asarray(X)
        n_samples = X.shape[0]

        predictions = np.zeros((n_samples, self.n_classes))
        for c in range(self.n_classes):
            predictions[:, c] = self.modules[c].predict(X)

        final_predictions = np.full(n_samples, -1)

        activation_counts = np.sum(predictions, axis=1)

        valid_mask = activation_counts == 1
        final_predictions[valid_mask] = np.argmax(predictions[valid_mask], axis=1)

        return final_predictions

    def predict_proba_activations(self, X) -> np.ndarray:
        """
        Retorna la matriz de activaciones crudas (0/1) de cada especialista.

        Útil para análisis detallado de la fusión.

        Returns
        -------
        activations : np.ndarray, shape (n_samples, n_classes)
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        activations = np.zeros((n_samples, self.n_classes))
        for c in range(self.n_classes):
            activations[:, c] = self.modules[c].predict(X)
        return activations


class ESMB_BR_Multiclass:
    """
    Modelo único multiclase — sin clasificadores paralelos, sin votación.

    Un solo GradientBoostingClassifier entrenado con labels multiclase.
    Predice directamente la clase, sin fusión ni rechazos.

    Parámetros
    ----------
    n_classes : int
        Número de clases.
    learning_cycles : int
        Número de estimadores (ciclos de LogitBoost).
    learning_rate : float
        Learning rate.
    max_depth : int
        Profundidad máxima de los árboles.
    semilla : int
        Semilla aleatoria.
    """

    def __init__(
        self,
        n_classes: int = 5,
        learning_cycles: int = 11,
        learning_rate: float = 0.12,
        max_depth: int = 10,
        semilla: int = 42,
    ):
        self.n_classes = n_classes
        self.learning_cycles = learning_cycles
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.semilla = semilla

        self.model = GradientBoostingClassifier(
            loss="log_loss",
            n_estimators=learning_cycles,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=semilla,
        )

    def fit(self, X, y):
        """
        Entrena el modelo único multiclase.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
            Labels 0-indexed (0..n_classes-1).
        """
        self.model.fit(X, np.asarray(y))
        return self

    def predict(self, X) -> np.ndarray:
        """
        Predice la clase directamente.

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Clase predicha (0..n_classes-1). Sin -1, sin rechazos.
        """
        return self.model.predict(np.asarray(X))
