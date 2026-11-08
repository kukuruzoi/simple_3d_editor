from dataclasses import dataclass
import numpy as np

@dataclass
class MeshData():
    """
    Хранит геометрические данные
    """
    vertices: np.ndarray # координаты вершин
    faces: np.ndarray # треугольные грани, заданные индексами вершин.

    def __post_init__(self):
        # Координаты вершин: массив N x 3
        self.vertices = np.asarray(
            self.vertices, dtype=np.float64
        )

        # Грани: массив M x 3 (треугольники)
        raw_faces = np.asarray(self.faces)

        if not np.issubdtype(raw_faces.dtype, np.integer):
            raise TypeError("Индексы граней должны быть целыми числами")

        self.faces = raw_faces.astype(np.int64, copy=True)

        self._validate()

    def _validate(self):
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("Вершины должны иметь форму (N, 3)")

        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("Грани должны иметь форму (M, 3)")

        if not np.all(np.isfinite(self.vertices)):
            raise ValueError("Координаты вершин должны быть конечными")

        if len(self.faces) > 0:
            if np.any(self.faces < 0):
                raise ValueError("Индексы граней не могут быть отрицательными")

            if np.any(self.faces >= len(self.vertices)):
                raise ValueError("Грань ссылается на несуществующую вершину")