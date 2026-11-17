import numpy as np

K_VALUE = 5

class KNN:
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.d: list[list] = []

    def fit(self, x, y):
        # Simpan seluruh data training
        self.x_train = np.asarray(x, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def predict(self, x:list):
        # Ubah data prediksi menjadi array NumPy
        x = np.asarray(x, dtype=float)

        # Hitung Manhattan distance ke seluruh data training
        distances = np.sum(np.abs(self.x_train - x), axis=1)

        # Jumlah neighbor
        neighbor_count = self.k

        # Gunakan seluruh indeks jika semua data menjadi tetangga
        if neighbor_count == len(self.x_train):
            nearest_indexes = np.arange(len(self.x_train))
        else:
            # Ambil indeks k jarak terkecil tanpa full sorting
            nearest_indexes = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]

        # Urutkan k-neigbouurs dengan jarak terdekat
        nearest_indexes = nearest_indexes[np.argsort(distances[nearest_indexes])]

        # Simpan pasangan jarak dan label neighbor
        self.d = [[distances[index], self.y_train[index]] for index in nearest_indexes]
        neighbor = self.d

        # Kalau ada yg sama ambil yg nilai y nya paling banyak
        labels = []
        for data in neighbor:
            labels.append(data[1])

        result = labels[0]
        max_label_count = 0

        for y in labels:
            label_count = labels.count(y)

            if label_count > max_label_count:
                max_label_count = label_count
                result = y

        return result

# Manhattan Distance for data numerik
def numeric_distance(a, b):
    total = 0

    for i in range(len(a)):
        total += abs(a[i] - b[i])

    return total

# Distance for data kategorikal
def categorical_distance(a, b):
    total = 0

    for i in range(len(a)):
        if a[i] != b[i]:
            total += 1

    return total