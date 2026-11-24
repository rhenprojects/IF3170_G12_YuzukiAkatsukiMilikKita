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

        # Ambil label dengan jarak nol
        zero_distance_labels = []

        for data in neighbor:
            distance = data[0]
            label = data[1]

            if distance == 0:
                zero_distance_labels.append(label)

        # Jika ada jarak nol, gunakan label yang paling banyak
        if len(zero_distance_labels) > 0:
            result = zero_distance_labels[0]
            max_label_count = 0

            for label in zero_distance_labels:
                label_count = zero_distance_labels.count(label)

                if label_count > max_label_count:
                    max_label_count = label_count
                    result = label

            return result

        # Hitung total distance weight setiap label
        label_weights = {}

        for data in neighbor:
            distance = data[0]
            label = data[1]

            weight = 1 / distance

            if label not in label_weights:
                label_weights[label] = 0

            label_weights[label] += weight

        # Ambil label dengan total weight terbesar
        result = max(label_weights, key=label_weights.get)
        return result

# Manhattan Distance for data numerik
def numeric_distance(a, b):
    total = 0

    for i in range(len(a)):
        total += abs(a[i] - b[i])

    return total