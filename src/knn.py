K_VALUE = 5

class KNN:
    def __init__(self, k=K_VALUE):
        self.k = k
        self.x_train: list[list] = []
        self.y_train: list = []
        self.d: list[list] = []

    def fit(self, x: list, y: list):
        self.x_train.append(x)
        self.y_train.append(y)

    def predict(self, x:list):
        self.d = []

        # Hitung jarak antara data masukin ke list d 
        for i in range(len(self.x_train)):
            d: float = numeric_distance(self.x_train[i], x)
            self.d.append([d, self.y_train[i]])

        # Urutkan daru jarak yang paling kecil
        self.d.sort()

        # Ambil K nilai teratas
        neighbor = self.d[:self.k]

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

# Euqlidean Distance for data numerik
def numeric_distance(a, b):
    total = 0

    for i in range(len(a)):
        total += (a[i] - b[i]) ** 2

    return total ** 0.5

# Euqlidean Distance for data kategorikal
def categorical_distance(a, b):
    total = 0

    for i in range(len(a)):
        if a[i] != b[i]:
            total += 1

    return total