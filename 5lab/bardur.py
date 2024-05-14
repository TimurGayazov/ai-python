import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift
from sklearn.metrics import silhouette_score

# Загрузка данных
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
column_names = ["area", "perimeter", "compactness", "length_of_kernel", "width_of_kernel",
                "asymmetry_coefficient", "length_of_kernel_groove", "target"]
data = pd.read_csv(url, sep=r"\s+", header=None, names=column_names)

# Масштабирование признаков
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop('target', axis=1))

# Применение алгоритмов кластеризации
# это методы машинного обучения, которые используются для группировки схожих объектов в наборе данных в отдельные кластеры.
# Целью кластеризации является разделение данных на группы таким образом, чтобы объекты внутри одного кластера были более
# похожи друг на друга, чем на объекты из других кластеров.
#  в медицине — для классификации пациентов на основе симптомов и характеристик заболевания
algorithms = {
    "KMeans": KMeans(n_clusters=3, random_state=42), # +скорость на больших наборах данных -для работы знать кол-во кластеров
    "Affinity Propagation": AffinityPropagation(), # +для работы не знать кол-во кластеров -нужно много вычислений
    "Mean Shift": MeanShift() #+сам определяет центры кластеров -вычислительно затратный
}
# 1)1. выбор центроидов - случайным образом выбирает точка в пространстве данных, которые будут служить начальными центрами кластеров
# 1)2. после каждая точка данных присваивается к ближайшему к ней центроиду, образуя первичный кластер
# 1)3.Центроиды перемещаются в центры своих кластеров, вычисляя среднее значение всех точек в кластере. Повтор пока кластеры не перестанут изменяться

# 2)1.каждая точка данных отправ. сообщения всем другим точкам оценивая их схожесть, она определяется на основе расстояния м-ду точками
# 2)2.получаем: доступность и ответственность. Доступность - пок. на сколько привлек точка как центр кластера, а Ответ.-точка предпочитает др т в кач ц
# 2)3.экземпляры кластеров обноваляются, чтобы определить наилучшие центры кластеров


#3) aлгоритм кластеризации, который ищет центры кластеров, перемещаясь в направлении наибольшего увеличения плотности точек данных.

# Обучение и оценка алгоритмов
for name, algorithm in algorithms.items():
    model = algorithm.fit(scaled_features)
    labels = model.labels_
    silhouette_avg = silhouette_score(scaled_features, labels)
    print(f"{name}: Silhouette Score = {silhouette_avg}") #метрика оценки качества кластеризации, которая помогает определить, насколько хорошо объекты кластеризованы.
# ближе к 1, хорошо кластеризованы, ближе к 0, указ на нахождение объекта на границе м-ду кластерами, если отриц. то объект неправильно присвоен к кластеру
    # Визуализация кластеров
    plt.scatter(data['area'], data['perimeter'], c=labels, cmap='viridis')
    plt.title(f"{name} Clustering")
    plt.xlabel('Area')
    plt.ylabel('Perimeter')
    plt.show()

