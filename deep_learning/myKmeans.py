import numpy as np
import matplotlib.pyplot as plt

class Cluster:
    def __init__(self, clsnum, centeroid):
        self.clsnum: int = clsnum
        self.centeroid: np.array = centeroid
        self.points = np.array([centeroid])

    def update_centeroid(self):
        if len(self.points) > 0:
            temp_centeroid = np.mean(self.points, axis=0)
            distances = np.linalg.norm(self.points - temp_centeroid, axis=1)
            center_idx = np.argmin(distances)
            self.centeroid = self.points[center_idx]


#随机初始化中心
def init_clusters(points, centeroid_num):
    indices = [np.random.randint(0, len(points) - 1) for i in range(centeroid_num)] #点序列中被取为簇中心的索引
    clusters = [Cluster(i, points[indices[i]]) for i in range(centeroid_num)]       #初始化簇类
    return clusters

#更新簇内的点
def update_clusters(points, clusters:list[Cluster]):
    for cluster in clusters:
        cluster.points = np.empty((0, 2))
    for i in range(len(points)):
        min_distance = (1 << 31) - 1
        for cluster in clusters:
            distance = np.linalg.norm(points[i] - cluster.centeroid)
            if min_distance > distance:
                min_distance = distance
                belong_to_cluster = cluster
        belong_to_cluster.points = np.vstack((belong_to_cluster.points, points[i]))
    for cluster in clusters:
        cluster.update_centeroid()

def centeroid_iteration(points, clusters, times):
    for i in range(times):
        update_clusters(points, clusters)
    return clusters

def myKmeans(points: np.array, clusternum: int, itertimes: int = 10):
    clusters = init_clusters(points, clusternum)
    clusters = centeroid_iteration(points, clusters, itertimes)
    all_points = []
    all_labels = []
    for cluster in clusters:
        if len(cluster.points) > 0:
            all_points.append(cluster.points)           #不同簇是分别同时插入簇的所有点，各簇的点数可能不同，不能直接转换为数组
            labels = np.ones(len(cluster.points)) * cluster.clsnum
            all_labels.append(labels)
    all_points = np.vstack(all_points)
    all_labels = np.hstack(all_labels)
    return all_points, all_labels, clusters

def print_centerpts():
    points = np.random.uniform(0, 5, (1000, 2))
    dummy1, dummy2, clusters= myKmeans(points, 10, 10)
    for cluster, i in zip(clusters, range(10)):
        print(f"centerX{i+1} = {cluster.centeroid[0]}, centerY{i+1} = {cluster.centeroid[1]}")


if __name__ == "__main__":
    points = np.random.uniform(0, 5, (1000, 2))
    all_points, all_labels, dummy = myKmeans(points, 10, 10)
    plt.scatter(all_points[:, 0], all_points[:, 1], s=18, c=all_labels)
    plt.colorbar()
    plt.show()
