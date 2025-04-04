##基于 Erdős–Rényi 模型生成随机图
import math
import random
from itertools import combinations

class GraphGenerator:
    @staticmethod
    def generate_random_bipartite(n_left, n_right, edge_prob=0.3):
        """
        生成不带权随机二分图    
        """
        total = n_left + n_right
        graph = {i: set() for i in range(total)}
        left = range(n_left)
        right = range(n_left, total)
        has_edge = False
        
        for u in left:
            for v in right:
                if random.random() < edge_prob:
                    graph[u].add(v)
                    graph[v].add(u)
                    has_edge = True
        
        if not has_edge:
            u = random.choice(left)
            v = random.choice(right)
            graph[u].add(v)
            graph[v].add(u)
        
        return graph

    @staticmethod
    def generate_random_weighted_bipartite(n_left, n_right, edge_prob=0.3, weight_range=(1, 10)):
        """
        生成均匀分布权重的随机二分图，边存在的概率为 p。
        """
        total = n_left + n_right
        graph = {i: {} for i in range(total)}
        left = range(n_left)
        right = range(n_left, total)
        has_edge = False
        low, high = weight_range
        
        for u in left:
            for v in right:
                if random.random() < edge_prob:
                    weight = random.randint(low, high)
                    graph[u][v] = weight
                    graph[v][u] = weight
                    has_edge = True
        
        if not has_edge:
            u = random.choice(left)
            v = random.choice(right)
            weight = random.randint(low, high)
            graph[u][v] = weight
            graph[v][u] = weight
        
        return graph
    
    @staticmethod
    def generate_random_graph(n, edge_prob=0.3):
        """
        生成不带权随机无向图
        """
        graph = {i: set() for i in range(n)}
        has_edge = False
        
        for u, v in combinations(range(n), 2):
            if random.random() < edge_prob:
                graph[u].add(v)
                graph[v].add(u)
                has_edge = True
        
        if not has_edge:
            u, v = random.sample(range(n), 2)
            graph[u].add(v)
            graph[v].add(u)
        
        return graph

    @staticmethod
    def generate_random_weighted_graph(n=5, p=0.3, weight_range=(1, 10)):
        """
        生成均匀分布权重的随机无向图，边存在的概率为 p
        """
        graph = {i: {} for i in range(n)}
        has_edge = False
        low, high = weight_range
        
        for u, v in combinations(range(n), 2):
            if random.random() < p:
                weight = random.randint(low, high)
                graph[u][v] = weight
                graph[v][u] = weight
                has_edge = True
        
        if not has_edge:
            u, v = random.sample(range(n), 2)
            weight = random.randint(low, high)
            graph[u][v] = weight
            graph[v][u] = weight
        
        return graph

    @staticmethod
    def generate_random_weighted_graph_powerlaw(n=5, p=0.3, exponent=2.5, scale=1):
        """
        生成幂律分布权重的随机无向图。
        使用 random.paretovariate(exponent) 生成权重，并乘以 scale 调整尺度。
        """
        graph = {i: {} for i in range(n)}
        has_edge = False
        
        for u, v in combinations(range(n), 2):
            if random.random() < p:
                weight = int(random.paretovariate(exponent) * scale)
                graph[u][v] = weight
                graph[v][u] = weight
                has_edge = True
        
        if not has_edge:
            u, v = random.sample(range(n), 2)
            weight = int(random.paretovariate(exponent) * scale)
            graph[u][v] = weight
            graph[v][u] = weight
        
        return graph

    @staticmethod
    def generate_random_weighted_graph_normal(n=5, p=0.3, mu=10, sigma=2):
        """
        生成正态分布权重的随机无向图。
        若生成的权重小于等于 0，则取 1 保证权重为正。
        """
        graph = {i: {} for i in range(n)}
        has_edge = False
        
        for u, v in combinations(range(n), 2):
            if random.random() < p:
                weight = int(random.gauss(mu, sigma))
                if weight <= 0:
                    weight = 1
                graph[u][v] = weight
                graph[v][u] = weight
                has_edge = True
        
        if not has_edge:
            u, v = random.sample(range(n), 2)
            weight = int(random.gauss(mu, sigma))
            if weight <= 0:
                weight = 1
            graph[u][v] = weight
            graph[v][u] = weight
        
        return graph

    @staticmethod
    def generate_sparse_random_weighted_graph(n=5, weight_range=(1, 10)):
        """
        生成稀疏图：边的存在概率设置为 log(n)/n。
        """
        p = math.log(n) / n if n > 1 else 1
        return GraphGenerator.generate_random_weighted_graph(n, p, weight_range)

    @staticmethod
    def generate_dense_random_weighted_graph(n=5, weight_range=(1, 10)):
        """
        生成较稠密图：边的存在概率较高，这里默认设置为 0.8。
        """
        p = 0.8
        return GraphGenerator.generate_random_weighted_graph(n, p, weight_range)

    @staticmethod
    def print_graph(graph):
        """
        打印图的邻接表表示形式。
        """
        for node in graph:
            print(f"{node}: {graph[node]}")

    @staticmethod
    def print_graph_VE(graph):
        """
        打印图的节点数和边数
        图的最大权重和最小权重
        最大度数和最小度数
        """
        num_vertices = len(graph)
        num_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
        print(f"节点数: {num_vertices}, 边数: {num_edges}")
        weights = [weight for neighbors in graph.values() for weight in neighbors.values()]
        if weights:
            max_weight = max(weights)
            min_weight = min(weights)
            print(f"最大权重: {max_weight}, 最小权重: {min_weight}")
        else:
            print("无权图")
        degrees = [len(neighbors) for neighbors in graph.values()]
        if degrees:
            max_degree = max(degrees)
            min_degree = min(degrees)
            print(f"最大度数: {max_degree}, 最小度数: {min_degree}")
        else:
            print("无边图")
        return num_vertices, num_edges, max_weight, min_weight, max_degree, min_degree