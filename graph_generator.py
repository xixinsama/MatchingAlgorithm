import random

class GraphGenerator:
    @staticmethod
    def generate_bipartite(n1, n2, p, weight_range=(1, 100)):
        """生成随机二分图"""
        U = [f"u{i}" for i in range(n1)]
        V = [f"v{j}" for j in range(n2)]
        edges = {}
        weights = {}
        for u in U:
            edges[u] = []
            for v in V:
                if random.random() < p:
                    edges[u].append(v)
                    edges[v] = edges.get(v, []) + [u]
                    weights[(u, v)] = random.randint(*weight_range)
                if random.random() < p:
                    edges[u].append(v)
                    edges[v] = edges.get(v, []) + [u]
                    weight = random.randint(*weight_range)
                    weights[(u, v)] = weight
                    weights[(v, u)] = weight  # 添加反向边权重
        return edges, weights, U, V

    @staticmethod
    def generate_general(n, p, weight_range=(1, 100)):
        """生成随机一般图"""
        nodes = [f"v{i}" for i in range(n)]
        edges = {v: [] for v in nodes}
        weights = {}
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    u, v = nodes[i], nodes[j]
                    edges[u].append(v)
                    edges[v].append(u)
                    weights[(u, v)] = random.randint(*weight_range)
                if random.random() < p:
                    edges[u].append(v)
                    edges[v] = edges.get(v, []) + [u]
                    weight = random.randint(*weight_range)
                    weights[(u, v)] = weight
                    weights[(v, u)] = weight  # 添加反向边权重
        return edges, weights