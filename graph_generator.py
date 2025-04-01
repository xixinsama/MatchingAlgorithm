import random
from itertools import combinations

class GraphGenerator:
    @staticmethod
    def generate_random_bipartite(n_left, n_right, edge_prob=0.3):
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
    def print_graph(graph):
        for node in graph:
            print(f"{node}: {graph[node]}")