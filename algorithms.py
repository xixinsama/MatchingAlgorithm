import sys
from collections import deque

class MatchingAlgorithms:
    @staticmethod
    def hungarian_bipartite(graph, U, V):
        """匈牙利算法求解二分图最大匹配"""
        M = set()
        visited = {}
        
        def dfs(u):
            for v in graph[u]:
                if not visited.get(v, False):
                    visited[v] = True
                    if v not in match or dfs(match[v]):
                        match[v] = u
                        return True
            return False
        
        match = {}
        for u in U:
            visited = {v: False for v in V}
            dfs(u)
        return {(match[v], v) for v in match}

    @staticmethod
    def blossom_algorithm(graph):
        """Edmonds开花算法求解一般图最大匹配"""
        def find_augmenting_path():
            pass  # 完整实现需处理花收缩，此处因复杂度较高建议参考正式文献
        
        M = set()
        # 实际实现需要处理奇环收缩和展开的复杂逻辑
        return M

    @staticmethod
    def gabow_weighted(graph, weights):
        """Gabow带权匹配算法框架（简化版）"""
        nodes = list(graph.keys())
        y = {v: max(weights[e] for e in graph[v]) for v in nodes}
        M = set()
        # 完整实现需要构建等式子图和标号调整
        return M

    @staticmethod
    def greedy_matching(graph, weights):
        """贪心算法求解带权匹配（1/2近似）"""
        edges = set()
        for u in graph:
            for v in graph[u]:
                if (u, v) not in edges and (v, u) not in edges:
                    edges.add((u, v))
        edges = sorted(edges, key=lambda e: weights[e], reverse=True)
        M = set()
        covered = set()
        for e in edges:
            u, v = e
            if u not in covered and v not in covered:
                M.add(e)
                covered.update({u, v})
        return M

    @staticmethod
    def path_growing(graph, weights):
        """路径增长算法求解带权匹配（1/2近似）"""
        M1 = set()
        M2 = set()
        current_graph = {v: set(neighbors) for v, neighbors in graph.items()}
        i = 1
        
        while current_graph:
            # 选择度至少为1的顶点
            x = next((v for v in current_graph if current_graph[v]), None)
            if not x:
                break
            
            while current_graph[x]:
                # 找到当前顶点x的最大权重边
                max_weight = -1
                best_y = None
                for y in current_graph[x]:
                    # 处理边方向问题
                    e = (x, y) if (x, y) in weights else (y, x)
                    if weights[e] > max_weight:
                        max_weight = weights[e]
                        best_y = y
                
                # 更新匹配
                if i == 1:
                    M1.add((x, best_y))
                else:
                    M2.add((x, best_y))
                i = 3 - i
                
                # 删除顶点x
                del current_graph[x]
                # 从其他顶点的邻接表中移除x
                for v in current_graph:
                    if x in current_graph[v]:
                        current_graph[v].remove(x)
                x = best_y
        
        # 返回权重更大的匹配
        weight1 = sum(weights[e] for e in M1)
        weight2 = sum(weights[e] for e in M2)
        return M1 if weight1 >= weight2 else M2

    @staticmethod
    def preis_linear(graph, weights):
        """Preis线性时间1/2近似算法"""
        M = set()
        E = sorted(graph.items(), key=lambda x: weights[x[0]], reverse=True)
        remaining = set(E)
        while remaining:
            u = next(iter(remaining))[0]
            max_edge = max(graph[u], key=lambda v: weights[(u, v)])
            M.add((u, max_edge))
            remaining -= {e for e in remaining if u in e or max_edge in e}
        return M