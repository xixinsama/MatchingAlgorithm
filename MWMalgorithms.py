import copy
import heapq
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

class MatchingAlgorithms:
    @staticmethod
    def calculate_weight(matching, graph):
        """计算匹配的总权重"""
        return sum(graph[u][v] for u, v in matching if u in graph and v in graph[u])

    @staticmethod
    def greedy_matching(graph_input):
        """改进的贪心算法（带局部优化）"""
        graph = copy.deepcopy(graph_input)
        matched = defaultdict(bool)
        matching = set()
        edges = []
        
        # 构建最大堆（按权重排序）
        for u in graph:
            for v, w in graph[u].items():
                if u < v:
                    heapq.heappush(edges, (-w, u, v))
        
        # 贪心选择阶段
        while edges:
            weight, u, v = heapq.heappop(edges)
            if not matched[u] and not matched[v]:
                matching.add((u, v))
                matched[u] = matched[v] = True
        
        # 局部优化阶段（2-opt改进）
        improved = True
        while improved:
            improved = False
            for u, v in list(matching):
                current_weight = graph[u][v]
                candidates = []
                
                # 收集可能的替换边
                for n in graph[u]:
                    if n != v and not matched[n]:
                        candidates.append((graph[u][n], u, n))
                for n in graph[v]:
                    if n != u and not matched[n]:
                        candidates.append((graph[v][n], v, n))
                
                # 寻找最优替换
                if candidates:
                    best_candidate = max(candidates, key=lambda x: x[0])
                    new_weight, a, b = best_candidate
                    if new_weight > current_weight:
                        matching.remove((u, v))
                        matching.add((a, b))
                        matched[u] = matched[v] = False
                        matched[a] = matched[b] = True
                        improved = True
        
        return matching

    @staticmethod
    def path_growing_algorithm(graph_input):
        # Create a deep copy of the graph to avoid modifying the original
        graph = {}
        for v in graph_input:
            graph[v] = {}
            for u in graph_input[v]:
                graph[v][u] = graph_input[v][u]

        M1 = set()
        M2 = set()
        i = 1

        def remove_vertex(x):
            # Remove the vertex x from the graph
            if x in graph:
                del graph[x]
            # Remove x from all other vertices' adjacency lists
            for v in list(graph.keys()):
                if x in graph[v]:
                    del graph[v][x]

        while True:
            # Find a vertex with at least one neighbor
            x = None
            for v in graph:
                if graph[v]:
                    x = v
                    break
            if x is None:
                break  # No more edges

            while True:
                # Check if x is still in the graph and has neighbors
                if x not in graph or not graph[x]:
                    break

                # Find the neighbor with the maximum edge weight
                neighbors = graph[x]
                y = max(neighbors.keys(), key=lambda k: neighbors[k])

                # Add the edge to the current matching
                current_set = M1 if i == 1 else M2
                edge = tuple(sorted((x, y)))
                current_set.add(edge)

                # Switch to the other matching
                i = 3 - i

                # Remove vertex x and its incident edges
                remove_vertex(x)

                # Move to vertex y
                x = y

        # Return the matching with the higher weight
        weight_m1 = MatchingAlgorithms.calculate_weight(M1, graph_input)
        weight_m2 = MatchingAlgorithms.calculate_weight(M2, graph_input)
        return M1 if weight_m1 >= weight_m2 else M2

    @staticmethod
    def improved_path_growing_algorithm(graph_input):
        """改进的路径增长算法实现（返回边集合）"""
        temp_graph = copy.deepcopy(graph_input)
        M_final = set()
        matched_nodes = set()

        # 步骤1：路径生成和动态规划
        while True:
            # 寻找仍有边的起始顶点
            start = None
            for node in temp_graph:
                if temp_graph[node]:
                    start = node
                    break
            if start is None:
                break

            # 构建路径（记录边和权重）
            path = []
            current = start
            while True:
                if current not in temp_graph or not temp_graph[current]:
                    break
                
                # 获取最大权重边
                max_neighbor, max_weight = None, -float('inf')
                for neighbor, weight in temp_graph[current].items():
                    if weight > max_weight:
                        max_weight = weight
                        max_neighbor = neighbor
                
                if max_neighbor is None:
                    break

                # 记录边（带权重用于动态规划）
                path.append((current, max_neighbor, max_weight))
                
                # 删除当前顶点及其边（保持无向图特性）
                del temp_graph[current]
                if max_neighbor in temp_graph:
                    del temp_graph[max_neighbor][current]
                
                current = max_neighbor

            # 动态规划处理路径
            if path:
                n = len(path)
                dp = [0] * (n + 1)
                selected = [[] for _ in range(n + 1)]
                
                # 初始化第一条边
                if n >= 1:
                    dp[1] = path[0][2]
                    selected[1] = [path[0]]

                # 动态规划递推
                for i in range(2, n+1):
                    option1 = dp[i-1]
                    option2 = dp[i-2] + path[i-1][2]
                    
                    if option1 > option2:
                        dp[i] = option1
                        selected[i] = selected[i-1]
                    else:
                        dp[i] = option2
                        selected[i] = selected[i-2] + [path[i-1]]

                # 添加最优边到匹配（仅记录节点对）
                for edge in selected[n]:
                    u, v, _ = edge
                    if u not in matched_nodes and v not in matched_nodes:
                        M_final.add((u, v))
                        matched_nodes.update([u, v])

        # 步骤2：扩展为极大匹配
        # 收集所有未处理的边（按权重降序）
        remaining_edges = []
        for u in graph_input:
            for v in graph_input[u]:
                if u < v and (u, v) not in M_final and (v, u) not in M_final:
                    remaining_edges.append((u, v))
        
        # 按权重降序排序
        remaining_edges.sort(key=lambda e: graph_input[e[0]][e[1]], reverse=True)

        # 添加不冲突的边
        for u, v in remaining_edges:
            if u not in matched_nodes and v not in matched_nodes:
                M_final.add((u, v))
                matched_nodes.update([u, v])

        return M_final
    
    @staticmethod
    def improved_path_growing_algorithm_optimized(graph_input):
        """优化后的改进路径增长算法（支持并行化）"""
        # 轻量级图副本（避免深拷贝）
        temp_graph = {u: dict(neighbors) for u, neighbors in graph_input.items()}
        M_final = set()
        matched_nodes = set()
        active_nodes = {u for u in temp_graph if temp_graph[u]}
        deleted_edges = set()  # 记录已删除边

        # 处理主循环
        while active_nodes:
            start = next(iter(active_nodes), None)
            if start not in temp_graph or not temp_graph[start]:
                active_nodes.discard(start)
                continue

            # 路径生成
            path = []
            current = start
            while True:
                if current not in temp_graph or not temp_graph[current]:
                    break
                max_neighbor, max_weight = max(
                    temp_graph[current].items(), 
                    key=lambda x: x[1], 
                    default=(None, -float('inf'))
                )
                if max_neighbor is None:
                    break

                path.append((current, max_neighbor, max_weight))
                deleted_edges.add((current, max_neighbor))
                deleted_edges.add((max_neighbor, current))

                # 动态删除边（更新active_nodes）
                del temp_graph[current]
                if max_neighbor in temp_graph:
                    del temp_graph[max_neighbor][current]
                    if not temp_graph[max_neighbor]:
                        del temp_graph[max_neighbor]
                current = max_neighbor
                active_nodes.discard(current)

            # 优化动态规划（滚动数组）
            if path:
                n = len(path)
                prev2, prev1 = 0, path[0][2] if n >=1 else 0
                selected_prev2, selected_prev1 = [], [path[0]] if n >=1 else []

                for i in range(2, n+1):
                    current_edge = path[i-1]
                    option1 = prev1
                    option2 = prev2 + current_edge[2]

                    if option1 > option2:
                        current_selected = selected_prev1
                        current_max = option1
                    else:
                        current_selected = selected_prev2 + [current_edge]
                        current_max = option2

                    prev2, prev1 = prev1, current_max
                    selected_prev2, selected_prev1 = selected_prev1, current_selected

                # 添加最优边
                for edge in selected_prev1:
                    u, v, _ = edge
                    if u not in matched_nodes and v not in matched_nodes:
                        M_final.add((u, v))
                        matched_nodes.update([u, v])

        # 极大匹配扩展（直接使用预存剩余边）
        remaining_edges = [
            (u, v) for u in graph_input for v in graph_input[u]
            if u < v and (u, v) not in deleted_edges
        ]
        remaining_edges.sort(key=lambda e: graph_input[e[0]][e[1]], reverse=True)

        for u, v in remaining_edges:
            if u not in matched_nodes and v not in matched_nodes:
                M_final.add((u, v))
                matched_nodes.update([u, v])

        return M_final

    @staticmethod
    def lam_max_weighted_matching(graph_input):
        """
        LAM算法求解最大权匹配问题
        
        参数:
        graph_input (dict): 邻接表表示的无向图，格式为 {u: {v: weight, ...}, ...}
        
        返回:
        set: 匹配的边集合，每个边表示为元组 (u, v) 且 u < v
        """
        vertices = list(graph_input.keys())
        U = set()  # 存储 (u, v) 且 u < v 的边
        incident = {v: set() for v in vertices}  # 每个顶点对应其在 U 中的边集合
        for u in vertices:
            for v, w in graph_input[u].items():
                if u < v:
                    edge = (u, v)
                    U.add(edge)
                    incident[u].add(edge)
                    incident[v].add(edge)
        R = set()
        MLAM = set()
        match = {}

        def weight(edge):
            u, v = edge
            return graph_input[u][v]

        def is_free(v):
            return v not in match

        def remove_edge(edge):
            if edge in U:
                U.remove(edge)
                u, v = edge
                incident[u].discard(edge)
                incident[v].discard(edge)

        def get_incident_edge(v):
            if incident[v]:
                edge = next(iter(incident[v]))
                remove_edge(edge)
                return edge
            return None

        def try_match(edge):
            a, b = edge
            C_a, C_b = set(), set()
            while is_free(a) and is_free(b) and (incident[a] or incident[b]):
                if is_free(a) and incident[a]:
                    edge_ac = get_incident_edge(a)
                    if edge_ac:
                        C_a.add(edge_ac)
                        if weight(edge_ac) > weight(edge):
                            try_match(edge_ac)
                if is_free(b) and incident[b]:
                    edge_bd = get_incident_edge(b)
                    if edge_bd:
                        C_b.add(edge_bd)
                        if weight(edge_bd) > weight(edge):
                            try_match(edge_bd)
                if not incident[a] and not incident[b]:
                    break

            if (not is_free(a)) and (not is_free(b)):
                R.update(C_a)
                R.update(C_b)
            elif (not is_free(a)) and is_free(b):
                R.update(C_a)
                free_edges = {e for e in C_b if is_free(e[0] if e[1] == b else e[1])}
                R.update(C_b - free_edges)
                for e in free_edges:
                    U.add(e)
                    u, v = e
                    incident[u].add(e)
                    incident[v].add(e)
            elif (not is_free(b)) and is_free(a):
                R.update(C_b)
                free_edges = {e for e in C_a if is_free(e[0] if e[1] == a else e[1])}
                R.update(C_a - free_edges)
                for e in free_edges:
                    U.add(e)
                    u, v = e
                    incident[u].add(e)
                    incident[v].add(e)
            else:
                R.update(C_a)
                R.update(C_b)
                MLAM.add(edge)
                match[a] = b
                match[b] = a

        while U:
            edge = U.pop()
            u, v = edge
            incident[u].discard(edge)
            incident[v].discard(edge)
            try_match(edge)

        return MLAM
