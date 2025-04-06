import copy
import heapq
from collections import defaultdict, deque

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
        # 使用深拷贝避免修改原图
        graph = copy.deepcopy(graph_input)
        M1 = set()
        M2 = set()
        weight_m1 = 0
        weight_m2 = 0
        i = 1
        
        # 使用集合维护有效顶点（存在边的顶点）
        active_vertices = {v for v in graph if graph[v]}
        
        def remove_vertex(x):
            """移除顶点并维护active_vertices集合"""
            if x not in graph:
                return
            
            # 删除x的所有邻边
            for y in list(graph[x].keys()):
                if y in graph:
                    del graph[y][x]
                    # 更新有效顶点集合
                    if not graph[y]:
                        active_vertices.discard(y)
            
            # 删除顶点本身
            del graph[x]
            active_vertices.discard(x)
        
        while active_vertices:
            # 任选一个有效顶点（快速获取）
            x = next(iter(active_vertices))
            
            # 严格遵循伪代码的双层循环结构
            while True:
                # 检查顶点有效性
                if x not in graph or not graph[x]:
                    break
                
                # 获取最大权重边
                y = max(graph[x], key=lambda k: graph[x][k], default=None)
                if not y:
                    break
                
                # 添加边到当前匹配集
                edge = tuple(sorted((x, y)))
                if i == 1:
                    M1.add(edge)
                    weight_m1 += graph_input[x][y]  # 从原始图获取权重
                else:
                    M2.add(edge)
                    weight_m2 += graph_input[x][y]
                
                # 切换匹配集（严格对应伪代码i := 3 - i）
                i = 3 - i
                
                # 删除当前顶点并移动指针
                remove_vertex(x)
                x = y  # 移动到下一个顶点
        
        return M1 if weight_m1 >= weight_m2 else M2


        """
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
    """

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
    
    @staticmethod
    def dynamic_programming_path_growth(graph_input):
        graph = copy.deepcopy(graph_input)
        M_final = set()

        def is_graph_empty(g):
            for u in g:
                if g[u]:
                    return False
            return True

        def select_vertex_with_degree_at_least_one(g):
            for u in g:
                if g[u]:
                    return u
            return None

        def remove_edges_connected_to(g, u):
            if u in g:
                for v in list(g[u].keys()):
                    if v in g and u in g[v]:
                        del g[v][u]
                del g[u]

        while not is_graph_empty(graph):
            x = select_vertex_with_degree_at_least_one(graph)
            if x is None:
                break
            P = []
            while True:
                max_weight = -float('inf')
                max_v = None
                current_edges = graph.get(x, {})
                for v in current_edges:
                    if current_edges[v] > max_weight:
                        max_weight = current_edges[v]
                        max_v = v
                if max_v is None:
                    break
                u_sorted = min(x, max_v)
                v_sorted = max(x, max_v)
                P.append((u_sorted, v_sorted, max_weight))
                remove_edges_connected_to(graph, x)
                x = max_v

            if P:
                k = len(P)
                take = [False] * k

                # 动态规划初始化
                dp_prev_prev = 0  # dp[0]
                if k >= 1:
                    dp_prev = P[0][2]
                    take[0] = True
                else:
                    dp_prev = 0

                for i in range(1, k):
                    current_weight = P[i][2]
                    if dp_prev_prev + current_weight > dp_prev:
                        new_dp = dp_prev_prev + current_weight
                        take[i] = True
                    else:
                        new_dp = dp_prev
                        take[i] = False
                    dp_prev_prev, dp_prev = dp_prev, new_dp

                # 回溯选中的边
                selected_edges = []
                i = k - 1
                while i >= 0:
                    if take[i]:
                        selected_edges.append(P[i])
                        i -= 2
                    else:
                        i -= 1

                # 更新匹配并删除相关边
                for edge in selected_edges:
                    u, v, _ = edge
                    M_final.add((u, v))
                    remove_edges_connected_to(graph, u)
                    remove_edges_connected_to(graph, v)

        # 处理剩余边
        for u in graph:
            for v in graph[u]:
                if u < v:
                    conflict = False
                    for (a, b) in M_final:
                        if u == a or u == b or v == a or v == b:
                            conflict = True
                            break
                    if not conflict:
                        M_final.add((u, v))

        return M_final
    
    @staticmethod
    def suitor_matching(graph_input):
        """SUITOR算法实现最大带权匹配"""
        graph = copy.deepcopy(graph_input)
        nodes = list(graph.keys())
        
        # 初始化数据结构
        suitor = {u: None for u in nodes}
        ws = {u: 0.0 for u in nodes}  # 记录每个顶点当前匹配边的权重
        
        # 预处理：按权重降序排列邻居列表（优化性能）
        for u in graph:
            # 添加顶点ID作为次要排序键（解决权重相等情况）
            neighbors = sorted(graph[u].items(), 
                            key=lambda x: (-x[1], x[0]))
            graph[u] = dict(neighbors)
        
        for u in nodes:
            current = u
            done = False
            
            while not done:
                current_partner = suitor[current]
                heaviest = ws[current]
                best_partner = None
                best_weight = 0
                
                # 遍历当前顶点的所有邻居（已按权重降序排列）
                for v, weight in graph[current].items():
                    # 检查是否满足两个条件（权重严格大于）
                    if weight > ws[v] and weight > best_weight:
                        best_partner = v
                        best_weight = weight
                        # 不立即break，继续寻找可能更优的（因ws动态变化）
                
                # 更新匹配关系
                if best_partner is not None:
                    displaced = suitor.get(best_partner)
                    
                    # 执行匹配更新
                    suitor[best_partner] = current
                    ws[best_partner] = best_weight
                    
                    if displaced is not None:
                        current = displaced  # 处理被替换的顶点
                        done = False
                    else:
                        done = True
                else:
                    done = True
        
        # 生成匹配集合（修正后的逻辑）
        matching = set()
        used = set()
        
        for u in nodes:
            if u not in used:
                partner = suitor[u]
                if partner is not None and suitor.get(partner) == u:
                    matching.add(tuple(sorted((u, partner))))
                    used.update({u, partner})
        
        return matching