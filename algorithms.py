import copy
import sys
from collections import deque, defaultdict

def _remove_vertex(vertex, graph):
    """从图中彻底删除顶点及其关联边"""
    if vertex in graph:
        # 删除顶点的所有邻接关系
        for neighbor in list(graph[vertex]):
            if neighbor in graph:
                del graph[neighbor][vertex]
        del graph[vertex]

class MatchingAlgorithms:    
    # 计算匹配权重的辅助函数
    @staticmethod
    def calculate_weight(matching, graph_input):
        total = 0
        for u, v in matching:
            if u in graph_input and v in graph_input[u]:
                total += graph_input[u][v]
            elif v in graph_input and u in graph_input[v]:
                total += graph_input[v][u]
        return total

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
    def gabow_weighted(graph, weights):
        """Gabow带权匹配算法框架（简化版）"""

    @staticmethod
    def greedy_matching(graph_input):
        """
        贪心算法求解最大权匹配问题
        
        参数:
        graph_input (dict): 邻接表表示的无向图，格式为 {u: {v: weight, ...}, ...}
        
        返回:
        set: 匹配的边集合，每个边表示为元组 (u, v) 且 u < v
        """
        graph = copy.deepcopy(graph_input)  # 深拷贝避免修改原图
        M = set()
        
        while True:
            # 查找当前权重最大的边
            max_weight = -1
            selected_edge = None
            for u in graph:
                for v in list(graph[u].keys()):
                    if u < v:  # 避免重复处理边
                        weight = graph[u][v]
                        if weight > max_weight:
                            max_weight = weight
                            selected_edge = (u, v)
            
            if selected_edge is None:
                break  # 没有剩余边
            
            M.add(selected_edge)
            u, v = selected_edge
            
            # 删除与u相连的所有边
            for w in list(graph[u].keys()):
                # 从邻接顶点的邻接表中删除u
                if w in graph and u in graph[w]:
                    del graph[w][u]
            graph[u].clear()  # 清空u的邻接表
            
            # 删除与v相连的所有边
            for w in list(graph[v].keys()):
                # 从邻接顶点的邻接表中删除v
                if w in graph and v in graph[w]:
                    del graph[w][v]
            graph[v].clear()  # 清空v的邻接表
        
        return M

    @staticmethod
    def linear_approx_matching(graph_input):
        """Preis的线性时间1/2近似算法实现"""
        # 初始化结构
        original_graph = {u: {v: w for v, w in adj.items()} for u, adj in graph_input.items()}
        U = set()
        for u in original_graph:
            for v in original_graph[u]:
                if u < v:
                    U.add((u, v))
        U = list(U)  # 转换为列表以便随机访问
        R = set()
        M = set()
        matched = set()  # 记录已匹配的顶点
        
        def is_free(u):
            return u not in matched
        
        def try_match(edge):
            a, b = edge
            local_Ca = set()
            local_Cb = set()
            
            while True:
                a_free = is_free(a)
                b_free = is_free(b)
                
                # 检查a的相邻边
                if a_free:
                    # 寻找a的未处理边
                    a_edges = [e for e in U if e[0] == a or e[1] == a]
                    for e in a_edges:
                        u, v = e
                        c = u if u != a else v
                        if e in U and e not in R and e not in local_Ca:
                            # 移动到local_Ca
                            local_Ca.add(e)
                            U.remove(e)
                            # 比较权重
                            if original_graph[a][c] > original_graph[a][b]:
                                try_match(e)
                
                # 检查b的相邻边（类似a的处理）
                if b_free:
                    b_edges = [e for e in U if e[0] == b or e[1] == b]
                    for e in b_edges:
                        u, v = e
                        d = u if u != b else v
                        if e in U and e not in R and e not in local_Cb:
                            local_Cb.add(e)
                            U.remove(e)
                            if original_graph[b][d] > original_graph[b][a]:
                                try_match(e)
                
                # 终止条件：a或b被匹配，或者没有更多边
                if not (a_free and b_free) or not (a_edges or b_edges):
                    break
            
            # 处理最终状态
            if not is_free(a) and not is_free(b):
                R.update(local_Ca)
                R.update(local_Cb)
            elif not is_free(a) and is_free(b):
                # 移动部分到R，部分回U
                R.update(local_Ca)
                for e in local_Cb:
                    u, v = e
                    d = u if u != b else v
                    if not is_free(d):
                        R.add(e)
                    else:
                        U.append(e)
            elif not is_free(b) and is_free(a):
                R.update(local_Cb)
                for e in local_Ca:
                    u, v = e
                    c = u if u != a else v
                    if not is_free(c):
                        R.add(e)
                    else:
                        U.append(e)
            else:
                # 添加当前边到匹配
                R.update(local_Ca)
                R.update(local_Cb)
                M.add((a, b))
                matched.add(a)
                matched.add(b)
        
        # 主循环
        while U:
            edge = U.pop()
            try_match(edge)
        
        return M




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

    """ 
    @staticmethod
    def improved_path_growing_algorithm(graph_input):
        original_graph = copy.deepcopy(graph_input)
        graph = copy.deepcopy(graph_input)
        original_edges = set()
        for u in original_graph:
            for v in original_graph[u]:
                if u < v:
                    original_edges.add((u, v))
        
        M_final = set()
        
        # 步骤1：路径生成与动态规划优化
        while True:
            # 检查是否还有边存在
            has_edges = any(len(neighbors) > 0 for neighbors in graph.values())
            if not has_edges:
                break
            
            # 选择起始顶点
            x = next((v for v in graph if graph[v]), None)
            if not x:
                break
            
            # 生成路径
            path = []
            current = x
            while True:
                if current not in graph or not graph[current]:
                    break
                
                # 选择最大权重边
                neighbors = graph[current]
                y = max(neighbors.keys(), key=lambda k: neighbors[k])
                edge = tuple(sorted((current, y)))
                path.append(edge)
                
                # 删除当前顶点
                _remove_vertex(current, graph)
                current = y
            
            if not path:
                continue
            
            # 动态规划求解路径最优匹配
            k = len(path)
            dp = [0] * (k + 1)
            selected = [set() for _ in range(k + 1)]
            
            if k >= 1:
                first_edge = path[0]
                dp[1] = original_graph[first_edge[0]][first_edge[1]]
                selected[1].add(first_edge)
            
            for i in range(2, k + 1):
                current_edge = path[i-1]
                weight = original_graph[current_edge[0]][current_edge[1]]
                
                if dp[i-1] >= dp[i-2] + weight:
                    dp[i] = dp[i-1]
                    selected[i] = selected[i-1].copy()
                else:
                    dp[i] = dp[i-2] + weight
                    selected[i] = selected[i-2].copy()
                    selected[i].add(current_edge)
            
            M_final.update(selected[k])
        
        # 步骤2：扩展为极大匹配
        vertices_in_matching = set()
        for u, v in M_final:
            vertices_in_matching.add(u)
            vertices_in_matching.add(v)
        
        for edge in original_edges:
            u, v = edge
            if (u not in vertices_in_matching and 
                v not in vertices_in_matching):
                M_final.add(edge)
                vertices_in_matching.add(u)
                vertices_in_matching.add(v)
        
        return M_final
    """

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