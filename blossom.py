from graph_generator import GraphGenerator as gg

class BlossomContractor:
    def __init__(self):
        self.contractions = {}
        self.blossom_counter = 0

    def contract(self, graph, blossom):
        """将花收缩为一个新顶点（带权图版本）"""
        new_node = f"B{self.blossom_counter}"
        self.blossom_counter += 1
        contracted_graph = {}
        mappings = {}
        
        # 创建映射关系
        for node in graph:
            if node in blossom:
                mappings[node] = new_node
            else:
                mappings[node] = node
        
        # 构建新图结构（保留权重）
        for node in graph:
            current = mappings[node]
            if current not in contracted_graph:
                contracted_graph[current] = {}
            
            # 处理原始图中的边和权重
            for neighbor, weight in graph[node].items():
                mapped_neighbor = mappings[neighbor]
                if mapped_neighbor != current:
                    # 保留最小权重（根据算法需求调整合并策略）
                    if mapped_neighbor not in contracted_graph[current] or \
                       weight < contracted_graph[current][mapped_neighbor]:
                        contracted_graph[current][mapped_neighbor] = weight
                        # 确保对称性
                        if mapped_neighbor not in contracted_graph:
                            contracted_graph[mapped_neighbor] = {}
                        contracted_graph[mapped_neighbor][current] = weight
        
        self.contractions[new_node] = blossom
        return contracted_graph, new_node

    def lift_path(self, path, blossom_node):
        """展开路径时保留权重信息"""
        if not path:
            return []
        
        expanded_path = []
        for node in path:
            if node in self.contractions:
                expanded_path.extend(self.contractions[node])
            else:
                expanded_path.append(node)
        return expanded_path

def find_augmenting_path(G, M, contractor=None):
    if contractor is None:
        contractor = BlossomContractor()
    
    parent = {}
    root = {}
    distance = {}
    marked_edges = set()
    marked_vertices = set()
    
    # 初始化暴露顶点
    exposed = [v for v in G if v not in M]
    for v in exposed:
        parent[v] = None
        root[v] = v
        distance[v] = 0
    
    # 标记匹配边（带权图适配）
    for u in M:
        if u in M and M[u] in G[u]:
            edge = tuple(sorted((u, M[u])))
            marked_edges.add(edge)
    
    while True:
        # 找到未标记的偶数距离顶点
        candidates = [v for v in parent if v not in marked_vertices and distance.get(v, 0) % 2 == 0]
        if not candidates:
            break
        
        v = candidates[0]
        marked_vertices.add(v)
        
        # 遍历邻居（带权图适配）
        for w in G[v].keys():  # 只关注邻居节点，忽略权重
            if w == v:  # 跳过自环
                continue
            
            e = tuple(sorted((v, w)))
            if e in marked_edges:
                continue
            
            marked_edges.add(e)
            
            if w not in parent:
                if w not in M:
                    continue
                x = M[w]
                parent[w] = v
                root[w] = root[v]
                distance[w] = distance[v] + 1
                parent[x] = w
                root[x] = root[w]
                distance[x] = distance[w] + 1
                marked_edges.add(tuple(sorted((w, x))))
            else:
                if distance[w] % 2 == 1:
                    continue
                
                if root[v] != root[w]:
                    # 构造增广路径
                    path_v = []
                    current = v
                    while current is not None:
                        path_v.append(current)
                        current = parent[current]
                    path_v.reverse()
                    
                    path_w = []
                    current = w
                    while current is not None:
                        path_w.append(current)
                        current = parent[current]
                    
                    return path_v + path_w
                else:
                    # 处理花的收缩（带权图版本）
                    path_to_root = []
                    current = v
                    while current != root[v]:
                        path_to_root.append(current)
                        current = parent[current]
                    path_to_root.append(root[v])
                    
                    blossom = []
                    a = v
                    b = w
                    while a != b:
                        if distance[a] > distance[b]:
                            blossom.append(a)
                            a = parent[a]
                        else:
                            blossom.append(b)
                            b = parent[b]
                    blossom.append(a)
                    
                    G_contracted, blossom_node = contractor.contract(G, blossom)
                    M_contracted = {}
                    for node in M:
                        if node in blossom:
                            continue
                        mapped_node = blossom_node if node in blossom else node
                        if M[node] in blossom:
                            M_contracted[mapped_node] = blossom_node
                        else:
                            M_contracted[mapped_node] = M[node]
                    
                    P_prime = find_augmenting_path(G_contracted, M_contracted, contractor)
                    if not P_prime:
                        return []
                    
                    return contractor.lift_path(P_prime, blossom_node)
    return []

def find_maximum_matching(G, initial_M=None):
    M = initial_M.copy() if initial_M else {}
    while True:
        P = find_augmenting_path(G, M)
        if not P:
            break
        
        # 增广路径（带权图适配）
        new_M = {}
        # 保留不在路径上的匹配
        for u in M:
            if u not in P and M[u] not in P:
                new_M[u] = M[u]
                new_M[M[u]] = u
        
        # 添加新的匹配边
        for i in range(0, len(P)-1, 2):
            u = P[i]
            v = P[i+1]
            new_M[u] = v
            new_M[v] = u
        M = new_M
    return M

if __name__ == "__main__":
    # 测试带权图匹配
    weighted_graph = gg.generate_random_weighted_bipartite(
        n_left=3, 
        n_right=3,
        edge_prob=0.6,
        weight_range=(1, 5)
    )
    
    print("Weighted Graph Structure:")
    for node in weighted_graph:
        print(f"{node}: {weighted_graph[node]}")
    
    max_matching = find_maximum_matching(weighted_graph)
    print("\nMaximum Matching:", max_matching)