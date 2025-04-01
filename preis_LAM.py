class LAMMatcher:
    def __init__(self, graph):
        self.graph = graph
        self.vertices = list(graph.keys())
        self.U = set()  # 存储 (u, v) 且 u < v 的边
        # incident: 每个顶点对应其在 U 中的边集合
        self.incident = {v: set() for v in self.vertices}
        for u in self.vertices:
            for v, w in graph[u].items():
                if u < v:
                    edge = (u, v)
                    self.U.add(edge)
                    self.incident[u].add(edge)
                    self.incident[v].add(edge)
        self.R = set()
        self.MLAM = set()
        self.match = {}

    def weight(self, edge):
        u, v = edge
        return self.graph[u][v]

    def is_free(self, v):
        return v not in self.match

    def remove_edge(self, edge):
        if edge in self.U:
            self.U.remove(edge)
            u, v = edge
            self.incident[u].discard(edge)
            self.incident[v].discard(edge)

    def get_incident_edge(self, v):
        # 直接从 incident[v] 中取一个边，无需扫描全局 U
        if self.incident[v]:
            edge = next(iter(self.incident[v]))
            self.remove_edge(edge)
            return edge
        return None

    def try_match(self, edge):
        a, b = edge
        C_a, C_b = set(), set()
        # 用 while 循环，直到 a 和 b 都无更多邻边
        while self.is_free(a) and self.is_free(b) and (self.incident[a] or self.incident[b]):
            if self.is_free(a) and self.incident[a]:
                edge_ac = self.get_incident_edge(a)
                if edge_ac:
                    C_a.add(edge_ac)
                    if self.weight(edge_ac) > self.weight(edge):
                        self.try_match(edge_ac)
            if self.is_free(b) and self.incident[b]:
                edge_bd = self.get_incident_edge(b)
                if edge_bd:
                    C_b.add(edge_bd)
                    if self.weight(edge_bd) > self.weight(edge):
                        self.try_match(edge_bd)
            # 如果 a 和 b 的 incident 都为空，则退出循环
            if not self.incident[a] and not self.incident[b]:
                break

        # 根据 a, b 的匹配状态处理 C_a, C_b
        if (not self.is_free(a)) and (not self.is_free(b)):
            self.R.update(C_a)
            self.R.update(C_b)
        elif (not self.is_free(a)) and self.is_free(b):
            self.R.update(C_a)
            free_edges = {e for e in C_b if self.is_free(e[0] if e[1]==b else e[1])}
            self.R.update(C_b - free_edges)
            for e in free_edges:
                self.U.add(e)
                u, v = e
                self.incident[u].add(e)
                self.incident[v].add(e)
        elif (not self.is_free(b)) and self.is_free(a):
            self.R.update(C_b)
            free_edges = {e for e in C_a if self.is_free(e[0] if e[1]==a else e[1])}
            self.R.update(C_a - free_edges)
            for e in free_edges:
                self.U.add(e)
                u, v = e
                self.incident[u].add(e)
                self.incident[v].add(e)
        else:
            self.R.update(C_a)
            self.R.update(C_b)
            self.MLAM.add(edge)
            self.match[a] = b
            self.match[b] = a

    def run(self):
        while self.U:
            edge = self.U.pop()
            # 同时更新 incident 字典
            u, v = edge
            self.incident[u].discard(edge)
            self.incident[v].discard(edge)
            self.try_match(edge)
        return self.MLAM

# 封装成函数，输入为图的邻接表，输出为匹配边集合（每个边为 (u, v) 且 u < v）
def lam_max_weighted_matching(graph):
    matcher = LAMMatcher(graph)
    matching = matcher.run()
    return matching