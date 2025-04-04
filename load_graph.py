# 文件处理和图数据加载示例
# 该代码用于从 COIL-RAG 数据集中加载图数据
# 该数据集包含边和权重信息，边存储在 COIL-RAG.edges 文件中，权重存储在 COIL-RAG.link_attrs 文件中
def load_COIL_RAG_graph(edges_filename, link_attrs_filename):
    """
    从 COIL-RAG.edges 文件和 COIL-RAG.link_attrs 文件中加载图数据，
    转换为字典格式：{ vertex: {neighbor: weight, ...}, ... }。

    参数：
      edges_filename: 存储边数据的文件名，每行格式为 "u,v"
      link_attrs_filename: 存储边属性（边权）的文件名，每行一个数值

    返回：
      graph: 表示无向图的字典
    """
    graph = {}
    edges = []
    
    # 读取边数据
    with open(edges_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        u = int(parts[0])
                        v = int(parts[1])
                        edges.append((u, v))
                    except ValueError:
                        continue

    # 读取边属性数据，转换为整数
    weights = []
    with open(link_attrs_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    weight = int(float(line)) # 将浮点数转换为整数，不能直接int()
                    weights.append(weight)
                except ValueError:
                    continue

    if len(weights) != len(edges):
        print("警告：边数与属性数不一致！边数 =", len(edges), "属性数 =", len(weights))

    # 构造图。由于为无向图，为避免重复（例如 1,2 与 2,1），只保留 u < v 的记录
    for ((u, v), w) in zip(edges, weights):
        if u > v:
            u, v = v, u

        if u in graph and v in graph[u]:
            continue

        if u not in graph:
            graph[u] = {}
        if v not in graph:
            graph[v] = {}

        graph[u][v] = w
        graph[v][u] = w

    return graph

""" 
if __name__ == "__main__":
    # 使用原始字符串避免转义问题
    edges_file = "D:\pyProj\matching\Algorithm\COIL-RAG\COIL-RAG.edges"
    link_attrs_file = "D:\pyProj\matching\Algorithm\COIL-RAG\COIL-RAG.link_attrs"
    
    graph = load_COIL_RAG_graph(edges_file, link_attrs_file)
    print(graph)
"""