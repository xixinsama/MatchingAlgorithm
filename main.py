import time

from MWMalgorithms import MatchingAlgorithms as mwma
from graph_generator import GraphGenerator as gg
import load_graph
from matplotlib import pyplot as plt
import networkx as nx
import copy

def exact_max_weight_matching(graph_input):
    """
    使用NetworkX的Blossom算法实现精确最大权匹配
    输入格式：{u: {v: weight, ...}, ...}
    输出格式：set of sorted tuples (u, v)
    
    """
    G = nx.Graph()
    for u in graph_input:
        for v, w in graph_input[u].items():
            if u < v:  # 避免重复添加边
                G.add_edge(u, v, weight=w)
    
    # 获取精确解（最大权匹配）
    exact_matching = nx.max_weight_matching(G, maxcardinality=False)
    
    # 转换为与你的算法相同的输出格式
    return set(tuple(sorted((u, v))) for u, v in exact_matching)

def run_algorithm(func, graph):
    start = time.time()
    mp = func(graph)
    elapsed = time.time() - start
    weight = mwma.calculate_weight(mp, graph)
    return mp, weight, elapsed

def is_valid_matching(matching):
    """
    检查匹配是否合法：
    1. 所有边互不相邻（无共同顶点）
    2. 边集合无重复
    """
    # 记录所有已使用的顶点
    used_nodes = set()
    for edge in matching:
        u, v = edge
        # 检查顶点是否已被占用
        if u in used_nodes or v in used_nodes:
            return False
        # 记录顶点
        used_nodes.add(u)
        used_nodes.add(v)
    return True


if __name__ == "__main__":
    start = time.time()
    #WG = gg.generate_random_weighted_graph(1000, 0.1, (1, 1000))
    WG = load_graph.load_COIL_RAG_graph("D:\\pyProj\\matching\\Algorithm\\COIL-RAG\\COIL-RAG.edges", "D:\\pyProj\\matching\\Algorithm\\COIL-RAG\\COIL-RAG.link_attrs")
    elapsed = time.time() - start
    print(f"随机图生成时间：{elapsed:.4f}秒")
    gg.print_graph_VE(WG)
    mp, w, t = run_algorithm(exact_max_weight_matching, WG)
    print(f"精确算法运行时间：{t:.4f}秒", "最大匹配权重：", w)
    #print(mp)
    print(is_valid_matching(mp))
    #gg.print_graph(WG)

    mp1, w1, t1 = run_algorithm(mwma.greedy_matching, WG)
    print(f"贪心算法运行时间：{t1:.4f}秒", "最大匹配权重：", w1)
    #print(mp1)
    print(is_valid_matching(mp1))

    mp2, w2, t2 = run_algorithm(mwma.path_growing_algorithm, WG)
    print(f"路径增长算法运行时间：{t2:.4f}秒", "最大匹配权重：", w2)
    #print(mp2)
    print(is_valid_matching(mp2))

    mp3, w3, t3 = run_algorithm(mwma.dynamic_programming_path_growth, WG)
    print(f"优化后的路径增长算法运行时间：{t3:.4f}秒", "最大匹配权重：", w3)
    #print(mp3)
    print(is_valid_matching(mp3))

    mp4, w4, t4 = run_algorithm(mwma.lam_max_weighted_matching, WG)
    print(f"preis_LAM算法运行时间：{t4:.4f}秒", "最大匹配权重：", w4)
    #print(mp4)
    print(is_valid_matching(mp4))

    mp5, w5, t5 = run_algorithm(mwma.suitor_matching, WG)
    print(f"s算法运行时间：{t5:.4f}秒", "最大匹配权重：", w5)
    #print(mp5)
    print(is_valid_matching(mp5))

    # 绘制运行时间对比图
    algorithms = ['Greedy', 'Path Growing', 'DP Path Growth', 'preis_LAM', 'suitor']
    times = [t1, t2, t3, t4, t5]
    weights = [w1-w, w2-w, w3-w, w4-w, w5-w]
    plt.figure(figsize=(6, 5))
    plt.bar(algorithms, times)
    plt.xlabel('algorithm')
    plt.ylabel('running time (s)')
    plt.title('comparison of running time of algorithms')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 绘制权重对比图
    # 只比较与贪心算法的权重的差异

    plt.figure(figsize=(6, 5))
    plt.bar(algorithms, weights)
    plt.xlabel('algorithm')
    plt.ylabel('weight difference')
    plt.title('comparison of maximum weight of algorithms')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()