import time

from MWMalgorithms import MatchingAlgorithms as mwma
from graph_generator import GraphGenerator as gg

def run_algorithm(func, graph):
    start = time.time()
    mp = func(graph)
    elapsed = time.time() - start
    weight = mwma.calculate_weight(mp, graph)
    return weight, elapsed

if __name__ == "__main__":
    start = time.time()
    WG = gg.generate_random_weighted_graph(100, 0.9, (1, 10))
    elapsed = time.time() - start
    print(f"随机图生成时间：{elapsed:.4f}秒")
    #gg.print_graph(WG)

    w1, t1 = run_algorithm(mwma.greedy_matching, WG)
    print(f"贪心算法运行时间：{t1:.4f}秒")
    print("最大匹配权重：", w1)

    w2, t2 = run_algorithm(mwma.path_growing_algorithm, WG)
    print(f"路径增长算法运行时间：{t2:.4f}秒")
    print("最大匹配权重：", w2)

    w3, t3 = run_algorithm(mwma.dynamic_programming_path_growth, WG)
    print(f"优化后的路径增长算法运行时间：{t3:.4f}秒")
    print("最大匹配权重：", w3)

    w4, t4 = run_algorithm(mwma.lam_max_weighted_matching, WG)
    print(f"preis_LAM算法运行时间：{t4:.4f}秒")
    print("最大匹配权重：", w4)

    w5, t5 = run_algorithm(mwma.suitor, WG)
    print(f"s算法运行时间：{t5:.4f}秒")
    print("最大匹配权重：", w5)