import time

from algorithms import MatchingAlgorithms as ma
from graph_generator import GraphGenerator as gg
import preis_LAM

if __name__ == "__main__":
    """
    B = gg.generate_random_bipartite(3, 2, 0.5)
    gg.print_graph(B)
    
    WB = gg.generate_random_weighted_bipartite(5, 10, 0.5, (1, 1000))
    gg.print_graph(WB)

    G = gg.generate_random_graph(5, 0.4)
    gg.print_graph(G)
    """
    start = time.time()
    WG = gg.generate_random_weighted_graph(1000, 0.5, (1, 50))
    elapsed = time.time() - start
    print(f"随机图生成时间：{elapsed:.4f}秒")
    #gg.print_graph(WG)

    ## 贪心算法测试
    #start = time.time()
    #matching = ma.greedy_matching(WG)
    #elapsed = time.time() - start
    #print(f"贪心算法运行时间：{elapsed:.4f}秒")
    #print("贪心算法找到的最大匹配权重：", ma.calculate_weight(matching, WG))

    ## 路径增长算法测试
    start = time.time()
    matching_path = ma.path_growing_algorithm(WG)
    elapsed = time.time() - start
    print(f"运行时间：{elapsed:.4f}秒")
    print("最大匹配权重：", ma.calculate_weight(matching_path, WG))

    ## 改进的路径增长算法测试
    #start = time.time()
    #matching_path_improved = ma.greedy_matching(WG)
    #elapsed = time.time() - start
    #print(f"运行时间：{elapsed:.4f}秒")
    #print("最大匹配权重：", ma.calculate_weight(matching_path_improved, WG))
    
    ## Preis算法测试
    start = time.time()
    match_path_LAM = preis_LAM.lam_max_weighted_matching(WG)
    elapsed = time.time() - start
    print(f"Preis算法运行时间：{elapsed:.4f}秒")
    print("最大匹配权重：", ma.calculate_weight(match_path_LAM, WG))