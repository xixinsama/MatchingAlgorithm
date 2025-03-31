import time

import pandas as pd

from algorithms import MatchingAlgorithms as ma
from graph_generator import GraphGenerator as gg


def evaluate_algorithms():
    results = []
    
    # 测试参数配置
    test_cases = [
        {"type": "bipartite", "n1": 50, "n2": 50, "p": 0.3},
        {"type": "general", "n": 100, "p": 0.2}
    ]
    
    for case in test_cases:
        if case["type"] == "bipartite":
            graph, weights, U, V = gg.generate_bipartite(
                case["n1"], case["n2"], case["p"]
            )
        else:
            graph, weights = gg.generate_general(
                case["n"], case["p"]
            )
        
        # 评估匈牙利算法（仅二分图）
        if case["type"] == "bipartite":
            start = time.time()
            M = ma.hungarian_bipartite(graph, U, V)
            elapsed = time.time() - start
            results.append({
                "Algorithm": "Hungarian",
                "Graph Type": case["type"],
                "Time": elapsed,
                "Matching Size": len(M)
            })
        
        # 评估贪心算法
        start = time.time()
        M_greedy = ma.greedy_matching(graph, weights)
        elapsed = time.time() - start
        results.append({
            "Algorithm": "Greedy",
            "Graph Type": case["type"],
            "Time": elapsed,
            "Matching Weight": sum(weights[e] for e in M_greedy)
        })
        
        # 评估Preis算法
        start = time.time()
        M_preis = ma.preis_linear(graph, weights)
        elapsed = time.time() - start
        results.append({
            "Algorithm": "Preis",
            "Graph Type": case["type"],
            "Time": elapsed,
            "Matching Weight": sum(weights[e] for e in M_preis)
        })
    
        # 评估路径增长算法
        start = time.time()
        M_path = ma.path_growing(graph, weights)
        elapsed = time.time() - start
        results.append({
            "Algorithm": "Path Growing",
            "Graph Type": case["type"],
            "Time": elapsed,
            "Matching Weight": sum(weights[e] for e in M_path)
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = evaluate_algorithms()
    print(df.to_markdown(index=False))