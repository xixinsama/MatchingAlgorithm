import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from MWMalgorithms import MatchingAlgorithms as ma
from graph_generator import GraphGenerator as gg

# 极大匹配验证函数（适配字典格式图）
def is_maximal_matching(graph, matching):
    all_edges = set()
    for u in graph:
        for v in graph[u]:
            if u < v:
                all_edges.add((u, v))
    
    matched = set()
    for u, v in matching:
        matched.update({u, v})
    
    for u, v in all_edges:
        if u not in matched and v not in matched:
            return False
    return True

# 实验参数配置
NODES_RANGE = [5, 20, 50, 100, 200]  # 节点数范围
EDGE_DENSITIES = [0.1, 0.3, 0.5, 0.7, 0.9]  # 边密度
WEIGHT_RANGES = [(1, 10), (1, 100), (1, 10000)]  # 权重范围
TRIALS = 3  # 每个配置重复次数

# 生成测试图
def generate_test_graphs():
    graphs = []
    for n, density, (w_min, w_max) in product(NODES_RANGE, EDGE_DENSITIES, WEIGHT_RANGES):
        graph = gg.generate_random_weighted_graph(n, density, (w_min, w_max))
        graphs.append({
            'nodes': n,
            'density': density,
            'weight_range': f"{w_min}-{w_max}",
            'graph': graph
        })
    return graphs

# 运行实验
def run_experiment():
    results = []
    test_graphs = generate_test_graphs()
    
    for config in test_graphs:
        n = config['nodes']
        density = config['density']
        w_range = config['weight_range']
        graph = config['graph']
        
        print(f"\nTesting n={n}, density={density:.1f}, weight_range={w_range}")
        
        for _ in range(TRIALS):
            # 运行算法
            start = time.time()
            mg_greedy = ma.greedy_matching(graph)
            t_greedy = time.time() - start
            
            start = time.time()
            mg_pga = ma.path_growing_algorithm(graph)
            t_pga = time.time() - start
            
            start = time.time()
            mg_ipga = ma.improved_path_growing_algorithm(graph)
            t_ipga = time.time() - start
            
            start = time.time()
            mg_preis = ma.lam_max_weighted_matching(graph)
            t_preis = time.time() - start
            
            # 计算指标
            weight_greedy = ma.calculate_weight(mg_greedy, graph)
            weight_pga = ma.calculate_weight(mg_pga, graph)
            weight_ipga = ma.calculate_weight(mg_ipga, graph)
            weight_preis = ma.calculate_weight(mg_preis, graph)
            
            maximal_greedy = is_maximal_matching(graph, mg_greedy)
            maximal_pga = is_maximal_matching(graph, mg_pga)
            maximal_ipga = is_maximal_matching(graph, mg_ipga)
            maximal_preis = is_maximal_matching(graph, mg_preis)
            
            # 存储结果
            results.append({
                'Nodes': n,
                'Density': density,
                'Weight_Range': w_range,
                'Algorithm': 'Greedy',
                'Time': t_greedy,
                'Weight': weight_greedy,
                'Maximal': maximal_greedy
            })
            
            results.append({
                'Nodes': n,
                'Density': density,
                'Weight_Range': w_range,
                'Algorithm': 'PGA',
                'Time': t_pga,
                'Weight': weight_pga,
                'Maximal': maximal_pga
            })
            
            results.append({
                'Nodes': n,
                'Density': density,
                'Weight_Range': w_range,
                'Algorithm': 'IPGA',
                'Time': t_ipga,
                'Weight': weight_ipga,
                'Maximal': maximal_ipga
            })
            
            results.append({
                'Nodes': n,
                'Density': density,
                'Weight_Range': w_range,
                'Algorithm': 'Preis',
                'Time': t_preis,
                'Weight': weight_preis,
                'Maximal': maximal_preis
            })
    
    return pd.DataFrame(results)

# 可视化函数
def plot_results(df):
    # 预处理：计算每个配置的平均值
    df_agg = df.groupby(['Nodes', 'Density', 'Weight_Range', 'Algorithm']).agg({
        'Time': 'mean',
        'Weight': 'mean',
        'Maximal': 'mean'
    }).reset_index()

    # 创建画布
    plt.figure(figsize=(25, 15))
    sns.set_context("notebook", font_scale=1.2)
    
    # 定义子图布局
    metrics = ['Time', 'Weight', 'Maximal']
    weight_ranges = df_agg['Weight_Range'].unique()
    
    # 为每个权重范围和指标创建子图
    for i, wr in enumerate(weight_ranges, 1):
        subset = df_agg[df_agg['Weight_Range'] == wr]
        
        # 时间对比
        plt.subplot(3, len(weight_ranges), i)
        sns.lineplot(data=subset, x='Nodes', y='Time', hue='Algorithm', palette='viridis')
        plt.title(f"Time (Weights {wr})")
        plt.yscale('log')  # 对数刻度处理大范围值
        
        # 权重对比
        plt.subplot(3, len(weight_ranges), i + len(weight_ranges))
        sns.barplot(data=subset, x='Nodes', y='Weight', hue='Algorithm', palette='viridis')
        plt.title(f"Weight (Weights {wr})")
        plt.yscale('log')  # 对数刻度处理大范围值
        
        # 极大匹配覆盖率
        plt.subplot(3, len(weight_ranges), i + 2*len(weight_ranges))
        sns.lineplot(data=subset, x='Nodes', y='Maximal', hue='Algorithm', palette='viridis')
        plt.title(f"Maximal Coverage (Weights {wr})")
        plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 统一图例
    plt.show()

if __name__ == "__main__":
    df = run_experiment()
    plot_results(df)