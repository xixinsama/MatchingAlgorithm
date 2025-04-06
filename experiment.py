import csv
import time
from itertools import product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from graph_generator import GraphGenerator as gg
from MWMalgorithms import MatchingAlgorithms as mwma


# ---------------------------
# 精确匹配函数（使用 NetworkX Blossom 算法）
def exact_max_weight_matching(graph_input):
    """
    使用 NetworkX 的 Blossom 算法实现精确最大权匹配
    输入格式：{u: {v: weight, ...}, ...}
    输出格式：set of sorted tuples (u, v)
    """
    G = nx.Graph()
    for u in graph_input:
        for v, w in graph_input[u].items():
            if u < v:
                G.add_edge(u, v, weight=w)
    exact_matching = nx.max_weight_matching(G, maxcardinality=False)
    return set(tuple(sorted((u, v))) for u, v in exact_matching)

# ---------------------------
# 评估各算法
def evaluate_algorithms(graph, exact_solution):
    results = {}
    algorithms = {
        "Greedy": mwma.greedy_matching,
        "LAM": mwma.lam_max_weighted_matching,
        "Path-Growing": mwma.path_growing_algorithm,
        "Improved-Path": mwma.dynamic_programming_path_growth,
        "Suitor": mwma.suitor_matching,
    }
    
    optimal_weight = mwma.calculate_weight(exact_solution, graph)
    
    for name, algorithm in algorithms.items():
        start = time.time()
        matching = algorithm(graph)
        end = time.time()
        weight = mwma.calculate_weight(matching, graph)
        ratio = weight / optimal_weight if optimal_weight > 0 else 0
        # 计算权重差距比例
        gap = (optimal_weight - weight) / optimal_weight if optimal_weight > 0 else 0
        results[name] = {
            "weight": weight,
            "ratio": ratio,
            "time": end - start,
            "gap": gap
        }
    # 另外也返回 optimal_weight 以便保存数据
    return results, optimal_weight

# ---------------------------
# 单次实验运行：根据节点数、边密度、权重分布选择相应的生成函数
def run_single_experiment(n, p, weight_range=(1, 10), dist="uniform"):
    if dist == "uniform":
        graph = gg.generate_random_weighted_graph(n=n, p=p, weight_range=weight_range)
    elif dist == "powerlaw":
        graph = gg.generate_random_weighted_graph_powerlaw(n=n, p=p, exponent=2.5, scale=1)
    elif dist == "normal":
        graph = gg.generate_random_weighted_graph_normal(n=n, p=p, mu=10, sigma=2)
    else:
        raise ValueError("未知的权重分布类型")
    
    exact = exact_max_weight_matching(graph)
    return evaluate_algorithms(graph, exact)

# ---------------------------
# 进行多次试验，聚合统计数据
def run_experiments(num_trials, n, p, weight_range=(1, 10), dist="uniform"):
    trial_results = []
    optimal_weights = []
    for _ in range(num_trials):
        result, optimal_weight = run_single_experiment(n, p, weight_range, dist)
        trial_results.append(result)
        optimal_weights.append(optimal_weight)
    return trial_results, np.mean(optimal_weights)

def aggregate_results(results, alg_names):
    agg = {name: {"weight": [], "ratio": [], "time": [], "gap": []} for name in alg_names}
    for res in results:
        for name in alg_names:
            agg[name]["weight"].append(res[name]["weight"])
            agg[name]["ratio"].append(res[name]["ratio"])
            agg[name]["time"].append(res[name]["time"])
            agg[name]["gap"].append(res[name]["gap"])
    for name in agg:
        agg[name]["weight"] = np.mean(agg[name]["weight"])
        agg[name]["ratio"] = np.mean(agg[name]["ratio"])
        agg[name]["time"] = np.mean(agg[name]["time"])
        agg[name]["gap"] = np.mean(agg[name]["gap"])
    return agg

# ---------------------------
# 数据导出函数：保存所有实验数据到 CSV 文件
def export_experiment_data(data, filename):
    """
    data: list，每个元素是一个 dict，包含 'n', 'p', 各算法指标
    """
    fieldnames = ['n', 'p', 'Algorithm', 'weight', 'ratio', 'time', 'gap']
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in data:
            writer.writerow(record)
    print(f"数据已导出到 {filename}")

# ---------------------------
# 绘图函数：折线图、热力图、雷达图
def plot_line(metric_results, metric, ylabel, xlabel="Graph Size (n)"):
    plt.figure(figsize=(8,6))
    for name, data in metric_results.items():
        plt.plot(data["n"], data[metric], marker='o', label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs. {xlabel}")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_heatmap(data, title, xlabel, ylabel):
    plt.figure(figsize=(8,6))
    sns.heatmap(data, annot=True, fmt=".4f", cmap="viridis")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_radar(data, metrics, title):
    import pandas as pd
    df = pd.DataFrame(data).T  # 算法为行，指标为列
    norm_df = (df - df.min()) / (df.max() - df.min())  # 归一化
    categories = list(norm_df.columns)
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)
    
    for index, row in norm_df.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=index)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title, size=15, y=1.1)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.show()

# ---------------------------
# 主实验报告函数
def experiment_report():
    num_trials = 10
    weight_range = (1, 10)
    alg_names = ["Greedy", "LAM", "Path-Growing", "Improved-Path", "Suitor"]
    
    export_data = []  # 用于导出所有数据
    
    # 1. 性能随节点规模变化图（固定 p=0.1, 均匀分布）
    n_values = list(range(60, 2060, 100))
    fixed_p = 0.1
    scale_results = {name: {"n": [], "weight": [], "ratio": [], "time": [], "gap": []} for name in alg_names}
    
    print("Running experiments: Varying n, fixed p=0.1, uniform weights")
    for n in n_values:
        print(f"n = {n}")
        results, optimal_weight = run_experiments(num_trials, n, fixed_p, weight_range, dist="uniform")
        agg = aggregate_results(results, alg_names)
        for name in alg_names:
            scale_results[name]["n"].append(n)
            scale_results[name]["weight"].append(agg[name]["weight"])
            scale_results[name]["ratio"].append(agg[name]["ratio"])
            scale_results[name]["time"].append(agg[name]["time"])
            scale_results[name]["gap"].append(agg[name]["gap"])
            # 保存到导出数据中，每个实验一行
            export_data.append({
                "n": n,
                "p": fixed_p,
                "Algorithm": name,
                "weight": agg[name]["weight"],
                "ratio": agg[name]["ratio"],
                "time": agg[name]["time"],
                "gap": agg[name]["gap"]
            })
        print(f"n = {n}:")
        for name in alg_names:
            print(f"{name}: weight={agg[name]['weight']:.2f}, ratio={agg[name]['ratio']:.4f}, time={agg[name]['time']:.6f}, gap={agg[name]['gap']:.4f}")
    
    # 修改绘制匹配权重图，改为绘制 gap（匹配差距）
    plot_line(scale_results, "gap", "Relative Weight Gap (Optimal - Alg) / Optimal", xlabel="Graph Size (n)")
    plot_line(scale_results, "ratio", "Matching Ratio (Approximation Quality)", xlabel="Graph Size (n)")
    plot_line(scale_results, "time", "Running Time (seconds)", xlabel="Graph Size (n)")
    
    # 2. 边密度敏感性分析（固定 n=500, 均匀分布）
    p_values = [0.1, 0.5, 0.9]
    fixed_n = 500
    p_results = {name: {"p": [], "ratio": []} for name in alg_names}
    
    print("\nRunning experiments: Varying p, fixed n=500, uniform weights")
    for p in p_values:
        print(f"p = {p}")
        results, _ = run_experiments(num_trials, fixed_n, p, weight_range, dist="uniform")
        agg = aggregate_results(results, alg_names)
        for name in alg_names:
            p_results[name]["p"].append(p)
            p_results[name]["ratio"].append(agg[name]["ratio"])
            # 同时保存到导出数据
            export_data.append({
                "n": fixed_n,
                "p": p,
                "Algorithm": name,
                "weight": agg[name]["weight"],
                "ratio": agg[name]["ratio"],
                "time": agg[name]["time"],
                "gap": agg[name]["gap"]
            })
    
    # 构造一个 DataFrame 用于热力图展示
    import pandas as pd
    heat_data = {}
    for name in alg_names:
        heat_data[name] = p_results[name]["ratio"]
    df_heat = pd.DataFrame(heat_data, index=[str(p) for p in p_values])
    plot_heatmap(df_heat, "Matching Ratio vs. Edge Probability (n=500, uniform)", "Algorithm", "Edge Probability (p)")
    
    # 3. 雷达图：固定 n=500, p=0.1, 均匀分布下各算法指标对比
    print("\nRunning experiments: Radar chart for n=500, p=0.1, uniform weights")
    results, _ = run_experiments(num_trials, 500, 0.1, weight_range, dist="uniform")
    agg = aggregate_results(results, alg_names)
    # 为雷达图准备数据：选取 Ratio, Speed (1/time), 和 Gap
    radar_data = {}
    for name in alg_names:
        radar_data[name] = {
            "Ratio": agg[name]["ratio"],
            "Speed": 1.0 / agg[name]["time"] if agg[name]["time"] > 0 else 0,
            "Gap": agg[name]["gap"]
        }
    plot_radar(radar_data, metrics=["Ratio", "Speed", "Gap"], title="Algorithm Comparison (n=500, p=0.1)")
    
    # 4. 导出所有实验数据到 CSV 文件
    export_experiment_data(export_data, "experiment_results.csv")
    
if __name__ == "__main__":
    experiment_report()
