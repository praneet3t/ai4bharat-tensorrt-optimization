import os
import onnx
import matplotlib.pyplot as plt

def get_model_info(onnx_path):
    # Get file size (including .data file if it exists)
    total_size = os.path.getsize(onnx_path)
    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        total_size += os.path.getsize(data_path)
    
    # Get node count
    model = onnx.load(onnx_path, load_external_data=False)
    node_count = len(model.graph.node)
    
    return total_size / (1024**2), node_count # Size in MB, Count

def visualize():
    # 1. Path Setup
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(root, "models", "onnx")
    report_dir = os.path.join(root, "reports", "onnx_plots")
    os.makedirs(report_dir, exist_ok=True)

    models = {
        "Original": os.path.join(model_dir, "model.onnx"),
        "Optimized": os.path.join(model_dir, "optimized_model.onnx"),
        "Quantized": os.path.join(model_dir, "quantized_model.onnx")
    }

    names, sizes, nodes = [], [], []

    print("\n--- Model Optimization Summary ---")
    for name, path in models.items():
        if os.path.exists(path):
            size_mb, node_cnt = get_model_info(path)
            names.append(name)
            sizes.append(size_mb)
            nodes.append(node_cnt)
            print(f"{name:10} | Size: {size_mb:7.2f} MB | Nodes: {node_cnt}")

    # 2. Plotting Size Comparison
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(names, sizes, color=['gray', 'blue', 'green'])
    plt.title("Model Size Comparison (MB)")
    plt.ylabel("Size (MB)")

    # 3. Plotting Node Count Comparison
    plt.subplot(1, 2, 2)
    plt.bar(names, nodes, color=['gray', 'blue', 'green'])
    plt.title("Node Count (Graph Complexity)")
    plt.ylabel("Number of Nodes")

    plt.tight_layout()
    plot_path = os.path.join(report_dir, "optimization_comparison.png")
    plt.savefig(plot_path)
    print(f"\n✅ Visualization saved to {plot_path}")

if __name__ == "__main__":
    visualize()