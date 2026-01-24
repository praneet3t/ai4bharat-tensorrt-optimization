import os
import onnx
import matplotlib.pyplot as plt

def get_stats(path):
    size = os.path.getsize(path) / (1024 * 1024)
    model = onnx.load(path)
    nodes = len(model.graph.node)
    return size, nodes

def visualize():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_dir = os.path.join(root, "reports", "onnx_plots")
    os.makedirs(report_dir, exist_ok=True)

    # Path to the three versions of our model
    files = {
        "Original": os.path.join(root, "models", "onnx", "model.onnx"),
        "Fused": os.path.join(root, "models", "onnx", "fused_model.onnx"),
        "Quantized": os.path.join(root, "models", "onnx", "quantized_model.onnx")
    }

    names, sizes, node_counts = [], [], []
    for name, path in files.items():
        if os.path.exists(path):
            s, n = get_stats(path)
            names.append(name)
            sizes.append(s)
            node_counts.append(n)

    # Plotting logic
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Model Size Comparison
    ax1.bar(names, sizes, color=['#e74c3c', '#3498db', '#2ecc71'])
    ax1.set_title("Model Size (MB)")
    ax1.set_ylabel("Megabytes")

    # Node Count Comparison (shows graph simplification)
    ax2.bar(names, node_counts, color=['#e74c3c', '#3498db', '#2ecc71'])
    ax2.set_title("Total Graph Nodes")
    ax2.set_ylabel("Number of Ops")

    plt.tight_layout()
    save_path = os.path.join(report_dir, "onnx_optimization_summary.png")
    plt.savefig(save_path)
    print(f"✅ Comparison plot saved to: {save_path}")

if __name__ == "__main__":
    visualize()