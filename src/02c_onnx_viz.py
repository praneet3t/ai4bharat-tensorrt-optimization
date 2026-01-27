import onnx
import os
import numpy as np
import matplotlib.pyplot as plt
from onnx import numpy_helper
from collections import Counter
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

model_path = r"C:\Users\apran\Videos\Cin\LIBRARY\ai4bharat-tensorrt-optimization\models\onnx\model.onnx"
report_dir = r"C:\Users\apran\Videos\Cin\LIBRARY\ai4bharat-tensorrt-optimization\reports"

os.makedirs(report_dir, exist_ok=True)

print(f"Loading ONNX model from {model_path}...")
model = onnx.load(model_path)
graph = model.graph

def get_shape_string(tensor):
    try:
        if not tensor.HasField("type"): return "Unknown"
        if not tensor.type.HasField("tensor_type"): return "Not Tensor"
        
        elem_type = tensor.type.tensor_type.elem_type
        type_str = str(TENSOR_TYPE_TO_NP_TYPE.get(elem_type, "Unknown"))
        
        dims = []
        if tensor.type.tensor_type.has_shape:
            for d in tensor.type.tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    dims.append(str(d.dim_value))
                elif d.HasField("dim_param"):
                    dims.append(d.dim_param)
                else:
                    dims.append("?")
        
        return f"{type_str} {dims}"
    except Exception:
        return "Error Parsing"

summary_path = os.path.join(report_dir, "model_summary_advanced.txt")
with open(summary_path, "w") as f:
    f.write(f"Model Producer: {model.producer_name}\n")
    f.write(f"Opset Version: {model.opset_import[0].version}\n")
    f.write(f"Graph Name: {graph.name}\n\n")
    
    f.write("--- INPUTS ---\n")
    for input in graph.input:
        f.write(f"{input.name}: {get_shape_string(input)}\n")
        
    f.write("\n--- OUTPUTS ---\n")
    for output in graph.output:
        f.write(f"{output.name}: {get_shape_string(output)}\n")

    f.write("\n--- NODE STATS ---\n")
    ops = [node.op_type for node in graph.node]
    for op, count in Counter(ops).most_common():
        f.write(f"{op}: {count}\n")

print(f"Saved summary to {summary_path}")

plt.figure(figsize=(14, 8))
labels, values = zip(*Counter(ops).most_common(20))
plt.bar(labels, values, color='#4e79a7')
plt.title(f'Top 20 Operators (Opset {model.opset_import[0].version})')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(report_dir, "operator_distribution.png"))
plt.close()

layer_stats = {}

print("Scanning Initializers...")
for initializer in graph.initializer:
    try:
        weight = numpy_helper.to_array(initializer)
        if weight.size > 0:
            layer_stats[initializer.name] = {
                'max': float(np.max(weight)),
                'min': float(np.min(weight))
            }
    except:
        continue

print("Scanning Constant Nodes (Hidden Weights)...")
for node in graph.node:
    if node.op_type == "Constant":
        for attr in node.attribute:
            if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                try:
                    weight = numpy_helper.to_array(attr.t)
                    if weight.size > 100: 
                        layer_stats[node.name] = {
                            'max': float(np.max(weight)),
                            'min': float(np.min(weight))
                        }
                except:
                    continue

if len(layer_stats) > 0:
    print(f"Found {len(layer_stats)} weight layers.")
    
    names = list(layer_stats.keys())
    max_vals = [v['max'] for v in layer_stats.values()]
    min_vals = [v['min'] for v in layer_stats.values()]

    sorted_indices = np.argsort(max_vals)[-50:]
    top_names = [names[i] for i in sorted_indices]
    top_max = [max_vals[i] for i in sorted_indices]
    top_min = [min_vals[i] for i in sorted_indices]

    plt.figure(figsize=(15, 8))
    plt.bar(top_names, top_max, label='Max', color='#e15759', alpha=0.7)
    plt.bar(top_names, top_min, label='Min', color='#59a14f', alpha=0.7)
    
    plt.axhline(y=65504, color='black', linestyle='--', linewidth=2, label='FP16 Limit')
    plt.axhline(y=-65504, color='black', linestyle='--', linewidth=2)
    
    plt.title('Weight Magnitudes (Check for FP16 Overflow)')
    plt.ylabel('Value')
    plt.xticks([]) 
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "weight_analysis_fixed.png"))
    plt.close()
    print("Saved weight_analysis_fixed.png")
else:
    print("❌ No weights found in graph. Model might be using external data references not loaded.")

print(f"✅ Reports generated in {report_dir}")