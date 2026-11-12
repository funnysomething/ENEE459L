import argparse
import json
from pathlib import Path

from assignment2 import Graph, Layer, arithmetic_intensity, load_graph


def build_synthetic_graph():
    # Example model with a skip connection and different dtypes.
    layers = [
        Layer(
            name="conv1", kind="conv",
            in_shape=(3, 32, 32), out_shape=(16, 32, 32),
            kernel=(3, 3), weight_dtype="fp16", act_dtype="fp16",
        ),
        Layer(
            name="bn1", kind="bn",
            in_shape=(16, 32, 32), out_shape=(16, 32, 32),
            weight_dtype="fp16", act_dtype="fp16",
        ),
        Layer(
            name="relu1", kind="relu",
            in_shape=(16, 32, 32), out_shape=(16, 32, 32),
            act_dtype="fp16",
        ),
        Layer(
            name="res_conv", kind="conv",
            in_shape=(16, 32, 32), out_shape=(16, 32, 32),
            kernel=(3, 3), reads=("relu1",),
            weight_dtype="fp16", act_dtype="fp16",
        ),
        Layer(
            name="add1", kind="add",
            in_shape=(16, 32, 32), out_shape=(16, 32, 32),
            reads=("relu1", "res_conv"), act_dtype="fp32",
        ),
        Layer(
            name="pool", kind="pool",
            in_shape=(16, 32, 32), out_shape=(16, 1, 1),
        ),
        Layer(
            name="flatten", kind="flatten",
            in_shape=(16, 1, 1), out_shape=(16,),
        ),
        Layer(
            name="fc", kind="linear",
            in_shape=(16,), out_shape=(10,), bias=True,
        ),
    ]
    return Graph("toy_resnet", (3, 32, 32), layers)


def main():
    parser = argparse.ArgumentParser(description="Assignment 2: arithmetic intensity")
    parser.add_argument("model", nargs="?", type=Path,
                        help="model JSON file; defaults to a synthetic graph")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("assignment2_results.json"))
    args = parser.parse_args()

    graph = load_graph(args.model) if args.model else build_synthetic_graph()
    results = arithmetic_intensity(graph)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        f.write("\n")

    print(f"Model: {graph.name}")
    print(f"Arithmetic intensity: {results['value']:.4f} FLOPs/byte")
    print(f"Results saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
