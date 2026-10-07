from __future__ import annotations

from typing import Any

from graph import (
    Graph,
    Layer,
    computed,
    dtype_bytes,
    is_answered,
    unknown,
)


FLOPS_PER_MAC = 2

# The conventions `to_flops` will honour by name. Anything else is unknown
# rather than an assumption, because the whole point of the parameter is that
# the caller has to say which one they mean.
FLOP_CONVENTIONS = {
    "mac_is_two_flops": FLOPS_PER_MAC,
    "mac_is_one_flop": FLOPS_PER_MAC // 2,
}

# Batch normalisation holds two learnable vectors per channel (scale and shift)
# and two non-learnable ones (running mean and variance). The first pair are
# parameters; the second pair are buffers. Both are in the file.
BN_PARAMS_PER_CHANNEL = 2
BN_BUFFERS_PER_CHANNEL = 2

# Buffers are kept in FP32 even when the weights are not. Halving them saves
# nothing worth having and a denormal running variance is a real failure mode.
BUFFER_DTYPE = "fp32"

# Below this many models there is no line to fit and no residual to report.
MIN_MODELS_FOR_FIT = 3

# Two floats are the same MAC count when they are the same integer. There is no
# tolerance here on purpose: MAC counts are integers, and a tolerance would let
# two genuinely different architectures be reported as tied.
TIE_EXACT = True


# ===========================================================================
# 1. How many numbers are stored
# ===========================================================================

def count_parameters(graph: Graph) -> dict[str, Any]:
    per_layer: dict[str, int] = {}

    for layer in graph:
        if layer.kind == "conv":
            if not layer.kernel or layer.groups < 1:
                return unknown(
                    f"{graph.name}: count_parameters",
                    f"{layer.name} is missing a convolution kernel or valid group count",
                )
            in_channels = layer.in_shape[0] if layer.in_shape else 0
            out_channels = layer.out_shape[0] if layer.out_shape else 0
            if (
                in_channels <= 0
                or out_channels <= 0
                or in_channels % layer.groups
                or out_channels % layer.groups
            ):
                return unknown(
                    f"{graph.name}: count_parameters",
                    f"{layer.name} has invalid channel dimensions for groups={layer.groups}",
                )
            weights = (
                out_channels
                * (in_channels // layer.groups)
                * layer.kernel[0]
                * layer.kernel[1]
            )
            per_layer[layer.name] = weights + (out_channels if layer.bias else 0)
        elif layer.kind == "linear":
            weights = layer.in_elements * layer.out_elements
            biases = layer.out_elements if layer.bias else 0
            per_layer[layer.name] = weights + biases
        elif layer.kind == "bn":
            if not layer.out_shape or layer.out_shape[0] <= 0:
                return unknown(
                    f"{graph.name}: count_parameters",
                    f"{layer.name} has no valid batch normalization channel dimension",
                )
            per_layer[layer.name] = BN_PARAMS_PER_CHANNEL * layer.out_shape[0]
        else:
            per_layer[layer.name] = 0

    return computed(
        sum(per_layer.values()),
        f"{graph.name}: {len(graph)} layers, shapes from the description",
        per_layer=per_layer,
        includes_bias=True,
        excludes_bn_buffers=True,
        bn_params_per_channel=BN_PARAMS_PER_CHANNEL,
    )


# ===========================================================================
# 2. What those numbers weigh, which is not the size of the file
# ===========================================================================


def model_size_bytes(graph: Graph) -> dict[str, Any]:
    """Bytes of stored tensors: parameters plus buffers, at their own dtypes.

    Lecture 04 slide 8 gives the formula as `#Parameters × bit width` and slide
    9 spends a page on why the file on disk is not that number. Three reasons,
    two of which this function has to get right:

      * a model is not stored in one dtype. `Layer.weight_dtype` is per layer
        and a network with FP16 weights and FP32 normalisation is completely
        ordinary. Multiplying a single total by a single bit width is the
        mistake, and on these four descriptions it is worth several per cent
      * buffers are in the file. Batch norm's running statistics are two
        vectors per channel that no optimiser ever touched, and they are still
        bytes you have to ship
      * the container is in the file too — the pickle framing, the state-dict
        keys, the archive directory. This function does *not* try to model
        that, and it says so in `container_overhead_excluded` rather than
        quietly letting the caller assume it did

    Returns a `computed` finding whose value is bytes, with the per-dtype
    breakdown that makes the first bullet checkable.
    """
    parameters = count_parameters(graph)
    if not is_answered(parameters):
        return parameters

    per_layer: dict[str, int | float] = {}
    per_dtype: dict[str, int | float] = {}
    buffer_bytes: int | float = 0

    for layer in graph:
        try:
            weight_bytes = dtype_bytes(layer.weight_dtype)
        except KeyError as exc:
            return unknown(
                f"{graph.name}: model_size_bytes",
                f"{layer.name}: {exc}",
            )

        parameter_count = parameters["per_layer"][layer.name]
        layer_bytes: int | float = parameter_count * weight_bytes
        per_dtype[layer.weight_dtype] = per_dtype.get(layer.weight_dtype, 0) + layer_bytes

        if layer.kind == "bn":
            buffer_count = BN_BUFFERS_PER_CHANNEL * layer.out_shape[0]
            try:
                buffer_size = buffer_count * dtype_bytes(BUFFER_DTYPE)
            except KeyError as exc:
                return unknown(f"{graph.name}: model_size_bytes", str(exc))
            layer_bytes += buffer_size
            buffer_bytes += buffer_size
            per_dtype[BUFFER_DTYPE] = per_dtype.get(BUFFER_DTYPE, 0) + buffer_size

        per_layer[layer.name] = layer_bytes

    return computed(
        sum(per_layer.values()),
        f"{graph.name}: per-layer dtypes, buffers at {BUFFER_DTYPE}",
        per_layer=per_layer,
        per_dtype=per_dtype,
        buffer_bytes=buffer_bytes,
        container_overhead_excluded=True,
        note="not the size of the file on disk; see the handout, Stage A step 3",
    )

# ===========================================================================
# 3. The memory nobody puts in the table
# ===========================================================================

def count_activations(graph: Graph) -> dict[str, Any]:
    """Total and peak activation footprint, in elements and in bytes.

    UNC COMP 790-150 Lec 2 p. 70 gives AlexNet as total 932,264 and peak
    440,928, and the two numbers answer two different questions. Total is what
    the whole forward pass produced. Peak is how much had to be resident at
    once, and peak is the one that decides whether the model runs.

    Peak is not `max(out_elements)`. Three things make it larger than that:

      * a layer's input is still resident while its output is being written.
        The live set at layer *i* contains both
      * a tensor consumed by a later layer stays resident in between. `add`
        layers name two inputs in `Layer.reads`, and the earlier one has been
        sitting in memory across every layer of the block. This is the residual
        connection and it is the single largest contributor to peak in
        ResNet-shaped networks
      * the network's own input is a tensor too

    The implementation is a liveness pass: work out the last layer that reads
    each tensor, then walk forward keeping a live set and taking the maximum of
    its total size. Anything simpler than that is wrong on any graph with a
    skip connection, and it is wrong quietly, in the direction that says the
    model fits.

    Returns a `computed` finding whose value is peak *bytes*, because bytes are
    what a memory budget is denominated in, with elements and the layer where
    the peak occurs alongside.
    """
    source = f"{graph.name}: liveness over {len(graph)} layers, input included"
    try:
        input_bytes = dtype_bytes(graph.precision)
        layer_bytes = {layer.name: dtype_bytes(layer.act_dtype) for layer in graph}
    except KeyError as exc:
        return unknown(source, str(exc))

    if any(dimension <= 0 for dimension in graph.input_shape):
        return unknown(source, "network input has an invalid shape")
    for layer in graph:
        if any(dimension <= 0 for dimension in layer.out_shape):
            return unknown(source, f"{layer.name} has an invalid output shape")

    layers = list(graph)
    positions = {layer.name: index for index, layer in enumerate(layers)}
    if len(positions) != len(layers):
        return unknown(source, "layer names must be unique to track activation lifetimes")
    consumers: dict[str, list[int]] = {layer.name: [] for layer in layers}
    for index, layer in enumerate(layers):
        if layer.reads:
            dependencies = layer.reads
        elif index:
            dependencies = (layers[index - 1].name,)
        else:
            dependencies = ()

        for dependency in dependencies:
            if dependency not in consumers:
                return unknown(source, f"{layer.name} reads unknown tensor {dependency!r}")
            if positions[dependency] >= index:
                return unknown(
                    source,
                    f"{layer.name} reads a tensor that is not earlier in the graph",
                )
            consumers[dependency].append(index)

    total_elements = sum(layer.out_elements for layer in layers)
    total_bytes: int | float = sum(
        layer.out_elements * layer_bytes[layer.name] for layer in layers
    )

    input_elements = 1
    for dimension in graph.input_shape:
        input_elements *= dimension

    # Keep a tensor through its last consumer. The input is live during the
    # first operation, and a layer output remains live if it is the final result.
    last_use = {
        name: max(indices) if indices else len(layers)
        for name, indices in consumers.items()
    }
    live: dict[str, tuple[int, int | float]] = {}
    peak_elements = input_elements
    peak_bytes = input_elements * input_bytes
    peak_at = "input"
    if layers:
        live["__input__"] = (input_elements, input_bytes)

    for index, layer in enumerate(layers):
        live[layer.name] = (layer.out_elements, layer_bytes[layer.name])
        resident_elements = sum(elements for elements, _ in live.values())
        resident_bytes = sum(elements * size for elements, size in live.values())
        if resident_bytes > peak_bytes:
            peak_elements = resident_elements
            peak_bytes = resident_bytes
            peak_at = layer.name

        dependencies = layer.reads or ((layers[index - 1].name,) if index else ())
        for dependency in dependencies:
            if last_use[dependency] == index:
                live.pop(dependency, None)
        if index == 0:
            live.pop("__input__", None)

    return computed(
        peak_bytes,
        source,
        peak_at=peak_at,
        peak_elements=peak_elements,
        total_elements=total_elements,
        total_bytes=total_bytes,
        includes_network_input=True,
        note="peak is the resident set, not the largest single tensor",
    )

# ===========================================================================
# 4. The factor of two that halves everybody's numbers
# ===========================================================================

def to_flops(macs: dict[str, Any], convention: str = "mac_is_two_flops") -> dict[str, Any]:
    """Convert a MAC finding to a FLOP finding, naming the convention used.

    A multiply-accumulate is one multiply and one add, so it is two
    floating-point operations. Roughly half the published literature calls a
    MAC one FLOP anyway, and the two conventions differ by exactly the factor
    that makes two papers' numbers incomparable.

    Three requirements, and the third is the graded one:

      * multiply once. `FLOPS_PER_MAC` exists so that the number 2 appears in
        this file exactly once
      * an unknown MAC count converts to an unknown FLOP count. It does not
        convert to zero and it does not raise
      * the convention goes in the finding. A FLOP count that does not say
        which convention produced it is not a FLOP count, it is a number, and
        `to_flops(x, "mac_is_one_flop")` has to be as clearly labelled as the
        default

    An unrecognised convention is `unknown`, not a default. The caller asked
    for something this function does not know how to do.
    """
    source = str(macs.get("source", "to_flops"))
    if convention not in FLOP_CONVENTIONS:
        return unknown(source, f"unknown MAC to FLOP convention {convention!r}")
    if not is_answered(macs):
        return unknown(
            source,
            macs.get("detail", "MAC count is not available"),
        )

    value = macs.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return unknown(source, "MAC count must be a non-negative number")

    factor = FLOP_CONVENTIONS[convention]
    per_layer = macs.get("per_layer")
    extra: dict[str, Any] = {}
    if isinstance(per_layer, dict):
        converted: dict[str, int | float] = {}
        for name, count in per_layer.items():
            if isinstance(count, bool) or not isinstance(count, (int, float)) or count < 0:
                return unknown(source, f"per-layer MAC count for {name!r} is invalid")
            converted[name] = count * factor
        extra["per_layer"] = converted

    return computed(
        value * factor,
        source,
        convention=convention,
        flops_per_mac=factor,
        **extra,
        note="a count of operations contains no unit of time",
    )
