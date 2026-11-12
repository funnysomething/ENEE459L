import json
import math


class Layer:
    def __init__(self, name, kind, in_shape, out_shape, kernel=None, stride=1,
                 groups=1, bias=False, weight_dtype="fp32", act_dtype="fp32",
                 reads=()):
        self.name = name
        self.kind = kind
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.kernel = kernel
        self.stride = stride
        self.groups = groups
        self.bias = bias
        self.weight_dtype = weight_dtype
        self.act_dtype = act_dtype
        self.reads = reads

    @property
    def in_elements(self):
        return math.prod(self.in_shape)

    @property
    def out_elements(self):
        return math.prod(self.out_shape)


class Graph:
    def __init__(self, name, input_shape, layers, precision="fp32"):
        self.name = name
        self.input_shape = input_shape
        self.layers = layers
        self.precision = precision

    def __iter__(self):
        return iter(self.layers)


# helper functions
def dtype_bytes(dtype):
    return {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}[dtype]


def computed(value, message, **metadata):
    return {"type": "computed", "value": value, "message": message, **metadata}


def _layer_parameters(ly):
    if ly.kind == "conv":
        if ly.kernel is None or ly.groups < 1:
            raise ValueError(f"{ly.name}: missing kernel or invalid group count")
        c_in = ly.in_shape[0]
        c_out = ly.out_shape[0]
        if c_in % ly.groups or c_out % ly.groups:
            raise ValueError(f"{ly.name}: groups must divide both channel counts")
        weights = c_out * (c_in // ly.groups) * math.prod(ly.kernel)
        return weights + (c_out if ly.bias else 0)
    if ly.kind == "linear":
        weights = ly.in_elements * ly.out_elements
        return weights + (ly.out_elements if ly.bias else 0)
    if ly.kind == "bn":
        return 2 * ly.out_shape[0]  # gamma and beta
    return 0


def _layer_macs(ly):
    # Only conv and linear layers count toward MACs here.
    if ly.kind == "conv":
        if ly.kernel is None or ly.groups < 1:
            raise ValueError(f"{ly.name}: missing kernel or invalid group count")
        return ly.out_elements * (ly.in_shape[0] // ly.groups) * math.prod(ly.kernel)
    if ly.kind == "linear":
        return ly.in_elements * ly.out_elements
    return 0


def load_graph(path):
    with open(path, encoding="utf-8") as f:
        model = json.load(f)

    precision = model.get("precision", "fp32")
    layers = []
    for ly in model["layers"]:
        layers.append(Layer(
            name=ly["name"],
            kind=ly["kind"],
            in_shape=tuple(ly["in_shape"]),
            out_shape=tuple(ly["out_shape"]),
            kernel=tuple(ly["kernel"]) if ly.get("kernel") else None,
            stride=ly.get("stride", 1),
            groups=ly.get("groups", 1),
            bias=ly.get("bias", False),
            weight_dtype=ly.get("weight_dtype", precision),
            act_dtype=ly.get("act_dtype", precision),
            reads=tuple(ly.get("reads", ())),
        ))

    return Graph(model["name"], tuple(model["input_shape"]), layers, precision)


def arithmetic_intensity(graph):
    layers=list(graph)
    seen = {}
    per_layer = {}
    total_flops=0
    total_bytes=0.0

    for i in range(len(layers)):
        ly = layers[i]
        if ly.name in seen:
            raise ValueError(f"duplicate layer name: {ly.name}")
        for name in ly.reads:
            if name not in seen:
                raise ValueError(f"{ly.name}: {name!r} is not an earlier layer")

        # input dtype
        if ly.reads:
            dt = seen[ly.reads[0]].act_dtype
        elif i > 0:
            dt=layers[i-1].act_dtype
        else:
            dt = graph.precision

        b1 = ly.in_elements * dtype_bytes(dt)
        if ly.kind == "add" and len(ly.reads) == 2:
            b1 = b1 * 2  # reads both inputs

        b2 = _layer_parameters(ly)*dtype_bytes(ly.weight_dtype)
        if ly.kind == "bn":
            b2 += 2 * ly.out_shape[0] * dtype_bytes("fp32")

        b3=ly.out_elements * dtype_bytes(ly.act_dtype)
        moved = float(b1+b2+b3)
        flops = 2*_layer_macs(ly)

        ai = 0
        if moved != 0:
            ai = round(flops/moved, 4)

        per_layer[ly.name] = {
            "flops": flops,
            "bytes": moved,
            "intensity": ai
        }
        total_flops = total_flops + flops
        total_bytes +=moved
        seen[ly.name]=ly

    intensity = 0
    if total_bytes != 0:
        intensity=total_flops/total_bytes

    return computed(
        round(intensity, 4),
        f"{graph.name}: FLOPs over bytes, every byte counted once",
        per_layer=per_layer,
        total_flops=total_flops,
        total_bytes=total_bytes,
        flops_per_mac=2,
        cache_reuse_assumed=False,
        note="no fusion and no cache reuse; both would lower the byte column",
    )
