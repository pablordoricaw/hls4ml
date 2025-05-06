from hls4ml.converters.flax_to_hls import (
    get_weights_data,
    flax_handler,
)


@flax_handler("Linear")
def parse_linear_layer(layer_name, flax_layer, input_names, input_shapes, data_reader):
    layer = {}
    output_shape = []

    layer["class_name"] = "Dense"
    layer["name"] = layer_name
    layer["inputs"] = input_names

    layer["weight_data"], layer["bias_data"] = get_weights_data(
        data_reader, layer_name, ["kernel", "bias"]
    )

    layer["use_bias"] = flax_layer.use_bias

    layer["n_in"] = layer["weight_data"].shape[0]
    layer["n_out"] = layer["weight_data"].shape[1]

    output_shape = input_shapes[0][:]
    output_shape[-1] = layer["n_out"]

    return layer, output_shape


activation_layers = ["ReLU", "Sigmoid", "Softmax"]


@flax_handler(*activation_layers)
def parse_activation_layer(
    layer_name, flax_layer, input_names, input_shapes, data_reader
):
    layer_type = flax_layer.__class__.__name__
    assert layer_type in activation_layers, (
        f"ERROR: Unsupported activation layer {layer_type}"
    )
    layer = {}
    output_shape = []

    layer["class_name"] = layer_type
    layer["activation"] = layer["class_name"].lower()
    layer["name"] = layer_name
    layer["inputs"] = input_names

    if layer["class_name"] in activation_layers:
        layer["class_name"] = "Activation"

    output_shape = input_shapes[0]

    return layer, output_shape
