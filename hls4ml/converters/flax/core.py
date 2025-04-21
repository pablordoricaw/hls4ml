from hls4ml.converters.flax_to_hls import (
    get_weights_data,
    flax_handler,
    parse_default_flax_layer,
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


activation_layers = ["ReLU"]


@flax_handler(*activation_layers)
def parse_activation_layer():
    layer = {}
    output_shape = []

    return layer, output_shape
