from hls4ml.converters.flax_to_hls import (
    get_weights_data,
    flax_handler,
    parse_default_flax_layer,
)


@flax_handler("Linear")
def parse_linear_layer(flax_layer, input_names, input_shapes, data_reader):
    layer = {}
    output_shape = []

    return layer, output_shape


activation_layers = ["ReLU"]


@flax_handler(*activation_layers)
def parse_activation_layer():
    layer = {}
    output_shape = []

    return layer, output_shape
