import numpy as np
from flax import nnx

from hls4ml.model import ModelGraph
from hls4ml.utils.dependency import requires


class FlaxReader:
    def get_weights_data(self, layer_name, var_name):
        raise NotImplementedError


class FlaxModelReader(FlaxReader):
    def __init__(self, flax_model):
        # flax_model contains both the model graph and weights and biases
        # Need to use Flax funcs to get the model weights and biases
        self.model = flax_model

    def get_weights_data(self, layer_name, var_name):
        """
        Extract the weights and bias from a Flax layer of a nnx.Sequential() or nnx.Module() model.

        The layer_name for a nnx.Sequential model is the number of the layer, while for a nnx.Module() model it's the name given in the constructor.
        """
        data = None

        model_params = nnx.variables(self.model, nnx.Param)

        if isinstance(self.model, nnx.Sequential):
            model_params = model_params.layers
            layer_name = int(layer_name)

        if layer_name in model_params.keys():
            data = model_params[layer_name][var_name].value
            return np.array(data)

        return data


def get_weights_data(data_reader, layer_name, var_name):
    """
    Extract the weights and bias from a Flax layer using the get_weights_data() method from the data_reader.

    The var_name parameter is a list with 'bias', and 'kernel' for weights in a Flax layer.

    This function is called by the Flax layer handlers.
    """

    if not isinstance(var_name, (list, tuple)):
        var_name = [var_name]

    data = [data_reader.get_weights_data(layer_name, var) for var in var_name]

    if len(data) == 1:
        return data[0]
    else:
        return (*data,)


layer_handlers = {}


def get_layer_handlers():
    return layer_handlers


def register_flax_layer_handler(layer_cname, handler_func):
    """Register a handler function for the given layer class name.

    The handler function should have the following signature:
        parse_func(keras_layer, input_names, input_shapes, data_reader, config):

    Args:
        layer_cname (str): The name of Keras layer (the 'class_name' property in the layer's config)
        handler_func (callable): The handler function

    Raises:
        Exception: If the layer class has already been registered.
    """
    if layer_cname in layer_handlers:
        raise Exception(f"Layer {layer_cname} already registered")
    else:
        layer_handlers[layer_cname] = handler_func


def get_supported_flax_layers():
    """Returns the list of Flax layers that the converter can parse.

    The returned list contains all Flax layers that can be parsed into the hls4ml internal representation. Support for
    computation of these layers may vary across hls4ml backends and conversion configuration.

    Returns:
        list: The names of supported Flax layers.
    """
    return list(layer_handlers.keys())


def flax_handler(*args):
    def decorator(function):
        function.handles = [arg for arg in args]
        return function

    return decorator


def parse_default_flax_layer(flax_layer, input_names):
    # Use in the ./flax/<layer>.py
    layer = {}

    return layer


def parse_flax_model(config, verbose=True):
    model = config["FlaxModel"]
    reader = FlaxModelReader(model)

    # This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []
    skip_layers = []

    supported_layers = get_supported_flax_layers() + skip_layers

    activation_layers = ["ReLU", "Sigmoid", "Softmax"]

    input_layers = None
    output_layers = None

    if isinstance(model, nnx.Sequential):
        print("Interpreting nnx.Sequential")

        layer_list.append(
            {
                "class_name": "InputLayer",
                "name": "dense_input",
                "data_format": "channels_last",
                "input_shape": [model.layers[0].in_features],
            }
        )

        for layer in model.layers:
            layer_type = layer.__class__.__name__
            if layer_type.lower() not in list(map(str.lower, supported_layers)):
                raise Exception(f"ERROR: Unsupported layer type: {layer_type}")

        if verbose:
            print("Topology:")
        for idx, layer in enumerate(model.layers):
            # case matters for layer_handler call below
            layer_type = layer.__class__.__name__

            layer_name = str(idx)  # in an nnx.Sequential model correspond to the idx
            previous_layer = model.layers[idx - 1]
            if layer_type in activation_layers:
                input_shapes = [[None, previous_layer.out_features]]
            else:
                input_shapes = [[None, layer.in_features]]

            if idx <= 0:
                input_names = []
            else:
                input_names = [str(idx - 1)]

            layer, output_shape = layer_handlers[layer_type](
                layer_name, layer, input_names, input_shapes, reader
            )
            if verbose:
                print(
                    "Layer name: _{}, layer type: {}, input shapes: {}, output shape: {}".format(
                        layer["name"], layer["class_name"], input_shapes, output_shape
                    )
                )
            layer_list.append(layer)

        # to convert to cpp HLS, layer names can't start with a number
        # so append '_'
        for layer in layer_list:
            if layer["class_name"] != "InputLayer":
                layer["name"] = f"_{layer['name']}"
                if "inputs" in layer.keys():
                    _inputs = []
                    for in_layer in layer["inputs"]:
                        _inputs.append(f"_{in_layer}")
                    layer["inputs"] = _inputs
            else:
                input_layers = [layer["name"]]

        output_layers = [f"_{str(len(model.layers) - 1)}"]
    else:
        raise Exception(
            "ERROR: Unsupported Flax model. Model needs to be an nnx.Sequential() model"
        )

    return layer_list, input_layers, output_layers


@requires("_flax")
def flax_to_hls(config):
    layer_list, input_layers, output_layers = parse_flax_model(config)
    print("Creating HLS model")
    hls_model = ModelGraph(
        config, layer_list, inputs=input_layers, outputs=output_layers
    )
    return hls_model
