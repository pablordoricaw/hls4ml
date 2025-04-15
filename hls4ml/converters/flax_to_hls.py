import numpy as np
from flax import nnx

from hls4ml.model import ModelGraph


class FlaxReader:
    def get_weights_data(self, layer_name, var_name):
        raise NotImplementedError


class FlaxModelReader(FlaxReader):
    def __init__(self, flax_model):
        # flax_model contains both the model graph and weights and biases
        # Could keep the state separate like the PyTorchModelReader or
        # keep in one like the KerasModelReader.
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

        if layer_name in model_params.keys():
            data = model_params[layer_name][var_name].value
            return np.array(data)

        return data


def get_weights_data(data_reader, layer_name, var_name):
    """
    Extract the weights and bias from a Flax layer using the get_weights_data() method from the data_reader.

    The var_name parameter is the 'bias', and 'kernel' for weights in a Flax layer.

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


def get_model_arch(config):
    if "FlaxModel" in config:
        flax_model = config["FlaxModel"]
        model_arch = nnx.graphdef(flax_model)
        reader = FlaxModelReader(flax_model)
    else:
        raise ValueError("No model found in config file.")

    return model_arch, reader


def parse_flax_model(model_arch, reader):
    # This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    skip_layers = []
    supported_layers = get_supported_flax_layers() + skip_layers

    # Traverse through all layers and check they all are supported

    # Traverse through layers
    #   Parse each layer with corresponding layer handler
    #   Append parsed layer to layer_list

    #   layer, output_shape = layer_handlers[flax_class](
    #       flax_class, layer_name, input_shapes, node, class_object, reader, config
    #   )
    #   layer_list.append(layer)


def flax_to_hls(config):
    model_arch, reader = get_model_arch(config)
    print(f"Model arch:\n{model_arch}")
    layer_list, input_layers, output_layers, _ = parse_flax_model(model_arch, reader)
    print("Creating HLS model")
    hls_model = ModelGraph(config, layer_list, input_layers, output_layers)
    return hls_model
