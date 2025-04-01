from hls4ml.model import ModelGraph


class FlaxReader:
    def get_weights_data(self, layer_name, var_name):
        raise NotImplementedError


class FlaxModelReader(FlaxReader):
    def __init__(self, flax_model):
        self.model = flax_model

    def get_weights_data(self, layer_name, var_name):
        return None


def get_weights_data(data_reader, layer_name, var_name):
    pass


layer_handlers = {}


def get_layer_handlers():
    return layer_handlers


def register_flax_layer_handler():
    pass


def get_supported_flax_layers():
    pass


def flax_handler(*args):
    def decorator(function):
        function.handles = [arg for arg in args]
        return function

    return decorator


def parse_default_flax_layer(flax_layer, input_names):
    layer = {}

    return layer


def get_model_arch(config):
    if "FlaxModel" in config:
        from flax import nnx

        flax_model = config["FlaxModel"]
        model_arch = nnx.graphdef(flax_model)
        reader = FlaxModelReader(flax_model)
    else:
        raise ValueError("No model found in config file.")

    return model_arch, reader


def parse_flax_model(model_arch, reader):
    pass


def flax_to_hls(config):
    model_arch, reader = get_model_arch(config)
    layer_list, input_layers, output_layers, _ = parse_flax_model(model_arch, reader)
    print("Creating HLS model")
    hls_model = ModelGraph(config, layer_list, input_layers, output_layers)
    return hls_model
