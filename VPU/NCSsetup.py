from openvino.inference_engine import IECore
import logging as log
import torch


class OpenVINOHandler:
    log.info("Loading Inference Engine")
    ie = IECore()

    def __init__(self, model_name, device='MYRIAD', root='VPU/IR/'):
        self.device = device

        model_xml = f'{root}{model_name}.xml'
        model_bin = f'{root}{model_name}.bin'

        self.net = self.ie.read_network(model=model_xml, weights=model_bin)
        log.info(f"Loaded network files:\n\t{model_xml}\n\t{model_bin}")

        for input_key in self.net.input_info:
            log.info("\tinput shape: " + str(self.net.input_info[input_key].input_data.shape))
            log.info("\tinput key: " + input_key)
            self.input_layout = self.net.input_info[input_key].input_data.layout
            self.input_shape = tuple(self.net.input_info[input_key].input_data.shape)
            self.input_key = input_key

        self.out_blob = next(iter(self.net.outputs))

        log.info("Loading model to the device")
        self.exec_net = self.ie.load_network(network=self.net, device_name=self.device)

        log.info(str(self))

    def infer(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = batch.numpy()

        if batch.shape != self.input_shape:
            raise ValueError(f"Batch shape does not match input! Expected {self.input_shape}, received {batch.shape}.")

        log.info("Creating infer request and starting inference")
        res = self.exec_net.infer(inputs={self.input_key: batch})
        return res[self.out_blob]

    def info(self):
        versions = self.ie.get_versions(self.device)
        return f"""{self.__class__.__name__}({self.device}):\n""" \
               f"""MKLDNNPlugin version: {versions[self.device].major}.{versions[self.device].minor}\n""" \
               f"""Build: {versions[self.device].build_number}\n""" \
               f"""Model info:\n""" \
               f"""\tInput layout: {self.input_layout}\n""" \
               f"""\tInput shape: {self.input_shape}"""

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{self.__class__.__name__}(device={self.device},\
             input_layout={self.input_layout}, input_shape={self.input_shape})"
