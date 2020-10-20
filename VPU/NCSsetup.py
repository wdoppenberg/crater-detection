from openvino.inference_engine import IECore
import logging as log
import torch


class NCSInferenceHandler:
    log.info("Loading Inference Engine")
    ie = IECore()

    def __init__(self, model_name, device='MYRIAD', IR_path='VPU/IR/'):
        self.device = device

        model_xml = f'{IR_path}{model_name}.xml'
        model_bin = f'{IR_path}{model_name}.bin'

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

    def __str__(self):
        return repr(self)

    def __repr__(self):
        versions = self.ie.get_versions(self.device)
        return f"""({self.device}):
        MKLDNNPlugin version: {versions[self.device].major}.{versions[self.device].minor}
        tBuild: {versions[self.device].build_number}
        tModel info:
        \tInput layout: {self.input_layout}
        \tInput shape: {self.input_shape}
        """
