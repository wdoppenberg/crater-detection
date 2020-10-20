from openvino.inference_engine import IECore
import logging as log
import torch
import numpy as np
import cv2

import sys
import os


class NCSInferenceHandler:
    log.info("Loading Inference Engine")
    ie = IECore()

    def __init__(self, model_name, device='MYRIAD', IR_path='VPU/IR/'):
        self.device = device

        model_xml = f'{IR_path}{model_name}.xml'
        model_bin = f'{IR_path}{model_name}.bin'

        self.net = self.ie.read_network(model=model_xml, weights=model_bin)
        print(f"Loaded network files:\n\t{model_xml}\n\t{model_bin}")

        for input_key in self.net.input_info:
            print("\tinput shape: " + str(self.net.input_info[input_key].input_data.shape))
            print("\tinput key: " + input_key)
            self.input_layout = self.net.input_info[input_key].input_data.shape
            self.input_shape = tuple(self.net.input_info[input_key].input_data.shape)
            self.input_key = input_key

        self.out_blob = next(iter(self.net.outputs))

    def infer(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = batch.numpy()

        if batch.shape != self.input_shape:
            raise ValueError("Batch shape does not match input!")

        print("Loading model to the device")
        exec_net = self.ie.load_network(network=self.net, device_name=self.device)
        print("Creating infer request and starting inference")
        res = exec_net.infer(inputs={self.input_key: batch})
        return res[self.out_blob]

    def __str__(self):
        return repr(self)

    def __repr__(self):
        versions = self.ie.get_versions(self.device)
        return f"""\t\t{self.device}
        \t\tMKLDNNPlugin version: {versions[self.device].major}.{versions[self.device].minor}
        \t\tBuild: {versions[self.device].build_number}
        \t\tModel info:
        \t\t\tInput layout: {self.input_layout}
        \t\t\tInput shape: {self.input_shape}
        """


"""
input_filepath = 'test_input.png'

images = np.ndarray(shape=(n, w, h, c))
images_hw = []
for i in range(n):
    image = cv2.imread(input_filepath, cv2.IMREAD_GRAYSCALE)
    ih, iw = image.shape
    images_hw.append((ih, iw))
    log.info("File was added: ")
    log.info("        {}".format(input_filepath))
    image = np.reshape(image, (w, h, 1))
    images[i] = image

out_blob = next(iter(net.outputs))
input_name, input_info_name = "", ""

for input_key in net.input_info:
    input_name = input_key
    log.info("Batch size is {}".format(net.batch_size))
    net.input_info[input_key].precision = 'FP16'
#
# sample_data = {}
# sample_data[input_name] = images
#
# output_name, output_info = "", net.outputs[next(iter(net.outputs.keys()))]
# output_info.precision = "FP16"
#
# log.info("Loading model to the device")
# exec_net = ie.load_network(network=net, device_name=DEVICE)
# log.info("Creating infer request and starting inference")
# res = exec_net.infer(inputs=sample_data)
"""
