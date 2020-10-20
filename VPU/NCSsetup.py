from openvino.inference_engine import IECore
import logging as log
import numpy as np
import cv2

import sys
import os

MODEL_NAME = 'deepmoon'
DEVICE = 'MYRIAD'

log.info("Loading Inference Engine")
ie = IECore()

model_xml = f'IR/{MODEL_NAME}.xml'
model_bin = f'IR/{MODEL_NAME}.bin'

log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
net = ie.read_network(model=model_xml, weights=model_bin)

log.info("Device info:")
versions = ie.get_versions(DEVICE)
print(f"\t\t{DEVICE}")
print(f"\t\tMKLDNNPlugin version ......... {versions[DEVICE].major}.{versions[DEVICE].minor}")
print(f"\t\tBuild ........... {versions[DEVICE].build_number}")

print("inputs number: " + str(len(net.input_info.keys())))

for input_key in net.input_info:
    print("input shape: " + str(net.input_info[input_key].input_data.shape))
    print("input key: " + input_key)
    if len(net.input_info[input_key].input_data.layout) == 4:
        n, c, h, w = net.input_info[input_key].input_data.shape

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
