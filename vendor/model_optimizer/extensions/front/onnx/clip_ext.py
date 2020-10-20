"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr, get_onnx_opset_version
from mo.ops.clamp import Clamp, AttributedClamp


class ClipFrontExtractor(FrontExtractorOp):
    op = 'Clip'
    enabled = True

    @classmethod
    def extract(cls, node):
        if get_onnx_opset_version(node) < 11:
            attrs = {
                'min': onnx_attr(node, 'min', 'f', np.finfo(np.float32).min),
                'max': onnx_attr(node, 'max', 'f', np.finfo(np.float32).max),
            }
            AttributedClamp.update_node_stat(node, attrs)
        else:
            Clamp.update_node_stat(node)
        return cls.enabled