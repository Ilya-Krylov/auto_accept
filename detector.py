import os

import numpy as np
from openvino.inference_engine import IECore

class Detector:
    def __init__(self, model_path):
        self.ie = IECore()
        bin_path = os.path.splitext(model_path)[0] + '.bin'
        self.net = self.ie.read_network(model_path, bin_path)

        self.device = 'CPU'
        self.exec_net = self.ie.load_network(network=self.net, device_name=self.device, num_requests=1)
        batch_size = self.net.input_info['image'].input_data.shape[0]
        assert batch_size == 1, 'Only batch 1 is supported.'

    def __call__(self, inputs):
        output = self.exec_net.infer(inputs)

        detection_out = output['detection_out']
        output['labels'] = detection_out[0, 0, :, 1].astype(np.int32)
        output['boxes'] = detection_out[0, 0, :, 3:] * np.tile(inputs['image'].shape[:1:-1], 2)
        output['boxes'] = np.concatenate((output['boxes'], detection_out[0, 0, :, 2:3]), axis=1)
        del output['detection_out']
        return output
