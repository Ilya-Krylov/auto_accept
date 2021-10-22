import argparse
import logging
import os
import time

import cv2
import numpy as np
import pyautogui

from detector import Detector

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder')

    return parser.parse_args()


class ScreenShotter:

    def __call__(self):
        image = np.array(pyautogui.screenshot())
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image


class ImageFolderReader:

    def __init__(self, folder):
        self.images_paths = [os.path.join(folder, name) for name in os.listdir(folder)]
        self.index = 0

    def __call__(self):
        if self.index < len(self.images_paths):
            image = cv2.imread(self.images_paths[self.index])
            self.index += 1
            return image

        else:
            return None

def main():
    model = Detector('alt_ssd_export/model.xml')
    net_width, net_height = 512, 512
    args = parse_args()

    if args.folder:
        source = ImageFolderReader(args.folder)
    else:
        source = ScreenShotter()


    mouse_positions = []

    while True:
        time.sleep(1)

        image = source()
        if image is None:
            break

        mouse_pos = pyautogui.position()
        mouse_positions.append([mouse_pos.x, mouse_pos.y])
        to_check = False
        averaging = 10
        if len(mouse_positions) > averaging:
            mouse_positions = mouse_positions[1:]
            s = np.sum(np.std(mouse_positions, axis=0))
            logging.info(f'{averaging}-sec averaged mouse activity {s}')
            if s == 0:
                logging.info('trying to find "accept" button')
                to_check = True
                mouse_positions = []
        else:
            logging.info('pause')

        if not to_check:
            continue

        orig_width, orig_height = image.shape[1], image.shape[0]

        orig_height_025 = orig_height // 4

        x_offset = orig_width // 2 - orig_height_025
        y_offset = orig_height_025

        central_crop = image[y_offset: y_offset + 2 * orig_height_025, x_offset : x_offset + 2 * orig_height_025]
        resized_central_crop = cv2.resize(central_crop, (net_width, net_height))

        det = model({'image': np.array([np.transpose(resized_central_crop, (2, 0, 1))])})
        for x1, y1, x2, y2, conf in det['boxes']:
            if conf > 0.99:
                x1_n = x1 / net_width
                x2_n = x2 / net_width
                y1_n = y1 / net_height
                y2_n = y2 / net_height

                x1_o = int(x_offset + x1_n * central_crop.shape[1])
                x2_o = int(x_offset + x2_n * central_crop.shape[1])
                y1_o = int(y_offset + y1_n * central_crop.shape[0])
                y2_o = int(y_offset + y2_n * central_crop.shape[0])
                x = (x1_o + x2_o) // 2
                y = (y1_o + y2_o) // 2

                logging.info(f'Clicked {x=} {y=} {conf=}')
                pyautogui.click(x, y)

            break


if __name__ == '__main__':
    main()
