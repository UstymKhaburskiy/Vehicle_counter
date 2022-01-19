import dtw
import sys
import os
import numpy as np

parent_dir_path = os.getcwd()
sys.path.append(os.path.dirname(parent_dir_path))
parent_dir_path = parent_dir_path.replace("\\", "/")

from Vehicle_counter.DataPreparation.IDMT_open import *
from Vehicle_counter.SoundMap.TimeDelay import *


class DTW_model:
    def __init__(self, template_right_path=None, template_left_path=None):
        self.parent_dir_path = parent_dir_path
        if template_right_path is None:
            template_right_path = f"{self.parent_dir_path}/Vehicle_counter/DataPreparation/TemplateAudios/Right_template.wav"
            template_left_path = f"{self.parent_dir_path}/Vehicle_counter/DataPreparation/TemplateAudios/Left_template.wav"
        self.template_right = time_delay_2d(import_template(template_right_path))
        self.template_left = time_delay_2d(import_template(template_left_path))

    @staticmethod
    def count_vehicles_one_side(signal, template, threshold_check=10, chunk_size=73, hop_size=36):
        current_signal = np.array([])
        vehicle_count = 0
        detected_result = []
        index_begin = 0
        for i in range(int(len(signal) // hop_size)):
            current_signal = np.append(current_signal, signal[i * hop_size:(i + 1) * hop_size])
            if len(current_signal) < chunk_size:
                continue
            signal_DTW = dtw.dtw(template, current_signal, keep_internals=True,
                                 step_pattern=dtw.rabinerJuangStepPattern(6, "c"), open_end=True, open_begin=True)
            vehicle_in_current_signal = signal_DTW.distance < threshold_check
            if vehicle_in_current_signal:
                vehicle_count += 1
                detected_indexes = signal_DTW.index2s + index_begin
                detected_result.append((detected_indexes[0], detected_indexes[-1]))
                index_begin += len(current_signal)
                current_signal = np.array([])
        return vehicle_count, detected_result

    def count_vehicles(self, signal_id, threshold_check=5, chunk_size=73, hop_size=36):
        self.signal_path = f"{self.parent_dir_path}/Vehicle_counter/DataPreparation/LongRecords/{signal_id}"
        signal = time_delay_2d(import_template(self.signal_path + ".wav"))
        self.signal = signal
        vehicle_count_left, detected_left = self.count_vehicles_one_side(signal, self.template_left, threshold_check,
                                                                         chunk_size, hop_size)
        vehicle_count_right, detected_right = self.count_vehicles_one_side(signal, self.template_right, threshold_check,
                                                                           chunk_size, hop_size)
        self.detected_cars_indexes = detected_left, detected_right
        self.vehicle_count_result = vehicle_count_left + vehicle_count_right

        with open(self.signal_path + ".txt", 'r') as signal_txt_file:
            labels = signal_txt_file.read().split("_")
        self.cars_left = []
        self.cars_right = []
        for i in range(len(labels)):
            side, indexes = labels[i].split("|")
            if side == 'L':
                self.cars_left.append(tuple(map(int, indexes.split(","))))
            elif side == 'R':
                self.cars_right.append(tuple(map(int, indexes.split(","))))
        return self.vehicle_count_result

    def plot_results(self):
        indexes = np.array([i for i in range(len(self.signal))])
        fig, axes = plt.subplots(2, 1, figsize=(len(self.signal) // 100, 7), sharex=True)
        axes[0].plot(indexes, self.signal)
        axes[1].plot(indexes, self.signal)
        for detected_region in self.detected_cars_indexes[0]:
            axes[0].axvspan(detected_region[0], detected_region[1], color='green', alpha=0.4)
        for detected_region in self.detected_cars_indexes[1]:
            axes[0].axvspan(detected_region[0], detected_region[1], color='yellow', alpha=0.4)

        for car_left in self.cars_left:
            axes[1].axvspan(car_left[0], car_left[1], color='green', alpha=0.4)
        for car_right in self.cars_right:
            axes[1].axvspan(car_right[0], car_right[1], color='yellow', alpha=0.4)
        plt.show()

    def f_measure(self):
        true_positive_right = 0
        true_positive_left = 0
        false_positive_right = 0
        false_positive_left = 0
        false_negative_right = 0
        false_negative_left = 0

        for true_right in self.detected_cars_indexes[1]:
            centr_index = (true_right[0] + true_right[1]) // 2
            true_positive_found = False
            for right in self.cars_right:
                if centr_index > right[0] and centr_index < right[1]:
                    true_positive_right += 1
                    true_positive_found = True
                    break
            if not true_positive_found:
                false_positive_right += 1

        for true_left in self.detected_cars_indexes[0]:
            centr_index = (true_left[0] + true_left[1]) // 2
            true_positive_found = False
            for left in self.cars_left:
                if centr_index > left[0] and centr_index < left[1]:
                    true_positive_left += 1
                    true_positive_found = True
                    break
            if not true_positive_found:
                false_positive_left += 1

        for right in self.cars_right:
            centr_index = (right[0] + right[1]) // 2
            true_positive_found = False
            for true_right in self.detected_cars_indexes[1]:
                if centr_index > true_right[0] and centr_index < true_right[1]:
                    true_positive_found = True
                    break
            if not true_positive_found:
                false_negative_right += 1

        for left in self.cars_left:
            centr_index = (left[0] + left[1]) // 2
            true_positive_found = False
            for true_left in self.detected_cars_indexes[0]:
                if centr_index > true_left[0] and centr_index < true_left[1]:
                    true_positive_found = True
                    break
            if not true_positive_found:
                false_negative_left += 1

        precision = (true_positive_left + true_positive_right) / (
                true_positive_left + true_positive_right + false_positive_left + false_positive_right)
        recall = (true_positive_left + true_positive_right) / (
                true_positive_left + true_positive_right + false_negative_left + false_negative_right)

        self.F_MEASURE = 2 * precision * recall / (precision + recall)
        print(f"Precision: {precision}; Recall: {recall}; F-measure:{self.F_MEASURE}")
        print(true_positive_left, true_positive_right, false_positive_left, false_positive_right, false_negative_left,
              false_negative_right)
        return self.F_MEASURE
