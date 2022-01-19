from tensorflow import keras
import tensorflow as tf
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

parent_dir_path = os.getcwd().replace("\\", "/")
sys.path.append(os.path.dirname(parent_dir_path))

class MobileNet_model:
    def __init__(self):
        self.parent_dir_path = parent_dir_path
        self.model = keras.models.load_model(self.parent_dir_path+'/Vehicle_counter/MobileNet_model/MobileNet_trained.h5')
        self.lab_enc = LabelEncoder()
        self.labels = self.lab_enc.fit_transform(['L', 'None', 'R'])
    
    def predict(self, file_name):
        test_data_file = self.parent_dir_path + "/Vehicle_counter/DataPreparation/MobileNet_Data/"+ file_name
        test_data = pd.read_hdf(test_data_file, "IDMT_traffic")
        test_soundmaps = tf.stack(test_data.SoundMap.to_numpy())
        test_preds = self.model.predict(test_soundmaps, 
                           batch_size=64, verbose=True)
        predictions = self.lab_enc.classes_[np.argsort(-test_preds, axis=1)[:, :1]]
        self.y_true = test_data.source_direction.to_list()
        self.y_pred = [' '.join([cat for cat in row]) for row in predictions]
        return self.y_pred
    
    def f_measure(self):
        Y_true, Y_pred = self.y_true, self.y_pred
        true_positives = 0
        false_positives = 0
        true_elements = Y_true.count('L') + Y_true.count('R')

        for i in range(len(Y_true)):
            if Y_true[i] == Y_pred[i]:
                if Y_pred[i] != 'None':
                    true_positives += 1
            elif Y_pred[i] != 'None':
                false_positives += 1
                print(f"index of false_positive: {i}")
            else:
                print(f"index of false_negative: {i}")

        precision = true_positives/(true_positives+false_positives)
        recall = true_positives/true_elements
        f = 2*precision*recall/(precision+recall)
        print(f"Precision: {precision}; Recall: {recall}; F-measure: {f}")
        return f
        