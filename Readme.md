# Acoustic vehicle count system 
##### (Project for AI course at Ukrainian Catholic University.)

 Acoustic vehicle count system using DTW (dynamic time warping) and MobileNet NN structure on "time delay" sound maps.
 

 
## Usage Example
Usage example also can be found in Usage_Example.ipynb

##### DTW model
```python
from Vehicle_counter.DTW_model.DTW import DTW_model

model_dtw = DTW_model()

#index - index(name) of audio file index.wav and its index.txt file of labels from Vehicle_counter/DataPreparation/LongRecords
#threshold_check - maximum distance between template sound map and checked chunk
model_dtw.count_vehicles(0, threshold_check=8)
model_dtw.plot_results()
model_dtw.f_measure()
```
##### Example of plot created:

First time series is sound map of signal with detected cars by model

Second time series is sound map with originally labeled cars

green is cars from left, yellow is cars from right

![DTW plot results](https://github.com/UstymKhaburskiy/Vehicle_counter/blob/master/DTW_plot_results_example.png)

##### MobileNet model
```python
from Vehicle_counter.MobileNet_model.MobileNet_model import MobileNet_model

# Data for model should be transformed to sound maps as in Veicle_counter/DataPreparatiom/MobileNet_prepo.ipynb example
#test_60_records_from_Schleusinger-Allee.hdf5 is a file with same records as in Longrecords/0.wav (60 small combined records)

file_name = "test_60_records_from_Schleusinger-Allee.hdf5"
model_mobile = MobileNet_model()
y_pred = model_mobile.predict(file_name)
print(y_pred)
model_mobile.f_measure()
```

## Files description
> **IDMT_traffic**: original data which is described below, but is not allowed to share, but can be found here: [IDMT_traffic](https://www.idmt.fraunhofer.de/en/publications/datasets/traffic.html)

> **DataPreparation**:

>> **LongRecords**: combined records from IDMT_traffic to 2min audios with their labels in txt files

>> **MobileNet_Data**: dataset of sound maps created from IDMT_traffic to train MobileNet model

>> **TemplateAudios**: examples of IDMT_traffic + two records which are taken as templates for DTW model

>> **CombineRecords.ipynb**: jupyter notebook to create LongRecords from IDMT_traffic

>> **IDMT_open.py**: functions to work with IDMT_traffic

>> **MobileNet_prepo.ipynb**: jupyter notebook to create sound maps for MobileNet_Data

> **DTW_model**:

>> **DTW.py**: file which contains class of main DTW_model

> **MobileNet_model**:

>> **MobileNet_train.ipynb**: jupyter notebook with training and creating of MobileNet model

>> **MobileNet_trained.h5**: pretrained model

>> **MobileNet_model.py**: contains class MobileNet_model which counts cars by pretrained model

> **SoundMap**:

>> **TimeDelay.py**: contains functions to create 2d and 3d sound maps

>> **gccestimating.py**: file with class GCC to compute 2d time_delays [GitHub link](https://github.com/SiggiGue/gccestimating)

## Problem Description

There are several ways to count the number of cars that have traveled along the road. One
such method, which is cheaper in comparison with others and insensitive to road lighting,
is a calculation based on data from two microphones. The idea of this project is to
explore several approaches to this problem and implement the method with the best
performance. The solution to this problem can be used in many practical applications. For
example, to study the congestion of certain sections of roads, or the calculation of noise
pollution and more

I have implemented two approaches to count cars on signal and detect side from which car is moving. One is by **DTW**(dynamic time wrapping) which calculates cars(including side of moving (from Left or from Right)) on any given stereo signal. And another is **MobileNet** model which detects car on two second signal (average time of car passing).

## Results

For both approaches I have calculated precision, recall and F-measure whcih includes both car detecting and side of moving.

On test signal (60 combined records of two seconds) MobileNet had given better results than DTW.

This test detecting can be fond in Usage_Example.ipynb

>F-measure for MovileNet model was 0.985

>F-measure for DTW model with treshold=8 was 0.885

## Data Description

Data which I used for training MobileNet model and testing DTW was [IDMT_traffic](https://www.idmt.fraunhofer.de/en/publications/datasets/traffic.html)
It is 17506 stereo audio files which are labeled with needed information:

• date = recording date (YYYY-MM-DD-HH-mm)

• location = recording location ([Fraunhofer-IDMT, Schleusinger-Allee, Langewiesener-Strasse, Hohenwarte])

• speed = recording speed ([30Kmh, 50Kmh, 70Kmh, unknownKmh])

• sampleposition = sampling postion in the original long-term audio recordings

• daytime = [M, A] (for morning and afternoon)

• weather = [D, W] (for dry and wet of the road condition)

• vehicle = [B, C, M, T] (bus, car, motorcycle, truck)

• direction = [L, R] (coming from left or right)

• microphonetype = [SE, ME] (SE=sE8, ME=MEMS-microphones (ICS-43434))

• channels = [12, 34] (for stereo pairs of channel 1+2 or 3+4)

**Examples of those files can be found in DataPreparation/TemplateAudios**

## Theory
#### Time Delay Sound Maps

For each model time delay sound maps were needed. It is delays calculated in ms between left and right microphone. DTW model is working with 2d delays while MonileNet with 3d
The best way to find time delays is to calculate Generalized Cross Correlation between signals from two microphones.
More theory about GCC for time delays can be found [here](https://arxiv.org/pdf/1910.08838.pdf)

Example of 2d sound map:
![SoundMap_2d](https://github.com/UstymKhaburskiy/Vehicle_counter/blob/master/SoundMap_2d.png)

Example of 3d sound map:
![SoundMap_3d](https://github.com/UstymKhaburskiy/Vehicle_counter/blob/master/SoundMap_3d.png)
 
#### DTW

Dynamic Time Warping is based on finding distances between two vectors. I first divide sound map 
sequences into fixed-length chunks. The first chunk (A) is then compared with a template using 
DTW. When the chunk (A) unmatches to the template, i.e., DTW distance is greater than a 
threshold, I combine the chunk (A) with the next chunk (B) and compare the combined chunk
with the template. When the combined chunk (A)+(B) matches to the template, the following 
chunk (C) is compared with the template

The templates for cars from left and right can be found in DataPreparation/TemplateAudios

#### MobileNet

MobileNet is a type of convolutional neural network designed for mobile and embedded vision applications. They are based on a streamlined architecture that uses depthwise separable convolutions to build lightweight deep neural networks that can have low latency for mobile and embedded devices.

Whole network architecture with correct input and output size can be found in jupyter notebook MobileNet_model/MobileNet_train.ipynb

The labeled sound maps which I used for training this mode:
![MobileNet_data](https://github.com/UstymKhaburskiy/Vehicle_counter/blob/master/MobileNet_data.png)

Validation accuracy after 200 epochs of training was 0.99.

## Conclusions and plans

In conclusion, the performance of the MobileNet model based on F-measure was better than DTW and it is close to ideal because on files in which this model made mistakes I couldn't recognize the vehicle as well.
However, both models can't recognize two cars when they pass simultaneously. So, to improve this model, I need to find better-labeled data or to create my own.
