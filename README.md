# QoS-UARAPC

The source code entitled, "Deep Learning-Aided User Association and Power Control
with Renewable Energy Sources". [Paper link(will be updated)]

## Dependencies

- Tensorflow 1.13.1
- Python>=3
- tqdm
- scipy

1. Clone Repository
2. Install dependencies
~~~
pip install tqdm
~~~

~~~
conda install tensorflow-gpu==1.13.1
~~~

~~~
pip install scipy
~~~

3. Run setup.py
~~~
python setup.py develop
~~~

## Dataset & NN Model (test)

1. You can download the dataset and NN model in the following link:

dataset: https://drive.google.com/file/d/1uZsHcxr3lrwpEnUTrBcav1kxCUZ97YEU/view?usp=sharing

model :https://drive.google.com/file/d/16y1QVesBemEWRj4k_FRGpIrpxS7MKstB/view?usp=sharing

2. Data location: ./DATA/
3. Model location: ./Models/BS4_UE40_Bh20
4. Run the test code
~~~
python tools/DDPG_test.py
~~~
## How to train the models?

- Will be uploaded after software registration & pending patent
