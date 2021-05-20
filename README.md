You can find the video for this project [here](https://photos.app.goo.gl/6uaa7BsiQKfcA94V7)

![Alt Text](https://user-images.githubusercontent.com/22652444/118917577-3167f580-b8ff-11eb-845b-538616d9cd33.gif)


# Installation

* System Setup

To setup tensorflow to work on Windows with GPU, check out [this](https://towardsdatascience.com/setting-up-your-pc-workstation-for-deep-learning-tensorflow-and-pytorch-windows-9099b96035cb) resource.

```
conda create --name tensorgpu python=3.7 
conda install -c anaconda tensorflow-gpu 
conda install -c conda-forge opencv 
conda install matplotlib 
conda install pandas 
conda install -y jupyter 
conda install -c conda-forge pickle5 
conda install -c conda-forge scikit-learn  
conda install -c conda-forge moviepy
pip install python-socketio=4.2.1
conda install -c conda-forge eventlet 
pip install flask
conda install -c anaconda keras-gpu 
```

* Project setup

You can find the simulation for your OS [here](https://github.com/udacity/self-driving-car-sim/blob/master/README.md)
       
To train the model:     

```
python model.py
```
This creates a model.h5 file.        
   
To use this model file in the simulator, open an instance of the simulator in autonomous mode and run

```
python drive.py model.h5
```
