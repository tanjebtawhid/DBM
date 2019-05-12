This project deals with the issue of extending an existing body model in order to drive the model through visual
inputs. The idea behind this is to enable an agent learn the mapping between visual representations and it's internal 
body model by observing another agent equipped with same underlying model performing body movements.

#### MMC Network
The body model is modeled as a MMC network, a type of recurrent neural network based on Mean of Multiple Computation (MMC)
principle. In the context of this project, the MMC network models movement of a three segmented arm. For details
about the underlying principle of MMC network as a body model please refer to
[Schilling, M.](https://link.springer.com/article/10.1007/s10514-011-9226-3)

#### Extending Body Model
The idea is to embed the MMC network into a convolutional-deconvolutional network or autoencoder. One way to accomplish 
this is to have the MMC network as an intermediary layer between the encoder and decoder of an autoencoder. This way,
the feature representations learned from input images will be input to the MMC network, which will then be processed
according to the dynamics of the MMC network to predict next arm configuration and reconstructed through the decoder.

#### Simple Movement
SimpleMovement models the dynamics of a single segmented arm in a two dimensional space. Given an initial position described 
in terms of angular displacement with respect to the coordinate system and initial velocity, aim is to reach a target
position also described in terms of angular displacement. In essence, simple movement mimics the behavior of the MMC
network but in a simpler setting. Primary reason for this is to better understand and analyze the behavior of the
autoencoder when trained to predict the next image on a sequence of images as MMC network has been empirically found to
be difficult to analyze.

#### Usage
##### Data generation
Training data for autoencoder can be generated by simulating the MMC/SimpleMovement network using the `DataGen` class for different targets given an initial configuration. Configurations are defined using JSON format. `config_mmc.json` and `config_simple_movement.json` contains example configuration for MMC and SimpleMovement respectively.
```
dgen = DataGen(100, 100, 2)
dgen.generate('mmc', <path to saving directory>)
```
In the above snippet parameters supplied to `DataGen` are the height, width and number of channels of the input image. For each target the network will be simulated for a fixed number of iterations and for each iteration corresponding arm configuration/position will be stored as a PNG image in the directory corresponding to the target. Example dataset can be found in the `data` folder.

##### Autoencoder 
The notebooks `autoencoder.ipynb` and `autoencoder_pytorch.ipynb` implements simple autoencoder to predict next image in the sequence given previous images. The purpose was to evaluate how well an autoencoder performs in predicting the next arm position given previous arm positions without having any knowledge of the underlying body model. Following figure shows an example output from test set:


##### Autoencoder with body model
The notebooks `extended_body_model.ipynb` extends simple autoencoder mentioned in the previous section with a latent layer implementing the body model. The latent layer performs one step or iteration of the MMC/SimpleMovement network on its input. Therefore, ideally it should be able to predict the next arm position more accurately compared to simple autoencoder.