# MGTST

## Key Designs

:star2: **Parallel Multi-Scale Architecture(PMSA)**: This approach involves creating multiple copies of the dataset. In each scale, the input data is embedded with different sequence lengths and stride lengths. The embedding tensors from each scale are then concatenated and projected to the output tensor.

:star2: **Temporal Embedding with Representation Tokens(TERT)**: In this method, the channel independent input is transformed into tensors, with a representation token attached at the end of each tensor.

:star2: **Cross-Channel Attention and Gated Mechanis(CCAGM)**: Tokens undergo self-attention to facilitate cross-channel interaction. The representation token is then used in a dot product operation with the embedding tensors.

:star2: **Channel grouping(CG)**:  When dealing with datasets that have a large number of channels, the channels are grouped together and the CCAGM approach is only applied within each group.


## Getting Started
![alt text]()




### Supervised Learning

1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download all the datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Create a seperate folder ```./dataset``` and put all the csv files in the directory.

3. Training. All the scripts are in the directory ```./scripts/```. The default code is in  ```./scripts/MGTST```. We also provide ablation script in ```./scripts/MGTST_scale_1``` ,  ```./scripts/MGTST_scale_1_gate_0``` and  ```./scripts/MGTST_scale_10_gate_0``` for abalation study. 


You can adjust the hyperparameters based on your needs (e.g. different patch length, different look-back windows and prediction lengths.). Remember to change the dataset location to where you save it.

