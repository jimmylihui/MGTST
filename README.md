# MGTST

## Getting Started



### Supervised Learning

1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download all the datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Create a seperate folder ```./dataset``` and put all the csv files in the directory.

3. Training. All the scripts are in the directory ```./scripts/```. The default code is in  ```./scripts/MGTST```. We also provide ablation script in ```./scripts/MGTST_scale_1``` ,  ```./scripts/MGTST_scale_1_gate_0``` and  ```./scripts/MGTST_scale_1o_gate_0```


You can adjust the hyperparameters based on your needs (e.g. different patch length, different look-back windows and prediction lengths.).

