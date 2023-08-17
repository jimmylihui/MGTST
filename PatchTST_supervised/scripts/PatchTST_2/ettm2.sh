data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
for pred_len in 96 192 336 720
do
    python -u /ssddata/data/jiahuili/PatchTST/PatchTST_supervised/run_longExp.py
      
done