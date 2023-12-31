if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=MSPTST

root_path_name=/ssddata/data/jiahuili/PatchTST/all_six_datasets/ETT-small/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2021
for pred_len in 96 192 336 720
do
    python -u /ssddata/data/jiahuili/PatchTST/PatchTST_supervised/run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 8 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --scale 1\
      --gate 0\
      --group 1\
      --itr 1 --batch_size 128 --learning_rate 0.0001 >/ssddata/data/jiahuili/PatchTST/logs/$model_name'_scale_1_gate_0'$model_id_name'_'$seq_len'_'$pred_len.log 
done