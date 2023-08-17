if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=MSPTST

root_path_name=/ssddata/data/jiahuili/PatchTST/all_six_datasets/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

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
      --enc_in 21 \
      --e_layers 1 \
      --n_heads 4 \
      --d_model 32 \
      --d_ff 64 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --learning_rate 0.0001\
      --patience 5\
      --scale 1\
      --gate 1\
      --group 1\
      --itr 1 --batch_size 128 >/ssddata/data/jiahuili/PatchTST/logs/$model_name'_'scale_1_$model_id_name'_'$seq_len'_'$pred_len.log 
done