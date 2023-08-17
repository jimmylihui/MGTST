if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=MGTST

root_path_name=./all_six_datasets/traffic/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

random_seed=2021
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
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
      --enc_in 862 \
      --e_layers 1 \
      --n_heads 8 \
      --d_model 64 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --lradj 'TST'\
      --train_epochs 100\
      --patience 10\
      --pct_start 0.2\
      --learning_rate 0.0001\
      --scale 1\
      --gate 1\
      --channel_dependent 0\
      --group 30\
      --itr 1 --batch_size 24  >logs/$model_name'_'scale_1_$model_id_name'_'$seq_len'_'$pred_len.log 
done