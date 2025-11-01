#python -u main_informer.py --model informer --data WTH --features M --attn prob --d_layers 2 --e_layers 3 --itr 3 --label_len 168 --pred_len 24 --seq_len 168 --des 'Exp'
# 训练 Normal 模型
python -u main_informer.py --model informer --data Normal --root_path ./data/FLEA/ --features S --embed fixed --freq t --seq_len 500 --label_len 50 --pred_len 50 --batch_size 64 --train_epochs 20 --des 'Exp_fixed_500' --itr 3

# 训练 Jam 模型（仅改 data_path 和 checkpoints）
python -u main_informer.py \
  --model informer \
  --data custom \
  --root_path ./data/FLEA/ \
  --data_path Jam.csv \
  --features S \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --target "Motor Y Voltage" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --batch_size 32 \
  --train_epochs 10 \
  --patience 3 \
  --itr 1 \
  --checkpoints ./checkpoints/informer_Jam/

# 训练 Position 模型（仅改 data_path 和 checkpoints）
python -u main_informer.py \
  --model informer \
  --data custom \
  --root_path ./data/FLEA/ \
  --data_path Position.csv \
  --features S \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --target "Motor Y Voltage" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --batch_size 32 \
  --train_epochs 10 \
  --patience 3 \
  --itr 1 \
  --checkpoints ./checkpoints/informer_Position/

# 训练 Spall 模型（仅改 data_path 和 checkpoints）
python -u main_informer.py \
  --model informer \
  --data custom \
  --root_path ./data/FLEA/ \
  --data_path Spall.csv \
  --features S \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --target "Motor Y Voltage" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --batch_size 32 \
  --train_epochs 10 \
  --patience 3 \
  --itr 1 \
  --checkpoints ./checkpoints/informer_Spall/
