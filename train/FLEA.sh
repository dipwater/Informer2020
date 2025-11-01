# 训练 Normal 模型
python -u main_informer.py --model informer --data Normal --root_path ./data/FLEA/ --features S --batch_size 64 --embed fixed --freq t --seq_len 500 --label_len 50 --pred_len 50 --train_epochs 20 --des 'Exp_fixed' --itr 3

python -u main_informer.py --model informer --data Normal --root_path ./data/FLEA/ --features MS --batch_size 64 --embed fixed --freq t --seq_len 500 --label_len 50 --pred_len 50 --train_epochs 20 --des 'Exp_fixed' --itr 3


# 训练 Jam 模型
python -u main_informer.py --model informer --data Jam --root_path ./data/FLEA/ --features S --batch_size 64 --embed fixed --freq t --seq_len 500 --label_len 50 --pred_len 50 --train_epochs 20 --des 'Exp_fixed' --itr 3

python -u main_informer.py --model informer --data Jam --root_path ./data/FLEA/ --features MS --batch_size 64 --embed fixed --freq t --seq_len 500 --label_len 50 --pred_len 50 --train_epochs 20 --des 'Exp_fixed' --itr 3


# 训练 Position 模型
python -u main_informer.py --model informer --data Position --root_path ./data/FLEA/ --features S --batch_size 64 --embed fixed --freq t --seq_len 500 --label_len 50 --pred_len 50 --train_epochs 20 --des 'Exp_fixed' --itr 3

python -u main_informer.py --model informer --data Position --root_path ./data/FLEA/ --features MS --batch_size 64 --embed fixed --freq t --seq_len 500 --label_len 50 --pred_len 50 --train_epochs 20 --des 'Exp_fixed' --itr 3


# 训练 Spall 模型
python -u main_informer.py --model informer --data Spall --root_path ./data/FLEA/ --features S --batch_size 64 --embed fixed --freq t --seq_len 500 --label_len 50 --pred_len 50 --train_epochs 20 --des 'Exp_fixed' --itr 3

python -u main_informer.py --model informer --data Spall --root_path ./data/FLEA/ --features MS --batch_size 64 --embed fixed --freq t --seq_len 500 --label_len 50 --pred_len 50 --train_epochs 20 --des 'Exp_fixed' --itr 3
