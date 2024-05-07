python src/script.py \
  --run_name "lower eps" \
  --steps_per_epoch 1000 \
  --buffer_size 100000 \
  --batch_size 32 \
  --n_envs 8 \
  --eps 0.15 \
  --min_eps 0.05 \
  --eps_anneal_steps 50000 \
  --gamma 0.99 \
  --lr 0.00025 \
  --min_lr 1e-6 \
  --epochs 500 \
  --eval_freq 1 \
  --save_freq 2 \
  --seed 0 \
  --max_eval_steps 3000 \
  --wandb_mode offline 

