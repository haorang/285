for seed in 12345 45678 12322; do
    python train_PPO.py --env walker_walk --seed $seed --lr 0.00005 --batch-size 64 --n-envs 32 --ent-coef 0.0 --n-steps 500 --total-timesteps 4000000 --num-layer 3 --hidden-dim 256 --clip-init 0.4 --gae-lambda 0.92 --eval-dir logs/PPO/eval/$seed/
done
