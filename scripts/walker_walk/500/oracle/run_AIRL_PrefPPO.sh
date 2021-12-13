for seed in 12345; do
    python train_AIRL_PrefPPO.py --env walker_walk --seed $seed  --lr 0.00005 --batch-size 64 --n-envs 32 --ent-coef 0.0 --n-steps 500 --total-timesteps 4000000 --num-layer 3 --hidden-dim 256 --clip-init 0.4 --gae-lambda 0.92  --re-feed-type 1 --re-num-interaction $1 --teacher-beta -1 --teacher-gamma 1 --teacher-eps-mistake 0 --teacher-eps-skip 0 --teacher-eps-equal 0 --re-segment 50 --unsuper-step 32000 --unsuper-n-epochs 50 --re-max-feed 500 --re-batch 50 --expert-demos logs/PPO/eval/12345/rollout_512000.npz
done
