# 3/1/22:15\
# 修复reset时传入相同种子的问题
# 参数同run1

python sac2.py \
    --exp-name "1" \
    --seed 42 \
    --total-timesteps 1000000 \
    --learning-starts 4500 \
    --eav_penalty_coef 0.5 \
    --comm_penalty_coef 1.5 \
    --buffer-size 1000000 \
    --gamma 0.99 \
    --tau 0.005 \
    --batch-size 512 \
    --policy-lr 3e-4 \
    --q-lr 1e-3 \
    --reward_scale 0.1 \
    --cuda "cuda:1" \