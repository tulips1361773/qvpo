# 3/1/22:15\
# 修复reset时传入相同种子的问题
# 参数同run1

# 3/10 0:59 适配myenv3环境  eav coaf:2-->5,其他参数不设置，想要追求run6.sh
# SAC基线实验，使用Agent端归一化（不启用myenv3的StateNormalizer）


python sac2.py \
    --exp-name "sac_myenv3_noarg" \
    --seed 42 \
    --total-timesteps 1000000 \
    --learning-starts 4500 \
    --buffer-size 1000000 \
    --gamma 0.99 \
    --tau 0.005 \
    --batch-size 512 \
    --policy-lr 3e-4 \
    --q-lr 1e-3 \
    --cuda "cuda:1"
    --eav_penalty_coef 5.0 \
    --eav_penalty_clip_max 1000.0 