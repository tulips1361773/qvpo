运行结果：
1、run8_sac_improved.sh:train reward和eval reward最后收敛到38.
当时的环境参数：
                 comm_penalty_coef=0.5,
                 comm_softplus_kappa: float = 2.0, # 稍微增加陡峭度
                 comm_penalty_clip_per_user=20.0,
                 comm_penalty_clip_total=50.0,

                 action_smooth_coef: float = 0.8,
其余参数见文件。


2、为了和main.py的qvpo算法做对比，sac作为基线实验。需要修改参数，和secure.sh一致，因此新建run9_sac.sh运行。
但是run9_sac.sh运行时，发现reward一路走低，train和eval时的reward收敛到-300.

sucure.sh的训练结果收敛到38左右。

任务：
给出sac不收敛的原因，先不修改代码，给出改进方案以及理由，并且给出修改位置以及代码对比。。
