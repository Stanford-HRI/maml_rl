from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from maml_examples.waterworld import WaterWorld

import csv
import joblib
import numpy as np
import pickle
import tensorflow as tf

stub(globals())

# horizon of 100
initial_params_file = 'data/local/vpg-maml-point100/trpomaml1_fbs20_mbs40_flr_0.5metalr_0.01_step11/params.pkl'

goals = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7]

step_size = 0.01

avg_returns = []
for goal in goals:
    env = TfEnv(normalize(WaterWorld(evader_params=[goal,0.05])))
    n_itr = 10
    policy = GaussianMLPPolicy(  # random policy
        name='policy',
        env_spec=env.spec,
        hidden_sizes=(100, 100),
    )
    policy = None
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = VPG(
        env=env,
        policy=policy,
        load_policy=initial_params_file,
        baseline=baseline,
        batch_size=4000,  # 2x
        max_path_length=100,
        n_itr=n_itr,
        optimizer_args={'init_learning_rate': step_size, 'tf_optimizer_args': {'learning_rate': 0.5*step_size}, 'tf_optimizer_cls': tf.train.GradientDescentOptimizer}
    )

    run_experiment_lite(
        algo.train(),
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=1,
        exp_prefix='vpg_maml_point100_test',
        exp_name='trpomaml1_theta'+str(goal)+'_test',
        plot=False,
        python_command='python3' 
    )

    with open('data/local/vpg-maml-point100-test/trpomaml1_theta'+str(goal)+'_test/progress.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        i = 0
        row = None
        returns = []
        j = 0
        for row in reader:
            i+=1
            if i ==1:
                while row[j] != 'AverageReturn':
                    j += 1
            else:
                returns.append(float(row[j]))
    avg_returns.append(returns)

print(avg_returns)