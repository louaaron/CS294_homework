import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--DAgger_iter', type = int, default=5)
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()
        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        training_obs = expert_data['observations']
        training_actions = expert_data['actions']

        print('observation shape', expert_data['observations'].shape)
        print('action shape', expert_data['actions'].shape)

        our_model = model.Model(training_obs, training_actions, args.envname[:-3], 'DAgger')
        our_model.train()

        for i in range(args.DAgger_iter):
            new_obs = []
            new_actions = []
            obs = env.reset()
            done = False
            while not done:
                action = our_model.sample(obs)
                obs, _, done, _ = env.step(action)
                if args.render:
                    env.render()
                corrected_action = policy_fn(obs[None, :])
                new_obs.append(obs)
                new_actions.append(corrected_action)

            training_obs = np.concatenate((training_obs, obs[None, :]), axis = 0)
            training_actions = np.concatenate((training_actions, corrected_action[None, :]), axis = 0)
            our_model.train(train_data = np.array(new_obs), test_data = np.array(new_actions), number = i)

if __name__ == '__main__':
    main()
