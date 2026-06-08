RL_TSC = ('dqn', 'ddpg', 'ppo')
TRADITIONAL_TSC = ('websters', 'sotl')
SUPPORTED_TSC = TRADITIONAL_TSC + RL_TSC


def is_rl_controller(tsc):
    return tsc in RL_TSC
