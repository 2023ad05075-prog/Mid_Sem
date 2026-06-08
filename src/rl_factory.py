from src.rlagents.dqnagent import DQNAgent
from src.rlagents.ddpgagent import DDPGAgent
from src.rlagents.ppoagent import PPOAgent

def rl_factory(rl_type, args, neural_network, exp_replay, rl_stats, n_actions, eps):

    if rl_type == 'dqn':
        return DQNAgent(
            neural_network,
            eps,
            exp_replay,
            n_actions,
            args.nsteps,
            args.batch,
            args.nreplay,
            args.gamma,
            rl_stats,
            args.mode,
            args.updates
        )

    elif rl_type == 'ddpg':
        return DDPGAgent(
            neural_network,
            eps,
            exp_replay,
            n_actions,
            args.nsteps,
            args.batch,
            args.nreplay,
            args.gamma,
            rl_stats,
            args.mode,
            args.updates
        )

    elif rl_type == 'ppo':
        return PPOAgent(
            neural_network,
            eps,
            exp_replay,
            n_actions,
            args.nsteps,
            args.batch,
            args.nreplay,
            args.gamma,
            rl_stats,
            args.mode,
            args.updates,
            clip_ratio=args.clip_ratio,
            ppo_epochs=args.ppo_epochs,
            entropy_coef=args.entropy_coef,
            lr=args.lr,
            lrc=args.lrc
        )

    else:
        assert 0, 'Supplied rl argument type ' + str(rl_type) + ' does not exist.'


'''from src.rlagents.dqnagent import DQNAgent
from src.rlagents.ddpgagent import DDPGAgent

def rl_factory(rl_type, args, neural_network, exp_replay, rl_stats, n_actions, eps):
    if rl_type == 'dqn':
        return DQNAgent(neural_network,
                        eps,                                     
                        exp_replay,                                   
                        n_actions,                                    
                        args.nsteps,                                  
                        args.batch,                                   
                        args.nreplay,                                 
                        args.gamma,                                   
                        rl_stats,
                        args.mode,
                        args.updates)                                     
    elif rl_type == 'ddpg':
        return DDPGAgent(neural_network,
                         eps,     
                         exp_replay,              
                         n_actions,                
                         args.nsteps,              
                         args.batch,               
                         args.nreplay,             
                         args.gamma,               
                         rl_stats,                
                         args.mode,
                         args.updates)                                     
    else:
        #raise not found exceptions
        assert 0, 'Supplied rl argument type '+str(rl_type)+' does not exist.'''
