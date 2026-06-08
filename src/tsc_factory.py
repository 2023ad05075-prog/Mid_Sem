from src.trafficsignalcontrollers.websterstsc import WebstersTSC
from src.trafficsignalcontrollers.sotltsc import SOTLTSC
from src.trafficsignalcontrollers.nextphaserltsc import NextPhaseRLTSC
from src.trafficsignalcontrollers.nextdurationrltsc import NextDurationRLTSC

def tsc_factory(tsc_type, tl, args, netdata, rl_stats, exp_replay, neural_network, eps, conn):
    if tsc_type == 'websters':
        return WebstersTSC(conn, tl, args.mode, netdata, args.r, args.y,
                           args.g_min, args.c_min,
                           args.c_max, args.sat_flow,
                           args.update_freq)
    elif tsc_type == 'sotl':
        return SOTLTSC(conn, tl, args.mode, netdata, args.r, args.y,
                       args.g_min, args.theta, args.omega,
                       args.mu )
    elif tsc_type == 'dqn':
        from src.rl_factory import rl_factory

        n_phases = len(netdata['inter'][tl]['green_phases'])
        n_duration_bins = 3 if args.g_max > args.g_min else 1
        n_actions = n_phases * n_duration_bins
        dqnagent = rl_factory(tsc_type, args,
                              neural_network, exp_replay, rl_stats, n_actions, eps)
        return NextPhaseRLTSC(conn, tl, args.mode, netdata, args.r, args.y,
                              args.g_min, dqnagent, gmax=args.g_max)
    elif tsc_type == 'ddpg':
        from src.rl_factory import rl_factory

        ddpgagent = rl_factory(tsc_type, args,
                                neural_network, exp_replay, rl_stats, 1, eps)
        return NextDurationRLTSC(conn, tl, args.mode, netdata, args.r, args.y,
                                 args.g_min, args.g_max, ddpgagent)
    elif tsc_type == 'ppo':
        from src.rl_factory import rl_factory

        n_phases = len(netdata['inter'][tl]['green_phases'])
        n_duration_bins = 3 if args.g_max > args.g_min else 1
        n_actions = n_phases * n_duration_bins
        ppoagent = rl_factory(tsc_type, args,
                                neural_network, exp_replay, rl_stats, n_actions, eps)
        # PPO uses same NextPhaseRLTSC interface as DQN
        return NextPhaseRLTSC(conn, tl, args.mode, netdata, args.r, args.y,
                              args.g_min, ppoagent, gmax=args.g_max)
    else:
        #raise not found exceptions
        assert 0, 'Supplied traffic signal control argument type '+str(tsc)+' does not exist.'
