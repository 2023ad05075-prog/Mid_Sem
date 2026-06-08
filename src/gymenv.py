import gymnasium as gym
import numpy as np
from src.sumosim import SumoSim

class SumoGymEnv(gym.Env):
    """Wraps SumoSim as a standard Gymnasium environment.
    Supports single-agent (one intersection) for DQN/PPO training.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, args, netdata, tsc_id, idx=0):
        super().__init__()
        self.args = args
        self.netdata = netdata
        self.tsc_id = tsc_id
        self.sim = SumoSim(args.cfg_fp, args.sim_len, args.tsc,
                           args.nogui, netdata, args, idx)
        
        n_lanes = len(netdata['inter'][tsc_id]['incoming_lanes'])
        n_phases = len(netdata['inter'][tsc_id]['green_phases'])
        
        # State: density + queue + wait + speed per lane + phase one-hot
        obs_size = n_lanes * 4 + n_phases
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        # Action: which green phase to activate next
        self.action_space = gym.spaces.Discrete(n_phases)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.gen_sim()
        # Create TSC without RL agent (we control actions externally)
        self.sim.create_tsc(
            rl_stats={self.tsc_id: {'updates': 0, 'n_exp': 0, 'max_r': 1.0,
                                    'online': None, 'target': None}},
            exp_replays={self.tsc_id: []},
            eps=0.0
        )
        obs = self.sim.tsc[self.tsc_id].get_state().astype(np.float32)
        return obs, {}

    def step(self, action):
        tsc = self.sim.tsc[self.tsc_id]
        # Apply chosen phase
        phase_str = tsc.green_phases[action]
        self.sim.conn.trafficlight.setRedYellowGreenState(self.tsc_id, phase_str)
        # Advance sim by g_min steps
        for _ in range(self.args.g_min):
            if self.sim.vehiclegen:
                self.sim.vehiclegen.run()
            self.sim.update_travel_times()
            self.sim.sim_step()

        obs     = tsc.get_state().astype(np.float32)
        reward  = float(tsc.get_reward())
        terminated = self.sim.t >= self.sim.sim_len
        truncated  = False
        info = {'t': self.sim.t}
        return obs, reward, terminated, truncated, info

    def close(self):
        self.sim.close()