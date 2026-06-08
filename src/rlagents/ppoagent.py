import numpy as np
import tensorflow as tf
from src.rlagent import RLAgent

class PPOAgent(RLAgent):
    """Proximal Policy Optimization agent for traffic signal control."""

    def __init__(self, networks, epsilon, exp_replay, n_actions,
                 n_steps, n_batch, n_exp_replay, gamma,
                 rl_stats, mode, updates,
                 clip_ratio=0.2, ppo_epochs=10, entropy_coef=0.01,
                 lr=1e-4, lrc=1e-3):
        super().__init__(networks, epsilon, exp_replay, n_actions,
                         n_steps, n_batch, n_exp_replay, gamma,
                         rl_stats, mode, updates)
        self.clip_ratio    = clip_ratio
        self.ppo_epochs    = ppo_epochs
        self.entropy_coef  = entropy_coef
        # PPO uses actor-critic: networks dict has 'actor' and 'critic'
        self.actor  = networks['actor']
        self.critic = networks['critic']
        self.actor_optimizer  = tf.keras.optimizers.Adam(learning_rate=lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lrc)

    def get_action(self, state):
        """Sample action from policy distribution (epsilon-greedy during training)."""
        if self.mode == 'train':
            self.retrieve_weights('online')
        if self.mode == 'train' and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        logits = self.actor.forward(state[np.newaxis], 'online')
        probs  = tf.nn.softmax(logits).numpy()[0]
        return np.random.choice(self.n_actions, p=probs)

    def train_batch(self, update_freq):
        if len(self.exp_replay) < self.n_batch:
            return
        batch = self.sample_replay()
        states, actions, returns, old_log_probs = self.process_batch(batch)

        for _ in range(self.ppo_epochs):
            with tf.GradientTape() as actor_tape, \
                 tf.GradientTape() as critic_tape:

                logits = self.actor.models['online'](states, training=True)
                values = self.critic.models['online'](states, training=True)

                # Manual log-prob computation (no deprecated tf.distributions)
                all_log_probs = tf.nn.log_softmax(logits)
                indices = tf.stack(
                    [tf.range(tf.shape(actions)[0]), actions], axis=1)
                log_probs = tf.gather_nd(all_log_probs, indices)

                # Entropy: H(p) = -sum(p * log p)
                probs   = tf.nn.softmax(logits)
                entropy = -tf.reduce_sum(probs * all_log_probs, axis=-1)

                # PPO clipped surrogate objective
                ratio = tf.exp(log_probs - old_log_probs)
                advantages = returns - tf.squeeze(values)
                advantages = (advantages - tf.reduce_mean(advantages)) / \
                             (tf.math.reduce_std(advantages) + 1e-8)

                surr1 = ratio * advantages
                surr2 = tf.clip_by_value(ratio,
                            1 - self.clip_ratio,
                            1 + self.clip_ratio) * advantages
                actor_loss  = -tf.reduce_mean(tf.minimum(surr1, surr2))
                actor_loss -= self.entropy_coef * tf.reduce_mean(entropy)

                critic_loss = tf.reduce_mean(
                    tf.square(returns - tf.squeeze(values))
                )

            actor_grads  = actor_tape.gradient(
                actor_loss, self.actor.models['online'].trainable_variables)
            critic_grads = critic_tape.gradient(
                critic_loss, self.critic.models['online'].trainable_variables)

            self.actor_optimizer.apply_gradients(
                zip(actor_grads, self.actor.models['online'].trainable_variables))
            self.critic_optimizer.apply_gradients(
                zip(critic_grads, self.critic.models['online'].trainable_variables))

        self.rl_stats['updates'] += 1
        self.rl_stats['n_exp'] -= 1
        self.send_weights('online')

        # Sync target networks periodically
        if self.rl_stats['updates'] % update_freq == 0:
            self.actor.transfer_weights()
            self.critic.transfer_weights()

    def process_batch(self, batch):
        """Flatten trajectories into state/action/return/log_prob tensors."""
        states, actions, returns, log_probs = [], [], [], []
        for traj in batch:
            rewards = [e['r'] for e in traj]
            # Bootstrap non-terminal trajectories using critic
            last_exp = traj[-1]
            if last_exp['terminal']:
                R = 0.0
            else:
                v = self.critic.forward(last_exp['next_s'][np.newaxis], 'online')
                R = float(v[0][0])
            traj_returns = self.compute_targets(rewards, R)
            for exp, ret in zip(traj, traj_returns):
                states.append(exp['s'])
                actions.append(exp['a'])
                returns.append(ret)
                logits = self.actor.forward(exp['s'][np.newaxis], 'online')
                lp = tf.nn.log_softmax(logits).numpy()[0][exp['a']]
                log_probs.append(lp)
        return (tf.constant(states, dtype=tf.float32),
                tf.constant(actions, dtype=tf.int32),
                tf.constant(returns, dtype=tf.float32),
                tf.constant(log_probs, dtype=tf.float32))

    def send_weights(self, nettype):
        self.rl_stats[nettype] = self.actor.get_weights(nettype)

    def retrieve_weights(self, nettype):
        self.actor.set_weights(self.rl_stats[nettype], nettype)