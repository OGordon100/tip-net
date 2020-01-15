from gym.envs.registration import register

register(
    id='tip_net-v0',
    entry_point='tip_net.envs:TipNet',
)