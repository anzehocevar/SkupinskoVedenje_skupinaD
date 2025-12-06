# Number of fish/agents
n_fish: int = 100

# Length of attraction
# Possible values: 1.5, 3, 5
# Default: 3
l_att: float = 3
# Length of alignment
# Possible values: 1.5, 3, 5
# Default: 3
l_ali: float = 3

# Attraction and alignment strength/intensity
# Possible values for gamma_att: [0.0, 0.6]
# Possible values for gamma_ali: [0.0, 1.2]
# k=1:
#   Swarming: gamma_att, gamma_ali = (0.6, 0.6)
#   Schooling: gamma_att, gamma_ali = (0.22, 0.6)
#   Milling: gamma_att, gamma_ali = (0.37, 0.2)
# k=2:
#   Swarming: gamma_att, gamma_ali = (0.6, 0.2)
#   Schooling: gamma_att, gamma_ali = (0.2, 0.3)
gamma_att: float = 0.6
gamma_ali: float = 0.6

# Number of neighbours to consider before every kick
# All 3 states are possible for k=1
# Only schooling and swarming are possible for k=2
k: int = 1

# Noise intensity
gamma_rand: float = 0.2

# Kick duration and length
tau_n_mean: float = 1.0
tau_n_std: float = 1.0
l_n_mean: float = 1.0

# Relaxation time
tau_0: float = 0.8

# Coefficient of anisotropy
eta: float = 0.8

# Randomness seed
seed: int = 123

# new extension constants/parameters
# The ratio between the duration of the burst phase and the total duration of both phases.
omega: float = 0.2

# The amount of decision instants within the burst phase. 
# Increasing this value will approximate a continuous decision making process within the burst phase.
n_omega: int = 5
