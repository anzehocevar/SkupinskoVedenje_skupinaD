from typing import Iterator
import numpy as np
import numpy.typing as npt
import src.constants

def initial_conditions(N: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Uniformly random placement of fish in circle with centre (0, 0) and radius R"""
    R: float = (src.constants.l_att/2.0) * np.sqrt(N/np.pi)
    r: npt.NDArray = R * np.sqrt(np.random.rand(N))
    angle: npt.NDArray = np.random.rand(N) * 2 * np.pi
    u_x: npt.NDArray = r * np.cos(angle)
    u_y: npt.NDArray = r * np.sin(angle)
    phi: npt.NDArray = np.random.rand(N) * 2 * np.pi
    return u_x, u_y, phi

def wrap_to_pi(x: npt.NDArray) -> npt.NDArray:
    x = np.where(x > +np.pi, x - 2*np.pi, x)
    return np.where(x < -np.pi, x + 2*np.pi, x)

def wrap_to_pi_single(x: float) -> float:
    x = x - 2*np.pi if x > np.pi else x
    return x + 2*np.pi if x < -np.pi else x

def compute_pairwise_distances(u_x: npt.NDArray, u_y: npt.NDArray) -> npt.NDArray:
    N: int = u_x.shape[0]
    d_ij: npt.NDArray = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d_ij[i, j] = np.sqrt(np.square(u_x[i]-u_x[j]) + np.square(u_y[i]-u_y[j]))
    d_ij = d_ij + d_ij.T
    return d_ij

def run_with_groups() -> Iterator[tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]]:
    dist_critical: float = 4*src.constants.l_att
    for u_x, u_y, phi, d_ij in run():
        N: int = u_x.shape[0]
        nearest_neighbours_indexes: npt.NDArray = np.zeros((N, N-1)).astype(int)    # N-1 because we know every fish is nearest to itself
        for i in range(N):
            nearest_neighbours_indexes[i] = np.argsort(d_ij[i])[1:]
        group: npt.NDArray = np.arange(N)
        last_in_sequence: npt.NDArray = group.copy()
        for i in range(N):
            i1: int = i
            i2: int = nearest_neighbours_indexes[i, 0]
            if d_ij[i1, i2] <= dist_critical:
                while nearest_neighbours_indexes[i2, 0] != i1:
                    i1 = i2
                    i2 = nearest_neighbours_indexes[i1, 0]
                last_in_sequence[i] = min(i1, i2)
        # group = group[last_in_sequence]
        for i in range(N):
            group[i] = group[last_in_sequence[i]]
        yield u_x, u_y, phi, group

def run() -> Iterator[tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]]:
    np.random.seed(src.constants.seed)
    u_x_last: npt.NDArray
    u_y_last: npt.NDArray
    phi: npt.NDArray
    N: int = src.constants.n_fish
    u_x_last, u_y_last, phi = initial_conditions(N)
    t_last: npt.NDArray = np.zeros(N)
    # tau: npt.NDArray = np.abs(np.random.normal(loc=src.constants.tau_n_mean, scale=src.constants.tau_n_std, size=N))
    tau: npt.NDArray = 0.5 * np.sqrt(2/np.pi) * np.sqrt(-2.0 * np.log(np.random.uniform(size=N) + 1e-16))
    t_next: npt.NDArray = t_last + np.abs(tau)
    d_ij: npt.NDArray = compute_pairwise_distances(u_x_last, u_y_last)

    while True:
        # Find time and fish of next kick
        i: int = int(np.argmin(t_next))
        t: float = t_next[i]

        # Compute time since all fish's last kicks
        s: npt.NDArray = t - t_last

        # Compute position of every fish at this time: u(t)
        phi_unitvec_x, phi_unitvec_y = np.cos(phi), np.sin(phi)
        scale: npt.NDArray = tau * (
            (1-np.exp(-s/src.constants.tau_0)) /
            (1-np.exp(-tau/src.constants.tau_0))
        )
        u_x: npt.NDArray = u_x_last + scale * phi_unitvec_x
        u_y: npt.NDArray = u_y_last + scale * phi_unitvec_y

        # Compute distances from fish i to all other fish
        u_x_i, u_y_i = u_x[i], u_y[i]
        d_i: npt.NDArray = np.sqrt(np.square(u_x_i - u_x) + np.square(u_y_i - u_y))

        # Compute angle(s) of perception for fish i
        u_x_relative, u_y_relative = u_x - u_x_i, u_y - u_y_i
        theta: npt.NDArray = np.arctan2(u_y_relative, u_x_relative)
        theta = wrap_to_pi(theta)
        psi: npt.NDArray = theta - phi
        psi = wrap_to_pi(psi)

        # Compute relative headings
        phi_relative: npt.NDArray = phi - phi[i]
        phi_relative = wrap_to_pi(phi_relative)

        # Compute the heading angle changes
        d_i_sq: npt.NDArray = np.square(d_i)
        delta_phi: npt.NDArray = (
            src.constants.gamma_att * ((d_i * np.sin(psi)) / (1+d_i_sq/np.square(src.constants.l_att))) +
            src.constants.gamma_ali * (1 + src.constants.eta * np.cos(psi)) * np.exp(-d_i_sq/np.square(src.constants.l_ali)) * np.sin(phi_relative)
        )
        influence: npt.NDArray = np.abs(delta_phi)
        top_k_indexes: npt.NDArray = np.argpartition(influence, -src.constants.k)[-src.constants.k:]

        # Compute new heading
        # phi[i] = phi[i] + src.constants.gamma_rand * np.random.normal(loc=0, scale=1) + np.sum(delta_phi[top_k_indexes])
        phi[i] = phi[i] + src.constants.gamma_rand * (
            np.sqrt(-2.0 * np.log(np.random.random()+1e-16)) * np.sin(2*np.pi*np.random.random())
        ) + np.sum(delta_phi[top_k_indexes])
        phi[i] = wrap_to_pi_single(phi[i])

        # Prepare for next kick
        # tau_i: float = np.abs(np.random.normal(loc=src.constants.tau_n_mean, scale=src.constants.tau_n_std))
        tau_i: float = 0.5 * np.sqrt(2/np.pi) * np.sqrt(-2.0 * np.log(np.random.uniform() + 1e-16))
        l_i: float = tau_i
        tau[i] = tau_i
        t_last[i] = t
        t_next[i] = t + tau_i
        u_x_new = u_x_last[i] + phi_unitvec_x[i] * l_i
        u_y_new = u_y_last[i] + phi_unitvec_y[i] * l_i
        u_x_last[i], u_x[i] = u_x_new, u_x_new
        u_y_last[i], u_y[i] = u_y_new, u_y_new

        # Compute new distances from fish i to all other fish
        d_i = np.sqrt(np.square(u_x_new - u_x) + np.square(u_y_new - u_y))
        d_ij[i] = d_i
        d_ij[:, i] = d_i
        yield u_x, u_y, phi, d_ij

if __name__ == "__main__":
    run()
