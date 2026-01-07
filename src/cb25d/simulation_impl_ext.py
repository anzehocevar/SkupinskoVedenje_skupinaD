# pyright:strict
import heapq
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Self, TypedDict, List, Tuple

import numpy as np
import pygame

from cb25d.render_environment import RenderEnvironment
from cb25d.simulation_framework import SimulationRecorder, SimulationRenderer

# Internal Event Codes for the Priority Queue
EVT_BURST_STEP = 0
EVT_START_COAST = 1
EVT_NEW_CYCLE = 2


@dataclass(kw_only=True, slots=True)
class SimulationImplExtended:
    """
    An extension of the Burst-and-Coast model introducing a specific Duty Cycle.
    
    Unlike the original model which treats kicks as instantaneous events (teleportation),
    this implementation integrates velocity over time. 
    
    Fish exist in two states:
    1. Burst: Active swimming (Linear motion). The fish can turn during this phase.
    2. Coast: Passive gliding (Exponential decay). The fish cannot turn.
    """
    cycle_tau_burst: np.ndarray
    """The duration of the burst phase for the current cycle."""

    # Model Constants

    c_n: int
    """Number of fish/agents."""
    
    c_l_att: float
    """Range of attraction interaction
    Possible values: 0.0, 0.6
    """
    
    c_l_ali: float
    """Range of alignment interaction 
    Possible values: 0.0, 1.2
    """
    
    c_gamma_att: float
    """Strength of attraction force."""

    # k=1:
    #   Swarming: gamma_att, gamma_ali = (0.6, 0.6)
    #   Schooling: gamma_att, gamma_ali = (0.22, 0.6)
    #   Milling: gamma_att, gamma_ali = (0.37, 0.2)
    # k=2:
    #   Swarming: gamma_att, gamma_ali = (0.6, 0.2)
    #   Schooling: gamma_att, gamma_ali = (0.2, 0.3)
    
    c_gamma_ali: float
    """Strength of alignment force."""
    
    c_gamma_rand: float
    """Intensity of random fluctuations (noise).
    
    Note: In the continuous extension, this is scaled by 1/sqrt(n_omega) 
    to maintain consistent diffusion characteristics.
    """
    
    c_k: int
    """Number of influential neighbours to consider."""
    
    c_tau_0: float
    """Relaxation time for the exponential speed decay during coasting."""
    
    c_tau_n_mean: float
    """Mean duration of a kick cycle (Burst + Coast)."""
    
    c_l_n_mean: float
    """Mean length traveled during a kick cycle."""
    
    c_eta: float
    """Coefficient of anisotropy (perception blind spot behind the fish)."""

    # --- Extension Parameters ---

    c_omega: float
    """Duty cycle: The ratio between the duration of the burst phase and the total duration.
    
    Range: [0, 1]
    - 0.0: Pure coasting (instantaneous kick limit).
    - 1.0: Continuous swimming (no gliding).
    """
    
    c_n_omega: int
    """The number of decision instants within the burst phase.
    
    Increasing this value approximates a continuous decision-making process.
    Forces are scaled by 1/n_omega to ensure the total turning capability 
    per burst remains comparable to the discrete model.
    """

    # Variables

    time: float
    """Current simulation time in seconds."""
    
    rng: np.random.Generator
    """Random number generator state."""

    # Physical State Arrays (Size N)
    u_x: np.ndarray
    """Current X position."""
    u_y: np.ndarray
    """Current Y position."""
    phi: np.ndarray
    """Current Heading angle (radians)."""

    # Extended Physics State (Size N)
    is_bursting: np.ndarray
    """Boolean mask. True if fish is currently in Burst phase, False if Coasting."""
    
    phase_start_time: np.ndarray
    """Absolute time when the current phase (Burst or Coast) began. 
    Used to calculate exponential decay during coasting."""
    
    cycle_v_peak: np.ndarray
    """The peak velocity calculated for the current cycle to satisfy distance constraints."""
    
    cycle_tau_coast: np.ndarray
    """The duration of the coasting phase for the current cycle."""
    
    burst_step_counter: np.ndarray
    """Integer counter tracking how many decision sub-steps have occurred in the current burst."""

    # Asynchronous Event Queue
    event_queue: List[Tuple[float, int, int]]
    """Min-Heap Priority Queue managing asynchronous events.
    Format: (time, fish_index, event_type)
    """

    _dirty: bool = False
    """Flag indicating if this state has been snapshotted and needs copying before modification."""

    def _undirty(self):
        """Creates a deep copy of arrays if this state is dirty."""
        if not self._dirty:
            return
        
        # Copy RNG state
        bg = self.rng.bit_generator.__class__(0)
        bg.state = self.rng.bit_generator.state
        self.rng = np.random.Generator(bg)
        
        # Copy arrays
        self.u_x = np.copy(self.u_x)
        self.u_y = np.copy(self.u_y)
        self.phi = np.copy(self.phi)
        
        # Copy extended state
        self.is_bursting = np.copy(self.is_bursting)
        self.phase_start_time = np.copy(self.phase_start_time)
        self.cycle_v_peak = np.copy(self.cycle_v_peak)
        self.cycle_tau_coast = np.copy(self.cycle_tau_coast)
        self.burst_step_counter = np.copy(self.burst_step_counter)
        self.cycle_tau_burst = np.copy(self.cycle_tau_burst)
        
        # Deep copy the heap
        self.event_queue = deepcopy(self.event_queue)
        
        self._dirty = False

    def _move_all_fish(self, t_target: float):
        """
        Integrates the physics of all fish from self.time to t_target.
        
        This handles the continuous motion:
        - Bursting fish move linearly: x += v_peak * dt
        - Coasting fish move logarithmically: integral of v_peak * exp(-t/tau0)
        """
        dt = t_target - self.time
        if dt <= 0:
            return

        # 1. Bursting Fish (Linear Motion)
        burst_mask = self.is_bursting
        dist_burst = self.cycle_v_peak * dt
        
        self.u_x[burst_mask] += dist_burst[burst_mask] * np.cos(self.phi[burst_mask])
        self.u_y[burst_mask] += dist_burst[burst_mask] * np.sin(self.phi[burst_mask])

        # 2. Coasting Fish (Exponential Decay)
        # We need the integral of velocity over the time step [t, t+dt]
        coast_mask = ~burst_mask
        if np.any(coast_mask):
            # Time elapsed since the start of the coast phase
            t_since_start = self.time - self.phase_start_time[coast_mask]
            
            # Velocity at the start of this specific integration step
            v_current = self.cycle_v_peak[coast_mask] * np.exp(-t_since_start / self.c_tau_0)
            
            # Integral(v * exp(-t/tau0) dt) -> v * tau0 * (1 - exp(-dt/tau0))
            dist_coast = v_current * self.c_tau_0 * (1 - np.exp(-dt / self.c_tau_0))
            
            self.u_x[coast_mask] += dist_coast * np.cos(self.phi[coast_mask])
            self.u_y[coast_mask] += dist_coast * np.sin(self.phi[coast_mask])

        self.time = t_target

    def step(self) -> None:
        """
        Advances the simulation by processing exactly one event from the priority queue.
        This provides O(log N) scheduling efficiency.
        """
        self._undirty()

        if not self.event_queue:
            return

        # Pop the next scheduled event
        t_event, i, event_type = heapq.heappop(self.event_queue)

        # 1. Integrate physics for everyone up to this event's time
        self._move_all_fish(t_event)

        # 2. Handle the event
        if event_type == EVT_NEW_CYCLE:
            self._handle_new_cycle(i)
        elif event_type == EVT_BURST_STEP:
            self._handle_burst_step(i)
        elif event_type == EVT_START_COAST:
            self._handle_start_coast(i)

    def _handle_new_cycle(self, i: int):
        """Start of a full Kick-Glide cycle. Samples statistics and sets peak velocity."""
        # Sample parameters from distributions
        # Rayliegh distribution approximation for tau
        tau = 0.5 * np.sqrt(2 / np.pi) * np.sqrt(-2.0 * np.log(self.rng.uniform() + 1e-16))

        # Gaussian for length
        # l_n = np.abs(self.rng.normal(self.c_l_n_mean, 0.2))
        l_n = tau
        
        # Calculate Phase durations
        tau_burst = self.c_omega * tau
        tau_coast = (1 - self.c_omega) * tau
        
        # Calculate v_peak such that total distance traveled equals l_n
        # Distance = Distance_Burst + Distance_Coast
        # Distance_Coast = Integral(v_peak * exp(-t/tau0)) from 0 to tau_coast
        coast_integral = self.c_tau_0 * (1 - np.exp(-tau_coast / self.c_tau_0))
        dist_factor = tau_burst + coast_integral
        
        v_peak = l_n / dist_factor if dist_factor > 0 else 0

        # Update State
        self.cycle_v_peak[i] = v_peak
        self.cycle_tau_coast[i] = tau_coast
        self.is_bursting[i] = True
        self.phase_start_time[i] = self.time
        self.burst_step_counter[i] = 0
        self.cycle_tau_burst[i] = tau_burst

        # Schedule First Action
        if self.c_omega < 1e-3:
            # If duty cycle is ~0, skip burst entirely
            tau_burst = 0.0
            self.is_bursting[i] = False
            heapq.heappush(self.event_queue, (self.time, i, EVT_START_COAST))
        else:
            # Schedule the first decision/turn step
            dt_step = tau_burst / max(1, self.c_n_omega)
            heapq.heappush(self.event_queue, (self.time + dt_step, i, EVT_BURST_STEP))

    def _handle_burst_step(self, i: int):
        """A decision instant within the burst phase. The fish interacts and turns."""
        # 1. Calculate Distances (Optimized O(N))
        dx = self.u_x - self.u_x[i]
        dy = self.u_y - self.u_y[i]
        dist_sq = dx**2 + dy**2
        dist = np.sqrt(dist_sq)
        
        # 2. Calculate Angles
        # Angle from i to j
        angle_to_j = np.arctan2(dy, dx)
        # Angle of perception (psi)
        psi = angle_to_j - self.phi[i]
        psi = (psi + np.pi) % (2 * np.pi) - np.pi # Wrap
        
        # Relative heading
        phi_rel = (self.phi - self.phi[i] + np.pi) % (2 * np.pi) - np.pi

        # 3. Calculate Forces
        att = (dist * np.sin(psi)) / (1 + dist_sq / (self.c_l_att**2))
        ali = (1 + self.c_eta * np.cos(psi)) * np.exp(-dist_sq / (self.c_l_ali**2)) * np.sin(phi_rel)
        
        delta_phi_all = self.c_gamma_att * att + self.c_gamma_ali * ali
        
        # 4. Filter Top K Influential Neighbors
        influence = np.abs(delta_phi_all)
        influence[i] = -1 # Exclude self
        
        if self.c_k >= self.c_n - 1:
            neighbors = np.delete(np.arange(self.c_n), i)
        else:
            neighbors = np.argpartition(influence, -self.c_k)[-self.c_k:]

            
        social_turn = np.sum(delta_phi_all[neighbors])
        
        # 5. Apply Turn
        # Scaling: Interactions divided by n_omega (additive)
        # Noise divided by sqrt(n_omega) (diffusive)
        n_steps = max(1, self.c_n_omega)
        noise = self.c_gamma_rand * self.rng.normal(0, 1)
        
        d_phi = (social_turn / n_steps) + (noise / np.sqrt(n_steps))
        self.phi[i] = (self.phi[i] + d_phi) % (2 * np.pi)

        # 6. Schedule Next Step
        self.burst_step_counter[i] += 1
        
        n_steps = max(1, self.c_n_omega)
        dt_step = self.cycle_tau_burst[i] / n_steps  # fixed per-substep interval

        if self.burst_step_counter[i] < n_steps:
            # schedule the next burst-step exactly dt_step later
            heapq.heappush(self.event_queue, (self.time + dt_step, i, EVT_BURST_STEP))
        else:
            # End of burst phase: schedule start coast at current time
            # (this.event time should equal phase_start_time + tau_burst if scheduling was consistent)
            heapq.heappush(self.event_queue, (self.time, i, EVT_START_COAST))

    def _handle_start_coast(self, i: int):
        """Transition from Burst to Coast."""
        self.is_bursting[i] = False
        self.phase_start_time[i] = self.time # Reset reference for exponential decay
        
        # Schedule the end of the coasting phase
        heapq.heappush(self.event_queue, (self.time + self.cycle_tau_coast[i], i, EVT_NEW_CYCLE))

    def snapshot(self):
        self._dirty = True
        return copy(self)

    def interpolate(self, other: Self, t: float):
        """Interpolates between two states for smooth rendering."""
        ret = self.snapshot()
        ret.time = (1 - t) * self.time + t * other.time
        
        # Linear interpolation of positions
        ret.u_x = (1 - t) * self.u_x + t * other.u_x
        ret.u_y = (1 - t) * self.u_y + t * other.u_y
        
        # Angular interpolation (handling wrap-around)
        diff = other.phi - self.phi
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        ret.phi = self.phi + t * diff
        
        return ret


# --- Helper Types & Functions ---

class _KwargsExtendedInit(TypedDict):
    """Type hint helper for the generator."""
    c_n: int
    c_l_att: float
    c_l_ali: float
    c_gamma_att: float
    c_gamma_ali: float
    c_gamma_rand: float
    c_k: int
    c_tau_0: float
    c_tau_n_mean: float
    c_l_n_mean: float
    c_eta: float
    c_omega: float
    c_n_omega: int
    time: float
    rng: np.random.Generator
    u_x: np.ndarray
    u_y: np.ndarray
    phi: np.ndarray
    is_bursting: np.ndarray
    phase_start_time: np.ndarray
    cycle_v_peak: np.ndarray
    cycle_tau_coast: np.ndarray
    cycle_tau_burst: np.ndarray
    burst_step_counter: np.ndarray
    event_queue: List[Tuple[float, int, int]]


def generate_extended_initial_conditions(
    *, 
    seed: int, 
    n: int, 
    l_att: float, 
    omega: float, 
    n_omega: int,
    # Standard Params defaulting to paper values
    l_ali: float = 3.0,
    gamma_att: float = 0.6,
    gamma_ali: float = 0.6,
    gamma_rand: float = 0.2,
    k: int = 1,
    tau_0: float = 0.8,
    tau_n_mean: float = 1.0,
    l_n_mean: float = 1.0,
    eta: float = 0.8,
) -> SimulationImplExtended:
    """Generates a SimulationImplExtended state with fish distributed in a circle."""
    
    rng = np.random.default_rng(seed)
    
    # Uniform placement in circle
    R = (l_att / 2.0) * np.sqrt(n / np.pi)
    r = R * np.sqrt(rng.random(n))
    angle = rng.random(n) * 2 * np.pi
    u_x = r * np.cos(angle)
    u_y = r * np.sin(angle)
    phi = rng.random(n) * 2 * np.pi
    
    # Initialize Event Queue
    # We stagger the start times so fish don't all kick at t=0
    events = []
    for i in range(n):
        # Random start time between 0 and 1s
        start_time = rng.uniform(0, 1.0)
        heapq.heappush(events, (start_time, i, EVT_NEW_CYCLE))

    return SimulationImplExtended(
        c_n=n, 
        c_l_att=l_att, 
        c_l_ali=l_ali,
        c_gamma_att=gamma_att,
        c_gamma_ali=gamma_ali,
        c_gamma_rand=gamma_rand,
        c_k=k,
        c_tau_0=tau_0,
        c_tau_n_mean=tau_n_mean,
        c_l_n_mean=l_n_mean,
        c_eta=eta,
        c_omega=omega, 
        c_n_omega=n_omega,
        rng=rng,
        time=0.0,
        u_x=u_x, 
        u_y=u_y, 
        phi=phi,
        is_bursting=np.zeros(n, dtype=bool),
        phase_start_time=np.zeros(n),
        cycle_v_peak=np.zeros(n),
        cycle_tau_coast=np.zeros(n),
        cycle_tau_burst=np.zeros(n),
        burst_step_counter=np.zeros(n, dtype=int),
        event_queue=events,
    )


@dataclass
class SimulationRecorderExtended(SimulationRecorder[SimulationImplExtended]):
    """Records statistics (Polarization, Milling, Dispersion) for the Extended model."""
    
    # Config
    skip_first_n: int = 0

    # Statistics Accumulators
    total_samples: int = 0
    total_dispersion: float = 0
    total_polarization: float = 0
    total_milling: float = 0

    def record(self, state: SimulationImplExtended):
        self.total_samples += 1
        if self.total_samples <= self.skip_first_n:
            return

        # 1. Extract State & Velocities
        u_x, u_y = state.u_x, state.u_y
        phi = state.phi
        
        # Determine velocity magnitudes based on current state (Burst vs Coast)
        speeds = state.cycle_v_peak.copy()
        coast_mask = ~state.is_bursting
        if np.any(coast_mask):
            t_elapsed = state.time - state.phase_start_time[coast_mask]
            t_elapsed = np.maximum(0, t_elapsed)
            speeds[coast_mask] *= np.exp(-t_elapsed / state.c_tau_0)
            
        v_x = speeds * np.cos(phi)
        v_y = speeds * np.sin(phi)

        # 2. Barycenter
        b_x, b_y = np.mean(u_x), np.mean(u_y)
        bv_x, bv_y = np.mean(v_x), np.mean(v_y)

        # 3. Compute Metrics
        
        # Dispersion: Mean distance to center
        dist_sq = (u_x - b_x)**2 + (u_y - b_y)**2
        dispersion = np.mean(np.sqrt(dist_sq)) # Using mean distance (vs squared)
        
        # Polarization: Norm of average heading vector
        pol_x = np.mean(np.cos(phi))
        pol_y = np.mean(np.sin(phi))
        polarization = np.sqrt(pol_x**2 + pol_y**2)
        
        # Milling: Rotation order parameter
        # Relative position
        r_x, r_y = u_x - b_x, u_y - b_y
        psi_bar = np.arctan2(r_y, r_x)
        # Heading relative to barycenter velocity
        rv_x, rv_y = v_x - bv_x, v_y - bv_y
        phi_bar = np.atan2(rv_y, rv_x)
        
        milling = np.abs(np.mean(np.sin(phi_bar - psi_bar)))

        self.total_dispersion += dispersion
        self.total_polarization += polarization
        self.total_milling += milling

    @property
    def samples(self) -> float:
        return self.total_samples - self.skip_first_n

    @property
    def results_available(self) -> bool:
        return self.samples > 0

    @property
    def dispersion(self) -> float:
        return self.total_dispersion / self.samples

    @property
    def polarization(self) -> float:
        return self.total_polarization / self.samples

    @property
    def milling(self) -> float:
        return self.total_milling / self.samples



@dataclass
class SimulationRendererExtended(SimulationRenderer[SimulationImplExtended]):
    """Renders the extended simulation state."""
    pos_size: float
    pos_color: tuple[int, int, int]
    dir_len: float
    dir_width: float
    dir_color: tuple[int, int, int]

    def draw(self, e: RenderEnvironment, state: SimulationImplExtended):
        # Draw fish based on state positions
        for x, y, p, is_burst in zip(state.u_x, state.u_y, state.phi, state.is_bursting):
            
            # Optional visual cue: Bursting fish are brighter/different color
            # If bursting, tint slightly towards Red/White
            col = self.pos_color
            if is_burst:
                # Simple tint logic (assuming green input)
                col = (min(255, col[0]+150), col[1], min(255, col[2]+150))
            
            pygame.draw.circle(e.screen, col, e.w2s((x, y)), self.pos_size / e.scale)
            
            # Direction vector
            ex, ey = np.cos(p), np.sin(p)
            pygame.draw.line(
                e.screen,
                self.dir_color,
                e.w2s((x, y)),
                e.w2s((x + ex * self.dir_len, y + ey * self.dir_len)),
                int(self.dir_width / e.scale),
            )