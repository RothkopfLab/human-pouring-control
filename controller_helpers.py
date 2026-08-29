from jax import jit, numpy as jnp
from functools import partial
import numpy as np
from nioc.control import gilqr
from nioc.control.policy import create_lqg_policy
from nioc.envs.wrappers import EKFWrapper

from nioc.envs import Pouring_MugBeaker, Pouring_ConicalBeaker, Pouring_GobletBeaker
from nioc.envs import Pouring_MugBottle, Pouring_ConicalBottle, Pouring_GobletBottle
from nioc.envs import Pouring_MugVase, Pouring_ConicalVase, Pouring_GobletVase


#============== SINDYc learned forward dynamics ========================================

def fwd_sim_beaker(state, action, dt=1/120):
    """
    Forward simulate one step using discovered dynamics

    state = [w, w_dot, theta_r]
    action = theta_r_vel (control)
    dt = timestep

    (w)' = 1.000 w_dot
    (w_dot)' = 0.178 w + -2.948 w_dot + -2850.083 theta_r + -28.630 theta_r_vel + 1.830 w theta_r 
        + -1.456 w_dot theta_r + -0.603 w_dot theta_r_vel + -146.497 theta_r^2 
        + -149.937 theta_r theta_r_vel + -1.597 theta_r_vel^2 
        + 2816.767 sin(1 theta_r) + 14.975 cos(1 theta_r)
          + -20.943 sin(1 theta_r_vel) + -46.252 cos(1 theta_r_vel)
    (theta_r)' = 1.000 theta_r_vel
    """
    w, wdot, theta = state
    theta_r_vel = action

    # dynamics from SINDy model
    dw = wdot
    dwdot = (
        0.178 *  w
        -2.948 * wdot
        - 2850.083 * theta
        - 28.630 * theta_r_vel
        + 1.830 * w * theta
        - 1.456 * wdot * theta
        - 0.603 * wdot * theta_r_vel
        - 146.497 * theta**2
        - 149.937 * theta * theta_r_vel
        - 1.597 * theta_r_vel**2
        + 2816.767 * np.sin(theta)
        + 14.975 * np.cos(theta)
        - 20.943 * np.sin(theta_r_vel)
        - 46.252 * np.cos(theta_r_vel)
    )
    dtheta = theta_r_vel

    # Euler integration
    newstate = np.array([
        w + dt*dw,
        max(0.0, wdot + dt*dwdot),  # clip to enforce positivity
        theta + dt*dtheta
    ])

    return newstate



def fwd_sim_vase(state, action, dt=1/120):
    """
    Forward simulate one step using discovered dynamics

    state = [w, w_dot, theta_r]
    action = theta_r_vel (control)
    dt = timestep

    (w)' = 1.000 w_dot
    (w_dot)' = 0.353 w + 0.544 w_dot + -5617.981 theta_r + -43.054 theta_r_vel 
            + 1.059 w theta_r + 0.801 w_dot theta_r + -1452.077 theta_r^2 
            + -55.277 theta_r theta_r_vel + -6.134 theta_r_vel^2 
            + 4877.735 sin(1 theta_r) + -113.413 cos(1 theta_r) 
            + -2.129 sin(1 theta_r_vel) + -20.894 cos(1 theta_r_vel)
    (theta_r)' = 1.000 theta_r_vel
    """
    w, wdot, theta = state
    theta_r_vel = action

    # dynamics from SINDy model
    dw = wdot
    dwdot = (
        0.353 *  w
        + 0.544 * wdot
        - 5617.981 * theta
        - 43.054 * theta_r_vel
        + 1.059 * w * theta
        +  0.801  * wdot * theta
        - 1452.077 * theta**2
        - 55.277 * theta * theta_r_vel
        - 6.134  * theta_r_vel**2
        + 4877.735 * np.sin(theta)
        - 113.413 * np.cos(theta)
        - 2.129 * np.sin(theta_r_vel)
        - 20.894 * np.cos(theta_r_vel)
    )
    dtheta = theta_r_vel

    # Euler integration
    newstate = np.array([
        w + dt*dw,
        max(0.0, wdot + dt*dwdot),  # clip to enforce positivity
        theta + dt*dtheta
    ])

    return newstate




def fwd_sim_bottle(state, action, dt=1/120):
    """
    Forward simulate one step using discovered dynamics

    state = [w, w_dot, theta_r]
    action = theta_r_vel (control)
    dt = timestep

    (w)' = 1.000 w_dot
    (w_dot)' = -0.413 w + -0.551 w_dot + -661.657 theta_r + -3.661 theta_r_vel + 
            -0.144 w theta_r + -0.240 w_dot theta_r + -0.631 w_dot theta_r_vel + -166.551 theta_r^2 + 
            -14.429 theta_r theta_r_vel + 1.648 theta_r_vel^2 
            + 582.182 sin(1 theta_r) + -26.813 cos(1 theta_r) 
            + -17.118 sin(1 theta_r_vel) + -10.937 cos(1 theta_r_vel)
    (theta_r)' = 1.000 theta_r_vel
    """
    w, wdot, theta = state
    theta_r_vel = action

    # dynamics from SINDy model
    dw = wdot
    dwdot = (
       -0.413 *  w
        -0.551 * wdot
        -661.657 * theta
        -3.661 * theta_r_vel
        -0.144 * w * theta
        -0.240  * wdot * theta
        -0.631  * wdot * theta_r_vel
        -166.551 * theta**2
        -14.429 * theta * theta_r_vel
        + 1.648  * theta_r_vel**2
        + 582.182 * np.sin(theta)
        -26.813 * np.cos(theta)
        -17.118 * np.sin(theta_r_vel)
        -10.937 * np.cos(theta_r_vel)
    )
    dtheta = theta_r_vel

    # Euler integration
    newstate = np.array([
        w + dt*dw,
        max(0.0, wdot + dt*dwdot),  # clip to enforce positivity
        theta + dt*dtheta
    ])

    return newstate


sindy_dynamics_functions_dict = {
    'Beaker': fwd_sim_beaker,
    'Vase': fwd_sim_vase,
    'Bottle': fwd_sim_bottle,
}

def simulate_dynamics(dynamics_func, x0, u):
    trajectory = np.array([x0])
    x = x0
    for i in range(len(u)):
        x_nxt = dynamics_func(x, u[:,0][i])
        trajectory = np.vstack((trajectory, x_nxt))
        x = x_nxt
    return trajectory

















#============== iLQG controller forward simulation ========================================

@partial(jit, static_argnums=(2, 3,))
def simulate_ilqg_MugBeaker(key, params, T, ntrial, uinit):

    env = Pouring_MugBeaker(T=T,)
    x0 = env._reset(None, None)
    b0 = (x0, jnp.eye(x0.shape[0]))
    u0=uinit
    gains, xbar, ubar = gilqr.solve(p=env,
    # gains, xbar, ubar = ilqr.solve(p=env,
                                    x0=x0, U_init=u0,
                                    params=params, max_iter=10)
    # create a policy and belief dynamics
    policy = create_lqg_policy(gains, xbar, ubar)
    
    ekf = EKFWrapper(Pouring_MugBeaker)(b0=b0)
        
    # simulate some trajectories
    xs, (xhat, covs), us, cost = ekf.simulate(key, steps=T, trials=ntrial, policy=policy, params=params)

    return xs, us

@partial(jit, static_argnums=(2, 3,))
def simulate_ilqg_ConicalBeaker(key, params, T, ntrial, uinit):

    env = Pouring_ConicalBeaker(T=T,)
    x0 = env._reset(None, None)
    b0 = (x0, jnp.eye(x0.shape[0]))
    u0=uinit
    gains, xbar, ubar = gilqr.solve(p=env,
                                    x0=x0, U_init=u0,
                                    params=params, max_iter=10)
    # create a policy and belief dynamics
    policy = create_lqg_policy(gains, xbar, ubar)
    
    ekf = EKFWrapper(Pouring_ConicalBeaker)(b0=b0)
        
    # simulate some trajectories
    xs, (xhat, covs), us, cost = ekf.simulate(key, steps=T, trials=ntrial, policy=policy, params=params)

    return xs, us


@partial(jit, static_argnums=(2, 3,))
def simulate_ilqg_GobletBeaker(key, params, T, ntrial, uinit):

    env = Pouring_GobletBeaker(T=T,)
    x0 = env._reset(None, None)
    b0 = (x0, jnp.eye(x0.shape[0]))
    u0=uinit
    gains, xbar, ubar = gilqr.solve(p=env,
                                    x0=x0, U_init=u0,
                                    params=params, max_iter=10)
    # create a policy and belief dynamics
    policy = create_lqg_policy(gains, xbar, ubar)
    
    ekf = EKFWrapper(Pouring_GobletBeaker)(b0=b0)
        
    # simulate some trajectories
    xs, (xhat, covs), us, cost = ekf.simulate(key, steps=T, trials=ntrial, policy=policy, params=params)

    return xs, us

@partial(jit, static_argnums=(2, 3,))
def simulate_ilqg_MugBottle(key, params, T, ntrial, uinit):

    env = Pouring_MugBottle(T=T,)
    x0 = env._reset(None, None)
    b0 = (x0, jnp.eye(x0.shape[0]))
    u0=uinit
    gains, xbar, ubar = gilqr.solve(p=env,
                                    x0=x0, U_init=u0,
                                    params=params, max_iter=10)
    # create a policy and belief dynamics
    policy = create_lqg_policy(gains, xbar, ubar)
    
    ekf = EKFWrapper(Pouring_MugBottle)(b0=b0)
        
    # simulate some trajectories
    xs, (xhat, covs), us, cost = ekf.simulate(key, steps=T, trials=ntrial, policy=policy, params=params)

    return xs, us

@partial(jit, static_argnums=(2, 3,))
def simulate_ilqg_ConicalBottle(key, params, T, ntrial, uinit):

    env = Pouring_ConicalBottle(T=T,)
    x0 = env._reset(None, None)
    b0 = (x0, jnp.eye(x0.shape[0]))
    u0=uinit
    gains, xbar, ubar = gilqr.solve(p=env,
                                    x0=x0, U_init=u0,
                                    params=params, max_iter=10)
    # create a policy and belief dynamics
    policy = create_lqg_policy(gains, xbar, ubar)
    
    ekf = EKFWrapper(Pouring_ConicalBottle)(b0=b0)
        
    # simulate some trajectories
    xs, (xhat, covs), us, cost = ekf.simulate(key, steps=T, trials=ntrial, policy=policy, params=params)

    return xs, us


@partial(jit, static_argnums=(2, 3,))
def simulate_ilqg_GobletBottle(key, params, T, ntrial, uinit):

    env = Pouring_GobletBottle(T=T,)
    x0 = env._reset(None, None)
    b0 = (x0, jnp.eye(x0.shape[0]))
    u0=uinit
    gains, xbar, ubar = gilqr.solve(p=env,
                                    x0=x0, U_init=u0,
                                    params=params, max_iter=10)
    # create a policy and belief dynamics
    policy = create_lqg_policy(gains, xbar, ubar)
    
    ekf = EKFWrapper(Pouring_GobletBottle)(b0=b0)
        
    # simulate some trajectories
    xs, (xhat, covs), us, cost = ekf.simulate(key, steps=T, trials=ntrial, policy=policy, params=params)

    return xs, us


@partial(jit, static_argnums=(2, 3,))
def simulate_ilqg_MugVase(key, params, T, ntrial, uinit):

    env = Pouring_MugVase(T=T,)
    x0 = env._reset(None, None)
    b0 = (x0, jnp.eye(x0.shape[0]))
    u0=uinit
    gains, xbar, ubar = gilqr.solve(p=env,
                                    x0=x0, U_init=u0,
                                    params=params, max_iter=10)
    # create a policy and belief dynamics
    policy = create_lqg_policy(gains, xbar, ubar)
    
    ekf = EKFWrapper(Pouring_MugVase)(b0=b0)
        
    # simulate some trajectories
    xs, (xhat, covs), us, cost = ekf.simulate(key, steps=T, trials=ntrial, policy=policy, params=params)

    return xs, us

@partial(jit, static_argnums=(2, 3,))
def simulate_ilqg_ConicalVase(key, params, T, ntrial, uinit):

    env = Pouring_ConicalVase(T=T,)
    x0 = env._reset(None, None)
    b0 = (x0, jnp.eye(x0.shape[0]))
    u0=uinit
    gains, xbar, ubar = gilqr.solve(p=env,
                                    x0=x0, U_init=u0,
                                    params=params, max_iter=10)
    # create a policy and belief dynamics
    policy = create_lqg_policy(gains, xbar, ubar)
    
    ekf = EKFWrapper(Pouring_ConicalVase)(b0=b0)
        
    # simulate some trajectories
    xs, (xhat, covs), us, cost = ekf.simulate(key, steps=T, trials=ntrial, policy=policy, params=params)

    return xs, us


@partial(jit, static_argnums=(2, 3,))
def simulate_ilqg_GobletVase(key, params, T, ntrial, uinit):

    env = Pouring_GobletVase(T=T,)
    x0 = env._reset(None, None)
    b0 = (x0, jnp.eye(x0.shape[0]))
    u0=uinit
    gains, xbar, ubar = gilqr.solve(p=env,
                                    x0=x0, U_init=u0,
                                    params=params, max_iter=10)
    # create a policy and belief dynamics
    policy = create_lqg_policy(gains, xbar, ubar)
    
    ekf = EKFWrapper(Pouring_GobletVase)(b0=b0)
        
    # simulate some trajectories
    xs, (xhat, covs), us, cost = ekf.simulate(key, steps=T, trials=ntrial, policy=policy, params=params)

    return xs, us


ilqg_functions_dict = {
    'MugBeaker':simulate_ilqg_MugBeaker,
    'MugBeaker_fast':simulate_ilqg_MugBeaker,
    'ConicalBeaker':simulate_ilqg_ConicalBeaker,
    'ConicalBeaker_fast':simulate_ilqg_ConicalBeaker,
    'GobletBeaker':simulate_ilqg_GobletBeaker,
    'GobletBeaker_fast':simulate_ilqg_GobletBeaker,

    'MugBottle':simulate_ilqg_MugBottle,
    'MugBottle_fast':simulate_ilqg_MugBottle,
    'ConicalBottle':simulate_ilqg_ConicalBottle,
    'ConicalBottle_fast':simulate_ilqg_ConicalBottle,
    'GobletBottle':simulate_ilqg_GobletBottle,
    'GobletBottle_fast':simulate_ilqg_GobletBottle,

    'MugVase':simulate_ilqg_MugVase,
    'MugVase_fast':simulate_ilqg_MugVase,
    'ConicalVase':simulate_ilqg_ConicalVase,
    'ConicalVase_fast':simulate_ilqg_ConicalVase,
    'GobletVase':simulate_ilqg_GobletVase,
    'GobletVase_fast':simulate_ilqg_GobletVase,
    }