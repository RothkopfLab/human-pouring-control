import h5py
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import statsmodels.api as sm

task_list = ['self-paced', 'fast']
vessel_list = ["Conical", "Mug", "Goblet", ]
container_list = ["Beaker", "Vase", "Bottle"]

max_heights = {'Conical' : 6.8, 'Mug' : 13.2, 'Goblet' : 12.6}

colors_dict_all = {
    # Containers
    'Bottle':  "#4A7004",  
    'Beaker':    "#21B007", 
    'Vase':  "#B09707" , 

    # Vessels
    'Mug':  "#098BA5",  
    'Conical':    "#3A86FF", 
    'Goblet':  "#3C0174" , 

    # Task
    'self-paced': "#884ea0",
    'fast':"#f39c12",
}

task_colors_dict = {'self-paced': "#884ea0",'fast':"#f39c12",}
task_colors = [ '#884ea0', '#f39c12',]



def simpleaxis(ax, constrained=False, polar=False, direction='in', tickLen=4):
    if not polar:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    ax.tick_params(direction=direction, length=tickLen, width=1, colors='black',
                   grid_color='black', grid_alpha=0.5)

    if not constrained:
        plt.tight_layout()


mocap_smp_rate = 120
dt = 1/mocap_smp_rate

def get_regression(x, y, add_intercept=False):
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    if not add_intercept:
        X = x_sorted.reshape(-1, 1)
    else:
        X = sm.add_constant(x_sorted)

    model = sm.OLS(y_sorted, X).fit()
    pred = model.get_prediction(X)
    ci = pred.summary_frame(alpha=0.05)  # 95% confidence
    return model, ci

def load_saved_trajectories(path):
    data_reloaded = {}
    with h5py.File(path, 'r') as f:
        for p, data_p in f.items():
            data_p_reload = {}
            for e, data_e in data_p.items():
                data_e_reload = {}
                for t, data_t in data_e.items():
                    data_trials = []
                    for ii in range(len(data_t.keys())):
                        data_trials.append(data_t[str(ii)][:])
                    data_e_reload[t] = data_trials
                data_p_reload[e] = data_e_reload
            data_reloaded[p] = data_p_reload
    return data_reloaded




def normalize_trajectories(trajs, T=None):
    """
    trajs: list of np arrays, each shape (t_i,)
    T:  optional, number of resampled time points. If None, uses average length.

    Returns:
        resampled: np.ndarray of shape (N, T, D) if D-dimensional input
    """
    if T is None:
        avg_length = int(np.round(np.mean([len(traj) for traj in trajs])))
        T = max(avg_length, 2)  

    resampled_trajs = []

    for traj in trajs:
        traj = np.asarray(traj)
        t_orig = np.linspace(0, 1, traj.shape[0])  # normalize original time axis
        t_target = np.linspace(0, 1, T)            # uniform resampled time axis

        # Determine if 1D or multidimensional trajectory
        if traj.ndim == 1:
            interp = interp1d(t_orig, traj, kind='linear', fill_value="extrapolate")
        elif traj.ndim == 2:
            interp = interp1d(t_orig, traj, axis=0, kind='linear', fill_value="extrapolate")
        else:
            raise ValueError(f"Unsupported trajectory shape: {traj.shape}")

        resampled = interp(t_target)
        resampled_trajs.append(resampled)

    resampled_trajs = np.stack(resampled_trajs)

    return resampled_trajs


from scipy.interpolate import make_interp_spline
csv_path = 'data/vessel_geometry'
# Load data from the CSV file
loaded_data = np.loadtxt(f'{csv_path}/conical_height_weight_data.csv', delimiter=',', skiprows=1)
weights = loaded_data[:, 0]
heights = loaded_data[:, 1]
spline_conical = make_interp_spline(weights, heights, k=3)

# Load data from the CSV file
loaded_data = np.loadtxt(f'{csv_path}/goblet_height_weight_data.csv', delimiter=',', skiprows=1)
weights = loaded_data[:, 0]
heights = loaded_data[:, 1]
spline_goblet = make_interp_spline(weights, heights, k=3)


def calculate_height_arr(cup_type, weights):
    if cup_type == "Conical":
        heights = spline_conical(weights)
    elif cup_type == "Mug":
        radius = 3.
        heights = weights / (np.pi * radius**2)
    elif cup_type == "Goblet":
        heights = spline_goblet(weights)
    else:
        print("Invalid cup type.")
    return heights

def get_vessel_name(exp_name):
    if "Mug" in exp_name:
        cup = "Mug" 
    elif "Conical" in exp_name:
        cup = "Conical"
    elif "Goblet" in exp_name:
        cup = "Goblet" 
    return cup

