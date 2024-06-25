import sys
import os
from multiprocessing import Pool, cpu_count
import numpy as np
import pickle

project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))  
sys.path.append(project_root)

from toy_example.algorithms import mypipla, mypgd, proximal_map_laplace_approx, pipgla

os.chdir(project_root)

output_file = "output_pipgla_toy.txt"

# Define the relative directory and filename
output_dir = "./toy_example/outputs"
os.makedirs(output_dir, exist_ok=True)

# Define the relative directory and filename
dir_experiment_data = "./toy_example/input"
# Construct the full path
experiment_data_filename = os.path.join(dir_experiment_data, "data_experiment_toy.pickle")
    
# Load the experiment data
with open(experiment_data_filename, 'rb') as f:
    data_experiment = pickle.load(f)


labels = data_experiment['labels']
x_unknown = data_experiment['x_unknown']


n_particles = 500
n_iterations = 2500


def run_algorithm_proximal(_):
    theta0 = np.random.randint(-15, 10)
    X0 = np.random.normal(loc=theta0, size=(50, n_particles))
    thetas_approx, _ = pipgla(proximal_map = proximal_map_laplace_approx, th = np.array([[theta0]]), X = X0, N = n_particles, data = labels, K = n_iterations, gamma = 0.005, h = 0.05)
    
    rmse = (thetas_approx[-1] - np.mean(x_unknown)) ** 2 / (np.mean(x_unknown)) ** 2       
    rmse_2 = (thetas_approx[-1] + 3) ** 2 / 3 ** 2    
    return rmse, rmse_2

# Define the total number of iterations
total_iterations = 100

def main():
    # Determine the number of processes
    num_processes = cpu_count()

    with Pool(num_processes) as pool:
        results = pool.map(run_algorithm_proximal, range(total_iterations))

    rmse_list, rmse_2_list = zip(*results)

    mean_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)

    mean_rmse_2 = np.mean(rmse_2_list)
    std_rmse_2 = np.std(rmse_2_list)

    # Construct the full path
    output_path = os.path.join(output_dir, output_file)
    
    with open(output_path, "w") as file:
        file.write(f"mean RMSE {mean_rmse}\n")
        file.write(f"std RMSE {std_rmse}\n")
        file.write(f"mean RMSE {mean_rmse_2}\n")
        file.write(f"std RMSE {std_rmse_2}\n")

if __name__ == "__main__":
    main()













