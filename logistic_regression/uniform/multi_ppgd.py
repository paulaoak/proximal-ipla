import sys
import os
from multiprocessing import Pool, cpu_count
import numpy as np
import pickle

project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(project_root)

from logistic_regression.algorithms import pip_ula, sig, mypipla, mypgd, prox_pgd, proximal_map_laplace_iteration_total, proximal_map_laplace_iterative, proximal_map_laplace_approx, proximal_map_laplace_approx_total, proximal_map_uniform, proximal_map_uniform_my, pipgla, proximal_map_uniform_new

os.chdir(project_root)

output_file = "output_ppgd_uniform.txt"

# Define the relative directory and filename
output_dir = "./logistic_regression/uniform/outputs"
os.makedirs(output_dir, exist_ok=True)

# Define the relative directory and filename
dir_experiment_data = "./logistic_regression/uniform/input"
# Construct the full path
experiment_data_filename = os.path.join(dir_experiment_data, "data_experiment_uniform.pickle")
    
# Load the experiment data
with open(experiment_data_filename, 'rb') as f:
    data_experiment = pickle.load(f)

design_matrix = data_experiment['design_matrix']
labels = data_experiment['labels']

n_particles = 50
n_iterations = 5000

def run_algorithm_proximal(_):
    theta0 = np.random.randint(-5, 7)
    X0 = np.random.normal(loc=theta0, size=(50, n_particles))
    thetas_approx, _ = prox_pgd(proximal_map = proximal_map_uniform_new, th = np.array([[theta0]]), X = X0, N = n_particles, design_matrix = design_matrix, data = labels, K = n_iterations, h = 0.03)
   
    rmse_2 = (thetas_approx[-1] - 1.5) ** 2 / (1.5) ** 2  

    rmse_3 = (np.mean(thetas_approx[-1500:]) - 1.5) ** 2 / (1.5) ** 2

    return rmse_2, rmse_3

# Define the total number of iterations
total_iterations = 500

def main():
    # Determine the number of processes
    num_processes = cpu_count()

    with Pool(num_processes) as pool:
        results = pool.map(run_algorithm_proximal, range(total_iterations))

    rmse_2_list, rmse_3_list = zip(*results)

    mean_rmse_2 = np.mean(rmse_2_list)
    std_rmse_2 = np.std(rmse_2_list)

    mean_rmse_3 = np.mean(rmse_3_list)
    std_rmse_3 = np.std(rmse_3_list)

    # Construct the full path
    output_path = os.path.join(output_dir, output_file)
    
    with open(output_path, "w") as file:
        file.write(f"last mean RMSE {mean_rmse_2}\n")
        file.write(f"last std RMSE {std_rmse_2}\n")
        file.write(f"avg mean RMSE {mean_rmse_3}\n")
        file.write(f"avg std RMSE {std_rmse_3}\n")

if __name__ == "__main__":
    main()
