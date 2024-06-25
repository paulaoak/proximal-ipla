import sys
import os
from multiprocessing import Pool, cpu_count
import numpy as np
import pickle

project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(project_root)

from logistic_regression.algorithms import pip_ula, sig, mypipla, mypgd, prox_pgd, proximal_map_laplace_iteration_total, proximal_map_laplace_iterative, proximal_map_laplace_approx, proximal_map_laplace_approx_total, pipgla

os.chdir(project_root)

output_file = "output_ppgd_approx_laplace.txt"

# Define the relative directory and filename
output_dir = "./logistic_regression/laplace/outputs"
os.makedirs(output_dir, exist_ok=True)

# Define the relative directory and filename
dir_experiment_data = "./logistic_regression/laplace/input"
# Construct the full path
experiment_data_filename = os.path.join(dir_experiment_data, "data_experiment_laplace.pickle")
    
# Load the experiment data
with open(experiment_data_filename, 'rb') as f:
    data_experiment = pickle.load(f)

design_matrix = data_experiment['design_matrix']
labels = data_experiment['labels']
x_unknown = data_experiment['x_unknown']

n_particles = 50
n_iterations = 5000

def run_algorithm_proximal(_):
    theta0 = np.random.randint(-15, 10)
    X0 = np.random.normal(loc=theta0, size=(50, n_particles))
    thetas_approx, _ = prox_pgd(proximal_map=proximal_map_laplace_approx_total, th=np.array([[theta0]]), X=X0, N=n_particles, design_matrix=design_matrix, data=labels, K=n_iterations, h=0.1)
    
    rmse = (thetas_approx[-1] - np.mean(x_unknown)) ** 2 / (np.mean(x_unknown)) ** 2       
    rmse_2 = (thetas_approx[-1] + 4) ** 2 / 4 ** 2    
    rmse_3 = (np.mean(thetas_approx[-1000:]) - np.mean(x_unknown)) ** 2 / (np.mean(x_unknown)) ** 2       
    rmse_4 = (np.mean(thetas_approx[-1000:]) + 4) ** 2 / 4 ** 2    
    return rmse, rmse_2, rmse_3, rmse_4

# Define the total number of iterations
total_iterations = 500

def main():
    # Determine the number of processes
    num_processes = cpu_count()

    with Pool(num_processes) as pool:
        results = pool.map(run_algorithm_proximal, range(total_iterations))

    rmse_list, rmse_2_list, rmse_3_list, rmse_4_list = zip(*results)

    mean_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)

    mean_rmse_2 = np.mean(rmse_2_list)
    std_rmse_2 = np.std(rmse_2_list)

    mean_rmse_3 = np.mean(rmse_3_list)
    std_rmse_3 = np.std(rmse_3_list)

    mean_rmse_4 = np.mean(rmse_4_list)
    std_rmse_4 = np.std(rmse_4_list)

    # Construct the full path
    output_path = os.path.join(output_dir, output_file)
    
    with open(output_path, "w") as file:
        file.write(f"last mean RMSE {mean_rmse}\n")
        file.write(f"last std RMSE {std_rmse}\n")
        file.write(f"last mean RMSE {mean_rmse_2}\n")
        file.write(f"last std RMSE {std_rmse_2}\n")

        file.write(f"avg mean RMSE {mean_rmse_3}\n")
        file.write(f"avg std RMSE {std_rmse_3}\n")
        file.write(f"avg mean RMSE {mean_rmse_4}\n")
        file.write(f"avg std RMSE {std_rmse_4}\n")

if __name__ == "__main__":
    main()
