import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Utility.decorators import timer

class IsingOptimiser:
    def __init__(self, data_path):
        """
        Initialiser for the IsingOptimiser class.

        Args:
        data_path (str): Path to the CSV file containing currency data.
        """
        self.df = pd.read_csv(data_path)
        self.symbols = self.df.columns.tolist()[1:]
        self.data_matrix = self.df.drop(columns=["Date"]).to_numpy()
        self._divide_into_subsets()

    def _divide_into_subsets(self, num_subsets=4):
        """
        Divide the data set of observed currencies into approximately equal subsets.

        Args:
        num_subsets (int): The number of subsets to divide the data into.
        """
        num_currencies = len(self.symbols)
        subset_size = num_currencies // num_subsets
        extra = num_currencies % num_subsets

        self.subsets_indices = {}
        start = 0
        for i in range(num_subsets):
            end = start + subset_size + (1 if i < extra else 0)
            self.subsets_indices[f'subset_{i+1}'] = list(range(start, end))
            start = end

        self.data_subsets = {key: self.data_matrix[:, indices] for key, indices in self.subsets_indices.items()}

        # Print information about how the currencies have been divided into subsets
        for key, indices in self.subsets_indices.items():
            print(f"{key}: Currencies {indices[0] + 1} to {indices[-1] + 1}")

    @timer
    def train(self, max_attempts=1000):
        """
        Executes the training process for the model using the specified optimisation method.
        The training involves optimising the model parameters in subsets ('optimise_all_subsets'),
        depending on the chosen method.

        Args:
            max_iterations (int): The maximum number of optimisation attempts.
        """

        # Initialise lists to store the results of successful optimisations
        J_results, h_results = [], []

        success_count = 0
        max_successes = max_attempts / 10 # Maximum number of successful optimisations required

        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1} of optimisation")

            # Perform the optimisation
            J_optimised, h_optimised, success = self._optimise_all_subsets(max_attempts)

            # Store and count successful optimisations
            if success:
                J_results.append(J_optimised)
                h_results.append(h_optimised)
                success_count += 1
                print("Optimisation successful.\n")

                # Stop if the required number of successes is achieved
                if success_count >= max_successes:
                    break
            else:
                print("Optimisation unsuccessful, restarting...\n")

        # Check and analyse the results if there were successful optimisations
        if J_results:
            # Calculate average results and sample standard deviation
            self.J_optimised = np.mean(J_results, axis=0)
            self.h_optimised = np.mean(h_results, axis=0)
            self.J_std = np.std(J_results, axis=0, ddof=1)
            self.h_std = np.std(h_results, axis=0, ddof=1)

            # Print statistical analysis results on J and h results
            print("Statistical Analysis of J_optimised and h_optimised:")
            print(f"Mean of h_optimised: {self.J_optimised}")
            print(f"Standard Deviation of J_optimised: {self.J_std}")
            print(f"Mean of h_optimised: {self.h_optimised}")
            print(f"Standard Deviation of h_optimised: {self.h_std}")

            print(f"Optimisation completed with averaged results after {success_count} successes.\n")

            self.scatter_plots()
        else:
            print("Optimisation was unsuccessful after maximum attempts.\n")

    def scatter_plots(self):
        d = len(self.symbols)
        reversed_symbols = [col[:3] if len(col) == 6 else col for col in self.symbols[::-1]]  # Reverse the order of symbols
        J_means = self.J_optimised[np.triu_indices(d, k=1)]
        J_std = self.J_std[np.triu_indices(d, k=1)]

        # Plotting scatter plots for optimised parameters
        plt.figure(figsize=(12, 6))

        # Scatter plot for J
        plt.subplot(1, 2, 1)
        plt.errorbar(J_means, range(len(J_means))[::-1], xerr=J_std, fmt='o', ecolor='r', capsize=5, label='J elements', markersize=5)
        plt.title("Coupling Constants (J) Between Observed Currencies")
        plt.ylabel("Element Index in J")
        plt.xlabel("Mean Values and Standard Deviations")
        plt.grid(True)
        plt.legend()

        # Scatter plot for h
        plt.subplot(1, 2, 2)
        plt.errorbar(self.h_optimised[::-1], reversed_symbols, xerr=self.h_std, fmt='o', ecolor='r', capsize=5, label='h elements')
        plt.title("External Fields (h) for Observed Currencies")
        plt.yticks(range(d), reversed_symbols)
        plt.xlabel("Mean Values and Standard Deviations")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.suptitle("Scatter Plots of Optimised Parameter Values Across Successful Optimisations", fontsize=16, y=1.02)
        plt.show()

    @timer
    def _optimise_all_subsets(self, max_attempts):
        optimised = False
        for attempt in range(max_attempts):
            J_initial = {key: np.random.uniform(-1, 1, (len(indices), len(indices)))
                            for key, indices in self.subsets_indices.items()}
            for key, matrix in J_initial.items():
                upper_tri = np.triu(matrix, k=1)  # Extract upper triangular part with k=1 to exclude the diagonal
                J_initial[key] = upper_tri + upper_tri.T  # Make the matrix symmetric with zeros on the diagonal
            h_initial = {key: np.random.uniform(-1, 1, len(indices))
                            for key, indices in self.subsets_indices.items()}
            optimised_results = {}
            for index, (key, data_subset) in enumerate(self.data_subsets.items(), 1):
                J_opt, h_opt, success = self._optimise_subset(data_subset, J_initial[key], h_initial[key], index)
                if not success:
                    print(f"Subset {key} failed to optimise. Restarting optimisation.\n")
                    break
                optimised_results[key] = (J_opt, h_opt, success)
            else:
                # Combine subsets 1 and 2, and subsets 3 and 4
                J_combined_1_2, h_combined_1_2 = (
                    self._combine_J_matrices(optimised_results['subset_1'][0], optimised_results['subset_2'][0]), 
                    np.hstack((optimised_results['subset_1'][1], optimised_results['subset_2'][1])))
                J_combined_3_4, h_combined_3_4 = (
                    self._combine_J_matrices(optimised_results['subset_3'][0], optimised_results['subset_4'][0]), 
                    np.hstack((optimised_results['subset_3'][1], optimised_results['subset_4'][1])))

                # Optimise the combined larger subsets
                J_optimised_1_2, h_optimised_1_2, success_1_2 = (
                    self._optimise_subset(self.data_matrix[:, self.subsets_indices['subset_1'] + self.subsets_indices['subset_2']], 
                                          J_combined_1_2, h_combined_1_2, '1_2'))
                if not success_1_2:
                    print("Combined subset 1_2 optimisation failed. Restarting optimisation.\n")
                    continue
                J_optimised_3_4, h_optimised_3_4, success_3_4 = (
                    self._optimise_subset(self.data_matrix[:, self.subsets_indices['subset_3'] + self.subsets_indices['subset_4']], 
                                          J_combined_3_4, h_combined_3_4, '3_4'))
                if not success_3_4:
                    print("Combined subset 3_4 optimisation failed. Restarting optimisation.\n")
                    continue

                # Final optimisation with the entire dataset
                J_final_combined = self._combine_J_matrices(J_optimised_1_2, J_optimised_3_4)
                h_final_combined = np.hstack((h_optimised_1_2, h_optimised_3_4))

                J_optimised_full, h_optimised_full, success_full = self._optimise_subset(self.data_matrix, J_final_combined, h_final_combined, 'full')

                # Check success for final optimisation
                if not success_full:
                    print("Final optimisation failed. Restarting optimisation.\n")
                    continue

                # print("Final optimisation completed.\n")
                return J_optimised_full, h_optimised_full, success_full
        print("Optimisation was unsuccessful after maximum attempts.\n")
        return optimised

    def _optimise_subset(self, data_subset, J_subset, h_subset, subset_index):
        """
        Optimise a subset of the Ising model using the L-BFGS-B algorithm.

        Args:
        subset_index (int): Index of the subset being optimised.
        """
        d = J_subset.shape[0]  # Number of spins (currencies)

        # Flatten the J matrix and h vector for the optimisation
        x0 = np.concatenate([J_subset[np.triu_indices(d, k=1)], h_subset])

        def objective_function(x):
            # Construct the symmetric J matrix and calculate the likelihood and gradients
            J, h = self._reconstruct_J_and_h(x, d)
            likelihood, grad_J, grad_h = self._log_pseudolikelihood_and_gradients(J, h, data_subset)
            # Combine and return the likelihood and flattened gradients
            return likelihood, np.concatenate([grad_J[np.triu_indices(d, k=1)], grad_h])
        
        # Execute the optimisation using the objective function and initial guesses
        res = minimize(objective_function, x0, method='L-BFGS-B', jac=True)

        # Reconstruct the optimised J matrix and h vector from the optimisation result
        J_optimised, h_optimised = self._reconstruct_J_and_h(res.x, d)

        return J_optimised, h_optimised, res.success

    @staticmethod
    def _reconstruct_J_and_h(flattened_array, d):
        """
        Helper function to reconstruct the symmetric J matrix and h vector from a flattened array.
        """
        J_upper_tri = flattened_array[:d * (d - 1) // 2]
        h = flattened_array[d * (d - 1) // 2:]

        # Construct the symmetric J matrix from the upper triangular part
        J = np.zeros((d, d))
        J[np.triu_indices(d, k=1)] = J_upper_tri
        J += J.T  # Symmetrise the J matrix

        return J, h

    @staticmethod
    def _log_pseudolikelihood_and_gradients(J, h, X):
        """
        Calculate the log-pseudo-likelihood for the Ising model and its gradients.
        """
        d = X.shape[1]  # Number of dimensions
        log_likelihood, grad_J, grad_h = 0, np.zeros_like(J), np.zeros_like(h)

        # Vectorised version
        J_diag = np.diag(J)  # Get the diagonal of J
        S_ij = X @ J - X * J_diag + h  # Compute S_ij vectorized, broadcasting h
        log_likelihood = (X * S_ij).sum() - np.sum(np.log(2 * np.cosh(S_ij)))

        tanh_S_ij = np.tanh(S_ij)
        grad_h = np.sum(X - tanh_S_ij, axis=0)

        # Compute the gradient for J, excluding the diagonal
        # Create a mask to zero out diagonal contributions in the grad_J calculation
        mask = np.ones_like(J) - np.eye(d)
        grad_J = ((X - tanh_S_ij).T @ X) * mask
        grad_J = (grad_J + grad_J.T)
        
        # Return the negative likelihood and gradients for minimisation
        return -log_likelihood, -grad_J, -grad_h

    @staticmethod
    def _combine_J_matrices(*matrices):
        """
        Combine smaller J matrices into a larger J matrix.
        """
        size = sum(m.shape[0] for m in matrices)
        J_combined = np.zeros((size, size))

        current_index = 0
        for m in matrices:
            m_size = m.shape[0]
            J_combined[current_index:current_index+m_size, current_index:current_index+m_size] = m
            current_index += m_size

        return J_combined

    def save_results(self, J_file_path, h_file_path, J_std_file_path, h_std_file_path):
        """
        Save the optimised J matrix and h vector to CSV files.
        """
        J_df = pd.DataFrame(self.J_optimised, columns=self.symbols, index=self.symbols)
        h_df = pd.DataFrame({'Symbol': self.symbols, 'h': self.h_optimised})

        J_df.to_csv(J_file_path, index=False)
        h_df.to_csv(h_file_path, index=False, header=False)

        print(f"The optimised J matrix has been saved to '{J_file_path}'.")
        print(f"The optimised h vector has been saved to '{h_file_path}'.")

        J_std_df = pd.DataFrame(self.J_std, columns=self.symbols, index=self.symbols)
        h_std_df = pd.DataFrame({'Symbol': self.symbols, 'h': self.h_std})

        J_std_df.to_csv(J_std_file_path, index=False)
        h_std_df.to_csv(h_std_file_path, index=False, header=False)

        print(f"The standard deviation of J matrix has been saved to '{J_std_file_path}'.")
        print(f"The standard deviation of h vector has been saved to '{h_std_file_path}'.")