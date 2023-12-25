import numpy as np
import pandas as pd
from scipy.optimize import minimize
from Utility.decorators import timer, exception_handler, debug

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

    def _divide_into_subsets(self):
        """
        Divide the currencies into approximately equal subsets.
        """
        num_currencies = len(self.symbols)
        subset_size = num_currencies // 4
        extra = num_currencies % 4

        self.subsets_indices = {}
        start = 0
        for i in range(4):
            end = start + subset_size + (1 if i < extra else 0)
            self.subsets_indices[f'subset_{i+1}'] = list(range(start, end))
            start = end

        self.data_subsets = {key: self.data_matrix[:, indices] for key, indices in self.subsets_indices.items()}

        # Print information about how the currencies have been divided into subsets
        for key, indices in self.subsets_indices.items():
            print(f"{key}: Currencies {indices[0]} to {indices[-1]}")

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

    @timer
    @exception_handler
    def _optimise_subset(self, data_subset, J_subset, h_subset, subset_index):
        """
        Optimise a subset of the Ising model using the L-BFGS-B algorithm.

        Args:
        subset_index (int): Index of the subset being optimised.
        """
        N = J_subset.shape[0]  # Number of spins (currencies)

        # Flatten the J matrix and h vector for the optimisation
        x0 = np.concatenate([J_subset[np.triu_indices(N, k=1)], h_subset])

        def objective_function(x):
            # Construct the symmetric J matrix and calculate the likelihood and gradients
            J, h = self._reconstruct_J_and_h(x, N)
            likelihood, grad_J, grad_h = self._log_pseudolikelihood_and_gradients(J, h, data_subset)
            # Combine and return the likelihood and flattened gradients
            return likelihood, np.concatenate([grad_J[np.triu_indices(N, k=1)], grad_h])
        
        # Execute the optimisation using the objective function and initial guesses
        res = minimize(objective_function, x0, method='L-BFGS-B', jac=True)

        # Reconstruct the optimised J matrix and h vector from the optimisation result
        J_optimised, h_optimised = self._reconstruct_J_and_h(res.x, N)

        # Print the optimisation results for the subset
        print(f"Optimised J matrix for subset {subset_index}: \n{J_optimised}")
        print(f"Optimised h vector for subset {subset_index}: \n{h_optimised}")
        print(f"Optimisation successful for subset {subset_index}: {res.success}")

        return J_optimised, h_optimised, res.success

    @staticmethod
    def _reconstruct_J_and_h(flattened_array, N):
        """
        Helper function to reconstruct the symmetric J matrix and h vector from a flattened array.
        """
        J_upper_tri = flattened_array[:N * (N - 1) // 2]
        h = flattened_array[N * (N - 1) // 2:]

        # Construct the symmetric J matrix from the upper triangular part
        J = np.zeros((N, N))
        J[np.triu_indices(N, k=1)] = J_upper_tri
        J += J.T  # Symmetrise the J matrix

        return J, h

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

    @staticmethod
    def _combine_h_vectors(*vectors):
        """
        Combine smaller h vectors into a larger h vector.
        """
        return np.concatenate(vectors)

    @timer
    def optimise_all_subsets(self):
        print("Optimising initial subsets...\n")
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
            optimised_results[key] = (J_opt, h_opt, success)
            print(f"Subset {key} optimised.\n")

        print("Combining into larger subsets for further optimisation...\n")
        # Combine subsets 1 and 2, and subsets 3 and 4
        J_combined_1_2, h_combined_1_2 = self._combine_J_matrices(optimised_results['subset_1'][0], optimised_results['subset_2'][0]), self._combine_h_vectors(optimised_results['subset_1'][1], optimised_results['subset_2'][1])
        J_combined_3_4, h_combined_3_4 = self._combine_J_matrices(optimised_results['subset_3'][0], optimised_results['subset_4'][0]), self._combine_h_vectors(optimised_results['subset_3'][1], optimised_results['subset_4'][1])

        # Optimise the combined larger subsets
        J_optimised_1_2, h_optimised_1_2, _ = self._optimise_subset(self.data_matrix[:, self.subsets_indices['subset_1'] + self.subsets_indices['subset_2']], J_combined_1_2, h_combined_1_2, '1_2')
        J_optimised_3_4, h_optimised_3_4, _ = self._optimise_subset(self.data_matrix[:, self.subsets_indices['subset_3'] + self.subsets_indices['subset_4']], J_combined_3_4, h_combined_3_4, '3_4')
        print("Larger subsets optimised.\n")

        # Final optimisation with the entire dataset
        print("Performing final optimisation with the entire dataset...\n")
        J_final_combined = self._combine_J_matrices(J_optimised_1_2, J_optimised_3_4)
        h_final_combined = self._combine_h_vectors(h_optimised_1_2, h_optimised_3_4)

        J_optimised_full, h_optimised_full, _ = self._optimise_subset(self.data_matrix, J_final_combined, h_final_combined, 'full')
        print("Final optimisation completed.\n")

        return J_optimised_full, h_optimised_full

    def save_results(self, J_matrix, h_vector, J_file_path, h_file_path):
        """
        Save the optimised J matrix and h vector to CSV files.
        """
        J_df = pd.DataFrame(J_matrix, columns=self.symbols, index=self.symbols)
        h_df = pd.DataFrame({'Symbol': self.symbols, 'h': h_vector})

        J_df.to_csv(J_file_path, index=False)
        h_df.to_csv(h_file_path, index=False, header=False)

        print(f"The optimised J matrix has been saved to '{J_file_path}'.")
        print(f"The optimised h vector has been saved to '{h_file_path}'.")