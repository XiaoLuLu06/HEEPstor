import numpy as np
from enum import Enum

class QuantMode(Enum):
    GLOBAL = 'global'
    PER_COLUMN = 'per_column'

def quantize_rhs_to_int8(matrix_fp32, quant_mode=QuantMode.GLOBAL, symmetric=False):
    """
    Quantize the right-hand side matrix from FP32 to INT8.
    
    Args:
        matrix_fp32: Input 2D matrix in FP32 format of shape (K, N)
        quant_mode: Quantization mode (GLOBAL or PER_COLUMN)
        symmetric: If True, use symmetric quantization (zero_points = 0)
    
    Returns:
        matrix_int8: Quantized INT8 matrix of shape (K, N)
        scales: Scaling factors (scalar for GLOBAL, array of N factors for PER_COLUMN)
        zero_points: Zero points (None if symmetric, scalar or array of N for asymmetric)
    """
    assert len(matrix_fp32.shape) == 2, "Input must be a 2D matrix"
    K, N = matrix_fp32.shape
    
    if quant_mode == QuantMode.GLOBAL:
        if symmetric:
            max_abs_val = np.max(np.abs(matrix_fp32))
            scales = max_abs_val / 127.0
            zero_points = None
            
            matrix_int8 = np.clip(
                np.round(matrix_fp32 / scales),
                -128, 127
            ).astype(np.int8)
        else:
            # First find zero point to shift data around zero
            min_val = np.min(matrix_fp32)
            max_val = np.max(matrix_fp32)
            zero_points = (max_val + min_val) / 2
            
            # Shift the data to be centered around zero
            matrix_centered = matrix_fp32 - zero_points
            
            # Find scaling factor based on the centered data
            max_abs = (max_val - min_val)/2
            scales = max_abs / 127.0
            
            # Quantize the centered data
            matrix_int8 = np.clip(
                np.round(matrix_centered / scales),
                -128, 127
            ).astype(np.int8)    
    else:  # PER_COLUMN
        scales = np.empty(N, dtype=np.float32)
        zero_points = None if symmetric else np.empty(N, dtype=np.float32)
        matrix_int8 = np.empty_like(matrix_fp32, dtype=np.int8)
        
        for col in range(N):
            if symmetric:
                max_abs_val = np.max(np.abs(matrix_fp32[:, col]))
                scales[col] = max_abs_val / 127.0
                matrix_int8[:, col] = np.clip(
                    np.round(matrix_fp32[:, col] / scales[col]),
                    -128, 127
                ).astype(np.int8)
            else:
                # First find zero point to shift data around zero
                min_val = np.min(matrix_fp32[:, col])
                max_val = np.max(matrix_fp32[:, col])
                zero_points[col] = (max_val + min_val) / 2

                # Shift the data to be centered around zero
                col_centered = matrix_fp32[:, col] - zero_points[col]
                
                # Find scaling factor based on the centered data
                max_abs = (max_val - min_val)/2
                scales[col] = np.float32(max_abs / 127.0)
                
                # Quantize the centered data
                matrix_int8[:, col] = np.clip(
                    np.round(col_centered / scales[col]),
                    -128, 127
                ).astype(np.int8)
    
    return matrix_int8, scales, zero_points

def dequantize_matmul_result(matmul_fp32_result, lhs, scales, zero_points=None, quant_mode=QuantMode.GLOBAL):
    """
    Dequantize the result of FP32×INT8 matrix multiplication.
    C = A × (B_int8 × scale + zero_point)
    = (A × B_int8) × scale + (A × ones) × zero_point
    
    Args:
        matmul_fp32_result: Result of FP32×INT8 matrix multiplication of shape (M, N)
        lhs: The LHS matrix used in the multiplication, needed for zero point correction
        scales: FP32 scaling factors (scalar for GLOBAL, array of N factors for PER_COLUMN)
        zero_points: FP32 zero points (None if symmetric quantization was used)
        quant_mode: Quantization mode used (GLOBAL or PER_COLUMN)
    
    Returns:
        result_fp32: Dequantized FP32 result
    """
    M, N = matmul_fp32_result.shape
    
    if quant_mode == QuantMode.GLOBAL:
        # For asymmetric quantization, correct for zero point
        if zero_points is not None:
            # Compute (A × ones) × zero_point
            # Sum of each row of A multiplied by zero_point returns a column vector that must be added
            #  to each column the result.
            zero_point_contribution = np.sum(lhs, axis=1, keepdims=True) * zero_points

            return matmul_fp32_result * scales + zero_point_contribution
        else:
            return matmul_fp32_result * scales
    
    else:  # PER_COLUMN
        result = np.empty_like(matmul_fp32_result)
        
        sum_of_each_lhs_row = np.sum(lhs, axis=1)
        for col in range(N):
            if zero_points is not None:
                # Compute (A × ones) × zero_point for this column
                zero_point_contribution = sum_of_each_lhs_row * zero_points[col]
                result[:, col] = matmul_fp32_result[:, col] * scales[col] + zero_point_contribution
            else:
                result[:, col] = matmul_fp32_result[:, col] * scales[col]
        return result

def test_fp32_int8_matmul(M=64, K=32, N=48, trials=100, quant_mode=QuantMode.GLOBAL, 
                         mean_A=0.0, std_A=1.0, mean_B=0.0, std_B=1.0):
    """
    Test FP32×INT8 matrix multiplication with multiple trials.
    
    Args:
        M, K, N: Matrix dimensions (A: M×K, B: K×N)
        trials: Number of random trials to run
        quant_mode: Quantization mode to test
        mean_A: Mean of the normal distribution for matrix A
        std_A: Standard deviation of the normal distribution for matrix A
        mean_B: Mean of the normal distribution for matrix B
        std_B: Standard deviation of the normal distribution for matrix B
    """
    # Initialize arrays to store errors across trials
    max_abs_errors = np.zeros((trials, 2))  # [:,0] for symmetric, [:,1] for asymmetric
    mean_abs_errors = np.zeros((trials, 2))
    mean_rel_errors = np.zeros((trials, 2))
    
    # Set random seed for reproducibility
    rng = np.random.default_rng(42)
    
    # Generate all matrices for all trials in advance
    A_matrices = (rng.standard_normal((trials, M, K)) * std_A + mean_A).astype(np.float32)
    B_matrices = (rng.standard_normal((trials, K, N)) * std_B + mean_B).astype(np.float32)
    
    for trial in range(trials):
        # Use pre-generated matrices
        A = A_matrices[trial]
        B = B_matrices[trial]
        
        # Ground truth (compute once per trial)
        C_true = np.matmul(A, B)
        
        # Test both symmetric and asymmetric quantization
        for idx, symmetric in enumerate([True, False]):
            # Quantize B matrix
            B_int8, scales, zero_points = quantize_rhs_to_int8(B, quant_mode, symmetric)
                        
            # Simulate FP32×INT8 matrix multiplication
            C_fp32 = np.matmul(A, B_int8.astype(np.float32))
            
            # Dequantize result
            C_dequant = dequantize_matmul_result(C_fp32, A, scales, zero_points, quant_mode)
            
            # Calculate errors
            abs_error = np.abs(C_true - C_dequant)
            rel_error = abs_error / (np.abs(C_true) + 1e-10)
            
            # Store errors for this trial
            max_abs_errors[trial, idx] = np.max(abs_error)
            mean_abs_errors[trial, idx] = np.mean(abs_error)
            mean_rel_errors[trial, idx] = np.mean(100 * rel_error)  # as percentage
    
    # Print statistical summary
    print(f"\nQuantization Mode: {quant_mode.value}")
    print(f"Distribution parameters:")
    print(f"Matrix A: mean={mean_A:.2f}, std={std_A:.2f}")
    print(f"Matrix B: mean={mean_B:.2f}, std={std_B:.2f}")
    
    for idx, mode in enumerate(['Symmetric', 'Asymmetric']):
        print(f"\n{mode} Quantization Statistics ({trials} trials):")
        print("\nMean Relative Error (%):")
        print(f"  Mean: {np.mean(mean_rel_errors[:, idx]):.2f}%")
        print(f"  Std:  {np.std(mean_rel_errors[:, idx]):.2f}%")
        print(f"  Min:  {np.min(mean_rel_errors[:, idx]):.2f}%")
        print(f"  Max:  {np.max(mean_rel_errors[:, idx]):.2f}%")

def run_distribution_comparison(M, K, N, trials, B_distribution_params):
    """
    Run quantization tests across different B distributions and generate comparison table.
    
    Args:
        M, K, N: Matrix dimensions (A: M×K, B: K×N)
        trials: Number of random trials to run
        B_distribution_params: List of tuples (mean_B, std_B, description) for B matrix distributions
    
    Returns:
        results: 2D array [num_distributions x 4] with mean relative errors for each method
    """
    # Initialize results storage
    # Format: [global_sym, global_asym, percol_sym, percol_asym] for each distribution
    results = np.zeros((len(B_distribution_params), 4))
    
    print("\nRunning tests...")
    print("Matrix A: Standard normal (mean=0.0, std=1.0) for all tests")
    
    # Run all tests and collect results
    for dist_idx, (mean_B, std_B, _) in enumerate(B_distribution_params):
        # Test both quantization modes
        col_idx = 0  # Index for the results array
        for mode in QuantMode:
            # Initialize arrays to store errors for this configuration
            mean_rel_errors = np.zeros((trials, 2))  # [:,0] for symmetric, [:,1] for asymmetric
            
            # Generate same random matrices for all quantization modes in this distribution
            rng = np.random.default_rng(42)
            A_matrices = (rng.standard_normal((trials, M, K)) * 1.0 + 0.0).astype(np.float32)
            B_matrices = (rng.standard_normal((trials, K, N)) * std_B + mean_B).astype(np.float32)
            
            for trial in range(trials):
                A = A_matrices[trial]
                B = B_matrices[trial]
                C_true = np.matmul(A, B)
                
                # Test both symmetric and asymmetric quantization
                for symmetric in [True, False]:
                    B_int8, scales, zero_points = quantize_rhs_to_int8(B, mode, symmetric)
                    C_fp32 = np.matmul(A, B_int8.astype(np.float32))
                    C_dequant = dequantize_matmul_result(C_fp32, A, scales, zero_points, mode)
                    
                    # Calculate relative error
                    abs_error = np.abs(C_true - C_dequant)
                    rel_error = abs_error / (np.abs(C_true) + 1e-10)
                    mean_rel_errors[trial, int(not symmetric)] = np.mean(100 * rel_error)  # as percentage
            
            # Store mean results for this mode and distribution
            results[dist_idx, col_idx:col_idx+2] = np.mean(mean_rel_errors, axis=0)
            col_idx += 2
    
    # Print results table
    TABLE_LENGTH = 99
    DISTRIBUTION_DESC_LENGTH = 35

    print("\nMean Relative Error (%) for different quantization methods and distributions")
    print("="*TABLE_LENGTH)
    
    # Column headers for different quantization modes
    headers = ["Global Symm", "Global Asymm", "Per-Col Symm", "Per-Col Asymm"]
    header_row = "Distribution".ljust(DISTRIBUTION_DESC_LENGTH) + " | "
    header_row += " | ".join(f"{h:^12}" for h in headers)
    print(header_row)
    print("-"*TABLE_LENGTH)
    
    # Print each distribution's results
    for (mean_B, std_B, desc), result_row in zip(B_distribution_params, results):
        row = f'{desc} (μ={round(mean_B, 2)}, σ={round(std_B, 2)})'.ljust(DISTRIBUTION_DESC_LENGTH) + " | "
        row += " | ".join(f"{err:^12.2f}" for err in result_row)
        print(row)
    
    print("="*TABLE_LENGTH)
    return results

if __name__ == "__main__":
    # Define matrix dimensions
    M, K, N = 64, 32, 48
    trials = 200
    
    # Keep A as standard normal, vary B distribution
    # Each tuple is (mean_B, std_B, description)
    B_distribution_params = [
        (0.0, 1.0, "Std"),
        (5.0, 1.0, "Shifted"),
        (0.0, 2.0, "Wide"),
        (3.0, 0.5, "Narrow"),
        (-2.0, 3.0, "Neg. wide"),
        (50.0, 10.0, "Far"),
        (500.0, 100.0, "Very far"),
        (5000.0, 1000.0, "Very very far"),

    ]
    
    # Run comparison and get results
    results = run_distribution_comparison(M, K, N, trials, B_distribution_params)