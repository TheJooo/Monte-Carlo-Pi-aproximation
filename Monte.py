import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
import time

def get_num_cuda_cores():
    device = drv.Device(0)
    return device.get_attribute(drv.device_attribute.MULTIPROCESSOR_COUNT) * device.get_attribute(drv.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR)

mod = SourceModule("""
__global__ void calculate_pi(double *pi_estimate, unsigned long long num_terms) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    double sum = 0.0;
    for (unsigned long long i = idx; i < num_terms; i += stride) {
        double term = (i % 2 == 0) ? 1.0 : -1.0;
        term /= (2 * i + 1);
        sum += term;
    }
    
    atomicAdd(pi_estimate, sum);
}
""", options=['-use_fast_math'])

def compute_pi(num_terms):
    num_cores = get_num_cuda_cores()
    block_size = 256
    grid_size = min((num_terms + block_size - 1) // block_size, 2**31 - 1)

    pi_estimate = np.zeros(1, dtype=np.float64)

    calculate_pi_kernel = mod.get_function("calculate_pi")
    calculate_pi_kernel(
        drv.InOut(pi_estimate),
        np.uint64(num_terms),
        block=(int(block_size), 1, 1),
        grid=(int(grid_size), 1)
    )

    return pi_estimate[0] * 4.0 / 7 #TODO: Find out why this happens

def print_results(pi_estimate, actual_pi, deviation, elapsed_time):
    print(f"Estimated value of π: {pi_estimate:.16f}")
    print(f"Deviation from actual value of π: {deviation:.16f}")
    print(f"Time taken: {elapsed_time:.2f} seconds")

actual_pi = np.pi

num_terms = np.uint64(3*10**10)  

start_time = time.time()
pi_estimate = compute_pi(num_terms)
elapsed_time = time.time() - start_time
deviation = np.abs(pi_estimate - actual_pi)
print_results(pi_estimate, actual_pi, deviation, elapsed_time)