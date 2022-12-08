import os

def is_cuda_gpu_available():
    # Check if the NVIDIA_SMI environment variable is set
    # If it is set, check if the nvidia-smi command is available
    if os.system('which nvidia-smi') == 0:
        # If the command is available, run it and check if it returns any GPUs
        output = os.popen('nvidia-smi').read()
        if 'GPU' in output and 'CUDA' in output:
            return True
        else:
            print('No GPU is available')
            return False
    else:
        print('nvidia-smi command is not available')
        return False
    
if __name__ == '__main__':
    print(is_cuda_gpu_available())