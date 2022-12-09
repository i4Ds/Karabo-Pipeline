import os


def is_cuda_available():
    # Check available GPU by invoking nvidia-smi
    try:
        output = os.popen("nvidia-smi").read()
        if "GPU" in output and "CUDA" in output:
            return True
        else:
            print("No GPU is available")
            return False
    except Exception as e:
        print("Error with nvidia-smi: ", e)
        return False


if __name__ == "__main__":
    print(is_cuda_available())
