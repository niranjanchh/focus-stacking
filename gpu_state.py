import cv2
import time
import threading

gpu_lock = threading.Lock()

_USE_GPU = False
cv2.ocl.setUseOpenCL(False)

def is_gpu_enabled():
    return _USE_GPU

def disable_gpu(reason):
    global _USE_GPU
    if _USE_GPU:
        print(f"[{time.strftime('%H:%M:%S')}] Disabling GPU: {reason}")
        _USE_GPU = False
        cv2.ocl.setUseOpenCL(False)

def enable_gpu():
    global _USE_GPU
    if cv2.ocl.haveOpenCL():
        _USE_GPU = True
        cv2.ocl.setUseOpenCL(True)
        
def get_gpu_vram_gb():
    try:
        if cv2.ocl.haveOpenCL():
            return cv2.ocl.Device.getDefault().globalMemSize() / (1024**3)
    except Exception:
        pass
    return 4.0  # Safe default fallback
        
def is_oom_error(e):
    """Check if a cv2.error is related to VRAM/OpenCL exhaustion."""
    err_str = str(e).lower()
    return any(keyword in err_str for keyword in [
        "cl_mem_object_allocation_failure",
        "cl_out_of_resources",
        "cl_out_of_host_memory",
        "allocate",
        "memory"
    ])
