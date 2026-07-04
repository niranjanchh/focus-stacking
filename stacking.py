import cv2
import numpy as np
import math
import gc
import os

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from PIL import Image
from gpu_state import is_gpu_enabled, disable_gpu, gpu_lock, is_oom_error

# Disable PIL image size limit warnings to support large 127MP files without crashing
Image.MAX_IMAGE_PIXELS = None

def _get_available_ram_gb():
    """Get available RAM in GB. Tries psutil first, falls back to wmic on Windows."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        try:
            import subprocess
            out = subprocess.check_output("wmic OS get FreePhysicalMemory /Value", shell=True).decode()
            for line in out.splitlines():
                if "FreePhysicalMemory" in line:
                    kb = int(line.split("=")[1].strip())
                    return kb / (1024 * 1024)
        except Exception:
            pass
        return 8.0

# Disable OpenCV internal thread pool globally to prevent massive thread oversubscription
# We will set it dynamically per-task instead of hardcoding it to 1.
# cv2.setNumThreads(1)

class FocusStacker:
    def __init__(self, pyramid_levels=8, kernel_size=5, num_threads=None, in_memory=False, method="pyramid", energy_scale=1.0):
        self.temp_dir = None  # No disk temp files created
        self.pyramid_levels = pyramid_levels
        self.method = method
        self.kernel_size = kernel_size
        if self.kernel_size > 1 and self.kernel_size % 2 == 0:
            self.kernel_size += 1
            
        self.num_threads = num_threads
        self.in_memory = in_memory
        self.energy_scale = max(0.1, min(1.0, energy_scale))

    def _effective_kernel_size(self):
        if self.kernel_size <= 1:
            return 0
        k = max(3, self.kernel_size)
        if k % 2 == 0:
            k += 1
        return k

    def _compute_energy_from_gray(self, gray):
        k = self._effective_kernel_size()
        gray_f = gray.astype(np.float32)
        
        if is_gpu_enabled():
            if gpu_lock.acquire(blocking=False):
                try:
                    u_gray_f = cv2.UMat(gray_f)
                    u_sobelx = cv2.Sobel(u_gray_f, cv2.CV_32F, 1, 0, ksize=3)
                    u_sobely = cv2.Sobel(u_gray_f, cv2.CV_32F, 0, 1, ksize=3)
                    sx = u_sobelx.get()
                    sy = u_sobely.get()
                    tenengrad = cv2.magnitude(sx, sy)
                    
                    lap_x_kernel = np.array([[-1, 2, -1]], dtype=np.float32)
                    lap_y_kernel = np.array([[-1], [2], [-1]], dtype=np.float32)
                    u_lap_x = cv2.filter2D(u_gray_f, cv2.CV_32F, lap_x_kernel)
                    u_lap_y = cv2.filter2D(u_gray_f, cv2.CV_32F, lap_y_kernel)
                    sml = np.abs(u_lap_x.get()) + np.abs(u_lap_y.get())
                    
                    focus_map = 0.7 * tenengrad + 0.3 * sml
                    
                    if k > 0:
                        # Use O(1) Box Filter for blazing fast energy aggregation on 120MP+ images
                        focus_map = cv2.boxFilter(focus_map, -1, (k, k))
                    return focus_map
                except cv2.error as e:
                    if is_oom_error(e):
                        disable_gpu(f"OpenCL Error (Out of VRAM?): {e}. Falling back to CPU for current run.")
                    else:
                        raise e
                finally:
                    gpu_lock.release()
                    
        # CPU Fallback path
        lap_x_kernel = np.array([[-1, 2, -1]], dtype=np.float32)
        lap_y_kernel = np.array([[-1], [2], [-1]], dtype=np.float32)
        
        H, W = gray.shape
        focus_map = np.empty((H, W), dtype=np.float32)
        tile_size = 2000
        padding = (k // 2) + 2

        for y_start in range(0, H, tile_size):
            y_end = min(H, y_start + tile_size)
            for x_start in range(0, W, tile_size):
                x_end = min(W, x_start + tile_size)
                
                y_start_pad = max(0, y_start - padding)
                y_end_pad = min(H, y_end + padding)
                x_start_pad = max(0, x_start - padding)
                x_end_pad = min(W, x_end + padding)
                
                gray_f_tile = gray[y_start_pad:y_end_pad, x_start_pad:x_end_pad].astype(np.float32)
                
                sobelx = cv2.Sobel(gray_f_tile, cv2.CV_32F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray_f_tile, cv2.CV_32F, 0, 1, ksize=3)
                tenengrad = cv2.magnitude(sobelx, sobely)
                del sobelx, sobely
                
                lap_x = cv2.filter2D(gray_f_tile, cv2.CV_32F, lap_x_kernel)
                lap_y = cv2.filter2D(gray_f_tile, cv2.CV_32F, lap_y_kernel)
                del gray_f_tile
                
                sml = np.abs(lap_x) + np.abs(lap_y)
                del lap_x, lap_y
                
                tile_focus_map = 0.7 * tenengrad + 0.3 * sml
                del tenengrad, sml
                
                if k > 0:
                    tile_focus_map = cv2.boxFilter(tile_focus_map, -1, (k, k))
                    
                y_off = y_start - y_start_pad
                x_off = x_start - x_start_pad
                focus_map[y_start:y_end, x_start:x_end] = tile_focus_map[y_off:y_off + (y_end - y_start), x_off:x_off + (x_end - x_start)]
                del tile_focus_map
                
        return focus_map

    def _compute_energy(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self._compute_energy_from_gray(gray)

    def _build_pyramids(self, img, weight):
        if is_gpu_enabled():
            if gpu_lock.acquire(blocking=False):
                try:
                    u_img = cv2.UMat(img)
                    u_weight = cv2.UMat(weight.astype(np.float32))
                    
                    gp_img = [u_img]
                    gp_weight = [u_weight]
                    
                    for _ in range(self.pyramid_levels - 1):
                        gp_img.append(cv2.pyrDown(gp_img[-1]))
                        gp_weight.append(cv2.pyrDown(gp_weight[-1]))
                        
                    lp_img = [gp_img[-1].get().astype(np.float32)]
                    for level in range(self.pyramid_levels - 1, 0, -1):
                        h, w = gp_img[level - 1].get().shape[:2]
                        expanded = cv2.pyrUp(gp_img[level], dstsize=(w, h))
                        
                        lap = np.subtract(gp_img[level - 1].get(), expanded.get(), dtype=np.float32)
                        lp_img.append(lap)
                        
                    return [w.get() for w in gp_weight], lp_img
                except cv2.error as e:
                    if is_oom_error(e):
                        disable_gpu(f"OpenCL Error (Out of VRAM?): {e}. Falling back to CPU for current run.")
                    else:
                        raise e
                finally:
                    gpu_lock.release()
                    
        # CPU Fallback path (executes if GPU is disabled, or if GPU threw an OOM error above)
        # Native CPU path: keep image in uint16/uint8 for 9x faster downsampling and 2x faster upsampling
        gp_img = [img]
        gp_weight = [weight.astype(np.float32)]
        
        for _ in range(self.pyramid_levels - 1):
            gp_img.append(cv2.pyrDown(gp_img[-1]))
            gp_weight.append(cv2.pyrDown(gp_weight[-1]))
            
        # Laplacian computation needs float32 to hold negative values securely
        lp_img = [gp_img[-1].astype(np.float32)]
        
        for level in range(self.pyramid_levels - 1, 0, -1):
            h, w = gp_img[level - 1].shape[:2]
            expanded = cv2.pyrUp(gp_img[level], dstsize=(w, h))
            lap = np.subtract(gp_img[level - 1], expanded, dtype=np.float32)
            lp_img.append(lap)
            
        gp_img[0] = None  # free last remaining level
        return gp_weight, lp_img

    def _process_pass1_single(self, img, i):
        if self.in_memory:
            energy = self._compute_energy(img)
            return img, energy
        else:
            # Memory optimization: Convert to grayscale, save and delete BGR immediately
            # to prevent both color image and float32 energy map from residing in RAM together.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_file = os.path.join(self.temp_dir, f"img_{i}.npy")
            np.save(img_file, img)
            
            del img
            
            energy = self._compute_energy_from_gray(gray)
            del gray
            
            temp_file = os.path.join(self.temp_dir, f"energy_{i}.npy")
            np.save(temp_file, energy)
            return img_file, temp_file

    def _enforce_channels(self, img, target_channels):
        if img is None: return None
        curr_channels = img.shape[2] if len(img.shape) > 2 else 1
        if curr_channels == target_channels:
            return img
        
        if target_channels == 3 and curr_channels == 4:
            return img[..., :3]
        elif target_channels == 4 and curr_channels == 3:
            alpha = np.full((img.shape[0], img.shape[1], 1), 
                            65535 if img.dtype == np.uint16 else 255, 
                            dtype=img.dtype)
            return np.concatenate((img, alpha), axis=2)
        elif target_channels == 1 and curr_channels >= 3:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if curr_channels == 4 else cv2.COLOR_BGR2GRAY)
        elif target_channels >= 3 and curr_channels == 1:
            code = cv2.COLOR_GRAY2BGRA if target_channels == 4 else cv2.COLOR_GRAY2BGR
            return cv2.cvtColor(img, code)
        return img

    def _normalize_exposure(self, img, i, ref_index, ref_mean, orig_dtype, cw, ch):
        if i != ref_index and ref_mean > 0:
            # High-performance NumPy slicing to avoid heavy cv2.resize downsample overhead (~0.2s per image)
            thumb = img[::32, ::32]
            cur_mean = np.mean(thumb)
            del thumb
            if cur_mean > 0:
                multiplier = np.float32(ref_mean / cur_mean)
                
                max_val = 65535 if orig_dtype == np.uint16 else 255
                img_f = img.astype(np.float32)
                img_f *= multiplier
                np.clip(img_f, 0, max_val, out=img_f)
                img = img_f.astype(orig_dtype)
                del img_f
        return img

    def stack_from_paths(self, image_paths, aligner, alignment_data):
        """Optimized stacking pipeline with parallel focus energy computation and image-sequential blending.
        
        Args:
            image_paths: list of image file paths
            aligner: ImageAligner instance (used for warp_and_crop_single)
            alignment_data: dict from aligner.compute_alignment()
        """
        transforms = alignment_data['transforms']
        ref_shape = alignment_data['ref_shape']
        crop_box = alignment_data['crop_box']
        orig_shapes = alignment_data['orig_shapes']
        ref_index = alignment_data['ref_index']
        run_use_gpu = alignment_data['run_use_gpu']
        enable_deghosting = alignment_data.get('enable_deghosting', False)
        enable_exposure_normalization = alignment_data.get('enable_exposure_normalization', False)
        valid_frames = alignment_data.get('valid_frames', [True] * len(image_paths))
        
        valid_indices = [i for i, v in enumerate(valid_frames) if v]
        if len(valid_indices) == 0:
            raise ValueError("All frames were rejected during alignment.")
            
        if not valid_frames[ref_index]:
            actual_ref_index = valid_indices[len(valid_indices) // 2]
        else:
            actual_ref_index = ref_index
            
        cx, cy, cw, ch = crop_box
        ref_img = aligner.warp_and_crop_single(
            image_paths[actual_ref_index], transforms[actual_ref_index], 
            orig_shapes[actual_ref_index], ref_shape, crop_box, run_use_gpu
        )
        
        if enable_deghosting:
            if len(ref_img.shape) == 2:
                ref_gray = ref_img
            elif ref_img.shape[2] == 4:
                ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGRA2GRAY)
            else:
                ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                
            if ref_gray.dtype == np.uint16:
                ref_gray = (ref_gray >> 8).astype(np.uint8)
                
            flow_scale = aligner._get_scale_factor(ch, cw)
            if flow_scale < 1.0:
                ref_gray_small = cv2.resize(ref_gray, (int(cw * flow_scale), int(ch * flow_scale)), interpolation=cv2.INTER_AREA)
            else:
                ref_gray_small = ref_gray
            del ref_gray
        else:
            ref_gray_small = None
            
        if enable_exposure_normalization:
            ref_thumb = ref_img[::32, ::32]
            ref_mean = float(np.mean(ref_thumb))
            del ref_thumb
        else:
            ref_mean = -1.0
            
        del ref_img
            
        ref_index = valid_indices.index(actual_ref_index)
        image_paths = [image_paths[i] for i in valid_indices]
        transforms = [transforms[i] for i in valid_indices]
        orig_shapes = [orig_shapes[i] for i in valid_indices]
        
        n_images = len(image_paths)
        cx, cy, cw, ch = crop_box
        pixels = cw * ch
        
        # Determine dtype and channels from first image without warping (faster)
        first_img_raw = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
        orig_dtype = first_img_raw.dtype
        channels = first_img_raw.shape[2] if len(first_img_raw.shape) > 2 else 1
        bytes_per_channel = 2 if orig_dtype == np.uint16 else 1
        del first_img_raw
        
        # Energy computation dimensions (reduced resolution)
        energy_h = max(64, int(ch * self.energy_scale))
        energy_w = max(64, int(cw * self.energy_scale))
        
        available_ram_gb = _get_available_ram_gb()
        
        if run_use_gpu:
            from gpu_state import get_gpu_vram_gb
            vram_gb = get_gpu_vram_gb()
            max_gpu_mp = vram_gb * 8.0  # e.g., 6.2GB VRAM -> ~50MP cap, 24GB VRAM -> ~192MP cap
            img_mp = pixels / 1_000_000
            
            if img_mp > max_gpu_mp:
                print(f"[{time.strftime('%H:%M:%S')}] Image size ({img_mp:.1f}MP) exceeds optimal GPU threshold ({max_gpu_mp:.1f}MP for {vram_gb:.1f}GB VRAM). Disabling GPU.")
                run_use_gpu = False
                disable_gpu("Exceeds optimal GPU size based on VRAM")
        
        # Check RAM guard for in_memory
        if self.in_memory:
            cache_gb = (n_images * pixels * bytes_per_channel * channels) / (1024**3)
            if cache_gb > available_ram_gb * 0.5:
                print(f"Warning: In-Memory Caching disabled. Requires {cache_gb:.1f} GB, but only {available_ram_gb:.1f} GB available (need 2x safety margin).")
                self.in_memory = False
                
        acc_multiplier = (12 + 12 * bytes_per_channel) if self.method == "pyramid" else (6 + 6 * bytes_per_channel)
        accumulator_gb = (pixels * acc_multiplier) / (1024**3)
        working_ram_gb = max(1.0, available_ram_gb - accumulator_gb)
        
        pass1_gb = ((pixels * (4 * bytes_per_channel + 8)) + (100 * 1024 * 1024)) / (1024**3)
        pass1_threads = max(1, int(working_ram_gb / pass1_gb))
        
        if self.method == "pyramid":
            pass3_gb = ((pixels * (12 + 12 * bytes_per_channel)) + (200 * 1024 * 1024)) / (1024**3)
            pass3_threads = max(1, int(working_ram_gb / pass3_gb))
        else:
            pass3_gb = ((pixels * (3 * bytes_per_channel + 4)) + (50 * 1024 * 1024)) / (1024**3)
            pass3_threads = max(1, int(working_ram_gb / pass3_gb))
            
        cpu_cores = os.cpu_count() or 4
        pass1_threads = min(cpu_cores, pass1_threads)
        pass3_threads = min(cpu_cores, pass3_threads)
        
        # System Capping for High-Core/Hybrid CPUs (e.g. 12-core i5 vs Workstations):
        # Spawning too many threads for images > 20MP saturates the memory bus and triggers E-cores.
        # - Standard/Hybrid CPUs: cap to 4 threads to run purely on fast P-cores.
        # - Workstations (>16 logical cores): cap to 8 threads to leverage higher bandwidth.
        if pixels > 20_000_000:
            max_safe_threads = max(4, cpu_cores // 2)
            pass1_threads = min(pass1_threads, max_safe_threads)
            pass3_threads = min(pass3_threads, max_safe_threads)
            
        print(f"[{time.strftime('%H:%M:%S')}] === Optimized Parallel Pipeline ===")
        print(f"[{time.strftime('%H:%M:%S')}] Images: {n_images} at {cw}x{ch} ({pixels/1_000_000:.1f}MP, {orig_dtype})")
        print(f"[{time.strftime('%H:%M:%S')}] Method: {self.method.upper()}")
        print(f"[{time.strftime('%H:%M:%S')}] Adaptive Threads: Pass 1 = {pass1_threads} threads | Pass 3 = {pass3_threads} threads (based on {available_ram_gb:.1f}GB RAM)")
        
        # ═══════════════════════════════════════════
        # PASS 1: Compute focus energy (Parallel)
        # ═══════════════════════════════════════════
        # Adaptive kernel: kernel=1 (default) = auto resolution-based, kernel>1 = user override
        if self.kernel_size <= 1:
            auto_k = max(5, int(cw / 1000))
            if auto_k % 2 == 0: auto_k += 1
            self.kernel_size = auto_k
        else:
            if self.kernel_size % 2 == 0: self.kernel_size += 1
            self.kernel_size = max(3, self.kernel_size)
        
        print(f"[{time.strftime('%H:%M:%S')}] Pass 1: Computing focus energy ({energy_w}x{energy_h}, kernel={self.kernel_size})...")
        print("[PROGRESS: 15]")
        
        energy_maps = [None] * n_images
        cached_images = [None] * n_images
        cache_lock = threading.Lock()
        
        def process_pass1_task(i):
            deghost = ref_gray_small if (enable_deghosting and i != ref_index) else None
            
            # Fast-path: Only read Grayscale if we are not saving the image
            is_grayscale_only = not self.in_memory
            
            img = aligner.warp_and_crop_single(
                image_paths[i], transforms[i], orig_shapes[i], ref_shape, crop_box, run_use_gpu,
                deghost_ref_gray_small=deghost, grayscale_only=is_grayscale_only
            )
            
            if not is_grayscale_only:
                img = self._enforce_channels(img, channels)
                
            img = self._normalize_exposure(img, i, ref_index, ref_mean, orig_dtype, cw, ch)
            
            if is_grayscale_only:
                gray = img
            else:
                if len(img.shape) == 2:
                    gray = img
                elif img.shape[2] == 4:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if self.in_memory:
                with cache_lock:
                    cached_images[i] = img
                
            if self.energy_scale < 1.0:
                gray_small = cv2.resize(gray, (energy_w, energy_h), interpolation=cv2.INTER_AREA)
            else:
                gray_small = gray
            del gray
            
            energy = self._compute_energy_from_gray(gray_small)
            if self.energy_scale < 1.0:
                del gray_small
            
            return i, energy
            
        # Dynamically set OpenCV threads to prevent oversubscription while maximizing core usage
        cv2.setNumThreads(max(1, cpu_cores // pass1_threads))
        with ThreadPoolExecutor(max_workers=pass1_threads) as executor:
            futures = {executor.submit(process_pass1_task, i): i for i in range(n_images)}
            for i_progress, f in enumerate(as_completed(futures)):
                idx, energy = f.result()
                energy_maps[idx] = energy
                print(f"[{time.strftime('%H:%M:%S')}][Pass 1] Frame {idx+1}/{n_images} complete")
                print(f"[PROGRESS: {15 + int(25 * ((i_progress + 1) / n_images))}]")
                
        # Force garbage collection to free all temporary gray/warped images from RAM
        gc.collect()
                
        # ═══════════════════════════════════════════
        # PASS 2: Compute Base Weights
        # ═══════════════════════════════════════════
        print(f"[{time.strftime('%H:%M:%S')}] Pass 2: Computing weights ({self.method.upper()})...")
        print("[PROGRESS: 45]")
        
        if self.method == "weighted_average":
            # Power-6 probabilities (Helicon Method A)
            global_max = 1e-6
            for i in range(n_images):
                m = np.max(energy_maps[i])
                if m > global_max:
                    global_max = m
                    
            tile_size = 2000
            for y_start in range(0, energy_h, tile_size):
                y_end = min(energy_h, y_start + tile_size)
                for x_start in range(0, energy_w, tile_size):
                    x_end = min(energy_w, x_start + tile_size)
                    
                    sum_w = np.zeros((y_end - y_start, x_end - x_start), dtype=np.float32)
                    powers = []
                    for i in range(n_images):
                        e = energy_maps[i][y_start:y_end, x_start:x_end].copy()
                        e /= global_max
                        e += 1e-5
                        np.power(e, 6.0, out=e)
                        sum_w += e
                        powers.append(e)
                        
                    zero_mask = sum_w < 1e-10
                    sum_w[zero_mask] = 1.0
                    
                    for i in range(n_images):
                        powers[i] /= sum_w
                        powers[i][zero_mask] = 1.0 / n_images
                        energy_maps[i][y_start:y_end, x_start:x_end] = powers[i]
                        
            weights_small = energy_maps
        else:
            # Hard Index Map (Winner-Takes-All) for Depth Map & Pyramid
            # This completely preserves micro-contrast (texture) because the sharpest pixel gets a weight of exactly 1.0
            max_index = np.zeros((energy_h, energy_w), dtype=np.uint8 if n_images <= 255 else np.uint16)
            tile_size = 2000
            for y_start in range(0, energy_h, tile_size):
                y_end = min(energy_h, y_start + tile_size)
                for x_start in range(0, energy_w, tile_size):
                    x_end = min(energy_w, x_start + tile_size)
                    
                    max_energy = np.zeros((y_end - y_start, x_end - x_start), dtype=np.float32)
                    tile_max_idx = np.zeros((y_end - y_start, x_end - x_start), dtype=max_index.dtype)
                    
                    for i in range(n_images):
                        e = energy_maps[i][y_start:y_end, x_start:x_end]
                        mask = e > max_energy
                        np.copyto(max_energy, e, where=mask)
                        tile_max_idx[mask] = i
                    max_index[y_start:y_end, x_start:x_end] = tile_max_idx
            
            # Topological smoothing on the low-res index map
            if n_images <= 255:
                max_index = cv2.medianBlur(max_index, 5)
                
            weights_small = []
            for i in range(n_images):
                # Create exact binary mask
                binary_mask = (max_index == i).astype(np.float32)
                weights_small.append(binary_mask)
                
            del max_index
            del energy_maps
        # Force garbage collection
        gc.collect()
        
        print(f"[{time.strftime('%H:%M:%S')}] Weights computed. Peak weight map: {max(w.max() for w in weights_small):.3f}")
        print("[PROGRESS: 50]")
        
        # ═══════════════════════════════════════════
        # PASS 3: Parallel blending
        # ═══════════════════════════════════════════
        print(f"[{time.strftime('%H:%M:%S')}] Pass 3: Blending ({self.method.upper()})...")
        print("[PROGRESS: 55]")
        
        current_ram_gb = _get_available_ram_gb()
        working_ram_gb = max(1.0, current_ram_gb - accumulator_gb)
        pass3_threads = max(1, int(working_ram_gb / pass3_gb))
        pass3_threads = min(cpu_cores, pass3_threads)
        if pixels > 20_000_000:
            max_safe_threads = max(4, cpu_cores // 2)
            pass3_threads = min(pass3_threads, max_safe_threads)
            
        print(f"[{time.strftime('%H:%M:%S')}] Re-evaluated Pass 3 Threads: {pass3_threads} (based on {current_ram_gb:.1f}GB RAM)")
        
        if self.method in ["weighted_average", "depth_map"]:
            if self.method == "depth_map":
                print(f"[{time.strftime('%H:%M:%S')}] Pass 2.5: Refining depth index map (commercial-grade)...")
                print("[PROGRESS: 52]")
                
                depth_dtype = np.uint8 if n_images <= 255 else np.uint16
                depth_index = np.zeros((ch, cw), dtype=depth_dtype)
                max_weight = np.zeros((ch, cw), dtype=np.float32)
                
                # 1. Raw Depth Map
                for i in range(n_images):
                    if self.energy_scale < 1.0:
                        w = cv2.resize(weights_small[i], (cw, ch), interpolation=cv2.INTER_LINEAR)
                    else:
                        w = weights_small[i]
                    mask = w > max_weight
                    np.copyto(max_weight, w, where=mask)
                    depth_index[mask] = i
                    
                del max_weight
                
                # 2. Refine depth map (Gentle Edge-Aware)
                if n_images <= 255:
                    # Smaller median filter prevents destruction of fine hairs
                    depth_index = cv2.medianBlur(depth_index, 5)
                    
                # Morphological cleanup (Hole filling and region growing)
                # Keep kernel size extremely tight (max 7) to preserve micro-texture boundaries
                m_size = min(7, max(3, int(math.sqrt(cw * ch) / 1000)))
                if m_size % 2 == 0: m_size += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m_size, m_size))
                
                depth_index = cv2.morphologyEx(depth_index, cv2.MORPH_CLOSE, kernel)
                depth_index = cv2.morphologyEx(depth_index, cv2.MORPH_OPEN, kernel)
                
                # 3. Edge Refinement (Ultra-tight Anti-Aliasing)
                refined_masks = []
                sum_mask = np.zeros((ch, cw), dtype=np.float32)
                # Dynamic feathering scaled to image resolution for clean transitions
                blur_size = max(5, int(math.sqrt(cw * ch) / 800))
                if blur_size % 2 == 0: blur_size += 1
                
                for i in range(n_images):
                    # Create binary mask for this layer
                    layer_mask = (depth_index == i).astype(np.float32)
                    # Feather the mask to soften edges
                    layer_mask = cv2.GaussianBlur(layer_mask, (blur_size, blur_size), 0)
                    refined_masks.append(layer_mask)
                    sum_mask += layer_mask
                    
                del depth_index
                
                # Normalize so they sum to exactly 1.0
                sum_mask[sum_mask == 0] = 1.0
                for i in range(n_images):
                    refined_masks[i] /= sum_mask
                    
                del sum_mask
                
                # Override weights_small for the blend loop, pretending it's weighted_average
                # but scaled to full res already!
                weights_small = refined_masks
                
            final_image_shape = (ch, cw, channels) if channels > 1 else (ch, cw)
            final_image = np.zeros(final_image_shape, dtype=np.float32)
            blend_lock = threading.Lock()
            
            def process_blend_direct(i):
                if self.in_memory and cached_images[i] is not None:
                    img = cached_images[i]
                    cached_images[i] = None  # Free immediately
                else:
                    deghost = ref_gray_small if (enable_deghosting and i != ref_index) else None
                    img = aligner.warp_and_crop_single(
                        image_paths[i], transforms[i], orig_shapes[i], ref_shape, crop_box, run_use_gpu,
                        deghost_ref_gray_small=deghost
                    )
                    img = self._enforce_channels(img, channels)
                    img = self._normalize_exposure(img, i, ref_index, ref_mean, orig_dtype, cw, ch)
                
                # Upscale weight to full resolution
                if self.method == "depth_map":
                    weight = weights_small[i]
                else:
                    if self.energy_scale < 1.0:
                        weight = cv2.resize(weights_small[i], (cw, ch), interpolation=cv2.INTER_LINEAR)
                    else:
                        weight = weights_small[i]
                
                img_f = img.astype(np.float32)
                if len(img_f.shape) > 2:
                    img_f *= weight[..., np.newaxis]
                else:
                    img_f *= weight
                del img, weight
                return i, img_f
                
            cv2.setNumThreads(max(1, cpu_cores // pass3_threads))
            with ThreadPoolExecutor(max_workers=pass3_threads) as executor:
                futures = {executor.submit(process_blend_direct, i): i for i in range(n_images)}
                for i_progress, f in enumerate(as_completed(futures)):
                    idx, img_f = f.result()
                    
                    with blend_lock:
                        final_image += img_f
                            
                    del img_f
                    print(f"[{time.strftime('%H:%M:%S')}][Pass 3] Blended frame {idx+1}/{n_images}")
                    print(f"[PROGRESS: {55 + int(40 * ((i_progress + 1) / n_images))}]")
                    
            del weights_small, cached_images
            gc.collect()
            
            max_val = 65535 if orig_dtype == np.uint16 else 255
            print("[PROGRESS: 100]")
            return np.clip(final_image, 0, max_val).astype(orig_dtype)
            
        else:
            # ═══════════════════════════════════════════
            # PYRAMID BLENDING (Image-Sequential)
            # ═══════════════════════════════════════════
            print(f"[{time.strftime('%H:%M:%S')}] Pass 4: Building Laplacians (Disk-Free Pipeline)...")
            print("[PROGRESS: 90]")
            
            t_pyramid_shapes = [(ch, cw)]
            cur_h, cur_w = ch, cw
            for _ in range(self.pyramid_levels - 1):
                cur_h = (cur_h + 1) // 2
                cur_w = (cur_w + 1) // 2
                t_pyramid_shapes.append((cur_h, cur_w))
                
            t_pyramid_shapes_c2f = list(reversed(t_pyramid_shapes))
            
            if channels > 1:
                pyramid_blended = [np.zeros((*s, channels), dtype=np.float32) for s in t_pyramid_shapes_c2f]
            else:
                pyramid_blended = [np.zeros(s, dtype=np.float32) for s in t_pyramid_shapes_c2f]
                
            blend_lock = threading.Lock()
            
            def process_pyramid_image(i):
                if cached_images[i] is not None:
                    img = cached_images[i]
                    cached_images[i] = None # Free from RAM immediately
                else:
                    deghost = ref_gray_small if (enable_deghosting and i != ref_index) else None
                    img = aligner.warp_and_crop_single(
                        image_paths[i], transforms[i], orig_shapes[i], ref_shape, crop_box, run_use_gpu,
                        deghost_ref_gray_small=deghost, grayscale_only=False
                    )
                    img = self._enforce_channels(img, channels)
                    img = self._normalize_exposure(img, i, ref_index, ref_mean, orig_dtype, cw, ch)
                    
                weight = weights_small[i]
                if self.energy_scale < 1.0:
                    weight = cv2.resize(weight, (cw, ch), interpolation=cv2.INTER_LINEAR)
                    # Restore hard edges after upscaling to preserve micro-texture
                    weight = (weight > 0.5).astype(np.float32)
                    
                gp_img = [img]
                gp_weight = [weight.astype(np.float32)]
                
                for _ in range(self.pyramid_levels - 1):
                    gp_img.append(cv2.pyrDown(gp_img[-1]))
                    gp_weight.append(cv2.pyrDown(gp_weight[-1]))
                    
                lap_base = gp_img[-1].astype(np.float32)
                w_layer = gp_weight[-1]
                if len(lap_base.shape) > 2:
                    w_layer = w_layer[..., np.newaxis]
                lap_base *= w_layer
                
                laps = [lap_base]
                
                for level in range(self.pyramid_levels - 1, 0, -1):
                    target_level = level - 1
                    h_dim, w_dim = gp_img[target_level].shape[:2]
                    expanded = cv2.pyrUp(gp_img[level], dstsize=(w_dim, h_dim))
                    
                    lap = np.subtract(gp_img[target_level], expanded, dtype=np.float32)
                    
                    w_layer = gp_weight[target_level]
                    if len(lap.shape) > 2:
                        w_layer = w_layer[..., np.newaxis]
                    lap *= w_layer
                    laps.append(lap)
                    
                with blend_lock:
                    for level_idx in range(self.pyramid_levels):
                        pyramid_blended[level_idx] += laps[level_idx]
                        
                del gp_img, gp_weight, laps
                return i
                
            cv2.setNumThreads(max(1, cpu_cores // pass3_threads))
            with ThreadPoolExecutor(max_workers=pass3_threads) as executor:
                futures = {executor.submit(process_pyramid_image, i): i for i in range(n_images)}
                for i_progress, f in enumerate(as_completed(futures)):
                    idx = f.result()
                    print(f"[{time.strftime('%H:%M:%S')}][Pass 3] Blended frame {idx+1}/{n_images}")
                    print(f"[PROGRESS: {55 + int(40 * ((i_progress + 1) / n_images))}]")
                    
            # Reconstruct final image from accumulator pyramid
            result = pyramid_blended[0]
            for level in range(1, self.pyramid_levels):
                h_dim, w_dim = pyramid_blended[level].shape[:2]
                result = cv2.pyrUp(result, dstsize=(w_dim, h_dim)) + pyramid_blended[level]
                
            final_image = result
            del pyramid_blended
            
            del weights_small, cached_images
            gc.collect()
            
            max_val = 65535 if orig_dtype == np.uint16 else 255
            print("[PROGRESS: 100]")
            return np.clip(final_image, 0, max_val).astype(orig_dtype)
