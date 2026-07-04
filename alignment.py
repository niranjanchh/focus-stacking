import os
import cv2
import numpy as np
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from gpu_state import is_gpu_enabled, disable_gpu, gpu_lock, is_oom_error

# Disable PIL image size limit warnings to support large 127MP files without crashing
Image.MAX_IMAGE_PIXELS = None

# Disable OpenCV internal thread pool globally to prevent massive thread oversubscription
# We will set it dynamically per-task instead of hardcoding it to 1.
# cv2.setNumThreads(1)

def get_available_ram_gb():
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

class ImageAligner:
    def __init__(self, max_align_dim=800, n_features=8000, max_shift_percent=0.10, ecc_threshold=0.85):
        self.max_align_dim = max_align_dim
        self.n_features = n_features
        self.max_shift_percent = max_shift_percent
        self.ecc_threshold = ecc_threshold
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def _get_scale_factor(self, h, w):
        max_dim = max(h, w)
        if max_dim > self.max_align_dim:
            return self.max_align_dim / max_dim
        return 1.0

    def _get_optimal_threads(self, img_path, requested_threads):
        cpu_cores = os.cpu_count() or 4
        bytes_per_channel = 1
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                
                if hasattr(img, 'mode') and ('16' in img.mode or img.mode in ['I', 'F']):
                    bytes_per_channel = 2
        except Exception:
            return 1, get_available_ram_gb()
            
        single_image_ram_gb = (h * w * 3 * bytes_per_channel) / (1024 ** 3)
        available_ram = get_available_ram_gb()
        ram_thread_limit = max(1, int((available_ram * 0.5) // single_image_ram_gb))
        
        target_threads = min(requested_threads if requested_threads is not None else cpu_cores, ram_thread_limit)
        # Cap threads to 8 to prevent network drive (Google Drive FUSE) and memory bus saturation
        target_threads = min(8, target_threads)
        return target_threads, available_ram

    def extract_features(self, img_path):
        # Use PIL to get dimensions without loading full pixel data
        with Image.open(img_path) as pil_img:
            w, h = pil_img.size
        
        scale = self._get_scale_factor(h, w)
        
        if scale < 1.0:
            sw, sh = int(w * scale), int(h * scale)
            # Use NEAREST resampling: 10x faster than Lanczos, preserves corners, and avoids mode crashes on 16-bit TIFFs
            try:
                resample_filter = Image.Resampling.NEAREST
            except AttributeError:
                resample_filter = Image.NEAREST
                
            with Image.open(img_path) as pil_img:
                pil_img.thumbnail((sw, sh), resample_filter)
                img_arr = np.array(pil_img)
                if len(img_arr.shape) == 2:
                    img_small = img_arr
                elif img_arr.shape[2] == 4:
                    img_small = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
                else:
                    img_small = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
            img_small = img

        if len(img_small.shape) == 2:
            gray = img_small
        else:
            if img_small.shape[2] == 4:
                gray = cv2.cvtColor(img_small, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        del img_small
        
        # Convert 16-bit to 8-bit for ORB feature extraction (OpenCV ORB only supports CV_8U)
        if gray.dtype == np.uint16:
            gray = (gray >> 8).astype(np.uint8)
        use_gpu = False
        if is_gpu_enabled():
            if gpu_lock.acquire(blocking=False):
                use_gpu = True
                try:
                    u_gray = cv2.UMat(gray)
                    u_gray_blur = cv2.GaussianBlur(u_gray, (11, 11), 0)
                    orb = cv2.ORB_create(nfeatures=self.n_features, fastThreshold=20)
                    kp, u_des = orb.detectAndCompute(u_gray_blur, None)
                    gray_blur = u_gray_blur.get()
                    des = u_des.get() if u_des is not None else None
                finally:
                    gpu_lock.release()
                    
        if not use_gpu:
            gray_blur = cv2.GaussianBlur(gray, (11, 11), 0)
            orb = cv2.ORB_create(nfeatures=self.n_features, fastThreshold=20)
            kp, des = orb.detectAndCompute(gray_blur, None)
        kp_pts = np.float32([k.pt for k in kp]) if len(kp) > 0 else np.empty((0, 2), dtype=np.float32)
        
        return (h, w), kp_pts, des, gray_blur

    def compute_pair_transform(self, kp1, des1, kp2, des2, gray1, gray2):
        M, inliers = None, None
        max_shift = max(gray1.shape) * self.max_shift_percent
        good_matches = []
        
        if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
            matches = self.bf.knnMatch(des1, des2, k=2)
            for m_n in matches:
                if len(m_n) != 2: continue
                m, n = m_n
                if m.distance < 0.8 * n.distance:
                    pt1 = kp1[m.queryIdx]
                    pt2 = kp2[m.trainIdx]
                    if math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1]) < max_shift:
                        good_matches.append(m)
                    
            if len(good_matches) >= 4:
                src_pts = kp1[[m.queryIdx for m in good_matches]].reshape(-1, 1, 2)
                dst_pts = kp2[[m.trainIdx for m in good_matches]].reshape(-1, 1, 2)
                M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        
        inlier_count = int(np.sum(inliers)) if inliers is not None else 0
        match_count = len(good_matches)
        
        # User requested no frame rejections.
        # If ORB completely fails or has very few inliers, fallback to identity matrix (no movement)
        if M is None or inlier_count < 10 or (match_count > 0 and (inlier_count / match_count) < 0.10):
            return np.eye(2, 3, dtype=np.float64), inlier_count, None
            
        gray1_f = np.float32(gray1)
        gray2_f = np.float32(gray2)
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4)
            M_init = M.astype(np.float32)
            cc, M_ecc = cv2.findTransformECC(gray1_f, gray2_f, M_init, cv2.MOTION_AFFINE, criteria)
            M = M_ecc.astype(np.float64)
            
            # Lowered ECC threshold and fallback to ORB instead of rejecting
            if cc < 0.5:
                M = M_init.astype(np.float64)
        except Exception as e:
            # Fallback to ORB if ECC throws an exception
            M = M_init.astype(np.float64)
            
        return M.astype(np.float64), inlier_count, None

    def get_alignment_matrices(self, image_paths, ref_index=0, num_threads=None, enable_crop=True):
        n = len(image_paths)
        if n == 0:
            return [], (0, 0), (0, 0, 0, 0), []
            
        opt_threads, available_ram = self._get_optimal_threads(image_paths[0], num_threads)
        
        cpu_cores = os.cpu_count() or 4
        # Dynamically scale OpenCV threads based on Python threads to maximize CPU utilization
        cv2.setNumThreads(max(1, cpu_cores // opt_threads))
        
        print(f"[{time.strftime('%H:%M:%S')}] Alignment: Using {opt_threads} threads based on {available_ram:.1f}GB RAM.")
        print("[PROGRESS: 5]")
        
        with ThreadPoolExecutor(max_workers=opt_threads) as executor:
            futures = {executor.submit(self.extract_features, path): idx for idx, path in enumerate(image_paths)}
            features = [None] * n
            for i_progress, f in enumerate(as_completed(futures)):
                idx = futures[f]
                features[idx] = f.result()
                filename = os.path.basename(image_paths[idx])
                print(f"[{time.strftime('%H:%M:%S')}][Alignment] Extracted features for {filename} ({idx+1}/{n})")
            
        print("[PROGRESS: 15]")
        valid_frames = [True] * n
        pair_transforms = [None] * n
        pair_transforms[0] = np.eye(2, 3, dtype=np.float64)
        last_valid_idx = 0
        for i in range(1, n):
            _, kp_prev, des_prev, gray_prev = features[last_valid_idx]
            _, kp_curr, des_curr, gray_curr = features[i]
            filename = os.path.basename(image_paths[i])
            M_pair, _, err_msg = self.compute_pair_transform(kp_prev, des_prev, kp_curr, des_curr, gray_prev, gray_curr)
            if M_pair is None:
                print(f"[{time.strftime('%H:%M:%S')}] Frame {i} ({filename}) rejected. Reason: {err_msg}")
                valid_frames[i] = False
                pair_transforms[i] = None
            else:
                pair_transforms[i] = M_pair
                last_valid_idx = i
            
        cumulative = [None] * n
        cumulative[ref_index] = np.eye(2, 3, dtype=np.float64)
        valid_frames[ref_index] = True
        
        for i in range(ref_index + 1, n):
            if not valid_frames[i]:
                cumulative[i] = None
                continue
                
            prev_idx = i - 1
            while prev_idx >= 0 and not valid_frames[prev_idx]:
                prev_idx -= 1
                
            prev = cumulative[prev_idx]
            pair = pair_transforms[i]
            R_pair, t_pair = pair[:2, :2], pair[:, 2]
            R_prev, t_prev = prev[:2, :2], prev[:, 2]
            cumulative[i] = np.hstack([R_pair @ R_prev, (R_pair @ t_prev + t_pair).reshape(2, 1)])
            
        for i in range(ref_index - 1, -1, -1):
            if not valid_frames[i]:
                cumulative[i] = None
                continue
                
            nxt_idx = i + 1
            while nxt_idx < n and not valid_frames[nxt_idx]:
                nxt_idx += 1
                
            nxt = cumulative[nxt_idx]
            pair = pair_transforms[nxt_idx]
            R_pair, t_pair = pair[:2, :2], pair[:, 2]
            try:
                R_inv = np.linalg.inv(R_pair)
            except np.linalg.LinAlgError:
                R_inv = R_pair.T
            t_inv = -R_inv @ t_pair
            R_nxt, t_nxt = nxt[:2, :2], nxt[:, 2]
            cumulative[i] = np.hstack([R_inv @ R_nxt, (R_inv @ t_nxt + t_inv).reshape(2, 1)])
            
        ref_h, ref_w = features[ref_index][0]
        ref_scale = self._get_scale_factor(ref_h, ref_w)
        ref_sh, ref_sw = int(ref_h * ref_scale), int(ref_w * ref_scale)
        actual_ref_scale = ref_sw / ref_w if ref_w > 0 else 1.0
        
        common_mask_small = np.ones((ref_sh, ref_sw), dtype=np.uint8) * 255
        
        for i in range(n):
            if i == ref_index or not valid_frames[i]: continue
            M_small = cumulative[i].copy()
            M_small_inv = cv2.invertAffineTransform(M_small)
            h, w = features[i][0]
            scale = self._get_scale_factor(h, w)
            sh, sw = int(h * scale), int(w * scale)
            
            mask_small = cv2.warpAffine(
                np.ones((sh, sw), dtype=np.uint8) * 255, M_small_inv, (ref_sw, ref_sh),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            common_mask_small = cv2.bitwise_and(common_mask_small, mask_small)
            
        x_s, y_s, cw_s, ch_s = cv2.boundingRect(common_mask_small)
        
        if enable_crop and cw_s > ref_sw // 2 and ch_s > ref_sh // 2:
            crop_x = int(round(x_s / actual_ref_scale))
            crop_y = int(round(y_s / actual_ref_scale))
            crop_w = int(round(cw_s / actual_ref_scale))
            crop_h = int(round(ch_s / actual_ref_scale))
            crop_x = max(0, min(crop_x, ref_w - 1))
            crop_y = max(0, min(crop_y, ref_h - 1))
            crop_w = max(1, min(crop_w, ref_w - crop_x))
            crop_h = max(1, min(crop_h, ref_h - crop_y))
        else:
            crop_x, crop_y, crop_w, crop_h = 0, 0, ref_w, ref_h
            
        return cumulative, (ref_h, ref_w), (crop_x, crop_y, crop_w, crop_h), [f[0] for f in features], valid_frames

    def warp_and_crop_single(self, image_path, M, orig_shape, ref_shape, crop_box, run_use_gpu=False, deghost_ref_gray_small=None, grayscale_only=False, deghost_preset="medium"):
        if grayscale_only:
            # Read directly as 1-channel grayscale while preserving bit depth (uint16/uint8)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        else:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
            
        cx, cy, cw, ch = crop_box
        
        # If transform is identity, bypass warpAffine completely to save CPU/GPU cycles
        if np.allclose(M, np.eye(2, 3)):
            cropped = img[cy:cy+ch, cx:cx+cw].copy()
        else:
            ref_h, ref_w = ref_shape
            h, w = orig_shape
            
            scale = self._get_scale_factor(h, w)
            scale_x = int(w * scale) / w if scale < 1.0 else 1.0
            
            M_full = M.copy()
            M_full[0, 2] /= scale_x
            M_full[1, 2] /= scale_x
            
            M_warp = cv2.invertAffineTransform(M_full)
            
            if run_use_gpu:
                try:
                    with gpu_lock:
                        umat_img = cv2.UMat(img)
                        # We don't delete img yet, so we have it for fallback if cv2.warpAffine fails
                        umat_warped = cv2.warpAffine(umat_img, M_warp, (ref_w, ref_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                        del umat_img
                        warped = umat_warped.get()
                        del img
                except cv2.error as e:
                    if is_oom_error(e):
                        disable_gpu(f"OpenCL Error (Out of VRAM?): {e}. Falling back to CPU.")
                        run_use_gpu = False
                    else:
                        raise e
                        
            if not run_use_gpu:
                warped = cv2.warpAffine(img, M_warp, (ref_w, ref_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                del img  # free original — warped is a separate allocation

            cropped = warped[cy:cy+ch, cx:cx+cw].copy()  # copy so warped can be freed
            del warped

        if deghost_ref_gray_small is not None:
            if len(cropped.shape) == 2:
                cur_gray = cropped
            elif cropped.shape[2] == 4:
                cur_gray = cv2.cvtColor(cropped, cv2.COLOR_BGRA2GRAY)
            else:
                cur_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                
            if cur_gray.dtype == np.uint16:
                cur_gray = (cur_gray >> 8).astype(np.uint8)
                
            flow_scale = self._get_scale_factor(ch, cw)
            if flow_scale < 1.0:
                cur_gray_small = cv2.resize(cur_gray, (int(cw * flow_scale), int(ch * flow_scale)), interpolation=cv2.INTER_AREA)
            else:
                cur_gray_small = cur_gray
            del cur_gray
            
            if deghost_preset == "high":
                dis_preset = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
            elif deghost_preset == "fast":
                dis_preset = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
            else:
                dis_preset = cv2.DISOPTICAL_FLOW_PRESET_FAST
                
            dis = cv2.DISOpticalFlow_create(dis_preset)
            flow_small = dis.calc(deghost_ref_gray_small, cur_gray_small, None)
            del cur_gray_small
            
            # Upscale flow
            flow_full = cv2.resize(flow_small, (cw, ch), interpolation=cv2.INTER_LINEAR)
            flow_full[:,:,0] *= (cw / flow_small.shape[1])
            flow_full[:,:,1] *= (ch / flow_small.shape[0])
            del flow_small
            
            # Add base coordinates using 1D broadcasting to eliminate massive memory spikes
            flow_full[:,:,0] += np.arange(cw, dtype=np.float32)
            flow_full[:,:,1] += np.arange(ch, dtype=np.float32)[:, np.newaxis]
            
            cropped = cv2.remap(cropped, flow_full[:,:,0], flow_full[:,:,1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            del flow_full

        return cropped



    def compute_alignment(self, image_paths, ref_index=0, num_threads=None, enable_deghosting=False, enable_exposure_normalization=False, enable_crop=True):
        """Compute alignment transforms without warping images.
        
        Returns a dict with alignment data that can be passed to
        FocusStacker.stack_from_paths() for the optimized pipeline.
        """
        cumulative, ref_shape, crop_box, orig_shapes, valid_frames = self.get_alignment_matrices(
            image_paths, ref_index, num_threads, enable_crop
        )
        opt_threads, _ = self._get_optimal_threads(image_paths[0], num_threads)
        run_use_gpu = is_gpu_enabled() and (opt_threads == 1)
        
        cw, ch = crop_box[2], crop_box[3]
        print(f"[{time.strftime('%H:%M:%S')}] Alignment complete. Crop: {cw}x{ch}")
        return {
            'transforms': cumulative,
            'ref_shape': ref_shape,
            'crop_box': crop_box,
            'orig_shapes': orig_shapes,
            'ref_index': ref_index,
            'run_use_gpu': run_use_gpu,
            'enable_deghosting': enable_deghosting,
            'enable_exposure_normalization': enable_exposure_normalization,
            'valid_frames': valid_frames,
        }
