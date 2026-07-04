import os
import cv2
import time
import numpy as np
import re
from dataclasses import dataclass
from typing import Optional
from alignment import ImageAligner
from stacking import FocusStacker
from gpu_state import enable_gpu

@dataclass
class PipelineOptions:
    method: str = "pyramid"
    kernel_size: int = 5
    pyramid_levels: int = 5
    energy_scale: float = 1.0
    in_memory: bool = False
    deghost: bool = False
    exposure_norm: bool = False
    num_threads: Optional[int] = None
    output_dir: Optional[str] = None
    enable_crop: bool = True
    deghost_preset: str = "medium"

def run_pipeline(folder_path: str, options: PipelineOptions):
    """
    Executes the focus stacking pipeline for a single folder.
    """
    # Reset GPU state for this new run (in case it was disabled during a previous run)
    enable_gpu()
    
    image_paths = [
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if f.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff', '.png'))
    ]
    # Natural sort
    image_paths.sort(key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', f)])
    
    if not image_paths:
        raise ValueError(f"No valid images found in {folder_path}")

    print(f"\n==============================================")
    print(f"Processing Folder: {folder_path}")
    print(f"Images: {len(image_paths)}")
    print(f"==============================================")
    
    start_time = time.time()
    
    aligner = ImageAligner()
    ref_idx = len(image_paths) // 2
    
    print(f"Using Stacking Method: {options.method.upper()}")
    print(f"Optimized pipeline settings (Energy scale: {options.energy_scale * 100:.0f}%, De-ghosting: {options.deghost}, Exposure normalization: {options.exposure_norm})")
    
    alignment_data = aligner.compute_alignment(
        image_paths, 
        ref_index=ref_idx, 
        num_threads=options.num_threads,
        enable_deghosting=options.deghost, 
        enable_exposure_normalization=options.exposure_norm,
        enable_crop=options.enable_crop
    )
    alignment_data['deghost_preset'] = options.deghost_preset
    
    stacker = FocusStacker(
        pyramid_levels=options.pyramid_levels, 
        kernel_size=options.kernel_size, 
        num_threads=options.num_threads, 
        in_memory=options.in_memory, 
        method=options.method,
        energy_scale=options.energy_scale
    )
    
    final_image = stacker.stack_from_paths(image_paths, aligner, alignment_data)
    
    if final_image is not None:
        elapsed = time.time() - start_time
        print(f"\nTotal processing time: {elapsed:.2f} seconds")
        
        # Setup output directory
        outputs_dir = options.output_dir or os.path.join(folder_path, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        
        folder_name = os.path.basename(folder_path.rstrip("/\\"))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"{folder_name}_{timestamp}"
        
        output_tiff_comp = os.path.join(outputs_dir, f"{base_name}_compressed.tif")
        output_tiff_uncomp = os.path.join(outputs_dir, f"{base_name}_uncompressed.tif")
        output_jpg = os.path.join(outputs_dir, f"{base_name}.jpg")
        
        # Save compressed TIFF (LZW)
        cv2.imwrite(output_tiff_comp, final_image, [cv2.IMWRITE_TIFF_COMPRESSION, 5])
        # Save uncompressed TIFF
        cv2.imwrite(output_tiff_uncomp, final_image, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        
        if final_image.dtype == np.uint16:
            jpg_img = (final_image >> 8).astype(np.uint8)
        else:
            jpg_img = final_image
            
        cv2.imwrite(output_jpg, jpg_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
        print(f"Saved LZW-compressed TIFF: {output_tiff_comp}")
        print(f"Saved uncompressed TIFF:   {output_tiff_uncomp}")
        print(f"Saved JPEG preview:        {output_jpg}")
        
        return {
            'tiff_comp': output_tiff_comp,
            'tiff_uncomp': output_tiff_uncomp,
            'jpg': output_jpg,
            'time_sec': elapsed
        }
    return None
