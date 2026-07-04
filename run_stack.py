import os
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
import argparse
from pipeline import PipelineOptions, run_pipeline
        
def main():
    parser = argparse.ArgumentParser(description="Adaptive FocusStack Pro CLI")
    parser.add_argument("folder", nargs="?", default="Small", help="Path to the folder containing images (or parent folder if --batch). Default is 'Small'.")
    parser.add_argument("--batch", action="store_true", help="Process all subfolders within the given folder.")
    parser.add_argument("--method", type=str, choices=["pyramid", "weighted_average", "depth_map"], default="pyramid", help="Stacking method (default: pyramid)")
    parser.add_argument("--kernel", type=int, default=5, help="Kernel size for energy mapping (default: 5)")
    parser.add_argument("--pyramid", type=int, default=5, help="Number of levels for laplacian pyramid (default: 5)")
    parser.add_argument("--energy", type=float, default=1.0, help="Energy map scale factor (default: 1.0)")
    parser.add_argument("--disk-cache", action="store_true", help="Force disk-backed caching instead of in-memory (saves RAM, slower)")
    parser.add_argument("--no-crop", action="store_true", help="Disable automatic cropping of aligned images")
    parser.add_argument("--deghost", action="store_true", help="Enable optical flow de-ghosting (prevents halos, slower)")
    parser.add_argument("--deghost-preset", type=str, choices=["high", "medium", "fast"], default="medium", help="Deghosting quality (default: medium)")
    parser.add_argument("--no-exposure", action="store_true", help="Disable exposure normalization")
    
    args = parser.parse_args()
    
    target_folder = os.path.abspath(args.folder)
    
    if not os.path.exists(target_folder):
        print(f"Error: The folder '{target_folder}' does not exist.")
        return
        
    print(f"=== Adaptive FocusStack Pro CLI ===")
    
    options = PipelineOptions(
        method=args.method,
        kernel_size=args.kernel,
        pyramid_levels=args.pyramid,
        energy_scale=args.energy,
        in_memory=not args.disk_cache,
        deghost=args.deghost,
        exposure_norm=not args.no_exposure,
        enable_crop=not args.no_crop,
        deghost_preset=args.deghost_preset
    )
    
    if args.batch:
        folders = []
        for d in os.listdir(target_folder):
            full_d = os.path.join(target_folder, d)
            if os.path.isdir(full_d) and os.path.basename(full_d) != "outputs":
                folders.append(full_d)
                
        if not folders:
            print("No subfolders found for batch processing.")
            return
            
        for current_folder in folders:
            try:
                options.output_dir = os.path.join(target_folder, "outputs")
                run_pipeline(current_folder, options)
            except Exception as e:
                print(f"\nCRITICAL ERROR in {current_folder}: {str(e)}")
                import traceback
                traceback.print_exc()
                print("Continuing to next folder...")
    else:
        try:
            run_pipeline(target_folder, options)
        except Exception as e:
            print(f"\nFATAL ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            
    print("\n=== Finished All Tasks ===")

if __name__ == "__main__":
    main()
