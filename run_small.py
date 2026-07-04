from pipeline import run_pipeline, PipelineOptions

options = PipelineOptions()
folder_path = r"C:\Users\padah\Downloads\Downloads\Bilinear_TIF_Small"

print("Running pipeline on small dataset...")
run_pipeline(folder_path, options)
print("Pipeline finished successfully!")
