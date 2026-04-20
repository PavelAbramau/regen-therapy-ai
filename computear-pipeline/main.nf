nextflow.enable.dsl=2

params.input_dir = "$baseDir/data/raw_images"
params.output_dir = "$baseDir/data/results"
params.weights_seg = "$baseDir/weights/best_seg.pt"
params.weights_pose = "$baseDir/weights/best_pose.pt"

process ANALYZE_IMAGES {
    container 'distance-tool:latest'
    publishDir params.output_dir, mode: 'copy', pattern: "*_annotated.jpg"

    input:
    path image_path
    path weights_seg
    path weights_pose

    output:
    path "*_annotated.jpg", emit: annotated_images
    path "result_*.csv", emit: mini_csvs

    script:
    """
    python /app/analyze_wound.py \\
        --image_path ${image_path} \\
        --output_dir . \\
        --weights_seg ${weights_seg} \\
        --weights_pose ${weights_pose}
    """
}

process MERGE_RESULTS {
    container 'distance-tool:latest'
    publishDir params.output_dir, mode: 'copy'

    input:
    path mini_csvs

    output:
    path "master_results.csv"

    script:
    """
    python - <<'PY'
import pandas as pd
from pathlib import Path

csv_files = sorted(Path('.').glob('result_*.csv'))
if not csv_files:
    raise SystemExit("No mini CSV files found to merge.")

merged_df = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files], ignore_index=True)
merged_df.to_csv('master_results.csv', index=False)
print(f"Merged {len(csv_files)} files into master_results.csv")
PY
    """
}

workflow {
    image_ch = Channel.fromPath("${params.input_dir}/**/*.{jpg,JPG,jpeg,JPEG,png,PNG}", checkIfExists: true)
    weights_seg_ch = Channel.value(file(params.weights_seg, checkIfExists: true))
    weights_pose_ch = Channel.value(file(params.weights_pose, checkIfExists: true))

    analyze_results = ANALYZE_IMAGES(image_ch, weights_seg_ch, weights_pose_ch)
    MERGE_RESULTS(analyze_results.mini_csvs.collect())
}