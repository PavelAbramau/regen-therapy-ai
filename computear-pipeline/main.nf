nextflow.enable.dsl=2

params.input_dir = "$baseDir/data/raw_images"
params.output_dir = "$baseDir/data/results"
params.weights_seg = "$baseDir/weights/yolo11seg.pt"
params.weights_pose = "$baseDir/weights/yolo11pose.pt"

process ANALYZE_IMAGES {
    container 'distance-tool:latest'
    publishDir params.output_dir, mode: 'copy'

    input:
    path input_dir
    path weights_seg
    path weights_pose

    output:
    path "master_results.csv"
    path "*_annotated.jpg"

    script:
    """
    python /app/wound_segmentation_dist_v2/analyze_wound.py \\
        --input_dir ${input_dir} \\
        --output_dir . \\
        --weights_seg ${weights_seg} \\
        --weights_pose ${weights_pose}
    """
}

workflow {
    input_ch = Channel.fromPath(params.input_dir, checkIfExists: true)
    weights_seg_ch = Channel.fromPath(params.weights_seg, checkIfExists: true)
    weights_pose_ch = Channel.fromPath(params.weights_pose, checkIfExists: true)
    ANALYZE_IMAGES(input_ch, weights_seg_ch, weights_pose_ch)
}