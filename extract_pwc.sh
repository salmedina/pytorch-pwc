#!/usr/bin/env bash

video_dir="/home/salvadom/Data/OF_Profiling/videos"
frames_dir="/home/salvadom/Data/OF_Profiling/frames"
output_dir="/home/salvadom/Data/OF_Profiling/turing"

rm -rf ${output_dir}/*

for video_path in ${video_dir}/*.mp4; do
    file_name=$(basename -- "$video_path")
    video_name="${file_name%.*}"
    video_frames_dir=${frames_dir}/${video_name}
    video_output_dir=${output_dir}/${video_name}
    echo "${video_output_dir}"
    mkdir -p ${video_output_dir}
    python runbatch.py --framesDir ${video_frames_dir} --outputDir ${video_output_dir}
done
