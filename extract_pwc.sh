#!/usr/bin/env bash

video_dir="/home/salvadom/Data/OF_Profiling/videos"
frames_dir="/home/salvadom/Data/OF_Profiling/frames"
output_dir="/home/salvadom/Data/OF_Profiling/pwc"

rm -rf ${output_dir}/*
python runbatch.py --framesDir=${video_frames_dir} --flowDir=${video_output_dir}

