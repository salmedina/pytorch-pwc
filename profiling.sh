#!/usr/bin/zsh

video_dir="/mnt/Alfheim/Data/DIVA_Proposals/OF_Profiling/videos"
frames_dir="/mnt/Alfheim/Data/DIVA_Proposals/trajectory_images/heu_neg_traj_props"
output_dir="/mnt/Alfheim/Data/DIVA_Proposals/OF_Profiling/pwcnet"

rm -rf ${output_dir}/*

for video_path in ${video_dir}/*.mp4; do
    video_name=${video_path:t:r}
    video_frames_dir=${frames_dir}/${video_name}
    video_output_dir=${output_dir}/${video_name}
    python runvideo.py --framesDir ${video_frames_dir} --outputDir ${video_output_dir}
done