#!/bin/bash
rm Results/GeneratedImages/*

rm Results/LerpVideos/lerp.mp4

python3 generate_lerp_video.py --framerate=30 --tracked_onset_cutoff=.15 --copied_frames_segment_modulus=2 --minimum_copy_frames_thresh=20 --copy_framerate_multiplier=1 --lerped_midpoint_search_modulus=1000 --model=55


ffmpeg -r 30 -f image2 -s 512x512 -i Results/GeneratedImages/image%04d.png  -vcodec libx264 -crf 25  -pix_fmt yuv420p Results/LerpedVideos/lerp.mp4