import ffmpeg
import cv2
from pprint import pprint

def convert_to_seconds(time):
    '''Converts time in format MM:SS to total number of seconds'''
    time = time.split(':')
    return int(time[0])*60 + int(time[1])

path = 'D:/Big Data/TikTok/train/TikTokVideos/9.mp4'
cap = cv2.VideoCapture(path)
num_frames_cv2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
metadata = ffmpeg.probe(path)['streams']
frame_rate = float(metadata[0]['avg_frame_rate'].split('/')[0])
duration = metadata[0]['duration']
num_frames_ffmpeg = float(metadata[0]['duration']) * frame_rate
print(f'Total frames sanity check:\n\tcv2:\t{num_frames_cv2}\n\tffmpeg:\t{num_frames_ffmpeg}')

_id = 0
timestamps = ['0:00', '0:15', '0:30']
for i in range(len(timestamps)-1):
    start = convert_to_seconds(timestamps[i])
    end = convert_to_seconds(timestamps[i+1])


    in_file = ffmpeg.input(path)
    vid = (
        in_file
        .trim(start=start, end=end)
        .setpts('PTS-STARTPTS')
        .output(f'out_{_id}.mp4')
        .run()
    )
    aud = (
        in_file
        .filter_('atrim', start=start, end=end)
        .filter_('asetpts', 'PTS-STARTPTS')
        .output(f'out_{_id}.wav')
        .run()
    )

    _id += 1
    
# in_file = ffmpeg.input(path)
# print(in_file.video)
# vid = (
#     ffmpeg
#     .input(path)
#     .trim(start=100, end=100)
#     .setpts('PTS-STARTPTS')
# )

# aud = (
#     ffmpeg
#     .input(path)
#     .trim(start=100, end=100)
#     .setpts('PTS-STARTPTS')
# )
