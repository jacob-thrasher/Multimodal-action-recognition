#!/root/venv/bin/python3
import ffmpeg
import os
import pandas as pd
import datetime
import csv
import json

def convert_to_seconds(time):
    '''Converts time in format MM:SS to total number of seconds'''
    time = time.split(':')
    return int(time[0])*60 + int(time[1])

def splice_by_timestamp(src, dst, csvpath):
    '''
    Splices videos by timestamps given in csv file. Preserves original data
    Expects src subfolders to be prepended with label
    
    Args:
        src - Path to data
        dst - Desired save destination
        csv - path to csv containing timestamps
    '''

    df = pd.read_excel(csvpath, sheet_name='time_marks_2')
    # print(df['video names']['Alex_150727_1.MPG'])
    # row = df.loc[df['video names'] == 'Alex_150727_1.MPG']
    # timestamps = []
    # for item in row.iloc[0][2:]:
    #     if type(item) is datetime.time:
    #         seconds = (item.hour*60 + item.minute)*60 + item.second
    #         timestamps.append(seconds)
    # print(timestamps)
    # f = open(os.path.join(dst, 'labels.csv'), 'w', newline='', encoding='UTF-8')
    # writer = csv.writer(f)
    # header = ['filename', 'label']
    # writer.writerow(header)
    labels = {}

    for folder in os.listdir(src):
        if folder[0] == '.': continue   #Skip hidden files
        this_folder = os.path.join(src, folder)
        for item in os.listdir(this_folder):
            if item[0] == '.': continue #Skip hidden files
            if item.endswith(('.MPG', '.mp4', '.m4v')):
                this_item = os.path.join(this_folder, item)
                label = folder[0]
                row = df.loc[df['video names'] == item]
                
                if row.empty: continue #Pass if no entry in csv

                timestamps = []
                for t in row.iloc[0][2:]:
                    if type(t) == datetime.time:
                        seconds = (t.hour*60 + t.minute)*60 + t.second
                        timestamps.append(seconds)
                timestamps.append(-1)
                
            _id = 0
            filename = item.split('.')[0]
            if 'other' in filename: continue #Skip others
            if label not in ['0', '1', '2']: continue #Skip unlabeled files

            for i in range(len(timestamps)-1):
                start = timestamps[i]
                end = timestamps[i+1]
                
                # writer.writerow([f'{filename}_{_id}', label])
                labels[f'{filename}_{_id}'] = label

                # in_file = ffmpeg.input(this_item) 
                # vid = (
                #     in_file
                #     .trim(start=start, end=end)
                #     .setpts('PTS-STARTPTS')
                #     .output(os.path.join(dst, 'video', f'{filename}_{_id}.mp4'))
                #     .run()
                # )
                # aud = (
                #     in_file
                #     .filter_('atrim', start=start, end=end)
                #     .filter_('asetpts', 'PTS-STARTPTS')
                #     .output(os.path.join(dst, 'audio', f'{filename}_{_id}.wav'))
                #     .run()
                # )

                _id += 1

    print(type(labels))
    with open(os.path.join(dst, 'labels.json'), 'w') as f:
        json.dump(labels, f)
    # f.close()
    return

# src = 'mnt/sdc/CalTech ADOS/2_ADOS_Data_videos'
src = '/mnt/sdc/CalTech ADOS/2_ADOS_Data_videos'
dst = '/mnt/sdc/jacob'
csvpath = '/mnt/sdc/CalTech ADOS/timestamps.xls'
splice_by_timestamp(src, dst, csvpath)