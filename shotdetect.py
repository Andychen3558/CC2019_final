import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
def colorHist(img):
    channels = ['b', 'g', 'r']
    hists = []
    for i, c in enumerate(channels):
        hist = cv2.calcHist([img], [i], None, [32], [0, 32])
        hist /= np.sum(hist)
        hists.extend(hist[:, 0])
    hists = np.array(hists)
    return hists

def shotdetect(video_name, min_duration_ms=2000):

    cap = cv2.VideoCapture(video_name)

    if not cap.isOpened():
        return

    count = 0
    p1 = -1
    p2 = -1
    diff = -1
    r = 0.3
    start_m_sec = 0
    m_sec = 0

    record = []

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        feature = colorHist(frame)

        if count != 0:
            diff = np.sum(np.abs(feature-pre_feature))
            
            if p1 < 0:
                p1 = diff

            m_sec = cap.get(cv2.CAP_PROP_POS_MSEC)
            
            if diff > 2*p1 and p2 < 0:
                p2 = p1
            
            if p1 < p2:
                p2 = -1

            if m_sec - start_m_sec > min_duration_ms:
                if (diff > 10*p1 or (diff > 5*p2 and p2 > 0)) and diff > 0.01:
                    ## shot detect
                    p1 = -1
                    p2 = -1
                    diff = -1
                    record.append((start_m_sec, m_sec))
                    start_m_sec = m_sec


            p1 = (1-r)*diff + r*p1 

        pre_feature = feature
        count += 1

    if start_m_sec != m_sec:
        record.append((start_m_sec, m_sec))

    record = [(int(i[0]), int(i[1])) for i in record]

    return record

def extract_key_frame(video_name, shots, out_dir):
    
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_name)

    if not cap.isOpened():
        return

    for shot in shots:
        t = int((shot[0] + shot[1])/2)
        cap.set(cv2.CAP_PROP_POS_MSEC, t)
        ret, frame = cap.read()
        v = video_name.split('/')[-1]
        img_name = f'{v}_{t}.jpg'
        print(img_name)
        cv2.imwrite(os.path.join(out_dir, img_name), frame)


if __name__ == '__main__':
    video_name = os.path.join('video', 'pentatonix.mp4')
    shots = shotdetect(video_name)
    extract_key_frame(video_name, shots, 'keyframes')