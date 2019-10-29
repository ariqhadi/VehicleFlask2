import os
import logging
import csv
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils_tf2


AREA_COLOR = (66, 183, 42)


class PipelineRunner(object):
    '''
        Very simple pipline.

        Just run passed processors in order with passing context from one to 
        another.

        You can also set log level for processors.
    '''

    def __init__(self, pipeline=None, log_level=logging.INFO):
        self.pipeline = pipeline or []
        self.context = {}
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(log_level)
        self.log_level = log_level


    def set_context(self, data):
        self.context = data

    def run(self):
        for p in self.pipeline:
            self.context = p(self.context)

        self.log.debug("Frame #%d processed.", self.context['frame_number'])
        # print("1")
        return self.context


class PipelineProcessor(object):
    '''
        Base class for processors.
    '''

    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        

class CapacityCounter(PipelineProcessor):

    def __init__(self):
        super(CapacityCounter, self).__init__()
        # self.AREA_PTS = AREA_PTS
    def calculate_capacity(self, frame, frame_number,base_bgrd):
        f_height,f_width,_ = frame.shape
        shape = (f_height,f_width)
        cl_y = round(1 / 2 * f_height)

    #DEFINE AREA HITUNG
        # AREA_PTS = np.array([(int(f_width/2)+150, 350), (int(f_width/2)+20, 350), (int(f_width/2)+10, f_height), (f_width, f_height-100)])
        # AREA_PTS1 =np.array([(int(f_width/2)-250, 350), (int(f_width/2), 350), (int(f_width/2), f_height), (0, f_height-100)])

        AREA_PTS = np.array([(int(f_width/2)+100, 250), (int(f_width/2)+20, 250), (int(f_width/2)+10, f_height), (f_width, f_height-100)])
        AREA_PTS1 =np.array([(int(f_width/2)-100, 250), (int(f_width/2), 250), (int(f_width/2), f_height), (-150, f_height-100)])


    #CREATE MASK UNTUK TEMP
        mask = np.zeros(frame.shape[:2], np.uint8)
        mask2 = np.zeros(frame.shape[:2], np.uint8)

    #ISI MASK SAMA AREA YANG DIHITUNG
        cv2.drawContours(mask, [AREA_PTS], -1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.drawContours(mask2, [AREA_PTS1], -1, (255, 255, 255), -1, cv2.LINE_AA)

    #CROP FRAME DENGAN MASK YANG TELAH SESUAI AREA HITUGN
        bgr = cv2.bitwise_and(base_bgrd, base_bgrd, mask=mask)
        bgr2 = cv2.bitwise_and(base_bgrd, base_bgrd, mask=mask2)

        dst = cv2.bitwise_and(frame, frame, mask=mask)
        dst2 = cv2.bitwise_and(frame, frame, mask=mask2)

        # cv2.imshow("Sfg",dst2)

        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # this used for noise reduction at night time

    #RUBAH WARNA
        dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
        dst2 = cv2.cvtColor(dst2, cv2.COLOR_RGB2GRAY)

        bgr= cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY)
        bgr2 = cv2.cvtColor(bgr2,cv2.COLOR_RGB2GRAY)

    #LIHAT PERBEDAAN ANTARA FRAME BACKGROUND DAN FRAME BARU
        difference = cv2.absdiff(bgr,dst)
        difference2 = cv2.absdiff(bgr2, dst2)
        
        
    #APABILA ADA YANG BEDA (DIFFERENCE) MAKA PIXEL DI JADIIN 255/PUTIH
        _, t = cv2.threshold(difference,35, 255,cv2.THRESH_BINARY)
        _, t2 = cv2.threshold(difference2,35, 255,cv2.THRESH_BINARY)
    
    #HITUNG SEMUA PIXEL BERWARNA PUTIH
        all1 = cv2.countNonZero(bgr)
        free1 = cv2.countNonZero(t)
        all2 = cv2.countNonZero(bgr2)
        free2 = cv2.countNonZero(t2)
        t3 = t + t2

        # img = np.zeros(frame.shape, frame.dtype)
        # img[:, :] = AREA_COLOR
        # mask = cv2.bitwise_and(img, img, mask= (mask))

        # cv2.addWeighted(mask, 1, frame, 1, 0, frame)
        # cv2.imshow("test",cv2.resize(t,(int(f_width/2),int(f_height/2))))
        # cv2.imshow("test1",cv2.resize(t2+t,(int(f_width/2),int(f_height/2))))
        # cv2.imshow("test2",cv2.resize(frame,(int(f_width/2),int(f_height/2))))

        
    #KAPASITAS DILIHAT DARI RATIO ANTARA JUMLAH PIXEL (PADA FRAME BARU) YANG KOSONG DAN JUMLAH PIXEL KESELURUHAN (PADA BACKGROUND)
        capacity = float(free1/all1)*100
        capacity2 = float(free2/all2)*100
        
        result = capacity,capacity2

        return result
        
    def __call__(self, context):
        frame = context['frame'].copy()
        frame_number = context['frame_number']
        base_bgrd = context['base_bgrd']
        # prev_cap = context['prev_cap']
        
        capacity = self.calculate_capacity(frame, frame_number,base_bgrd)

        context['capacity1'] = capacity[1]
        context['capacity2'] = capacity[0]

        return context
        
        
class ContextCsvWriter(PipelineProcessor):

    def __init__(self, path, start_time=0, data=None, field_names=[], fps=30, faster=1, diff=False):
        super(ContextCsvWriter, self).__init__()

        self.fp = open(os.path.join(path), 'w')
        self.writer = csv.DictWriter(
            self.fp, fieldnames=['time']+field_names)
        self.writer.writeheader()
        self.start_time = start_time
        self.field_names = field_names
        self.fps = fps
        self.path = path
        self.prev = None
        self.data = data
        self.faster = faster
        self.diff = diff

    def __call__(self, context):
        frame_number = context['frame_number']
        count = context.get(self.data) or context
        count = {k:v for k,v in count.items() if k in self.field_names}

        _count = count        
        if self.diff:
            if not self.prev:
                self.prev = count
            else:
                _count = {k: v - self.prev[k] for k, v in count.iteritems()}
                self.prev = count
                
        if self.faster > 1:
            _count['time'] = (self.start_time + int(frame_number*self.faster/self.fps)) 
        else:
            _count['time'] = ((self.start_time + int(frame_number / self.fps)) * 100 + int(100.0 / self.fps) * (frame_number % self.fps))

        self.writer.writerow(_count)
        
        return context





