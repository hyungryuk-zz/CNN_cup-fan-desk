# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:40:17 2017

@author: ryuk
"""

import cv2
vidcap = cv2.VideoCapture('종이컵.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  cv2.imwrite("C:/Users/ryuk/.spyder-py3/images/cup/cup%d.jpg" % count, image)     # save frame as JPEG file
  count += 1