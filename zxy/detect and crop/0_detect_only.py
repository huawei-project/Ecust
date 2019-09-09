"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import tensorflow as tf
import numpy as np
from toolkit import detect_face
import cv2
import glob

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
    
def detect_RGB(indoors=True):

    failedface_filename = rootdir + '/failedrgb.txt'

    # 如果数据图片的目录格式发生了改变，此处读取数据的程序需要做相应改变
    if indoors == True:
        # indoors
        rgb_paths_1_6 = glob.glob(os.path.join(rootdir, '*/rgb/*/*[1,6]/*'), recursive=True)
        rgb_paths_5 = glob.glob(os.path.join(rootdir, '*/rgb/*/*jpg'), recursive=True)
        rgbfiles = rgb_paths_1_6 + rgb_paths_5
    else:
        # outdoor
        rgb_paths_1= glob.glob(os.path.join(rootdir, '*/rgb/*1/*'), recursive=True)
        rgb_paths_6 = glob.glob(os.path.join(rootdir, '*/rgb/*jpg'), recursive=True)
        rgbfiles = rgb_paths_1 + rgb_paths_6

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    with open(failedface_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0 

        for image_path in rgbfiles:
            nrof_images_total += 1 
            # to transform the \\ in windows to / in unix
            image_path = image_path.replace('\\','/')
            try:
                img = misc.imread(image_path)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image_path, e)
                print(errorMessage)
            else:
                if img.ndim<2: # 数据的维度<2 ，说明都不是一张图片
                    print('Unable to align "%s"' % image_path)
                    continue
                if img.ndim == 2: # 数据维度 = 2 ，灰度图，复制3次转化成rgb 3通道图片
                    img = to_rgb(img)

                img = img[:,:,0:3]

                bounding_boxes, landmark = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                if nrof_faces > 0: # 检测出的框>0，说明检测到了 如果图片中只有一个人的话，nrof_faces = 1
                    nrof_successfully_aligned += 1    
                else: # 没有检测到
                    print('Unable to align "%s"' % image_path)
                    text_file.write('%s\n' % (image_path))
        
        text_file.write('Total number of images: %d \n' % nrof_images_total)
        text_file.write('Number of successfully aligned images: %d' % nrof_successfully_aligned)

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

def detect_Multi(indoors=True):

    failedface_filename = rootdir + '/failemultifile.txt'
    failedface_filrdir = rootdir + '/failemultdir.txt'

    # 如果数据图片的目录格式发生了改变，此处读取数据的程序需要做相应改变
    if indoors == True:
        # indoors
        multi_paths_1_6 = glob.glob(os.path.join(rootdir, '*/multi/*/*[1,6]/*'), recursive=True)
        multi_paths_5 = glob.glob(os.path.join(rootdir, '*/multi/*/*5'), recursive=True)
        multidirs = multi_paths_1_6 + multi_paths_5
    else:
        # outdoor
        multi_paths_1 = glob.glob(os.path.join(rootdir, '*/multi/*1/*'), recursive=True)
        multi_paths_6 = glob.glob(os.path.join(rootdir, '*/multi/*6'), recursive=True)
        multidirs = multi_paths_1 + multi_paths_6

    nrof_images_total = len(multidirs)

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    nrof_successfully_aligned = 0 

    with open(failedface_filrdir, "w") as dirs:
        with open(failedface_filename, "w") as files:
            for imgdir in multidirs:
                # to transform the \\ in windows to / in unix
                imgdir = imgdir.replace('\\','/')
                imgpathlist = [os.path.join(imgdir, imgpath) for imgpath in os.listdir(imgdir)]
                nrof_successfully_aligned_band = 0
                for image_path in imgpathlist:
                    # to transform the \\ in windows to / in unix
                    image_path = image_path.replace('\\','/')
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim<2: # 数据的维度<2 ，说明都不是一张图片
                            print('Unable to align "%s"' % image_path)
                            continue
                        if img.ndim == 2: # 数据维度 = 2 ，灰度图，复制3次转化成rgb 3通道图片
                            img = to_rgb(img)
                        img = img[:,:,0:3]
                        
                        bounding_boxes, landmark = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        if nrof_faces>0: # 检测出的框>0，说明检测到了 如果图片中只有一个人的话，nrof_faces = 1
                            nrof_successfully_aligned_band +=1
                        else:
                            # print('Unable to align "%s"' % image_path)
                            files.write('%s\n' % (image_path))

                # 如果有一个波段被检测出来了，就算是整个波段都被检测出来了
                if nrof_successfully_aligned_band > 0: 
                    nrof_successfully_aligned += 1
                else:
                    print('Unable to align "%s"' % imgdir)
                    dirs.write('%s\n' % (imgdir))

        dirs.write('Total number of images: %d \n' % nrof_images_total)
        dirs.write('Number of successfully aligned images: %d' % nrof_successfully_aligned)

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

if __name__ == '__main__':
    rootdir = 'E:/Desktop/Outdoor20190810'
    detect_RGB(indoors=False)
    detect_Multi(indoors=False)