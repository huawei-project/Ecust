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
    path_save = os.path.join(rootdir, "rgbdetect.txt")

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
    dict_save = dict()
    

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
                    # [n,4] 人脸框
                    nrof_successfully_aligned += 1
                    det = bounding_boxes[:,0:4]
                    img_size = np.asarray(img.shape)[0:2]
                    if nrof_faces>1:
                        '''
                        如果检测出来很多张脸，但是要选一个最大的，采用的方法如下：
                        用人脸框大小-偏移平方的2倍，得到的结果那个大就选哪个
                        即人脸框越大越好，偏移量越小越好
                        '''
                        # det [n,0:4]是矩阵
                        bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1]) # (x2-x1)*(y2-y1) 人脸框大小
                        img_center = img_size / 2 # 原图片中心
                        offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                        index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                        bounding_boxes = bounding_boxes[index,:]   
                        landmark = landmark[:,index]   
                        bbox = bounding_boxes.tolist()
                        score = bbox[4]
                        bbox = bbox[0:4]
                        new_landmark_modify = np.zeros(landmark.shape)
                        new_landmark_modify[0:10:2] = landmark[0:5]
                        new_landmark_modify[1:11:2] = landmark[5:10] 
                        landmark = new_landmark_modify.tolist()
                    
                    else:
                        # 把bbox，landmark的数据格式整理成统一的格式
                        bbox = bounding_boxes.tolist()
                        bbox = [y for x in bbox for y in x]
                        score = bbox[4]
                        bbox = bbox[0:4]
                        # 其中landmark的顺序和我们需要的不一样
                        new_landmark_modify = np.zeros(landmark.shape)
                        new_landmark_modify[0:10:2] = landmark[0:5]
                        new_landmark_modify[1:11:2] = landmark[5:10]
                        landmark = new_landmark_modify.tolist()
                        landmark = [y for x in landmark for y in x]

                else: # 没有检测到
                    print('Unable to align "%s"' % image_path)
                    text_file.write('%s\n' % (image_path))

                dict_save[image_path[len(rootdir):].split('.')[0]] = (score, bbox, landmark)\
                                    if (nrof_faces != 0) else (None,None,None)
        
        text_file.write('Total number of images: %d \n' % nrof_images_total)
        text_file.write('Number of successfully aligned images: %d' % nrof_successfully_aligned)

    # 把bbox，sorce，landmark写到txt中
    with open(path_save,'w') as f:
        f.write(str(dict_save))
        f.write('\n')

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

def detect_Multi(indoors=True):

    failedface_filename = rootdir + '/failemultifile.txt'
    failedface_filrdir = rootdir + '/failemultdir.txt'
    path_save = os.path.join(rootdir, "multidetect.txt")

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
    dict_save = dict() # 保存的bbox,landmark,sorce

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
                            bounding_boxes_temp = bounding_boxes
                            landmark_temp = landmark
                            nrof_faces_temp = nrof_faces
                        else:
                            # print('Unable to align "%s"' % image_path)
                            files.write('%s\n' % (image_path))

                # 如果有一个波段被检测出来了，就算是整个波段都被检测出来了
                if nrof_successfully_aligned_band > 0: 
                    nrof_successfully_aligned += 1
                    bounding_boxes = bounding_boxes_temp
                    landmark = landmark_temp
                    nrof_faces = nrof_faces_temp

                    if nrof_faces > 0: # 检测出的框>0，说明检测到了 如果图片中只有一个人的话，nrof_faces = 1
                    # [n,4] 人脸框
                        det = bounding_boxes[:,0:4]
                        img_size = np.asarray(img.shape)[0:2]
                        if nrof_faces>1:
                            '''
                            如果检测出来很多张脸，但是要选一个最大的，采用的方法如下：
                            用人脸框大小-偏移平方的2倍，得到的结果那个大就选哪个
                            即人脸框越大越好，偏移量越小越好
                            '''
                            # det [n,0:4]是矩阵
                            bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1]) # (x2-x1)*(y2-y1) 人脸框大小
                            img_center = img_size / 2 # 原图片中心
                            offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                            offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                            index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                            bounding_boxes = bounding_boxes[index,:]   
                            landmark = landmark[:,index]   
                            bbox = bounding_boxes.tolist()
                            score = bbox[4]
                            bbox = bbox[0:4]
                            new_landmark_modify = np.zeros(landmark.shape)
                            new_landmark_modify[0:10:2] = landmark[0:5]
                            new_landmark_modify[1:11:2] = landmark[5:10] 
                            landmark = new_landmark_modify.tolist()
                        
                        else:
                            # 把bbox，landmark的数据格式整理成统一的格式
                            bbox = bounding_boxes.tolist()
                            bbox = [y for x in bbox for y in x]
                            score = bbox[4]
                            bbox = bbox[0:4]
                            # 其中landmark的顺序和我们需要的不一样
                            new_landmark_modify = np.zeros(landmark.shape)
                            new_landmark_modify[0:10:2] = landmark[0:5]
                            new_landmark_modify[1:11:2] = landmark[5:10]
                            landmark = new_landmark_modify.tolist()
                            landmark = [y for x in landmark for y in x]

                else:
                    print('Unable to align "%s"' % imgdir)
                    dirs.write('%s\n' % (imgdir))

                dict_save[imgdir[len(rootdir):].split('.')[0]] = (score, bbox, landmark)\
                                if (nrof_faces != 0) else (None,None,None)

        dirs.write('Total number of images: %d \n' % nrof_images_total)
        dirs.write('Number of successfully aligned images: %d' % nrof_successfully_aligned)

    # 把bbox，sorce，landmark写到txt中
    with open(path_save,'w') as f:
        f.write(str(dict_save))
        f.write('\n')

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

if __name__ == '__main__':
    rootdir = 'E:/Desktop/Outdoor20190810'
    detect_RGB(indoors=False)
    detect_Multi(indoors=False)