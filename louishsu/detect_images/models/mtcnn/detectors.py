# coding:utf-8

import sys
import cv2
import time
import numpy as np
import tensorflow as tf

def py_nms(dets, thresh, mode="Union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        #keep
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class FcnDetector(object):
    """

    Attributes:
        cls_prob:   {tensor(batch_size, H, W, 2)} or {tensor(H, W, 2)}
        bbox_pred:  {tensor(batch_size, H, W, 4)} or {tensor(H, W, 4)}
    Notes:
        - For P-Net
    """

    def __init__(self, net_factory, model_path):

        graph = tf.Graph()
        with graph.as_default():        # 新生成的图作为整个`tensorflow`运行环境的默认

            self.image_op = tf.placeholder(tf.float32, name='input_image')
            self.width_op = tf.placeholder(tf.int32, name='image_width')
            self.height_op = tf.placeholder(tf.int32, name='image_height')
            image_reshape = tf.reshape(self.image_op, [1, self.height_op, self.width_op, 3])
            self.cls_prob, self.bbox_pred, _ = net_factory(image_reshape, training=False)
            
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            
            saver = tf.train.Saver()
            # ----- check -----
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print(model_path)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            # print("restore models' param")
            # -----------------
            saver.restore(self.sess, model_path)

            # print(self.sess.graph.get_operations())

    def predict(self, databatch):
        height, width, _ = databatch.shape
        cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred],
                                            feed_dict={
                                                self.image_op: databatch,
                                                self.width_op: width,
                                                self.height_op: height})
        return cls_prob, bbox_pred


class Detector(object):
    """
    
    Attributes:
        cls_prob:   {tensor(batch_size, 2)}
        bbox_pred:  {tensor(batch_size, 4)}
        data_size:  24 for R-Net and 48 for O-Net
        batch_size: 
    Notes:
        - For R-Net and O-Net
    """

    def __init__(self, net_factory, data_size, batch_size, model_path):

        graph = tf.Graph()
        with graph.as_default():

            self.image_op = tf.placeholder(tf.float32, 
                                            shape=[batch_size, data_size, data_size, 3], 
                                            name='input_image')
            self.cls_prob, self.bbox_pred, self.landmark_pred = net_factory(self.image_op, training=False)
            

            self.sess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            

            saver = tf.train.Saver()
            # ----- check -----
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print(model_path)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            # print("restore models' param")
            # -----------------
            saver.restore(self.sess, model_path)

        self.data_size = data_size
        self.batch_size = batch_size


    def predict(self, databatch):
        """
        
        Args:
            databatch: {tensor(batch_size, data_size, data_size)}
        """

        scores = []
        batch_size = self.batch_size


        minibatch = []
        cur = 0
        n = databatch.shape[0]
        while cur < n:
            minibatch.append(databatch[cur:min(cur + batch_size, n), :, :, :])
            cur += batch_size


        cls_prob_list = []
        bbox_pred_list = []
        landmark_pred_list = []
        for idx, data in enumerate(minibatch):

            m = data.shape[0]
            real_size = self.batch_size


            if m < batch_size:  # the last batch
                keep_inds = np.arange(m)
                gap = self.batch_size - m
                while gap >= len(keep_inds):
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds, keep_inds))
                if gap != 0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                data = data[keep_inds]
                real_size = m


            cls_prob, bbox_pred,landmark_pred = self.sess.run([self.cls_prob, self.bbox_pred,self.landmark_pred], feed_dict={self.image_op: data})


            cls_prob_list.append(cls_prob[:real_size])
            bbox_pred_list.append(bbox_pred[:real_size])
            landmark_pred_list.append(landmark_pred[:real_size])


        return np.concatenate(cls_prob_list, axis=0),\
                np.concatenate(bbox_pred_list, axis=0),\
                np.concatenate(landmark_pred_list, axis=0)


class MtcnnDetector(object):

    def __init__(self,
                 detectors,
                 min_face_size=20,
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.79,
                 # scale_factor=0.709,#change
                 slide_window=False):

        self.pnet_detector = detectors[0]
        self.rnet_detector = detectors[1]
        self.onet_detector = detectors[2]
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor
        self.slide_window = slide_window

    def convert_to_square(self, bbox):
        """ 以图像中心为基准，以长边为边长，划出新的正方形框

        Parameters:
            bbox: {ndarray(n_boxes, 5)}
        Returns:
            square_bbox: {ndarray(n_boxes, 5)}
        """

        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        max_side = np.maximum(h, w)

        square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
        square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
        square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1

        return square_bbox

    def calibrate_box(self, bbox, reg):
        """ calibrate bboxes
        
        Args:
            bbox: {ndarray(n_boxes, 5)} input bboxes
            reg:  {ndarray(n_boxes, 4)} bboxes adjustment
        Returns:
            bbox_c: {} bboxes after refinement
        """

        bbox_c = bbox.copy()

        # 获得每个框的大小信息
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)

        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug

        return bbox_c


    def generate_bbox(self, cls_map, reg, scale, threshold):
        """ 根据`P-Net`的输出，生成框

        Args:
            cls_map:    {ndarray(batch_size, H, W, 2)} probability of boxes
            reg:        {ndarray(batch_size, H, W, 4)} offsets
            scale:      {float} current scale
            threshold:  {float} threshold of probability
        """

        stride = 2
        cellsize = 12

        t_index = np.where(cls_map > threshold)
        if t_index[0].size == 0: return np.array([])

        # offset
        dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]
        reg = np.array([dx1, dy1, dx2, dy2])
        score = cls_map[t_index[0], t_index[1]]

        # generate
        x1 = np.round((stride * t_index[1]) / scale)
        y1 = np.round((stride * t_index[0]) / scale)
        x2 = np.round((stride * t_index[1] + cellsize) / scale)
        y2 = np.round((stride * t_index[0] + cellsize) / scale) 
        boundingbox = np.vstack([x1, y1, x2, y2, score, reg])

        return boundingbox.T


    def processed_image(self, img, scale):
        """ rescale/resize the image according to the scale

        Args:
            img:    {ndarray(H, W, C)}
            scale:  {float}
        Returns:
            img_resized: {ndarray(H*scale, W*scale, C)}
        """

        height, width, channels = img.shape
        new_height = int(height * scale)
        new_width = int(width * scale)
        new_dim = (new_width, new_height)

        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
        img_resized = (img_resized - 127.5) / 128

        return img_resized


    def pad(self, bboxes, w, h):
        """ 按输入的框生成空白框位置，其位置在原图的左上角

        Args:
            bboxes: {ndarray(n_boxes, 5)} [x1, y1, x2, y2, score] 
            w:      {float} width of the input image
            h:      {float} height of the input image
        Returns :
            return_list: [list[ndarray]]
        Notes:
            dy, dx:     {ndarray(n_boxes, 1)} start point of the bbox in target image
            edy, edx:   {ndarray(n_boxes, 1)} end point of the bbox in target image
            y, x :      {ndarray(n_boxes, 1)} start point of the bbox in original image
            ex, ex :    {ndarray(n_boxes, 1)} end point of the bbox in original image
            tmph, tmpw: {ndarray(n_boxes, 1)} height and width of the bbox
        """

        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
        num_box = bboxes.shape[0]

        # 初始化`target`框到图像左上角
        dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
        edx, edy = tmpw.copy() - 1, tmph.copy() - 1
        # `origin`框
        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]


        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0


        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]


        return return_list

    def detect_pnet(self, im):
        """ Get face candidates through pnet

        Args:
            im: {ndarray(batch_size, H, W, C)}
        Returns:
            boxes:  {ndarray(batch_size, 5)} 
                        - [x1, y1, x2, y2, score] 
                        - detected boxes before calibration
            boxes_c:{ndarray(batch_size, 5)} 
                        - [x1, y1, x2, y2, score]
                        - boxes after calibration
        Notes:
            - cls_cls_map:  {ndarray(batch_size, H, W, 2)} 
            - reg:          {ndarray(batch_size, H, W, 4)} 
            - boxes         {ndarray(batch_size, 9)} 
                                [x1, y1, x2, y2, score, 
                                    x1_offset, y1_offset, x2_offset, y2_offset ]
        """

        h, w, c = im.shape
        net_size = 12

        current_scale = float(net_size) / self.min_face_size    # find initial scale
        im_resized = self.processed_image(im, current_scale)    # resize
        current_height, current_width, _ = im_resized.shape


        # 改变图像的尺度，在多尺度下进行搜索框，并使用NMS算法合并框
        all_boxes = list()
        while min(current_height, current_width) > net_size:

            # generate boxes using P-Net
            cls_cls_map, reg = self.pnet_detector.predict(im_resized)
            boxes = self.generate_bbox(cls_cls_map[:, :, 1], reg, current_scale, self.thresh[0])

            # resize image
            current_scale *= self.scale_factor
            im_resized = self.processed_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            # merging boxes
            if boxes.size == 0:
                continue
            keep = py_nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None, None
        all_boxes = np.vstack(all_boxes)


        # merge the detection from first stage
        keep = py_nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes = all_boxes[keep]
        boxes = all_boxes[:, :5]

        # refine the boxes
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        x1_rf = all_boxes[:, 0] + all_boxes[:, 5] * bbw
        y1_rf = all_boxes[:, 1] + all_boxes[:, 6] * bbh
        x2_rf = all_boxes[:, 2] + all_boxes[:, 7] * bbw
        y2_rf = all_boxes[:, 3] + all_boxes[:, 8] * bbh
        score = all_boxes[:, 4]

        boxes_c = np.vstack([x1_rf, y1_rf, x2_rf, y2_rf, score])
        boxes_c = boxes_c.T

        return boxes, boxes_c, None

    def detect_rnet(self, im, dets):
        """ Get face candidates using rnet

        Parameters:
            im:     {ndarray(batch_size, H, W, C)}
            dets:   {ndarray(batch_size, n_boxes, 5)}
                        output of last stage, [x1, y1, x2, y2, score]
        Returns:
            boxes:  {ndarray(batch_size, 5)} 
                        - [x1, y1, x2, y2, score] 
                        - detected boxes before calibration
            boxes_c:{ndarray(batch_size, 5)} 
                        - [x1, y1, x2, y2, score]
                        - boxes after calibration
        """

        h, w, c = im.shape


        dets = self.convert_to_square(dets)
        dets[:, 0: 4] = np.round(dets[:, 0: 4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        n_boxes = dets.shape[0]

        
        # 按P-Net输出结果，切割原图中的人脸
        cropped_ims = np.zeros((n_boxes, 24, 24, 3), dtype=np.float32)
        for i in range(n_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (24, 24)) - 127.5) / 128


        cls_scores, reg, _ = self.rnet_detector.predict(cropped_ims)

        # 筛选出概率大于阈值的结果
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > self.thresh[1])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]             # 筛选出P-Net的框
            boxes[:, 4] = cls_scores[keep_inds] # 更新框的评分
            reg = reg[keep_inds]                # 筛选出框的offsets
        else: return None, None, None

        keep = py_nms(boxes, 0.6)
        boxes = boxes[keep]

        boxes_c = self.calibrate_box(boxes, reg[keep])

        return boxes, boxes_c, None



    def detect_onet(self, im, dets):
        """Get face candidates using onet

        Args:
            im:     {ndarray(batch_size, H, W, C)}
            dets:   {ndarray(batch_size, n_boxes, 5)}
                        output of last stage, [x1, y1, x2, y2, score]
        Returns:
            boxes:  {ndarray(batch_size, 5)} 
                        - [x1, y1, x2, y2, score] 
                        - detected boxes before calibration
            boxes_c:{ndarray(batch_size, 5)} 
                        - [x1, y1, x2, y2, score]
                        - boxes after calibration
            landmark:{ndarray(batch_size, 10)}
        """

        h, w, c = im.shape


        dets = self.convert_to_square(dets)
        dets[:, 0: 4] = np.round(dets[:, 0: 4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        n_boxes = dets.shape[0]

        
        # 按R-Net输出结果，切割原图中的人脸
        cropped_ims = np.zeros((n_boxes, 48, 48, 3), dtype=np.float32)
        for i in range(n_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] =\
                                    im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48)) - 127.5) / 128


        cls_scores, reg, landmark = self.onet_detector.predict(cropped_ims)


        # 筛选出概率大于阈值的结果
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > self.thresh[2])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]             # 筛选出R-Net的框
            boxes[:, 4] = cls_scores[keep_inds] # 更新框的评分
            reg = reg[keep_inds]                # 筛选出框的offsets
            landmark = landmark[keep_inds]      # 筛选出框的landmark
        else: return None, None, None


        w = boxes[:, 2] - boxes[:, 0] + 1
        h = boxes[:, 3] - boxes[:, 1] + 1

        landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
        landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
        
        
        # NMS删除多余的框
        boxes_c = self.calibrate_box(boxes, reg)
        keep = py_nms(boxes, 0.6, "Minimum")
        boxes = boxes[keep]

        keep = py_nms(boxes_c, 0.6, "Minimum")
        boxes_c = boxes_c[keep]

        landmark = landmark[keep]


        return boxes, boxes_c, landmark


    def detect(self, img):
        """ 

        Args: 
            img: {ndarray(H, W, C)}
        Returns:

        Notes:
            - Detect face over image
        """

        boxes = None
        t = time.time()

        # pnet
        t1 = 0
        if self.pnet_detector:
            boxes, boxes_c, _ = self.detect_pnet(img)
            if boxes_c is None:
                return np.array([]), np.array([])

            t1 = time.time() - t
            t = time.time()

        # rnet
        t2 = 0
        if self.rnet_detector:
            boxes, boxes_c, _ = self.detect_rnet(img, boxes_c)
            if boxes_c is None:
                return np.array([]), np.array([])

            t2 = time.time() - t
            t = time.time()

        # onet
        t3 = 0
        if self.onet_detector:
            boxes, boxes_c, landmark = self.detect_onet(img, boxes_c)
            if boxes_c is None:
                return np.array([]), np.array([])

            t3 = time.time() - t
            t = time.time()
            # print(
            #    "time cost " + '{:.3f}'.format(t1 + t2 + t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2,
            #                                                                                                  t3))

        return boxes_c, landmark
        

    def detect_face(self, test_data):
        all_boxes = []  # save each image's bboxes
        landmarks = []
        batch_idx = 0

        sum_time = 0
        t1_sum = 0
        t2_sum = 0
        t3_sum = 0
        num_of_img = test_data.size
        empty_array = np.array([])
        # test_data is iter_
        s_time = time.time()
        for databatch in test_data:
            # databatch(image returned)
            batch_idx += 1
            if batch_idx % 100 == 0:
                c_time = (time.time() - s_time )/100
                print("%d out of %d images done" % (batch_idx ,test_data.size))
                print('%f seconds for each image' % c_time)
                s_time = time.time()


            im = databatch
            # pnet


            if self.pnet_detector:
                st = time.time()
                # ignore landmark
                boxes, boxes_c, landmark = self.detect_pnet(im)

                t1 = time.time() - st
                sum_time += t1
                t1_sum += t1
                if boxes_c is None:
                    print("boxes_c is None...")
                    all_boxes.append(empty_array)
                    # pay attention
                    landmarks.append(empty_array)

                    continue
                #print(all_boxes)

            # rnet

            if self.rnet_detector:
                t = time.time()
                # ignore landmark
                boxes, boxes_c, landmark = self.detect_rnet(im, boxes_c)
                t2 = time.time() - t
                sum_time += t2
                t2_sum += t2
                if boxes_c is None:
                    all_boxes.append(empty_array)
                    landmarks.append(empty_array)

                    continue
            # onet

            if self.onet_detector:
                t = time.time()
                boxes, boxes_c, landmark = self.detect_onet(im, boxes_c)
                t3 = time.time() - t
                sum_time += t3
                t3_sum += t3
                if boxes_c is None:
                    all_boxes.append(empty_array)
                    landmarks.append(empty_array)

                    continue

            all_boxes.append(boxes_c)
            landmark = [1]
            landmarks.append(landmark)
        print('num of images', num_of_img)
        print("time cost in average" +
            '{:.3f}'.format(sum_time/num_of_img) +
            '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1_sum/num_of_img, t2_sum/num_of_img,t3_sum/num_of_img))


        # num_of_data*9,num_of_data*10
        print('boxes length:',len(all_boxes))
        return all_boxes, landmarks

    def detect_single_image(self, im):
        all_boxes = []  # save each image's bboxes

        landmarks = []

       # sum_time = 0

        t1 = 0
        if self.pnet_detector:
          #  t = time.time()
            # ignore landmark
            boxes, boxes_c, landmark = self.detect_pnet(im)
           # t1 = time.time() - t
           # sum_time += t1
            if boxes_c is None:
                print("boxes_c is None...")
                all_boxes.append(np.array([]))
                # pay attention
                landmarks.append(np.array([]))


        # rnet

        if boxes_c is None:
            print('boxes_c is None after Pnet')
        t2 = 0
        if self.rnet_detector and not boxes_c is  None:
           # t = time.time()
            # ignore landmark
            boxes, boxes_c, landmark = self.detect_rnet(im, boxes_c)
           # t2 = time.time() - t
           # sum_time += t2
            if boxes_c is None:
                all_boxes.append(np.array([]))
                landmarks.append(np.array([]))


        # onet
        t3 = 0
        if boxes_c is None:
            print('boxes_c is None after Rnet')

        if self.onet_detector and not boxes_c is  None:
          #  t = time.time()
            boxes, boxes_c, landmark = self.detect_onet(im, boxes_c)
         #   t3 = time.time() - t
          #  sum_time += t3
            if boxes_c is None:
                all_boxes.append(np.array([]))
                landmarks.append(np.array([]))


        #print(
         #   "time cost " + '{:.3f}'.format(sum_time) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2, t3))

        all_boxes.append(boxes_c)
        landmarks.append(landmark)

        return all_boxes, landmarks