#include "crop_align.h"

/*
 * @returns {landmark}
 * @notes 
 * -    image size: 112 x 96
 */
landmark initAligned()
{
    landmark aligned = {0};

    aligned.x1 = 30.2946; aligned.y1 = 51.6963;
    aligned.x2 = 65.5318; aligned.y2 = 51.5014;
    aligned.x3 = 48.0252; aligned.y3 = 71.7366;
    aligned.x4 = 33.5493; aligned.y4 = 92.3655;
    aligned.x5 = 62.7299; aligned.y5 = 92.2041;

    return aligned;
}

/*
 * @returns {landmark}
 * @notes 
 * -    image size: 112 x 96
 */
landmark initAlignedOffset()
{
    int h = 112, w = 96;
    landmark offset = {0};

    offset.x1 = 30.2946 / w; offset.y1 = 51.6963 / h;
    offset.x2 = 65.5318 / w; offset.y2 = 51.5014 / h;
    offset.x3 = 48.0252 / w; offset.y3 = 71.7366 / h;
    offset.x4 = 33.5493 / w; offset.y4 = 92.3655 / h;
    offset.x5 = 62.7299 / w; offset.y5 = 92.2041 / h;

    return offset;
}

/*
 * @params
 * -    offset: 相对于边长h, w的偏置
 * -    h, w:   边长
 * @returns {landmark}
 */
landmark dstLandmark(landmark offset, float h, float w)
{
    landmark dst = {0};
    
    dst.x1 = offset.x1 * w; dst.y1 = offset.y1 * h;
    dst.x2 = offset.x2 * w; dst.y2 = offset.y2 * h;
    dst.x3 = offset.x3 * w; dst.y3 = offset.y3 * h;
    dst.x4 = offset.x4 * w; dst.y4 = offset.y4 * h;
    dst.x5 = offset.x5 * w; dst.y5 = offset.y5 * h;

    return dst;
}

/*
 * @params
 * -    mark: 原图中的坐标
 * -    x, y: 该关键点所在回归框的左上角坐标
 * @return
 *      mark: 在回归框中的坐标
 */
landmark substract_offset(landmark mark, float x, float y)
{
    mark.x1 -= x; mark.y1 -= y;
    mark.x2 -= x; mark.y2 -= y;
    mark.x3 -= x; mark.y3 -= y;
    mark.x4 -= x; mark.y4 -= y;
    mark.x5 -= x; mark.y5 -= y;
    return mark;
}

/*
 * @params
 * -    mark: 原图中的坐标
 * -    scale: 缩放比例
 * @return
 *      mark: 在回归框(h, w)中的坐标
 */
landmark multiply_scale(landmark mark, float scale)
{
    mark.x1 *= scale; mark.y1 *= scale;
    mark.x2 *= scale; mark.y2 *= scale;
    mark.x3 *= scale; mark.y3 *= scale;
    mark.x4 *= scale; mark.y4 *= scale;
    mark.x5 *= scale; mark.y5 *= scale;
    return mark;
}

CvMat* landmark_to_cvMat(landmark mark)
{
    CvMat* mat = cvCreateMat(5, 2, CV_32FC1);
    mat->data.fl[0] = mark.x1; mat->data.fl[1] = mark.y1;
    mat->data.fl[2] = mark.x2; mat->data.fl[3] = mark.y2;
    mat->data.fl[4] = mark.x3; mat->data.fl[5] = mark.y3;
    mat->data.fl[6] = mark.x4; mat->data.fl[7] = mark.y4;
    mat->data.fl[8] = mark.x5; mat->data.fl[9] = mark.y5;
    return mat;
}

/*
 * @params  
 * -    im:     原图
 * -    box:    回归框
 * -    h, w:   期望得到的图片尺寸
 * -    scale:  原图(hh, ww)缩放到(h, w)的比例
 * @returns
 *      resized: 切割后的图像
 * @notes
 * -    若h == 0, w == 0则不进行resize
 */
image image_crop(image im, bbox box, int h, int w, float* scale)
{
    float cx = (box.x2 + box.x1) / 2;   // centroid
    float cy = (box.y2 + box.y1) / 2;

    // padding
    float w_ = box.x2 - box.x1 + 1;
    float h_ = box.y2 - box.y1 + 1;

    h = (h == 0)? h_: h; w = (w == 0)? w_: w;

    float ratio_src = h_ / w_;
    float ratio_dst = (float)h / (float)w;
    int ww_ = 0, hh_ = 0;
    if (ratio_src < ratio_dst){
        // 原图h为较短边，以h为基准截取 h/w 比例的人脸
        hh_ = (int)h_; ww_ = (int)(hh_ / ratio_dst);
    } else {
        // 原图w为较短边，以w为基准截取 h/w 比例的人脸
        ww_ = (int)w_; hh_ = (int)(ww_ * ratio_dst);
    }

    int x1 = (int)box.x1;
    int x2 = (int)box.x2;
    int y1 = (int)box.y1; 
    int y2 = (int)box.y2;
    int xx1 = 0, yy1 = 0;
    int xx2 = ww_ - 1;
    int yy2 = hh_ - 1;
    if (x1 < 0){xx1 = - x1; x1 = 0;}
    if (y1 < 0){yy1 = - y1; y1 = 0;}
    if (x2 > im.w - 1){xx2 = (x2-x1+1) + im.w - x2 - 2; x2 = im.w - 1;}
    if (y2 > im.h - 1){yy2 = (y2-y1+1) + im.h - y2 - 2; y2 = im.h - 1;}
    
    // crop
    image croped = make_image(ww_, hh_, im.c);
    for (int k = 0; k < im.c; k++ ){
        for (int j = yy1; j < yy2 + 1; j++ ){
            for (int i = xx1; i < xx2 + 1; i++ ){
                int x = x1 + i; int y = y1 + j;
                float val = im.data[x + y*im.w + k*im.w*im.h];
                croped.data[i + j*ww_ + k*ww_*hh_] = val;
            }
        }
    }

    *scale = (float)w / (float)ww_;
    image resized = resize_image(croped, w, h);
    free_image(croped);
    return resized;
}

/*
 * @params  
 * -    im:     原图
 * -    box:    回归框
 * -    h, w:   期望得到的图片尺寸
 * @returns
 *      resized: 切割并矫正后的图像
 * @notes 
 */
image image_crop_aligned(image im, bbox box, landmark srcMk, landmark offset, int h, int w, int mode)
{
    float scale = -1.;
    // 以回归框将人脸切出
    image croped = image_crop(im, box, h, w, &scale);
    
    // 计算在剪切出的人脸图像中，关键点的坐标
    float x1 = (box.x1 + box.x2 - croped.w / scale) / 2.;
    float y1 = (box.y1 + box.y2 - croped.h / scale) / 2.;

    landmark dstMk = dstLandmark(offset, croped.h, croped.w);
    srcMk = substract_offset(srcMk, x1, y1);
    srcMk = multiply_scale(srcMk, scale);
    
    // 计算变换矩阵
    CvMat* srcPtMat = landmark_to_cvMat(srcMk);
    CvMat* dstPtMat = landmark_to_cvMat(dstMk);
    CvMat* M = cp2form(srcPtMat, dstPtMat, mode);
    cvReleaseMat(&srcPtMat); cvReleaseMat(&dstPtMat); 

    // 用矩阵变换图像
    IplImage* srcIpl = image_to_ipl(croped);
    IplImage* dstIpl = cvCloneImage(srcIpl);
    cvWarpAffine(srcIpl, dstIpl, M, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
    free_image(croped); cvReleaseMat(&M);
    // 返回
    image warped = ipl_to_image(dstIpl);
    cvReleaseImage(&srcIpl); cvReleaseImage(&dstIpl);

    return warped;
}

image image_aligned_v2(image im, landmark src, landmark dst, int h, int w, int mode)
{
    // 计算变换矩阵
    CvMat* srcPtMat = landmark_to_cvMat(src);
    CvMat* dstPtMat = landmark_to_cvMat(dst);
    CvMat* M = cp2form(srcPtMat, dstPtMat, mode);
    cvReleaseMat(&srcPtMat); cvReleaseMat(&dstPtMat); 
    // 变换图像
    IplImage* srcIpl = image_to_ipl(im);
    IplImage* dstIpl = cvCloneImage(srcIpl);
    cvWarpAffine(srcIpl, dstIpl, M, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
    // 截取图像
    cvSetImageROI(dstIpl, cvRect(0, 0, w, h));
    IplImage* warpedIpl = cvCreateImage(cvSize(w, h), dstIpl->depth, dstIpl->nChannels);
    cvCopy(dstIpl, warpedIpl, NULL); cvResetImageROI(dstIpl);
    cvReleaseImage(&srcIpl); cvReleaseImage(&dstIpl); cvReleaseMat(&M);

    image warped = ipl_to_image(warpedIpl);
    return warped;
}