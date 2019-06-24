/*
 * @Author: louishsu
 * @Date: 2019-05-31 11:30:23 
 * @Last Modified by:   louishsu 
 * @Last Modified time: 2019-05-31 11:30:23 
 */
#ifndef __CROP_ALIGN_H
#define __CROP_ALIGN_H

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "darknet.h"
#include "mtcnn.h"
#include "cp2form.h"

landmark initAligned();
landmark initAlignedOffset();
image image_crop(image im, bbox box, int h, int w, float* scale);
image image_crop_aligned(image im, bbox box, landmark srcMk, landmark offset, int h, int w, int mode);
image image_aligned_v2(image im, landmark src, landmark dst, int h, int w, int mode);

#endif