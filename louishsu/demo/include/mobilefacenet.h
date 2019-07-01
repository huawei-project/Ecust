/*
 * @Author: louishsu
 * @Date: 2019-05-31 11:31:08 
 * @Last Modified by:   louishsu
 * @Last Modified time: 2019-05-31 11:31:08 
 */
#ifndef __MOBILEFACENET_H
#define __MOBILEFACENET_H

#include <opencv/cv.h>
#include "darknet.h"
#include "mtcnn.h"
#include "util.h"

#define H 112
#define W 96
#define N 128

network* load_mobilefacenet();
image convert_mobilefacenet_image(image im);
int verify(network* net, image im1, image im2, float* cosine);

int verify_input_images(int argc, char** argv);
int verify_video_demo(int argc, char** argv);
int verify_lfw_images(int argc, char** argv);

#endif