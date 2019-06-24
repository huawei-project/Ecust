/*
 * @Author: louishsu
 * @Date: 2019-05-31 11:31:24 
 * @Last Modified by:   louishsu
 * @Last Modified time: 2019-05-31 11:31:24 
 */
#ifndef __UTIL_H
#define __UTIL_H

#include <opencv/cv.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include "darknet.h"


IplImage *image_to_ipl(image im);
image     ipl_to_image(IplImage* src);

void  show_ipl(IplImage* ipl, char* winname, int pause);
void  show_im(image im, char* winname, int pause);

image rgb_to_bgr(image im);
image resize_image_scale(image im, float scale);

static inline float _min(float a, float b){return a<b? a: b;}
static inline float _max(float a, float b){return a>b? a: b;}
static inline int _ascending(const void * a, const void * b){return ( *(int*)a - *(int*)b );}
static inline int _descending(const void * a, const void * b){return ( *(int*)b - *(int*)a );}

void find_max_min(float* x, int n, float* max, float* min);
void norm(float* vector, int n);
float distCosine(float* vec1, float* vec2, int n);
float distEuclid(float* vec1, float* vec2, int n);

#endif