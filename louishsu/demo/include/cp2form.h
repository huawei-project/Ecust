#ifndef __CP2FORM_H
#define __CP2FORM_H

#include <stdio.h>
#include <opencv/cv.h>

CvMat* cp2form(const CvMat* src, const CvMat* dst, int mode);

#endif