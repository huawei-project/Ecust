/*
 * @Author: louishsu 
 * @Date: 2019-05-31 13:53:34 
 * @Last Modified by: louishsu
 * @Last Modified time: 2019-05-31 17:12:34
 */
#include "cp2form.h"

void printCvMat(CvMat* M)
{
    for (int r = 0; r < M->rows; r++){
        for (int c = 0; c < M->cols; c++){
            printf("%.2f ", M->data.fl[r*M->cols + c]);
        }    
        printf("\n");
    }
        printf("\n\n");
}

/* 
 * @param
 *      xy: [x, y]， Nx2
 * @return
 *      ret: 2Nx4
 * @notes
 *      x  y 1 0
 *      y -x 0 1
 */
CvMat* _stitch(const CvMat* xy)
{
    int rows = xy->rows;
    CvMat* C = cvCreateMat(rows, 1, CV_32F);

    // x, y, 1, 0
    cvGetCol(xy, C, 0); CvMat* x = cvCloneMat(C);
    cvGetCol(xy, C, 1); CvMat* y = cvCloneMat(C);
    CvMat* ones  = cvCreateMat(rows, 1, CV_32F);
    CvMat* zeros = cvCreateMat(rows, 1, CV_32F);
    for (int i = 0; i < rows; i++){
        ones ->data.fl[i] = 1.; zeros->data.fl[i] = 0.;
    }
      
    CvMat* X = cvCreateMat(2*xy->rows, 4, CV_32F);
    cvGetSubRect(X, C, cvRect(0,    0, 1, rows)); cvCopy(    x, C, NULL);
    cvGetSubRect(X, C, cvRect(1,    0, 1, rows)); cvCopy(    y, C, NULL);
    cvGetSubRect(X, C, cvRect(2,    0, 1, rows)); cvCopy( ones, C, NULL);
    cvGetSubRect(X, C, cvRect(3,    0, 1, rows)); cvCopy(zeros, C, NULL);
    for (int i = 0; i < rows; i++) x->data.fl[i] *= -1.;
    cvGetSubRect(X, C, cvRect(0, rows, 1, rows)); cvCopy(    y, C, NULL);
    cvGetSubRect(X, C, cvRect(1, rows, 1, rows)); cvCopy(    x, C, NULL);
    cvGetSubRect(X, C, cvRect(2, rows, 1, rows)); cvCopy(zeros, C, NULL);
    cvGetSubRect(X, C, cvRect(3, rows, 1, rows)); cvCopy( ones, C, NULL);

    cvReleaseMat(&C);
    cvReleaseMat(&x); cvReleaseMat(&y);
    cvReleaseMat(&ones); cvReleaseMat(&zeros);

    return X;   
}


/* 
 * @param
 *      M:  2x3
 *      uv: [u, v]， Nx2
 * @return
 *      xy: Nx2
 * @notes
 *      xy = [uv, 1] * M^T, Nx2
 */
CvMat* _tformfwd(const CvMat* M, const CvMat* uv)
{
    int rows = uv->rows;
    int cols = uv->cols;
    CvMat* mat = cvCreateMat(rows, cols, CV_32F);

    CvMat* UV = cvCreateMat(rows, cols + 1, CV_32F);

    cvGetSubRect(UV, mat, cvRect(0, 0, cols, rows));

    cvCopy(uv, mat, NULL);
    for (int r = 0; r < rows; r++){
        UV->data.fl[r*(cols+1) + cols] = 1.;
    }

    CvMat* MT = cvCreateMat(M->cols, M->rows, CV_32F); 
    CvMat* xy = cvCreateMat(rows, cols, CV_32F);
    cvTranspose(M, MT);
    cvMatMul(UV, MT, xy);

    cvReleaseMat(&UV); cvReleaseMat(&mat); cvReleaseMat(&MT);
    return xy;
}

float _matrixNorm2(CvMat* Mat)
{
    CvMat* U = cvCreateMat(Mat->rows, Mat->rows, CV_32F);
    CvMat* W = cvCreateMat(Mat->rows, Mat->cols, CV_32F);
    CvMat* V = cvCreateMat(Mat->cols, Mat->cols, CV_32F);

    cvSVD(Mat, W, U, V, CV_SVD_V_T);

    float s = FLT_MIN;
    for (int i = 0; i < W->rows*W->cols; i++){
        float val = W->data.fl[i];
        if (val > s){
            s = val;
        }
    }

    cvReleaseMat(&U); cvReleaseMat(&W); cvReleaseMat(&V);
    return s;
}

/* 
 * @param
 *      uv: [u, v]， Nx2
 *      xy: [x, y]， Nx2
 * @return
 * @notes
 * -    Xr = Y   ===>  r = (X^T X + \lambda I)^{-1} X^T Y
 */
CvMat* _findNonreflectiveSimilarity(const CvMat* uv, const CvMat* xy)
{
    CvMat* X = _stitch(xy);                         // 2N x  4

    CvMat* XT = cvCreateMat(X->cols, X->rows, CV_32F);  
    cvTranspose(X, XT);                             //  4 x 2N

    CvMat* XTX = cvCreateMat(XT->rows, X->cols, CV_32F);
    cvMatMul(XT, X, XTX);                           //  4 x  4
    for (int i = 0; i < XTX->rows; i++) XTX->data.fl[i*XTX->rows + i] += 1e-15;

    CvMat* XTXi = cvCreateMat(XTX->rows, XTX->cols, CV_32F); 
    cvInvert(XTX, XTXi, CV_LU);                     //  4 x  4

    // -----------------------------------------------------------------------
    
    CvMat* uvT = cvCreateMat(uv->cols, uv->rows, CV_32F); 
    cvTranspose(uv, uvT);                           //  2 x  N
    CvMat header; 
    CvMat* YT = cvReshape(uvT, &header, 0, 1);      //  1 x 2N    TODO
    CvMat* Y = cvCreateMat(YT->cols, YT->rows, CV_32F);  
    cvTranspose(YT, Y);                             // 2N x  1
    
    CvMat* XTXiXT = cvCreateMat(XTXi->rows, XT->cols, CV_32F);
    CvMat* r = cvCreateMat(XTXiXT->rows, Y->cols, CV_32F);
    cvMatMul(XTXi, XT, XTXiXT); cvMatMul(XTXiXT, Y, r);       //  4 x  1

    // -----------------------------------------------------------------------

    cvReleaseMat(&X); cvReleaseMat(&XT); 
    cvReleaseMat(&XTX); cvReleaseMat(&XTXi); cvReleaseMat(&XTXiXT);
    cvReleaseMat(&uvT); cvReleaseMat(&Y);

    // =======================================================================

    CvMat* R = cvCreateMat(3, 3, CV_32F);
    R->data.fl[0 * 3 + 0] = r->data.fl[0]; R->data.fl[0 * 3 + 1] = -r->data.fl[1]; R->data.fl[0 * 3 + 2] = 0.;
    R->data.fl[1 * 3 + 0] = r->data.fl[1]; R->data.fl[1 * 3 + 1] =  r->data.fl[0]; R->data.fl[1 * 3 + 2] = 0.;
    R->data.fl[2 * 3 + 0] = r->data.fl[2]; R->data.fl[2 * 3 + 1] =  r->data.fl[3]; R->data.fl[2 * 3 + 2] = 1.;
    
    CvMat* Ri = cvCreateMat(R->cols, R->rows, CV_32F);
    cvInvert(R, Ri, CV_LU);

    CvMat* MT = cvCreateMat(3, 2, CV_32F);
    cvGetSubRect(Ri, MT, cvRect(0, 0, 2, 3));
    
    CvMat* M = cvCreateMat(MT->cols, MT->rows, CV_32F);
    cvTranspose(MT, M);

    // -----------------------------------------------------------------------

    cvReleaseMat(&r); cvReleaseMat(&R); cvReleaseMat(&Ri); cvReleaseMat(&MT);

    return M;
}

/* 
 * @param
 *      uv: [u, v]， Nx2
 *      xy: [x, y]， Nx2
 * @return
 * @notes
 */
CvMat* _findReflectiveSimilarity(const CvMat* uv, const CvMat* xy)
{
    CvMat* xyR = cvCloneMat(xy);
    for (int r = 0; r < xyR->rows; r++) xyR->data.fl[r*xyR->cols] *= -1;

    CvMat* M1 = _findNonreflectiveSimilarity(uv, xy);
    CvMat* M2 = _findNonreflectiveSimilarity(uv, xyR);

    cvReleaseMat(&xyR);

    for (int r = 0; r < M2->rows; r++) M2->data.fl[r*M2->cols] *= -1;

    CvMat* xy1 = _tformfwd(M1, uv);
    CvMat* xy2 = _tformfwd(M2, uv);
    cvSub(xy1, xy, xy1, NULL); cvSub(xy2, xy, xy2, NULL);

    float norm1 = _matrixNorm2(xy1);
    float norm2 = _matrixNorm2(xy2);

    cvReleaseMat(&xy1); cvReleaseMat(&xy2);

    if (norm1 < norm2){
        cvReleaseMat(&M2);
        return M1;
    } else {
        cvReleaseMat(&M1);
        return M2;
    }
}

/* 
 * @param
 *      src: 原始坐标点 [[x1, y1], [x2, y2], ..., [xn, yn]]
 *      dst: 对齐对标点 [[x1, y1], [x2, y2], ..., [xn, yn]]
 *      mode:模式 
 * @return
 * @notes
 */
CvMat* cp2form(const CvMat* src, const CvMat* dst, int mode)
{
    CvMat* M;

    if (mode == 0){
        M = _findNonreflectiveSimilarity(src, dst);
    } else if (mode == 1){
        M = _findReflectiveSimilarity(src, dst);
    } else {
        printf("Mode %d not supported!\n", mode);
    }
    
    return M;
}

// 1.02 -0.09 0.88 
// 0.09 1.02 7.36 


// 0.92 -0.01 -2.13 
// 0.01 0.92 17.92