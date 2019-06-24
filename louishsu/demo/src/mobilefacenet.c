#include "mobilefacenet.h"
#include "mtcnn.h"
#include "parser.h"
#include "activations.h"

network* load_mobilefacenet()
{
    network* net = load_network("cfg/mobilefacenet.cfg", 
                    "weights/mobilefacenet.weights", 0);
    return net;
}

/*
 * Args:
 *      im: {image} RGB image, range[0, 1]
 * Returns:
 *      cvt:{image} BGR image, range[-1, 1]
 * */
image convert_mobilefacenet_image(image im)
{
    int size = im.h*im.w*im.c;

    image cvt = copy_image(im);         // RGB, 0~1
    for (int i = 0; i < size; i++ ){
        float val = im.data[i]*255.;
        val = (val - 127.5) / 128.;
        cvt.data[i] = val; 
    }

    rgbgr_image(cvt);                   // BGR, -1~1
    return cvt;
}

/*
 * Args:
 *      net:    {network*}  MobileFaceNet
 *      im1/2:  {image}     image of size `3 x H x W`
 *      cosine: {float*}    threshold of verification, will be replaced with cosion distance.
 * Returns:
 *      isOne:  {int}       if the same, return 1; else 0.
 * */
int verify(network* net, image im1, image im2, float* cosine)
{
    assert(im1.w == W && im1.h == H);
    assert(im2.w == W && im2.h == H);

    float* X;

    float* feat1 = calloc(N*2, sizeof(float));
    image cvt1 = convert_mobilefacenet_image(im1);
    X = network_predict(net, cvt1.data);
    memcpy(feat1, X, N*sizeof(float));
    flip_image(cvt1);
    X = network_predict(net, cvt1.data);
    memcpy(feat1 + N, X, N*sizeof(float));
    
    float* feat2 = calloc(N*2, sizeof(float));
    image cvt2 = convert_mobilefacenet_image(im2);
    X = network_predict(net, cvt2.data);
    memcpy(feat2, X, N*sizeof(float));
    flip_image(cvt2);
    X = network_predict(net, cvt2.data);
    memcpy(feat2 + N, X, N*sizeof(float));

    float dist = distCosine(feat1, feat2, N*2);
    int is_one = dist < *cosine? 0: 1;

    *cosine = dist;

    free(feat1); free(feat2);
    free_image(cvt1); free_image(cvt2);

    return is_one;
}

