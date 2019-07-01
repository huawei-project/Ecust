#include "mtcnn.h"
#include "mobilefacenet.h"
#include "crop_align.h"

#include <m3api/xiApi.h> // Linux, OSX

static int g_videoDone = 0;
static char* winname = "frame";
static CvFont font;

static HANDLE g_xiCap = NULL;
static XI_RETURN g_xiStat = XI_OK;
static int g_iExposure = 20000;

static image g_imFrame[3];
static int g_index = 0;

static double g_time;
static double g_fps;
static int g_running = 0;
static detect* g_dets = NULL;
static int g_ndets = 0;

static params p;
static network* pnet;
static network* rnet;
static network* onet;

#define THRESH 0.3
#define N 128

static network* mobilefacenet;
static landmark g_aligned = {0};
static int g_mode = 0;
static int g_initialized = 0;
static float* g_feat_saved = NULL;
static float* g_feat_toverify = NULL;
static float g_cosine = 0;
static float g_thresh = 0.5;
static int g_isOne = -1;

#define HandleResult(res,place) if (res!=XI_OK) {printf("Error after %s (%d)\n",place,res);goto finish;}

image xiImage_to_image(XI_IMG src)
{
    int h = src.height;
    int w = src.width;
    int c = 25;

    if (w == 0 || h == 0){
        printf("get image 0 x 0\n");
    }

    h /= 5; w /= 5; 

    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src.bp;

    int idx_i = 0, idx_j = 0;
    for (int i = 0; i < h; i++){
        idx_i = i*5;
        
        for (int j = 0; j < w; j++){
            idx_j = j*5;
            
            for (int k = 0; k < 25; k++){
                int idx = (idx_i + k / 5) * src.width + (idx_j + k % 5);
                im.data[k*w*h + i*w + j] = (float)(data[idx]) / 255.;
            }
        }
    }
    return im;
}

void _equalizeHist(image* im)
{
    IplImage* ipl = image_to_ipl(*im);
    cvEqualizeHist(ipl, ipl);
    image equalized = ipl_to_image(ipl);
    memcpy(im->data, equalized.data, im->h*im->w*sizeof(float));
    free_image(equalized);
}

image _frame(int hist)
{
    XI_IMG xiFrame;
	memset(&xiFrame, 0, sizeof(xiFrame));
	xiFrame.size = sizeof(XI_IMG);
    g_xiStat = xiGetImage(g_xiCap, 5000, &xiFrame);
    image frame = xiImage_to_image(xiFrame);

    if (hist){  // 直方图均衡化
        image temp = make_image(frame.w, frame.h, 1);
        int size = frame.w*frame.h;
        for (int i = 0; i < frame.c; i++){
            memcpy(temp.data, frame.data + i*size, size*sizeof(float));
            _equalizeHist(&temp);
            memcpy(frame.data + i*size, temp.data, size*sizeof(float));
        }
        free_image(temp);
    }

    return frame;
}

void* read_frame_in_thread(void* ptr)
{
    free_image(g_imFrame[g_index]);
    g_imFrame[g_index] = _frame(1);
    if (g_imFrame[g_index].data == 0){
        g_videoDone = 1;
        return 0;
    }
    return 0;
}

void* detect_frame_in_thread(void* ptr)
{
    g_running = 1;

    image frame = g_imFrame[(g_index + 2) % 3];
    g_dets = realloc(g_dets, 0); g_ndets = 0;

    image temp = make_image(frame.w, frame.h, 3);
    int i = 0; int size = frame.w*frame.h;
    // while (g_ndets == 0){   // 当有一个波段检测到人脸即可结束
    //     for (int k = 0; k < temp.c; k++){
    //         memcpy(temp.data + k*size, frame.data + i*size, size*sizeof(float));
    //     }
    //     detect_image(pnet, rnet, onet, temp, &g_ndets, &g_dets, p);
    //     if(++i == frame.c){
    //         break;
    //     }
    // }

    for (int k = 0; k < temp.c; k++){
        memcpy(temp.data + k*size, frame.data + i*size, size*sizeof(float));
    }
    detect_image(pnet, rnet, onet, temp, &g_ndets, &g_dets, p);
    
    free_image(temp);
    g_running = 0;
}

void _generate_feature_v1(image im, landmark mark, float* X)
{
    float* x = NULL;
    image warped = image_aligned_v2(im, mark, g_aligned, H, W, g_mode);
    image cvt = convert_mobilefacenet_image(warped);
    
    x = network_predict(mobilefacenet, cvt.data);
    memcpy(X,     x, N*sizeof(float));

    flip_image(cvt);
    x = network_predict(mobilefacenet, cvt.data);
    memcpy(X + N, x, N*sizeof(float));

    free_image(warped); free_image(cvt);
}

/*
 * 每个波段产生一个N*2的向量，共25个通道 
 */
void _generate_feature_v2(image im, landmark mark, float* X)
{
    float* x = NULL;
    int size = im.w*im.h;
    image warped = image_aligned_v2(im, mark, g_aligned, H, W, g_mode);
    image cvt = convert_mobilefacenet_image(warped);
    image temp = make_image(cvt.w, cvt.h, 3);

    for (int i = 0; i < cvt.c; i++){
        for (int c = 0; c < im.c; c++){
            memcpy(temp.data + c*size, cvt.data + i*size, size*sizeof(float));
        }
        x = network_predict(mobilefacenet, temp.data); memcpy(X + i*N, x, N*sizeof(float));
        flip_image(temp);
        x = network_predict(mobilefacenet, temp.data); memcpy(X + (i+1)*N, x, N*sizeof(float));
    }

    free_image(warped); free_image(cvt); free_image(temp);
}

void* display_frame_in_thread(void* ptr)
{
    while(g_running);

    image src = g_imFrame[(g_index + 1) % 3]; int size = src.w*src.h;
    image im = make_image(src.w, src.h, 3);
    for (int i = 0; i < im.c; i++){
        memcpy(im.data + i*size, src.data, size*sizeof(float));
    }

    IplImage* iplFrame = image_to_ipl(im);
    for (int i = 0; i < g_ndets; i++ ){
        detect det = g_dets[i];
        float score = det.score;
        bbox bx = det.bx;
        landmark mk = det.mk;

        char buff[256];
        sprintf(buff, "%.2f", score);
        cvPutText(iplFrame, buff, cvPoint((int)bx.x1, (int)bx.y1),
                    &font, cvScalar(0, 0, 255, 0));

        cvRectangle(iplFrame, cvPoint((int)bx.x1, (int)bx.y1),
                    cvPoint((int)bx.x2, (int)bx.y2),
                    cvScalar(255, 255, 255, 0), 1, 8, 0);

        cvCircle(iplFrame, cvPoint((int)mk.x1, (int)mk.y1),
                    1, cvScalar(255, 255, 255, 0), 1, 8, 0);
        cvCircle(iplFrame, cvPoint((int)mk.x2, (int)mk.y2),
                    1, cvScalar(255, 255, 255, 0), 1, 8, 0);
        cvCircle(iplFrame, cvPoint((int)mk.x3, (int)mk.y3),
                    1, cvScalar(255, 255, 255, 0), 1, 8, 0);
        cvCircle(iplFrame, cvPoint((int)mk.x4, (int)mk.y4),
                    1, cvScalar(255, 255, 255, 0), 1, 8, 0);
        cvCircle(iplFrame, cvPoint((int)mk.x5, (int)mk.y5),
                    1, cvScalar(255, 255, 255, 0), 1, 8, 0);
    }
    image draw = ipl_to_image(iplFrame);
    cvReleaseImage(&iplFrame);

    int c = show_image(draw, winname, 1);

    if (c != -1) c = c%256;
    if (c == 27) {          // Esc
        g_videoDone = 1;
        return 0;
    } else if (c == 's') {  // save feature
        image _im = g_imFrame[(g_index + 1) % 3];
        int idx = keep_one(g_dets, g_ndets, _im);
        if (idx < 0) return 0;
        landmark mark = g_dets[idx].mk;
        _generate_feature_v2(_im, mark, g_feat_saved);
        g_initialized = 1;
    } else if (c == 'v') {  // verify
        image _im = g_imFrame[(g_index + 1) % 3];
        int idx = keep_one(g_dets, g_ndets, _im);
        if (idx < 0) return 0;
        landmark mark = g_dets[idx].mk;
        _generate_feature_v2(_im, mark, g_feat_toverify);

        g_cosine = distCosine(g_feat_saved, g_feat_toverify, N*2*25);
        g_isOne = g_cosine < g_thresh? 0: 1;
    } else if (c == '[') { 
        g_thresh -= 0.05;
    } else if (c == ']') {
        g_thresh += 0.05;
    } else if (c == ';') { 
        g_iExposure -= 100;
        xiSetParamInt(g_xiCap, XI_PRM_EXPOSURE, g_iExposure);
    } else if (c == '\'') {
        g_iExposure += 100;
        xiSetParamInt(g_xiCap, XI_PRM_EXPOSURE, g_iExposure);
    }

    printf("\033[2J"); printf("\033[1;1H\n");
    printf("FPS:%.1f\n", g_fps);
    printf("Exposure: %d\n\n", g_iExposure);
    printf("Objects:%d\n", g_ndets);
    printf("Initialized:%d\n", g_initialized);
    printf("Threshold:%.4f\n", g_thresh);
    printf("Cosine:%.4f\n", g_cosine);
    printf("Verify:%d\n", g_isOne);

    free_image(im); free_image(draw);
    return 0;
}

int verify_video_demo(int argc, char **argv)
{
    pnet = load_mtcnn_net("PNet");
    rnet = load_mtcnn_net("RNet");
    onet = load_mtcnn_net("ONet");
    mobilefacenet = load_mobilefacenet();
    printf("\n\n");

    // ======================================================================
    printf("Initializing Capture...");
    g_xiStat = xiOpenDevice(0, &g_xiCap); 
    HandleResult(g_xiStat, "xiOpenDevice");
    g_xiStat = xiSetParamInt(g_xiCap, XI_PRM_IMAGE_DATA_FORMAT, XI_MONO8);
    HandleResult(g_xiStat, "xiSetParamInt (image format)");
    g_xiStat = xiSetParamInt(g_xiCap, XI_PRM_EXPOSURE, g_iExposure); 
    HandleResult(g_xiStat, "xiSetParamInt (exposure set)");
    g_xiStat = xiStartAcquisition(g_xiCap); 
    HandleResult(g_xiStat, "xiStartAcquisition");
    
    g_imFrame[0] = _frame(1);
    g_imFrame[1] = copy_image(g_imFrame[0]);
    g_imFrame[2] = copy_image(g_imFrame[0]);

    cvNamedWindow(winname, CV_WINDOW_AUTOSIZE);
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 1, 2, 8);
    printf("OK!\n");

    // ======================================================================
    printf("Initializing detection...");
    p = initParams(argc, argv);
    g_dets = calloc(0, sizeof(detect)); g_ndets = 0;
    printf("OK!\n");

    // ======================================================================
    printf("Initializing verification...");
    // g_aligned = initAlignedOffset();
    g_aligned = initAligned();
    g_mode = find_int_arg(argc, argv, "--mode", 1);
    g_feat_saved    = calloc(N*2*25, sizeof(float));
    g_feat_toverify = calloc(N*2*25, sizeof(float));
    printf("OK!\n");

    // ======================================================================
    pthread_t thread_read;
    pthread_t thread_detect;
    pthread_t thread_display;

    // image im = _frame(1);
    // image tmp = make_image(im.w, im.h, 3);
    // for (int i = 0; i < im.c; i++){
    //     memcpy(tmp.data + 0*tmp.w*tmp.h, im.data + i*im.w*im.h, im.w*im.h*sizeof(float));
    //     memcpy(tmp.data + 1*tmp.w*tmp.h, im.data + i*im.w*im.h, im.w*im.h*sizeof(float));
    //     memcpy(tmp.data + 2*tmp.w*tmp.h, im.data + i*im.w*im.h, im.w*im.h*sizeof(float));
    //     show_image(tmp, "tmp", 0);
    // }
    // free_image(tmp);
    // exit(0);

    g_time = what_time_is_it_now();
    while(!g_videoDone){
        g_index = (g_index + 1) % 3;

        if(pthread_create(&thread_read, 0, read_frame_in_thread, 0)) error("Thread read create failed");
        if(pthread_create(&thread_detect, 0, detect_frame_in_thread, 0)) error("Thread detect create failed");
        if(pthread_create(&thread_display, 0, display_frame_in_thread, 0)) error("Thread detect create failed");
        
        g_fps = 1./(what_time_is_it_now() - g_time);
        g_time = what_time_is_it_now();
        // display_frame_in_thread(0);
        
        pthread_join(thread_read, 0);
        pthread_join(thread_detect, 0);
        pthread_join(thread_display, 0);
    }
    for (int i = 0; i < 3; i++ ){
        free_image(g_imFrame[i]);
    }
    free(g_dets);

    xiStopAcquisition(g_xiCap);
	xiCloseDevice(g_xiCap);

    cvDestroyWindow(winname);
    
    free(g_feat_saved); free(g_feat_toverify);

finish:
    free_network(pnet);
    free_network(rnet);
    free_network(onet);
    free_network(mobilefacenet);

    return 0;
}

