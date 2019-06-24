#include "mtcnn.h"
#include "mobilefacenet.h"

#if 1

int main(int argc, char** argv)
{
    int help = find_arg(argc, argv, "--help");
    if(help){
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "    ./demo <function>\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Optional:\n");
        fprintf(stderr, "  for MobileFacenet:\n");
        fprintf(stderr, "    --mode     align mode      default `1`, find similarity;\n");
        fprintf(stderr, "  for MTCNN:\n");
        fprintf(stderr, "    -v         video mode,     default `0`, image mode;\n");
        fprintf(stderr, "    --path     file path,      default `../images/test.*`;\n");
        fprintf(stderr, "    --index    camera index,   default `0`;\n");
        fprintf(stderr, "    -p         thresh for PNet,default `0.8`;\n");
        fprintf(stderr, "    -r         thresh for RNet,defalut `0.8`;\n");
        fprintf(stderr, "    -o         thresh for ONet,defalut `0.8`;\n");
        fprintf(stderr, "    --minface  minimal face,   default `96.0`;\n");
        fprintf(stderr, "    --scale    resize factor,  default `0.79`;\n");
        fprintf(stderr, "    --stride                   default `2`;\n");
        fprintf(stderr, "    --cellsize                 default `12`;\n");
        fprintf(stderr, "    --softnms                  default `0`;\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Example:\n");
        fprintf(stderr, "    ./demo\n");
        fprintf(stderr, "\n");
        return 0;
    }

    verify_video_demo(argc, argv);
    return 0;
}

#endif

#if 0
#include <stdio.h>
#include "parser.h"

int main(int argc, char** argv)
{
    image im = load_image_color("images/patch_112x96.jpg", 0, 0);   // RGB, 0.~1.
    image cvt = convert_mobilefacenet_image(im);                    // BGR, -1~1

    network* net = load_mobilefacenet();
    network_predict(net, cvt.data);
    
    FILE* fp = fopen("images/patch_112x96_c.txt", "w");
#if 0
    for (int i = 0; i < cvt.c*cvt.h*cvt.w; i++ ){
        fprintf(fp, "%.8f\n", cvt.data[i]);
    }
#else
    for (int i = 0; i < net->n; i++ ){
        if ( i == 0 || i == 1 ||i == 2 || i == 5 || i == 13 || i == 22 || i == 31 || i == 40 || i == 49 ||
             i == 57 || i == 66 || i == 75 || i == 84 || i == 93 || i == 102 || i == 111 ||
              i == 119 || i == 128 || i == 137 || i == 140 || i == 141 || i == 142 || i == 143 || i == 144){
            layer l = net->layers[i];
            fprintf(fp, "[%d]%s =================\n", i, layer_type_to_string(l.type));
            for (int j = 0; j < l.outputs; j++ ){
                fprintf(fp, "%.8f\n", l.output[j]);
            }
        }
    }
#endif
    fclose(fp);
    free_image(im);
    free_image(cvt);
    return 0;
}

#endif