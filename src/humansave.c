#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>

#define DEMO 1
#define CSVFile "./data/DataRecord.csv"

#ifdef OPENCV
//#include "opencv2/videoio/videoio_c.h"
//#include "opencv2/highgui/highgui_c.h"
//#include "opencv2/imgproc/imgproc_c.h"

//void image_to_iplimage(image p, IplImage *disp);


static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static long int videotime;
static float **probs;
static box *boxes;
static network *net;
static image buff [3];
static image buff_letter[3];
static int buff_index = 0;
static CvCapture * cap;
static IplImage  * ipl;
static float fps = 0;
static float demo_thresh = 0.24;
static float demo_hier = .5;
static int running = 0;
static float begintime;


static int demo_frame = 3;
static int demo_detections = 0;
static float **predictions;
static int demo_index = 0;
static int demo_done = 0;
static float *avg;
double demo_time;
FILE *fp;
FILE *videofp;
//For write video
static CvVideoWriter *DetectVideo = NULL;

save_boxes(int boxes_num)
{
    int i = 0, j, k, human_num = 0, color_num, class, left, right, top, bot;
    float prob, r, g, b;
    image display = buff[(buff_index+2) % 3];

    for(i=0; i< boxes_num; i++)
    {
        class = max_index(probs[i], demo_classes);
        prob = probs[i][class];
        if(prob > demo_thresh)
	    human_num++;
    }
    fprintf(fp,"%d,",human_num);

    for(i=0; i< boxes_num; i++)
    {
        class = max_index(probs[i], demo_classes);
        prob = probs[i][class];
        if(prob > demo_thresh)
        {
            box bx = boxes[i];
            left = (bx.x-bx.w/4.)*display.w;
            right = (bx.x+bx.w/4.)*display.w;
            top = (bx.y-bx.h/4.)*display.h;
            bot = bx.y*display.h;
	    if(left < 0) left = 0;
            if(right > display.w-1) right = display.w-1;
            if(top < 0) top = 0;
            if(bot > display.h-1) bot = display.h-1;
	    color_num = 0;
	    r = 0.0;
            g = 0.0;
            b = 0.0;
	    for(j = left; j<=right; j++)
		for(k = top; k<=bot; k++){
		    r = r + display.data[j + k*display.w + 0*display.w*display.h];
		    g = g + display.data[j + k*display.w + 1*display.w*display.h];
		    b = b + display.data[j + k*display.w + 2*display.w*display.h];
		    color_num++;
		}
	    if(color_num>1){
	    	r = r/((float)color_num);
	    	g = g/((float)color_num);
            	b = b/((float)color_num);
	    }
	    if(i < human_num-1)
            	fprintf(fp,"%f,%f,%f,%f,%f,%f,%f,", bx.x, bx.y, bx.w, bx.h, r, g, b);
	    if(i == human_num-1)
            	fprintf(fp,"%f,%f,%f,%f,%f,%f,%f", bx.x, bx.y, bx.w, bx.h, r, g, b);
            printf("%f,%f,%f,%f,%f,%f,%f,",bx.x,bx.y, bx.w, bx.h, r, g, b);
        }
    }
    fprintf(fp,"\n");
    printf("\n");
}

void *human_save_detect_in_thread(void *ptr)
{
    running = 1;
    float nms = .4;

    layer l = net->layers[net->n-1];
    float *X = buff_letter[(buff_index+2)%3].data;
    float *prediction = network_predict(net, X);

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, demo_frame, l.outputs, avg);
    l.output = avg;
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
    } else if (l.type == REGION){
        get_region_boxes(l, buff[0].w, buff[0].h, net->w, net->h, demo_thresh, probs, boxes, 0, 0, 0, demo_hier, 1);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");
    image display = buff[(buff_index+2) % 3];

    save_boxes(l.w*l.h*l.n);

    draw_detections(display, demo_detections, demo_thresh, boxes, probs, 0, demo_names, demo_alphabet, demo_classes);

    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
}

void *human_save_fetch_in_thread(void *ptr)
{
    int status = fill_image_from_stream(cap, buff[buff_index]);
    letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
    if(status == 0) demo_done = 1;
    return 0;
}

void *human_save_display_in_thread(void *ptr)
{
    show_image_cv(buff[(buff_index + 1)%3], "Demo", ipl);
    int c = cvWaitKey(1);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}

void *human_save_display_loop(void *ptr)
{
    while(1){
        human_save_display_in_thread(0);
    }
}

void *human_save_detect_loop(void *ptr)
{
    while(1){
        human_save_detect_in_thread(0);
    }
}

void humansave(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{

    char *findtime = (char*)malloc(50);
    findtime = strcpy(findtime, filename);

    char *time;
    time = strtok(findtime,"/");
    time = strtok (NULL, "/");
    sscanf(time, "%ld", &videotime);
    begintime =(float)(videotime%100+((videotime/100)%100)*60+((videotime/10000)%100)*3600);
    printf("    The time record the video:%f\n", begintime);

    char *directory = (char*)malloc(50);
    strcpy(directory, filename);
    directory = strtok(directory,"/");
    directory = strcat(directory,"/");
    directory = strcat(directory,time);
    directory = strcat(directory,"/");
    printf("    The directory of video file:%s\n",directory);

    char *videocsvfile = (char*)malloc(50);
    videocsvfile = strcpy(videocsvfile, directory);
    videocsvfile = strcat(videocsvfile, time);
    videocsvfile = strcat(videocsvfile, ".csv");
    printf("    The frame time recording file of video:%s\n",videocsvfile);




    demo_frame = avg_frames;
    predictions = calloc(demo_frame, sizeof(float*));
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
//  demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;

    srand(2222222);

    if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
        if ( !(cap = cvCaptureFromFile(filename))) {
		fprintf(stderr, "Can not open video file %s\n", filename);
		return -2;
	} 

	printf("capture from file\n");
    }else{
        cap = cvCaptureFromCAM(cam_index);

        if(w){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
        }
        if(h){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if(frames){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
        }
    }

    if(!cap) error("Couldn't connect to webcam.\n");
 
    //----------------------For Reading CSV File--------------------------
    videofp = NULL;

    videofp = fopen(videocsvfile, "r");
    if(videofp == NULL)
    {
        printf("The video csv file or box csv file cannot be read!\n");
        exit(1);
    }
 
    /*-------------------------For Writing CSV File------------------------*/
    fp = NULL;
    fp = fopen(CSVFile, "a");
    if(NULL == fp)
    {
        printf("Cannot open the csv file for writing!");
        exit(1);
    }

    /*---------------------------------------------------------------------*/


    layer l = net->layers[net->n-1];
    demo_detections = l.n*l.w*l.h;
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    printf("demo_frame: %d\n", demo_frame);
    for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

  
/*    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
        if(fullscreen){
            cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Demo", 0, 0);
            cvResizeWindow("Demo", 1352, 1013);
        }
    }

    demo_time = what_time_is_it_now();*/

    int count  = 0;
    float frametime = 0;
    int framenum = 0;

    /*----------------------For Recording Video file------------------*/
    DetectVideo = cvCreateVideoWriter("./data/PersonDete.avi", CV_FOURCC('P','I','M','1'),30.0,cvSize(buff[0].w,buff[0].h),1);
   /*------------------------------------------------------------------*/ 

    //printf("start while loop\n");
    while(!demo_done){
	++count;
	fprintf(fp, "%d,",count);
	fscanf(videofp, "%d, %f\n",&framenum, &frametime);
	fprintf(fp, "%f,",frametime);
        printf("\n------------------Count: %d------------------\n",count);
        buff_index = (buff_index + 1) %3;
        if(pthread_create(&fetch_thread, 0, human_save_fetch_in_thread, 0)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, human_save_detect_in_thread, 0)) error("Thread creation failed");
      /*if(!prefix){
            fps = 1./(what_time_is_it_now() - demo_time);
            demo_time = what_time_is_it_now();
            //human_save_display_in_thread(0);
        }else{
            char name[256];
            sprintf(name, "%s_%08d", prefix, count);
            save_image(buff[(buff_index + 1)%3], name);
        }*/
 
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
	/*---------------Save images---------------*/

	char name[256];
	printf("Here Save the image!\n");
        sprintf(name, "./data/Moving/%d", count);
	save_image(buff[(buff_index + 2)%3], name);   
	/*------------Save video--v---------------*/
        IplImage *videoframe = cvCreateImage(cvSize(buff[(buff_index + 2)%3].w,buff[(buff_index + 2)%3].h), IPL_DEPTH_8U, 3);
        image_to_iplimage(buff[(buff_index + 2)%3], videoframe);
        cvWriteFrame(DetectVideo, videoframe);
        cvReleaseImage(&videoframe);
    }
    //printf("Releasing Video Writer\n");
    cvReleaseVideoWriter(&DetectVideo);
    //printf("Video Writer released\n");
    fclose(fp);
    fclose(videofp);
    fp = NULL;
    //printf("humansave function closed\n");
}

#else
void humansave(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

