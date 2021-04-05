#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <unistd.h>

#if 1
    #define MAX_DEEP 3600
#else
    #define MAX_DEEP 100
#endif

#define CUT_WIDTH 1540
#define CUT_HEIGHT 890


extern int ms_key;
extern int ms_posx, ms_posy;

cv::VideoCapture cap(0);
cv::Mat frame, grayImage, cutImage;

typedef struct {
    int name;
    int key;
    int posx;
    int posy;
    cv::Mat map;
} HDMI_FRAME_STRCT;

HDMI_FRAME_STRCT m_hdmi[MAX_DEEP];

int HdmiInit()
{
    if (!cap.isOpened()) {
        std::cout << "VideoCapture(0) initial failure." << std::endl;
        return -1;
    }

    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(CV_CAP_PROP_FPS, 60);

    cap >> frame;
    cv::cvtColor(frame, grayImage, CV_RGB2GRAY);
    cutImage = grayImage(cv::Rect(0,0,1530,890));
    return 0;
}


int HdmiCapture(int name, int num)
{
    cap >> frame;

    cv::cvtColor(frame, grayImage, CV_RGB2GRAY);
//    frame.copyTo(grayImage);

    if (num < MAX_DEEP) {
        m_hdmi[num].name = name;
        m_hdmi[num].key = ms_key;
        m_hdmi[num].posx = ms_posx;
        m_hdmi[num].posy = ms_posy;
        cutImage.copyTo(m_hdmi[num].map);
        return 0;
    } else return 1;
}

void HdmiSave()
{
    char szname[80];

    for (int i=0; i<MAX_DEEP; i++) {
        sprintf(szname, "./games/ss%04i.png", m_hdmi[i].name);
        cv::imwrite(szname, m_hdmi[i].map);
        printf(".");
        fflush(stdout);
    }

    FILE *file = fopen("./games/mouse.dat", "w");
    if (file != NULL) {
        for (int i=0; i<MAX_DEEP; i++) {
            fwrite(&m_hdmi[i].name, 1, 2, file);
            fwrite(&m_hdmi[i].key, 1, 2, file);
            fwrite(&m_hdmi[i].posx, 1, 2, file);
            fwrite(&m_hdmi[i].posy, 1, 2, file);
        }
        fclose(file);
    }
}

