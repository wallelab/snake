#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if 0
    #define MAX_DEEP 3600
#else
    #define MAX_DEEP 100
#endif

#define CUT_WIDTH 1530
#define CUT_HEIGHT 890

cv::VideoCapture cap(1);
cv::Mat frame, grayImage, cutImage;

typedef struct {
    int name;
    int keyst;
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
        cutImage.copyTo(m_hdmi[num].map);
        return 0;
    } else return 1;

}

void HdmiSave()
{
    char szname[80];

    for (int i=0; i<MAX_DEEP; i++) {
        sprintf(szname, "./pics/hdmi%04i.png", m_hdmi[i].name);
        cv::imwrite(szname, m_hdmi[i].map);
    }
}

