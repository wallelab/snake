#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#define MAX_DEEP 3600
#define CUT_WIDTH 1200
#define CUT_HEIGHT 720

cv::VideoCapture cap(1);

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

    return 0;
}


int HdmiCapture(int name, int num)
{
    cv::Mat frame, grayImage;

    cap >> frame;

//    cv::cvtColor(frame, grayImage, CV_RGB2GRAY);
    frame.copyTo(grayImage);

    if (num < MAX_DEEP) {
        m_hdmi[num].name = name;
        grayImage.copyTo(m_hdmi[num].map);
        return 0;
    } else return 1;

}

void HdmiSave()
{
    char szname[80];
    cv::Mat image;

    for (int i=0; i<MAX_DEEP; i++) {
        sprintf(szname, "./pics/hdmi%04i.png", m_hdmi[i].name);
        image = m_hdmi[i].map(cv::Rect(0,0,CUT_WIDTH, CUT_HEIGHT));
        cv::imwrite(szname, m_hdmi[i].map);
    }
}

