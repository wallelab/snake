//  Read games and collect maps

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <unistd.h>
#include <math.h>

typedef struct {
    int light;
    int newx;
    int newy;
    int angle;

    int name;
    int key;
    int posx;
    int posy;
    cv::Mat map;
} HDMI_FRAME_STRCT;


#define MAX_DEEP 3600
#define MAX_WIDTH 1530
#define MAX_HEIGHT 890

HDMI_FRAME_STRCT m_pic[MAX_DEEP];
int n_pic = 0;

int ReadMouse()
{
    int i;
    FILE *file = fopen("./game/mouse.dat", "rb");
    if (file == NULL) {
        printf("Open file ./game/mouse.dat failure.\n");
        return -1;
    }

    for (i=0; i<MAX_DEEP; i++) {
        fread(&m_pic[i].name, 1, 2, file);
        fread(&m_pic[i].key, 1, 2, file);
        fread(&m_pic[i].posx, 1, 2, file);
        fread(&m_pic[i].posy, 1, 2, file);
        if (feof(file)) break;
    }
    n_pic = i;
    fclose(file);
    return 0;
}

void ReadPics()
{
    char szname[64];

    for (int i=0; i<n_pic; i++) {
        sprintf(szname, "./game/ss%04i.png", m_pic[i].name);
        m_pic[i].map = cv::imread(szname);
        printf(".");
        fflush(stdout);
    }
}

//  center at (770,460)
#define O_CX 770
#define O_CY 460
#define O_RADIO 200
// set gap = 5 degree.
#define RAD_GAP (3.1415926/36)
#define RAD_90 (1.5707964)
#define RAD_180 (3.1415926)
#define RAD_270 (4.7123890)
#define RAD_360 (6.2831853)

void RecalcaMouse()
{
    float dx, dy, tgs, alpha;
    int sec, ang;

    for (int i=0; i<n_pic; i++) {
        dx = m_pic[i].posx - O_CX;
        dy = m_pic[i].posy - O_CY;
        if (dy >= 0) {
            if (dx >= 0) sec = 1;
            else sec = 2;
        } else {
            if (dx >= 0) sec = 4;
            else sec = 3;
        }
        dx = fabs(dx);
        dy = fabs(dy);
        if (dx >= dy) {
            tgs = dy/dx;
            alpha = atan(tgs);
        } else {
            tgs = dx/dy;
            alpha = RAD_90 - atan(tgs);
        }
        switch (sec)
        {
        case 2:
            alpha = RAD_180 - alpha;
            break;
        case 3:
            alpha = RAD_180 + alpha;
            break;
        case 4:
            alpha = RAD_360 - alpha;
            break;
        }
        ang = alpha/RAD_GAP;
        alpha = ang * RAD_GAP;
//        printf("%i.", ang*5);
        m_pic[i].angle = ang;
        m_pic[i].newx = O_CX + cos(alpha) * O_RADIO;
        m_pic[i].newy = O_CY + sin(alpha) * O_RADIO;
    }
}

void SaveData()
{
    int i;
    FILE *file = fopen("./data/mouse.dat", "w");
    if (file == NULL) {
        printf("Error open file ./data/mouse.dat failure.\n");
        return;
    }

    for (i=0; i<n_pic; i++) {
        fwrite(&m_pic[i].light, 1, 2, file);
        fwrite(&m_pic[i].name, 1, 2, file);
        fwrite(&m_pic[i].key, 1, 2, file);
        fwrite(&m_pic[i].angle, 1, 2, file);
    }
    fclose(file);

    char szname[64];
    cv::Mat image, uimg, roi1, roi2;

    for (int i=0; i<n_pic; i++) {
        sprintf(szname, "./data/ss%04i.png", m_pic[i].name);
        roi1 = m_pic[i].map(cv::Rect(0,0,240,260));
        roi1.setTo(cv::Scalar(0, 0, 0));
        roi2 = m_pic[i].map(cv::Rect(1230,0,MAX_WIDTH-1230,390));
        roi2.setTo(cv::Scalar(0, 0, 0));

        cv::resize(m_pic[i].map, image, cv::Size(MAX_WIDTH/10, MAX_HEIGHT/10));
        cv::threshold(image, uimg, 64, 255, 0);
        cv::imwrite(szname, uimg);
        printf(".");
        fflush(stdout);
    }

    return;
}

int main()
{
    if (ReadMouse()) return -1;
    RecalcaMouse();

    ReadPics();

    cv::Mat image, img;
    int seq = 0;
    int brun = 1;
    while (brun) {
        printf("org(%i,%i) => (%i,%i) - %i\n", m_pic[seq].posx, m_pic[seq].posy,
               m_pic[seq].newx, m_pic[seq].newy, m_pic[seq].angle*5);
        m_pic[seq].map.copyTo(image);
        img = image(cv::Rect(m_pic[seq].newx, m_pic[seq].newy, 20, 20));
        if (m_pic[seq].light) {
            img.setTo(cv::Scalar(0, 192, 0));
        } else {
            img.setTo(cv::Scalar(0, 0, 192));
        }

        cv::imshow("mapedit", image);

        int c = cv::waitKey();
        switch (c)
        {
        case 32:
            m_pic[seq].light = 1;
            printf("select pic %i\n", seq);
            if (seq < n_pic-1) seq++;
            break;
        case 'D':
        case 'd':
            m_pic[seq].light = 0;
            printf("remove pic %i\n", seq);
            if (seq < n_pic-1) seq++;
            break;
        case 27:
            brun = 0;
            break;
        case 81:
            if (seq > 0) seq--;
            break;
        case 83:
            if (seq < n_pic-1) seq++;
            break;
        }
    }
    cv::destroyAllWindows();


    SaveData();

    return 0;
}

