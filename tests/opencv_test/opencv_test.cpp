#include <opencv2/opencv.hpp>


int main() {
    cv::Mat img = cv::Mat::zeros(300, 300, CV_8UC3);
    cv::circle(img, {150, 150}, 100, {0, 255, 0}, -1);

    cv::imshow("Green Circle", img);
    cv::waitKey(0);

    return 0;
}
