#include <opencv2/opencv.hpp>
int main()
{
    cv::Mat image = 256*cv::Mat::eye(100, 100, CV_8UC3);
    cv::imwrite("test.png", image);

    return EXIT_SUCCESS;
}
