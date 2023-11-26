#include <ros/ros.h>
#include <iostream>

int main(int argc, char* argv[]) {


    
    ros::init(argc, argv, "test");



    ros::NodeHandle nh;

    std::cout << "[+] Everything good!" << std::endl;

    ros::spin();

    return 0;
}