cmake_minimum_required(VERSION 3.8)
project(star_cs_agent)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package()

install(
    DIRECTORY launch rviz data
    DESTINATION share/${PROJECT_NAME}
)

ament_package()