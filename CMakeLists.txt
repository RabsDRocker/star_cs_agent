cmake_minimum_required(VERSION 3.8)
project(star_cs_agent)

# Find the necessary ROS 2 build tools
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# This tells colcon to look for a setup.py file to handle the Python specifics
ament_python_install_package()

# This ensures your launch, rviz, and data files get installed correctly
# so that 'ros2 launch' can find them.
install(
    DIRECTORY launch rviz data
    DESTINATION share/${PROJECT_NAME}
)

ament_package()
