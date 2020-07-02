cmake_minimum_required (VERSION 3.5)

# Setup projectname
project (myDnn VERSION 1.0.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -ldl -pthread")
# Compile and link main.cpp and foo.cpp


set(OPENCV_INCLUDE_DIR /home/ali/installations/opencv/include/opencv4)
set(OPENCV_LIB_DIR /home/ali/installations/opencv/lib)
set(CMAKE_PREFIX_PATH /home/ali/installations/opencv/lib/cmake/opencv4)

find_package( OpenCV REQUIRED )

set(NUMCPP_INCLUDES /home/ali/installations/NumCpp/include)
set(Boost_INCLUDE_DIR /home/ali/installations/boost_1_66_0/)
set(Boost_LIBRARY_DIRS /home/ali/installations/boost_1_66_0/libs)
include_directories( include )


# file(GLOB_RECURSE tiny_dnn_headers "${CMAKE_SOURCE_DIR}/tiny_dnn/*.h")
# include_directories(tiny_dnn)
file(GLOB_RECURSE tiny_dnn_headers "${CMAKE_SOURCE_DIR}/tiny_dnn/*.h")
file(RELATIVE_PATH tiny_dnn_relpath ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_SOURCE_DIR}/tiny_dnn")
# string(REPLACE "/" "\\" tiny_dnn_relpath "${tiny_dnn_relpath}")
# foreach(_source IN ITEMS ${tiny_dnn_headers})
#     file(RELATIVE_PATH _source_rel ${CMAKE_CURRENT_SOURCE_DIR} "${_source}")
#     get_filename_component(_source_path "${_source_rel}" PATH)
#     string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
#     string(REPLACE ${tiny_dnn_relpath} "tiny_dnn" _source_path_msvc ${_source_path_msvc})
#     source_group("${_source_path_msvc}" FILES "${_source}")
# endforeach()


include_directories( tiny_dnn )
include_directories( ${tiny_dnn_headers} )
include_directories( ${tiny_dnn_relpath} )


add_library( foolib
    SHARED
    include/foo.cpp
    include/foo.hpp
)


include_directories	(${Boost_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBS})
link_directories(${Boost_LIBRARY_DIRS})


add_executable( ${PROJECT_NAME} src/main.cpp    )

target_link_libraries(  ${PROJECT_NAME}  foolib  ${OpenCV_LIBS} ${Boost_LIBRARY_DIRS})
target_include_directories( ${PROJECT_NAME} PUBLIC ${NUMCPP_INCLUDES} )
target_include_directories( ${PROJECT_NAME} PUBLIC ${Boost_INCLUDE_DIRS}) 