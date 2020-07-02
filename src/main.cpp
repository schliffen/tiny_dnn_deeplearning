//
// CUDA GPU programming
//
#include "NumCpp.hpp"
#include <NumCpp/Functions/matmul.hpp>
#include <NumCpp/Functions/multiply.hpp>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <stdio.h> 
#include <time.h>
#include "foo.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include "opencv2/core/core.hpp"
#include "tiny_dnn.h"
#include <random>

// typedef struct ali
// {
//     /* data */
//     int a;
//     float b;
//     char c;
// } vr;

//     vr t1,t2;

//     t1.a = 1; t2.b=1.2;


//     h.hello();
//     h1.hello1();
//     h2.hello2();



// void set_input_buffer(std::vector<cv::Mat>& input_channels,
// 	float* input_data, const int height, const int width)
// {
// 	for (int i = 0; i < 3; ++i) {
// 		cv::Mat channel(height, width, CV_32FC1, input_data);
// 		input_channels.push_back(channel);
// 		input_data += width * height;
// 	}
// }

// connection table, see Table 1 in [LeCun1998]
#define O true
#define X false
static const bool tbl [] = {
    O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
    O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
    O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
    X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
    X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
    X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
};
#undef O
#undef X

void check_data(std::vector<std::string> data, std::vector<int> target){
    for (int i=0; i<data.size(); i++){
        std::cout<< "data: "<< data[i] << " target: "<< target[i]<< std::endl;
    }

}

void put_image(std::vector<std::string> imagefilename,
                   double minv,
                   double maxv,
                   int w,
                   int h,
                   tiny_dnn::vec_t &data) {

  for (int i=0; i<imagefilename.size(); i++  ){     
        std::cout<< "reading image!\n";
        tiny_dnn::image<> img(imagefilename[i], tiny_dnn::image_type::grayscale);
        tiny_dnn::image<> resized = resize_image(img, w, h);

        resized.save("test_img_"+ std::to_string(i) + ".jpg"  );

        // mnist dataset is "white on black", so negate required
        std::transform(
            resized.begin(), resized.end(), std::back_inserter(data),
            [=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });

}
std::cout<< "image set!\n";
}

void train_lenet( std::vector<std::string> train_list, 
                        std::vector<std::string> test_list, 
                        tiny_dnn::vec_t train_labels, tiny_dnn::vec_t test_labels
                         ) {
    // specify loss-function and learning strategy
    tiny_dnn::timer t;
    tiny_dnn::network<tiny_dnn::sequential> net;
    //core::backend_t backend_type = core::default_engine();
    
    //
    net << tiny_dnn::layers::conv(32, 32, 5, 1, 6) << tiny_dnn::activation::tanh()  // in:32x32x1, 5x5conv, 6fmaps
        << tiny_dnn::layers::ave_pool(28, 28, 6, 2) << tiny_dnn::activation::tanh() // in:28x28x6, 2x2pooling
        << tiny_dnn::layers::fc(14 * 14 * 6, 120) << tiny_dnn::activation::tanh()   // in:14x14x6, out:120
        << tiny_dnn::layers::fc(120, 2);   
            
    
    tiny_dnn::adagrad optimizer;

    std::cout << "load models..." << std::endl;

    // load MNIST dataset
    // std::vector<label_t> train_labels, train_labels;
    // std::vector<vec_t> train_images, train_images;

    std::cout << "start training" << std::endl;

    int minibatch_size = 10;
    int num_epochs = 2;
    int epochs = 10;
    int batch_size = 2;
    int iteration = std::floor( train_list.size()/batch_size);
    int w=32, h=32;
    double minv=-1.0, maxv=1.0; 
     
    tiny_dnn::progress_display disp(static_cast<unsigned long>( batch_size ) );
    // timer t;

    optimizer.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(minibatch_size));

    tiny_dnn::vec_t train_data, test_data;


    // vectors for iteration on the image list
    // std::vector<std::string>::const_iterator batchListBegin, batchListEnd;
    std::vector<std::string> tempvect;
    tempvect.resize(batch_size);

    for(int ei=0; ei<num_epochs-1; ei++){ 

        // reading the images
        // selecting trima images from the list
        // batchListBegin = train_list.begin() + ei * batch_size;
        // batchListEnd = train_list.begin() + (ei + 1) * batch_size;
        // tempvect =  train_list(batchListBegin, batchListEnd);

        std::cout<< "mem begin: "<< ei * batch_size << " mem end: "<< (ei + 1) * batch_size<< std::endl;

        memcpy( &tempvect[0], &train_list[ ei * batch_size],  (ei + 1) * batch_size*sizeof(std::string) );


        std::cout<< ei << "  here!---------- \n";
    
        
        put_image( tempvect, minv, maxv, w, h, train_data); 


        
        tempvect.clear();
        // free(tempvect);
        std::cout<< train_data.size() << " function end \n";
        
        // create callback
        auto on_enumerate_epoch = [&](){
            std::cout << t.elapsed() << "s elapsed." << std::endl;
            tiny_dnn::result res = net.test(train_data, train_labels);

            std::cout << res.num_success << "/" << res.num_total << std::endl;
            
            disp.restart(static_cast<unsigned long>(train_data.size()));
            t.restart();
        };

        auto on_enumerate_minibatch = [&](){
            disp += minibatch_size;
        };

        // training
        net.train<tiny_dnn::mse>(optimizer, train_data, train_labels, minibatch_size, num_epochs,
                on_enumerate_minibatch, on_enumerate_epoch);

        std::cout << "end training." << std::endl;
    }

        /*
        // test and show results
        net.test(test_images, test_labels).print_detail(std::cout);

    } // end of the epoch

    // save network model & trained weights
    net.save("LeNet-model");
    */
}



int main(){
    //
    // float * arr ; //= new float[2];   
    //
    nc::NdArray<float> sMeasured = { 0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.74 , 1.2};
    nc::NdArray<float> rateMeasured = { 0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317 };

    // create a random image with NumCpp
    std::string img_dir = "/home/ali/ProjLAB/deeplearning/FaceDetectionProject/sample-images/Howard.jpg", imgsDir = "/home/ali/ProjLAB/LearningCpp/dataset/dogvscat/test_set/";
    cv::Mat img=cv::imread(img_dir);
    cv::resize(img, img, cv::Size(256, 256));

    auto ncArray = nc::random::randInt<nc::uint8>({ 128, 128 }, 0, nc::DtypeInfo<nc::uint8>::max());

    auto cvArray = cv::Mat( ncArray.numRows(), ncArray.numCols(), CV_8SC1, ncArray.data());


    auto arrImg = nc::NdArray<nc::uint8>(img.data, img.rows, img.cols); 

    const float coff=.2; 
    // ncArray = arrImg +  nc::multiply (coff, ncArray) ;
    // cv::imshow("cvArray", cvArray);
    // cv::waitKey(0);
    // std::array<nc::NdArray> arr;

    // reading list of image names:
    std::vector<std::string> imgList, cats, dogs;
    cv::glob(imgsDir + "cats/*.jpg" , cats, false);
    cv::glob(imgsDir + "dogs/*.jpg" , dogs, false);
    //
    std::cout<< "cats data size: "<< cats.size()<< " dogs data size: " << dogs.size() << std::endl; 
    std::vector< std::string > input_train, input_test;
    tiny_dnn::vec_t target_train, target_test;
    int distributor;
    for (int i=0; i<cats.size() + dogs.size(); i++ ){
        distributor = (random() % 100); 
        if (  distributor < 40)
        {
            // reading the 
            // std::cout<<" data name: "<< cats.at(i%cats.size()) << std::endl;
            input_train.push_back( cats.at(i%cats.size()) );
            target_train.push_back({0}); // cats zero
            
        }else if (distributor < 80)
        {
            input_train.push_back( dogs.at(i%dogs.size()) );
            target_train.push_back({1}); // cats zero
        }else if (distributor < 90){
            input_test.push_back( cats.at(i%cats.size()) );
            target_test.push_back({0}); // cats zero
        }else{
            input_test.push_back( dogs.at(i%dogs.size()) );
            target_test.push_back({1}); // cats zero
        };

        
    };
    //
    // check_data( input_train, target_train);

    // checking the data
    std::cout<<" train data name: "<< input_train.size() << std::endl;
    // 
    // preparing the model
    //
    //network

    // training ---     
    train_lenet( input_train, 
                 input_test, 
                 target_train, 
                 target_test
                 );

    std::cout<< "main end  \n";
    



    return 0;
}

/*
    std::string img_dir = "/home/ali/ProjLAB/deeplearning/FaceDetectionProject/sample-images/Howard.jpg";
    cv::Mat img=cv::imread(img_dir);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(8, 8), 0, 0, cv::INTER_LINEAR);
    resized.convertTo(resized, CV_32FC3);
	resized = (resized - 127.5)*0.0078125;
    int scale_h = 8;
    int scale_w = 8;
    // scale_h = 128; scale_w=128;
	float * newarr = new float[3*scale_h*scale_w];
    std::vector<float> input1(3 * scale_h*scale_w);
	std::vector<cv::Mat> input_channels;
    cv::Mat tensor;

    //    
	set_input_buffer(input_channels, newarr, scale_h, scale_w);
    cv::split(resized, input_channels);
    // cv::dnn::blobFromImage(resized, tensor, 1.0,cv::Size(scale_h, scale_w),cv::Scalar(0,0,0),true);
    //
    for (int i=0; i< input_channels.size(); i++)
         std::cout<< i << " cnl -> " << input_channels[i] << std::endl;  
    for (int i=0; i< resized.rows; i++)
    for (int j=0; j< resized.cols; j++)
         std::cout<< i << " img -> " << resized.at<Voc3d>(i,j) << std::endl;
    
    std::cout<< " img -> " << resized.size() << std::endl;
    std::cout<< " img -> " << resized << std::endl;
    
    cv::imshow("img", img);
    cv::waitKey(0);
*/