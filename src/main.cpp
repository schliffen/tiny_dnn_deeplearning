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
// rescale output to 0-100
template <typename Activation>
double rescale(double x) {
  Activation a(1);
  return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

//
void check_data(std::vector<std::string> data, std::vector<int> target){
    for (int i=0; i<data.size(); i++){
        std::cout<< "data: "<< data[i] << " target: "<< target[i]<< std::endl;
    }

}

void put_image(std::vector<std::string> imagefilename,
               std::vector<tiny_dnn::vec_t>    *train_images,
               std::vector<tiny_dnn::label_t>  *train_labels,
                double scale_min = 0,
                double scale_max = 1,
                int target_w =32,
                int target_h =36,
                int channels =3,
                int x_padding =0,
                int y_padding =0)
    {

    if (x_padding < 0 || y_padding <0)
        throw tiny_dnn::nn_error( "padding cannot be negative! ");
    if ( scale_min > scale_max )
        throw tiny_dnn::nn_error(" min scale is bigger than max scale");
  
    for (std::vector<std::string>::iterator imgname = imagefilename.begin(); imgname != imagefilename.end(); ++imgname ){     
        std::cout<< "reading image!\n";
        
        std::cout <<  *imgname <<std::endl;
        
        
        tiny_dnn::image<> temp( *imgname, tiny_dnn::image_type::rgb);
        tiny_dnn::image<> resized = resize_image(temp, target_w, target_h);
        //
        resized.save("test_img_"+ std::to_string( random() ) + "_.jpg"  ); // saving the sample image
        //
        std::cout<< "resized size: " << resized.shape() << std::endl;    

        tiny_dnn::vec_t img;    
        //
    
        int w = target_w + 2 * x_padding;
        int h = target_h + 2 * y_padding;

        img.resize(w * h * channels, scale_min);

        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < target_h; y++) {
                for (int x = 0; x < target_w; x++) {
                    // std::cout<< "c="<< c << " y="<<y << " x="<< x<< std::endl;
                    img[c * w * h + (y + y_padding) * w + x + x_padding] = scale_min + 
                    (scale_max - scale_min) * resized[ x, y, c] / 255;
                }
            }
        
        
        
        }

        // train_images->push_back(img);
        // train_labels->push_back(label);
        
        
        
        }
        // }


        // std::transform(
        //     resized.begin(), resized.end(), std::back_inserter(data),
        //     [=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });

        

}


  





template <typename N>
void construct_net(N &nn) {
  using conv    = tiny_dnn::convolutional_layer;
  using pool    = tiny_dnn::max_pooling_layer;
  using fc      = tiny_dnn::fully_connected_layer;
  using relu    = tiny_dnn::relu_layer;
  using softmax = tiny_dnn::softmax_layer;

  const size_t n_fmaps  = 32;  ///< number of feature maps for upper layer
  const size_t n_fmaps2 = 64;  ///< number of feature maps for lower layer
  const size_t n_fc = 64;  ///< number of hidden units in fully-connected layer

  nn << conv(32, 32, 5, 3, n_fmaps, tiny_dnn::padding::same)  // C1
     << pool(32, 32, n_fmaps, 2)                              // P2
     << relu(16, 16, n_fmaps)                                 // activation
     << conv(16, 16, 5, n_fmaps, n_fmaps, tiny_dnn::padding::same)  // C3
     << pool(16, 16, n_fmaps, 2)                                    // P4
     << relu(8, 8, n_fmaps)                                        // activation
     << conv(8, 8, 5, n_fmaps, n_fmaps2, tiny_dnn::padding::same)  // C5
     << pool(8, 8, n_fmaps2, 2)                                    // P6
     << relu(4, 4, n_fmaps2)                                       // activation
     << fc(4 * 4 * n_fmaps2, n_fc)                                 // FC7
     << fc(n_fc, 2) << softmax(2);                               // FC10
}


void train_lenet( std::vector<std::string> train_list, 
                  std::vector<std::string> test_list, 
                  std::vector<tiny_dnn::label_t> train_labels, 
                  std::vector<tiny_dnn::label_t> test_labels) {
    // specify loss-function and learning strategy
    tiny_dnn::timer t;
    tiny_dnn::network<tiny_dnn::sequential> net;
    //core::backend_t backend_type = core::default_engine();
    
    //
    construct_net( net );  
            
    
    tiny_dnn::adagrad optimizer;

    // std::cout << "load models..." << std::endl;

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

    // tiny_dnn::vec_t train_data, test_data;


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
        std::vector<tiny_dnn::vec_t>    train_images;
        std::vector<tiny_dnn::label_t>  train_labels;

        std::cout<< "mem begin: "<< ei * batch_size << " mem end: "<< (ei + 1) * batch_size<< std::endl;

        memcpy( &tempvect[0], &train_list[ ei * batch_size],  (ei + 1) * batch_size*sizeof(std::string) );


        std::cout<< ei << "  here!---------- \n";
    
        
        put_image( tempvect, &train_images, &train_labels); 
        
        // tempvect.clear();
        /*
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

        */    
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
    std::string img_dir = "/home/ali/ProjLAB/deeplearning/FaceDetectionProject/sample-images/Howard.jpg", 
                imgsDir = "/home/ali/ProjLAB/LearningCpp/dataset/dogvscat/test_set/";
    cv::Mat img=cv::imread(img_dir);
    cv::resize(img, img, cv::Size(256, 256));

    // auto ncArray = nc::random::randInt<nc::uint8>({ 128, 128 }, 0, nc::DtypeInfo<nc::uint8>::max());
    // auto cvArray = cv::Mat( ncArray.numRows(), ncArray.numCols(), CV_8SC1, ncArray.data());
    auto arrImg = nc::NdArray<nc::uint8>(img.data, img.rows, img.cols);  // opencv to Ndarray

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
    std::vector<tiny_dnn::label_t> target_train, target_test;
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