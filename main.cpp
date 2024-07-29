// based on https://ffmpeg.org/doxygen/trunk/remuxing_8c-example.html

#include "TransformVideo.h"

#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>

//#include <Python.h>

//#define Py_CPYTHON_FRAMEOBJECT_H

//#include <pybind11/pybind11.h>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include "cvnp/cvnp.h"

#include <functional>
#include <iostream>



const cv::String keys =
    "{help h usage ? |      | print this message   }"
    "{@input         |      | input file           }"
    "{@output        |      | output file          }"
//    "{m model      | fsrcnn | desired model name   }"
    "{f file | img2img.py | desired model file   }"
//    "{u upscale      | 2    | upscale factor       }"
;


namespace py = pybind11;

int main(int argc, char **argv)
{    
    py::scoped_interpreter guard{};  // Start the Python interpreter

    // Evaluate in scope of main module
    py::object scope = py::module_::import("__main__").attr("__dict__");

    try {

        cv::CommandLineParser parser(argc, argv, keys);
        parser.about("Movie Upscaler Application");
        if (parser.has("help"))
        {
            parser.printMessage();
            return 0;
        }

        const auto input = parser.get<cv::String>(0);
        const auto output = parser.get<cv::String>(1);

        //const auto modelName = parser.get<cv::String>("m");
        const auto modelFile = parser.get<cv::String>("f");
        enum { downscale = 1 };//parser.get<int>("u");

        // Import the python script
        //py::module python_script = py::exec(modelFile.c_str());

        // Evaluate the statements in an separate Python file on disk
        py::eval_file(modelFile.c_str(), scope, scope);

        auto fun = scope["process_image"];

        auto lam = [&fun](cv::Mat& image) {
            // Convert the OpenCV image to a numpy array
            //py::array_t<uint8_t> numpy_image(image.total()* image.elemSize(),
            //    image.data);

            auto numpy_image = cvnp::mat_to_nparray(image);

            // Call the Python function with the numpy array as an argument
            //py::array_t<uint8_t> 
            py::array output_image = fun(numpy_image);

            // Convert the returned numpy array back to an OpenCV image
            //cv::Mat processed_image(image.size(), CV_8UC3, const_cast<uint8_t*>(output_image.data()));
            auto processed_image = cvnp::nparray_to_mat(output_image);
            image = processed_image.clone();
            cv::imshow("Output", image);
            cv::waitKey(1);
        };

        return TransformVideo(input.c_str(), output.c_str(), lam, 1, downscale);
    }
    catch (const std::exception& ex) {
        std::cerr << "Exception " << typeid(ex).name() << ": " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
}
