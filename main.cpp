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


namespace {

#if 0

// Parses the value of the active python exception
// NOTE SHOULD NOT BE CALLED IF NO EXCEPTION
std::string parse_python_exception()
{
    namespace py = pybind11;

    PyObject *type_ptr = NULL, *value_ptr = NULL, *traceback_ptr = NULL;
    // Fetch the exception info from the Python C API
    PyErr_Fetch(&type_ptr, &value_ptr, &traceback_ptr);

    // Fallback error
    std::string ret("Unfetchable Python error");
    // If the fetch got a type pointer, parse the type into the exception string
    if(type_ptr != NULL){
        try {
            py::handle h_type(type_ptr);
            py::str type_pstr(h_type);
            ret = type_pstr;
        }
        catch (...) {
            ret = "Unknown exception type";
        }
    }
    // Do the same for the exception value (the stringification of the exception)
    if(value_ptr != NULL){
        try {
            py::handle h_val(value_ptr);
            py::str a(h_val);
            ret += ": " + std::string(a);
        }
        catch (...) {
            ret += ": Unparseable Python error: ";
        }
    }
    // Parse lines from the traceback using the Python traceback module
    if(traceback_ptr != NULL){
        py::handle h_tb(traceback_ptr);
        // Load the traceback module and the format_tb function
        py::object tb(py::import("traceback"));
        py::object fmt_tb(tb.attr("format_tb"));
        // Call format_tb to get a list of traceback strings
        py::object tb_list(fmt_tb(h_tb));
        // Join the traceback strings into a single string
        py::object tb_str(py::str("\n").join(tb_list));
        // Extract the string, check the extraction, and fallback in necessary
        py::extract<std::string> returned(tb_str);
        if(returned.check())
            ret += ": " + returned();
        else
            ret += ": Unparseable Python traceback";
    }
    return ret;
}


// convert a cv::mat to an np.array
py::array to_array(const cv::Mat& im) {
    const auto channels = im.channels();
    const auto height = im.rows;
    const auto width = im.cols;
    const auto dim = sizeof(uchar) * height * width * channels;
    auto data = new uchar[dim];
    std::copy(im.data, im.data + dim, data);
    return py::array_t<uchar>(
        py::buffer_info(
            data,
            sizeof(uchar), //itemsize
            py::format_descriptor<uchar>::format(),
            channels, // ndim
            std::vector<ssize_t> { height, width, channels }, // shape
            std::vector<ssize_t> { width* channels, channels, sizeof(uchar) } // strides
    ),
        py::capsule(data, [](void* f) {
            // handle releasing data
            delete[] reinterpret_cast<uchar*>(f);
            })
    );
}


// convert an np.array to a cv::mat
cv::Mat from_array(const py::array& ar) {
    if (!ar.dtype().is(py::dtype::of<uchar>())) {
        std::cout << "error unsupported dtype!" << std::endl;
        return cv::mat();
    }

    auto shape = ar.shape();
    int rows = shape[0];
    int cols = shape[1];
    int channels = shape[2];
    int type = cv_maketype(cv_8u, channels); // cv_8uc3
    cv::mat mat = cv::mat(rows, cols, type);
    memcpy(mat.data, ar.data(), sizeof(uchar) * rows * cols * channels);

    return mat;
}

#endif


} // namespace



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
        enum { downscale = 2 };//parser.get<int>("u");

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
        };

        return TransformVideo(input.c_str(), output.c_str(), lam, 1, downscale);
    }
    //catch (const py::error_already_set& ex) {
    //    std::cerr << "Exception " << typeid(ex).name() << ": " << py::detail::error_string() << '\n';
    //}
    catch (const std::exception& ex) {
        std::cerr << "Exception " << typeid(ex).name() << ": " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
}
