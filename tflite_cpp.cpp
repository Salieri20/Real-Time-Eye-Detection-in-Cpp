#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/string_util.h>
#include <thread>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <fstream>
#include <memory>


int counter = 0;

int main(int argc, char* argv[]) {

    // Load Model path.
    std::string face_model_file = "assets/face_landmark.tflite";
    std::string eyes_model_file = "assets/eye_model.tflite";
    bool show_windows = true;

    //Load Face Model from it's path
    auto FaceModel = tflite::FlatBufferModel::BuildFromFile(face_model_file.c_str());
    if (!FaceModel) {
        throw std::runtime_error("Failed to load FaceTFLite model"); //Check for the model existing
    }
    //Set up an interpreter with a custom operation resolver for Face Detection model.
    tflite::ops::builtin::BuiltinOpResolver face_op_resolver;
    std::unique_ptr<tflite::Interpreter> face_interpreter;
    tflite::InterpreterBuilder(*FaceModel, face_op_resolver)(&face_interpreter);
    //Check if tensor allocation was successful before printing the interpreter's state.
    if (face_interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Failed to allocate tensors");
    }
    //Printing the face landmark interpreter's state.
    tflite::PrintInterpreterState(face_interpreter.get());




    //Set up an interpreter with a custom operation resolver for Eye model.
    auto EyesModel = tflite::FlatBufferModel::BuildFromFile(eyes_model_file.c_str());
    if (!EyesModel) {
        throw std::runtime_error("Failed to load EyesTFLite model");
    }
    tflite::ops::builtin::BuiltinOpResolver eyes_op_resolver;
    std::unique_ptr<tflite::Interpreter> eyes_interpreter;
    tflite::InterpreterBuilder(*EyesModel, eyes_op_resolver)(&eyes_interpreter);
    //Check if tensor allocation was successful before printing the interpreter's state.
    if (eyes_interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Failed to allocate tensors");
    }
    //Printing the eyes interpreter's state.
    tflite::PrintInterpreterState(eyes_interpreter.get());
    //////////////////////////FACE//////////////////////////////////////////////////////////////////////
   ///////////////////////////inputs_details////////////////////////
    auto input_face = face_interpreter->inputs()[0];
    auto input_face_batch_size = face_interpreter->tensor(input_face)->dims->data[0];
    auto input_face_height = face_interpreter->tensor(input_face)->dims->data[1];
    auto input_face_width = face_interpreter->tensor(input_face)->dims->data[2];
    auto input_face_channels = face_interpreter->tensor(input_face)->dims->data[3];

    std::cout << "The input tensor of face has the following dimensions: [" << input_face_batch_size << ","
        << input_face_height << ","
        << input_face_width << ","
        << input_face_channels << "]" << std::endl;
    ///////////////////////////outputs_details////////////////////////
    auto output_face = face_interpreter->outputs()[0];

    auto face_dim0 = face_interpreter->tensor(output_face)->dims->data[0];
    auto face_dim1 = face_interpreter->tensor(output_face)->dims->data[1];//height
    auto face_dim2 = face_interpreter->tensor(output_face)->dims->data[2];//width
    auto face_dim3 = face_interpreter->tensor(output_face)->dims->data[3];
    std::cout << "The output tensor of face has the following dimensions: [" << face_dim0 << ","
        << face_dim1 << ","
        << face_dim2 << ","
        << face_dim3 << "]" << std::endl;
    //////Eye//////////////////////////////////////////////////////////////////////
      //inputs_details///
    auto input_eye = eyes_interpreter->inputs()[0];

    auto input_eye_batch_size = eyes_interpreter->tensor(input_eye)->dims->data[0];
    auto input_eye_height = eyes_interpreter->tensor(input_eye)->dims->data[1];
    auto input_eye_width = eyes_interpreter->tensor(input_eye)->dims->data[2];
    auto input_eye_channels = eyes_interpreter->tensor(input_eye)->dims->data[3];

    std::cout << "The input tensor of eye has the following dimensions: [" << input_eye_batch_size << ","
        << input_eye_height << ","
        << input_eye_width << ","
        << input_eye_channels << "]" << std::endl;

    auto output_eye = eyes_interpreter->outputs()[0];

    auto eye_dim0 = eyes_interpreter->tensor(output_eye)->dims->data[0];
    auto eye_dim1 = eyes_interpreter->tensor(output_eye)->dims->data[1];
    auto eye_dim2 = eyes_interpreter->tensor(output_eye)->dims->data[2];
    auto eye_dim3 = eyes_interpreter->tensor(output_eye)->dims->data[3];
    std::cout << "The output tensor of eye has the following dimensions: [" << eye_dim0 << ","
        << eye_dim1 << ","
        << eye_dim2 << ","
        << eye_dim3 << "]" << std::endl;
    int threads = 8;
    std::cout << "number of threads [" << threads << "]" << std::endl;
    eyes_interpreter->SetNumThreads(threads);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video capture" << std::endl;
        return -1;
    }
    cv::Mat frame;
    cv::Mat resized_img;
    int frame_count = 0;
    double total_face_inference_time = 0.0;
    double total_eye_inference_time = 0.0;

    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        auto frame_start = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point st, en;
        st = std::chrono::steady_clock::now();
        cap.read(frame);
        cv::flip(frame, frame, 1);
        // Frame size is [480 x 640] 
        int image_width = frame.size().width;
        int image_height = frame.size().height;
        int square_dim = std::min(image_width, image_height);
        int delta_height = (image_height - square_dim) / 2;
        int delta_width = (image_width - square_dim) / 2;

        // Crop the frame to the region of interest (ROI) 
        cv::Rect roi(delta_width, delta_height, square_dim, square_dim); // (x, y, width, height)
        cv::Mat cropped_frame = frame(roi);

        // Resize the cropped frame to match the input shape
        cv::Mat input_data; //Normalized data
        cv::resize(cropped_frame, resized_img, cv::Size(input_face_width, input_face_height));

        // Show output of Original , Cropped and resized frames  


        // Normalize input frame 
        resized_img.convertTo(input_data, CV_32FC1); //Convert input data to Float 32bit   
        input_data /= 255.0f;


        memcpy(face_interpreter->typed_input_tensor<float>(0), input_data.data, input_data.total() * input_data.elemSize());
        
        auto face_inference_start = std::chrono::steady_clock::now();
        
        // inference
        if (face_interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Inference failed" << std::endl;
            return -1;
        }

        auto face_inference_end = std::chrono::steady_clock::now();
        double face_inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(face_inference_end - face_inference_start).count();
        total_face_inference_time += face_inference_time;

        float* output_data = face_interpreter->typed_output_tensor<float>(0);
        const TfLiteIntArray* output_dims = face_interpreter->tensor(0)->dims;

        // Calculate the total number of elements in the original tensor
        int total_elements = 1;
        for (int i = 0; i < output_dims->size; ++i) {
            total_elements *= output_dims->data[i];
        }

        // Check if the total number of elements is divisible by 3
        if (total_elements % 3 != 0) {
            std::cerr << "Error: Total number of elements in the original tensor is not divisible by 3." << std::endl;
            return -1;
        }

        // Reshape the output data to have dimensions [468, 3]
        int num_rows = total_elements / 3; // Calculate the number of rows
        int num_cols = 3; // Number of columns
        std::vector<std::vector<float>> reshaped_output(num_rows, std::vector<float>(3));
        for (int i = 0; i < num_rows; ++i) {
            for (int j = 0; j < 3; ++j) {
                reshaped_output[i][j] = output_data[i * 3 + j];
            }
        }
        //crop left eye
        int y_coord_top_left = int(reshaped_output[299][1] * cropped_frame.size().width / input_face_width);
        int y_coord_bot_left = int(reshaped_output[330][1] * cropped_frame.size().width / input_face_width);
        int x_coord_left_left = int(reshaped_output[6][0] * cropped_frame.size().height / input_face_height);
        int x_coord_right_left = int(reshaped_output[383][0] * cropped_frame.size().height / input_face_height);

        cv::Rect roi_left(x_coord_left_left, y_coord_top_left, x_coord_right_left - x_coord_left_left, y_coord_bot_left - y_coord_top_left); // (x, y, width, height)
        cv::Mat left_eye = cropped_frame(roi_left);



        //crop right eye                                                              
        int y_coord_top_right = int(reshaped_output[69][1] * cropped_frame.size().width / input_face_width);
        int y_coord_bot_right = int(reshaped_output[101][1] * cropped_frame.size().width / input_face_width);
        int x_coord_left_right = int(reshaped_output[156][0] * cropped_frame.size().height / input_face_height);
        int x_coord_right_right = int(reshaped_output[6][0] * cropped_frame.size().height / input_face_height);

        cv::Rect roi_right(x_coord_left_right, y_coord_top_right, x_coord_right_right - x_coord_left_right, y_coord_bot_right - y_coord_top_right); // (x, y, width, height)
        cv::Mat right_eye = cropped_frame(roi_right);


        cv::imshow("Original",resized_img); 
        cv::imshow("right_eye", right_eye);
        cv::imshow("left_eye", left_eye);
       /////////////////////////////// Inference for right eye //////////////////////////////////////

            // Resize the cropped frame to match the input shape
        cv::Mat resized_right_eye; //resized 
        cv::Mat input_right; //Normalized data input for interpreter
        cv::resize(right_eye, resized_right_eye, cv::Size(input_eye_width, input_eye_height));

        // Normalize input frame 
        resized_right_eye.convertTo(input_right, CV_32FC1); //Convert input data to Float 32bit   
        input_right /= 255.0f;


        memcpy(eyes_interpreter->typed_input_tensor<float>(0), input_right.data, input_right.total() * input_right.elemSize());

        // inference
        std::chrono::steady_clock::time_point start, end;
        start = std::chrono::steady_clock::now();

        auto eye_inference_start = std::chrono::steady_clock::now();
        if (eyes_interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Inference failed" << std::endl;
            return -1;
        }
        auto eye_inference_end = std::chrono::steady_clock::now();
        double eye_inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(eye_inference_end - eye_inference_start).count();
        total_eye_inference_time += eye_inference_time;

        end = std::chrono::steady_clock::now();
        auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "processing time: " << processing_time << " ms" << std::endl;

        float* output_right = eyes_interpreter->typed_output_tensor<float>(0);
        if (output_right[0] <= 0.5)
            std::cout << "Right eye prediction = [" << "Closed" << "]" << std::endl;
        else
            std::cout << "Right eye prediction = [" << "Open" << "]" << std::endl;

        ////////////////////////////// Inference for left eye //////////////////////////////////////

            // Resize the cropped frame to match the input shape
        cv::Mat resized_left_eye; //resized 
        cv::Mat input_left; //Normalized data input for interpreter
        cv::resize(left_eye, resized_left_eye, cv::Size(input_eye_width, input_eye_height));

        // Normalize input frame 
        resized_left_eye.convertTo(input_left, CV_32FC1); //Convert input data to Float 32bit   
        input_left /= 255.0f;


        memcpy(eyes_interpreter->typed_input_tensor<float>(0), input_left.data, input_left.total() * input_left.elemSize());

        // inference
        if (eyes_interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Inference failed" << std::endl;
            return -1;
        }
        float* output_left = eyes_interpreter->typed_output_tensor<float>(0);
        if (output_left[0] <= 0.5)
            std::cout << "Left eye prediction = [" << "Closed" << "]" << std::endl;
        else
            std::cout << "Left eye prediction = [" << "Open" << "]" << std::endl;

        if (output_left[0] <= 0.5 || output_right[0] <= 0.5) {
            counter++;
        }
        std::cout << "times the eyes are closed = [" << counter << "]" << std::endl;
        if (counter >= 8) {
            std::cout << "take action" << std::endl;
        }
        auto frame_end = std::chrono::steady_clock::now();
        double frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start).count();
        frame_count++;
        std::cout << "Frame time: " << frame_time << " ms" << std::endl;

        int key = cv::waitKey(1);
        if (key == 27) // Press 'Esc' to exit the loop
            break;
    }
    auto end_time = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    double avg_face_inference_time = frame_count > 0 ? total_face_inference_time / frame_count : 0;
    double avg_eye_inference_time = frame_count > 0 ? total_eye_inference_time / frame_count : 0;
    double frame_rate = frame_count > 0 ? frame_count / total_time : 0;

    std::cout << "Average Face Inference Time: " << avg_face_inference_time << " ms" << std::endl;
    std::cout << "Average Eye Inference Time: " << avg_eye_inference_time << " ms" << std::endl;
    std::cout << "Average Frame Rate: " << frame_rate << " fps" << std::endl;
    return 0;
}


