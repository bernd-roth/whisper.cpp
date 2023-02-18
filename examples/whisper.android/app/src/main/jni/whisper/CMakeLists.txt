cmake_minimum_required(VERSION 3.4.1)

add_library(whisperb
        SHARED
        ggml.c)

target_include_directories(whisperb PRIVATE ../../../../../../..)

target_link_libraries(whisperb
        # Link with libneuralnetworks.so for NN API
        neuralnetworks
        android
        log)
