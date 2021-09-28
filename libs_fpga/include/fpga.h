//
// Created by chi on 9/22/21.
//

#ifndef HIPC21_FPGA_H
#define HIPC21_FPGA_H

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

// #define L1 8 

//L1
// #define L1 8 //L1
// #define L2 64 //L2
// #define L3 4 //L3

#include <vector>
#include <CL/cl2.hpp>


#include <iostream>
#include <fstream>
#include <CL/cl_ext_xilinx.h>
#include <unistd.h>
#include <limits.h>
#include <sys/stat.h>
#include <stdio.h>
#include <ap_int.h>
#include <cstdlib>
#include <ctime>
#include <iterator>
#include <string>
#include <cfloat>
#include <CL/cl_ext.h>


#include "./topl_new.h"
#include "./rmm.h"

using namespace std;
// function for aligning the address
template <typename Tour>
struct aligned_allocator
{
  using value_type = Tour;
  Tour* allocate(std::size_t num)
  {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(Tour)))
      throw std::bad_alloc();
    return reinterpret_cast<Tour*>(ptr);
  }
  void deallocate(Tour* p, std::size_t num)
  {
    free(p);
  }
};

namespace xcl {
static std::vector<cl::Device> get_devices(const std::string& vendor_name) {

    size_t i;
    cl_int err;
    std::vector<cl::Platform> platforms;
    // cl::OCL_CHECK(err, err = cl::Platform::get(&platforms));
    cl::Platform::get(&platforms);
    cl::Platform platform;
    for (i  = 0 ; i < platforms.size(); i++){
        platform = platforms[i];
        // cl::OCL_CHECK(err, std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err));
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err);
        if (platformName == vendor_name){
            std::cout << "Found Platform" << std::endl;
            std::cout << "Platform Name: " << platformName.c_str() << std::endl;
            break;
        }
    }
    if (i == platforms.size()) {
        std::cout << "Error: Failed to find Xilinx platform" << std::endl;
        exit(EXIT_FAILURE);
    }
   
    //Getting ACCELERATOR Devices and selecting 1st such device 
    std::vector<cl::Device> devices;
    // OCL_CHECK(err, err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices));
    platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
    return devices;
}


static std::vector<cl::Device> get_xilinx_devices() 
{
    size_t i;
    cl_int err;
    std::vector<cl::Platform> platforms;
    err = cl::Platform::get(&platforms);
    cl::Platform platform;
    for (i  = 0 ; i < platforms.size(); i++){
        platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err);
        if (platformName == "Xilinx"){
            std::cout << "INFO: Found Xilinx Platform" << std::endl;
            break;
        }
    }
    if (i == platforms.size()) {
        std::cout << "ERROR: Failed to find Xilinx platform" << std::endl;
        exit(EXIT_FAILURE);
    }
   
    //Getting ACCELERATOR Devices and selecting 1st such device 
    std::vector<cl::Device> devices;
    err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
    return devices;
}
   
static char* read_binary_file(const std::string &xclbin_file_name, unsigned &nb) 
{
    if(access(xclbin_file_name.c_str(), R_OK) != 0) {
        printf("ERROR: %s xclbin not available please build\n", xclbin_file_name.c_str());
        exit(EXIT_FAILURE);
    }
    //Loading XCL Bin into char buffer 
    std::cout << "INFO: Loading '" << xclbin_file_name << "'\n";
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);
    return buf;
}

}

namespace rlu::fpga {
    // define a global variable here. must be static
    static cl_int err;
    static cl::Program program;
    static cl::CommandQueue q;
	// Host buffers for data-rearrangement (resize & link once in init bitxm)
    static std::vector<blockvec, aligned_allocator<blockvec>> In_rows;
    static std::vector<blockvec, aligned_allocator<blockvec>> In_rows_snt;
    static std::vector<w1blockvec, aligned_allocator<w1blockvec>> Out_w1bram;
    static std::vector<w3blockvec, aligned_allocator<w3blockvec>> Out_w2bram;
    static std::vector<actvec, aligned_allocator<actvec>> In_actions;
    static std::vector<blockvec, aligned_allocator<blockvec>> In_rewards;
    static std::vector<bsbit, aligned_allocator<bsbit>> In_dones;
    static std::vector<float, aligned_allocator<float>> Out_bias1;
    static std::vector<float, aligned_allocator<float>> Out_bias2;
    static std::vector<float, aligned_allocator<float>> Out_Q;
    static std::vector<float, aligned_allocator<float>> Out_Loss;

    static std::vector<int, aligned_allocator<int>> insert_ind;
    static std::vector<float, aligned_allocator<float>> init_priority;
    static std::vector<int, aligned_allocator<int>> ind_o_out;
    // DDR Bank Pointers (link once in init bitxm)
    static cl_mem_ext_ptr_t InrExt;
    static cl_mem_ext_ptr_t InrExt2;
    static cl_mem_ext_ptr_t InrExt3;
    static cl_mem_ext_ptr_t InrExt4;
    static cl_mem_ext_ptr_t InrExt5;
    static cl_mem_ext_ptr_t OutExt;
    static cl_mem_ext_ptr_t OutExt2;
    static cl_mem_ext_ptr_t OutExt3;
    static cl_mem_ext_ptr_t OutExt4;
    static cl_mem_ext_ptr_t OutExt5;
    static cl_mem_ext_ptr_t OutExt6;

    static cl_mem_ext_ptr_t RepInExt1;
    static cl_mem_ext_ptr_t RepInExt2;
    static cl_mem_ext_ptr_t RepoutExt; 
    // buffers for transferring data
	static cl::Buffer in1_buf;
	static cl::Buffer in2_buf;
	static cl::Buffer in3_buf;
	static cl::Buffer in4_buf;
	static cl::Buffer in5_buf;
	static cl::Buffer out1_buf;
	static cl::Buffer out2_buf;
	static cl::Buffer out3_buf;
	static cl::Buffer out4_buf;
	static cl::Buffer out5_buf;
	static cl::Buffer out6_buf;

	static cl::Buffer insind_buf;
	static cl::Buffer inpn_buf;
	static cl::Buffer out_buf;

    static int insert_signal_in = 1;
    static int update_signal=0;
    static int sample_signal = 0;
    static int load_seed = 0;
    static float gamma=0.5; //actor->gamma?????????????????
    static float alpha=0.1; //tau????????????????
    static int wsync = 1; //wsync mechanism??????????????????????
}


#endif //HIPC21_FPGA_H
