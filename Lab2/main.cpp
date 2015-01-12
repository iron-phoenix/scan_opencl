#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <assert.h>

#define BLOCK_SIZE 256

struct scanner {
	scanner(cl::Context& context, cl::Program& program, cl::CommandQueue& queue) :
		context(context),
		program(program),
		queue(queue)
	{
		scan_kernel = cl::Kernel(program, "scan_blelloch");
		add_kernel = cl::Kernel(program, "blocks_sum");
	}

	void scan(cl::Buffer const & buffer_input, cl::Buffer const & buffer_output, size_t const data_size, size_t const block_size) {
		size_t range = round_block_size(data_size, block_size);
		cl::Buffer buffer_block(context, CL_MEM_READ_WRITE, sizeof(float) * range / block_size);
		if (data_size <= block_size) {
			cl::KernelFunctor scan_b(scan_kernel, queue, cl::NullRange, cl::NDRange(range), cl::NDRange(block_size));
			cl::Event event = scan_b(buffer_input, buffer_output, cl::__local(sizeof(float) * block_size), buffer_block, data_size);
			event.wait();
		} else {
			cl::Buffer buffer_scan(context, CL_MEM_READ_WRITE, sizeof(float) * data_size);
			cl::Buffer buffer_add(context, CL_MEM_READ_WRITE, sizeof(float) * range / block_size);
			cl::KernelFunctor scan_f(scan_kernel, queue, cl::NullRange, cl::NDRange(range), cl::NDRange(block_size));
			cl::Event event = scan_f(buffer_input, buffer_scan, cl::__local(sizeof(float) * block_size), buffer_block, data_size);
			event.wait();
			scan(buffer_block, buffer_add, range / block_size, block_size);
			cl::KernelFunctor add(add_kernel, queue, cl::NullRange, cl::NDRange(range), cl::NDRange(block_size));
			event = add(buffer_scan, buffer_output, buffer_add);
			event.wait();
		}
	}
private:
	cl::Context context;
	cl::Program program;
	cl::CommandQueue queue;

	cl::Kernel scan_kernel;
	cl::Kernel add_kernel;

	static size_t round_block_size(size_t n, size_t block_size) {
		size_t result = n;
		if (n % block_size != 0) result += (block_size - n % block_size);
		return result;
	}
};

cl::Platform get_platform() {
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0){
		std::cout << " No platforms found. Check OpenCL installation!" << std::endl;
		exit(1);
	}
	cl::Platform platform = all_platforms[1];
	std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
	return platform;
}

cl::Device get_device(cl::Platform& platform) {
	std::vector<cl::Device> all_devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0){
		std::cout << " No devices found. Check OpenCL installation!" << std::endl;
		exit(1);
	}
	cl::Device device = all_devices[0];
	std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
	return device;
}

cl::Program get_program(cl::Device& device, cl::Context& context, std::string const & program_file_name) {
	std::ifstream kernel_file(program_file_name);
	std::string kernel_code(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));
	cl::Program::Sources sources(1, { kernel_code.c_str(), kernel_code.length() });

	cl::Program program(context, sources);
	if (program.build({ device }) != CL_SUCCESS){
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		exit(1);
	}
	return program;
}

void read_data(std::ifstream& in, std::vector<float>& data, size_t size) {
	data.resize(size);
	for (size_t i = 0; i < size; ++i)
		in >> data[i];
}

void write_data(std::ofstream& out, std::vector<float> const & data) {
	out << std::fixed << std::setprecision(3);
	for (size_t i = 0; i < data.size(); ++i)
		out << data[i] << " ";
	out << std::endl;
}



void main_program(cl::Device& device, cl::Context& context, cl::Program& program) {
	size_t size = 0;
	size_t const block_size = BLOCK_SIZE;

	std::ifstream in("input.txt");
	std::ofstream out("output.txt");
	in >> size;

	std::vector<float> input;
	std::vector<float> output(size);

	read_data(in, input, size);

	cl::CommandQueue queue(context, device);

	cl::Buffer buffer_input(context, CL_MEM_READ_ONLY, sizeof(float) * input.size());
	cl::Buffer buffer_output(context, CL_MEM_READ_ONLY, sizeof(float) * input.size());

	queue.enqueueWriteBuffer(buffer_input, CL_TRUE, 0, sizeof(float) * input.size(), &input[0]);
	queue.finish();

	scanner s(context, program, queue);
	s.scan(buffer_input, buffer_output, size, block_size);

	queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, sizeof(float) * input.size(), &output[0]);

	write_data(out, output);
}

int main()
{
	try {
		cl::Platform default_platform = get_platform();
		cl::Device default_device = get_device(default_platform);
		cl::Context context({ default_device });
		cl::Program program = get_program(default_device, context, "scan.cl");

		main_program(default_device, context, program);
	}
	catch (std::exception &e) {
		std::cout << std::endl << e.what() << std::endl;
	}


	return 0;
}