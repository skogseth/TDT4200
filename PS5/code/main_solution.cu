#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <signal.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

typedef struct pixel_struct {
	unsigned char r;
	unsigned char g;
	unsigned char b;
	unsigned char a;
} pixel;

#define cudaErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


//--------------------------------------------------------------------------------------------------
//--------------------------bilinear interpolation--------------------------------------------------
//--------------------------------------------------------------------------------------------------
__device__ void bilinear(pixel* Im, float row, float col, pixel* pix, int width, int height) {
	int cm, cn, fm, fn;
	double alpha, beta;

	cm = (int)ceil(row);
	fm = (int)floor(row);
	cn = (int)ceil(col);
	fn = (int)floor(col);
	alpha = ceil(row) - row;
	beta = ceil(col) - col;

	pix->r = (unsigned char)(alpha*beta*Im[fm*width+fn].r
			+ (1-alpha)*beta*Im[cm*width+fn].r
			+ alpha*(1-beta)*Im[fm*width+cn].r
			+ (1-alpha)*(1-beta)*Im[cm*width+cn].r );
	pix->g = (unsigned char)(alpha*beta*Im[fm*width+fn].g
			+ (1-alpha)*beta*Im[cm*width+fn].g
			+ alpha*(1-beta)*Im[fm*width+cn].g
			+ (1-alpha)*(1-beta)*Im[cm*width+cn].g );
	pix->b = (unsigned char)(alpha*beta*Im[fm*width+fn].b
			+ (1-alpha)*beta*Im[cm*width+fn].b
			+ alpha*(1-beta)*Im[fm*width+cn].b
			+ (1-alpha)*(1-beta)*Im[cm*width+cn].b );
	pix->a = 255;
}


__global__ void bilinear_kernel(pixel* d_pixels_in, pixel* d_pixels_out, int in_width, int in_height, int out_width, int out_height) {
	pixel new_pixel;
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < out_height && j < out_width) {
		float row = i * (in_height-1)/(float)out_height;
		float col = j * (in_width-1)/(float)out_width;
		bilinear(d_pixels_in, row, col, &new_pixel, in_width, in_height);
		d_pixels_out[i*out_width+j] = new_pixel;
	}
}



int main(int argc, char** argv) {
	stbi_set_flip_vertically_on_load(true);
	stbi_flip_vertically_on_write(true);

	int in_width;
	int in_height;

	pixel* h_pixels_in;
	int channels;
	h_pixels_in = (pixel*) stbi_load(argv[1], &in_width, &in_height, &channels, STBI_rgb_alpha);
	if (h_pixels_in == NULL) exit(1);
	printf("Image dimensions: %dx%d\n", in_width, in_height);

	double scale_x = argc > 2 ? atof(argv[2]) : 2;
	double scale_y = argc > 3 ? atof(argv[3]) : 8;

	int out_width = in_width * scale_x;
	int out_height = in_height * scale_y;

	pixel* h_pixels_out = (pixel*) malloc(sizeof(pixel)*out_width*out_height);

	pixel* d_pixels_in;
	pixel* d_pixels_out;

	cudaError_t error;
	error = cudaMalloc(&d_pixels_in, sizeof(pixel)*in_width*in_height);
	if (error != cudaSuccess) printf("memory allocation failed (device in)\n");
	error = cudaMalloc(&d_pixels_out, sizeof(pixel)*out_width*out_height);
	if (error != cudaSuccess) printf("memory allocation failed (device out)\n");

   	cudaEvent_t start_transfer, stop_transfer;
       	cudaEventCreate(&start_transfer);
        cudaEventCreate(&stop_transfer);
	cudaEventRecord(start_transfer);

	error = cudaMemcpy(d_pixels_in, h_pixels_in, sizeof(pixel)*in_width*in_height, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) printf("memcpy from host to device failed (input)\n");
	error = cudaMemcpy(d_pixels_out, h_pixels_out, sizeof(pixel)*out_width*out_height, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) printf("memcpy from host to device failed (output)\n");

	dim3 blockSize(32,32);
	dim3 gridSize(out_width/blockSize.x, out_height/blockSize.y);

   	cudaEvent_t start, stop;
    	cudaEventCreate(&start);
        cudaEventCreate(&stop);
	cudaEventRecord(start);

	bilinear_kernel<<<gridSize, blockSize>>>(d_pixels_in, d_pixels_out, in_width, in_height, out_width, out_height);

	cudaEventRecord(stop);
	cudaDeviceSynchronize();

	error = cudaGetLastError();
	if (error != cudaSuccess) printf("bilinear_kernel failed\n%s\n", cudaGetErrorString(error));

	cudaDeviceSynchronize();
	cudaEventSynchronize(stop);
	float spentTime = 0.0;
	cudaEventElapsedTime(&spentTime, start, stop);
	printf("Time spent %.3f seconds\n", spentTime/1000);

	error = cudaMemcpy(h_pixels_out, d_pixels_out, sizeof(pixel)*out_width*out_height, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) printf("memcpy from device to host failed\n%s\n", cudaGetErrorString(error));

	cudaEventRecord(stop_transfer);
	cudaEventSynchronize(stop_transfer);
	float spentTimeTransfer = 0.0;
	cudaEventElapsedTime(&spentTimeTransfer, start_transfer, stop_transfer);
	printf("Time spent including transfer: %.3f seconds\n", spentTimeTransfer/1000);

	// Writes the host-side data to the output file.
	stbi_write_png("output.png", out_width, out_height, STBI_rgb_alpha, h_pixels_out, sizeof(pixel) * out_width);

	free(h_pixels_in);
	free(h_pixels_out);
	cudaFree(d_pixels_in);
	cudaFree(d_pixels_out);

	return 0;
}
