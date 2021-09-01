


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <signal.h>
#include "mpi.h"

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



//--------------------------------------------------------------------------------------------------
//--------------------------bilinear interpolation--------------------------------------------------
//--------------------------------------------------------------------------------------------------
void bilinear(pixel* Im, float row, float col, pixel* pix, int width, int height)
{
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
//---------------------------------------------------------------------------

//Helper function to locate the source of errors
void
SEGVFunction( int sig_num)
{
	printf ("\n Signal %d received\n",sig_num);
	exit(sig_num);
}

int main(int argc, char** argv)
{
	signal(SIGSEGV, SEGVFunction);
	stbi_set_flip_vertically_on_load(true);
	stbi_flip_vertically_on_write(true);

//TODO 1 - init
    int comm_size;
    int rank;
//TODO END


	pixel* pixels_in;

	int in_width;
	int in_height;
	int channels;


//TODO 2 - broadcast
	pixels_in = (pixel *) stbi_load(argv[1], &in_width, &in_height, &channels, STBI_rgb_alpha);
	if (pixels_in == NULL) {
		exit(1);
	}
	printf("Image dimensions: %dx%d\n", in_width, in_height);
//TODO END


	double scale_x = argc > 2 ? atof(argv[2]): 2;
	double scale_y = argc > 3 ? atof(argv[3]): 8;

	int out_width = in_width * scale_x;
	int out_height = in_height * scale_y;
	
//TODO 3 - partitioning
	int local_width = in_width;
	int local_height = in_height;

	int local_out_width = out_width;
	int local_out_height = out_height;

	pixel* local_out = (pixel *) malloc(sizeof(pixel) * local_out_width * local_out_height);
//TODO END


//TODO 4 - computation
	for(int i = 0; i < local_out_height; i++) {
		for(int j = 0; j < local_out_width; j++) {
			pixel new_pixel;

			float row = i * (in_height-1) / (float)out_height;
			float col = j * (in_width-1) / (float)out_width;

			bilinear(pixels_in, row, col, &new_pixel, in_width, in_height);

			local_out[i*out_width+j] = new_pixel;
		}
	}
//TODO END


//TODO 5 - gather
	pixel* pixels_out = local_out;
	stbi_write_png("output.png", out_width, out_height, STBI_rgb_alpha, pixels_out, sizeof(pixel) * out_width);
//TODO END


//TODO 1 - init
//TODO END
	return 0;
}
