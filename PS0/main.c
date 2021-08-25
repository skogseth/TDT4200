#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

//#include <windows.h>
//#include <magick_wand.h>

typedef struct{
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
} pixel;

int main(int argc, char** argv)
{
    stbi_set_flip_vertically_on_load(true);
	stbi_flip_vertically_on_write(true);

	int width;
	int height;
	int channels;
    unsigned char* char_pixels_1 = stbi_load(argv[1], &width, &height, &channels, STBI_rgb_alpha);
    unsigned char* char_pixels_2 = stbi_load(argv[2], &width, &height, &channels, STBI_rgb_alpha);

    printf("height:% d, width: %d\n", height, width);
    if (char_pixels_1 == NULL || char_pixels_2 == NULL)
    {
        exit(1);
    }

    //TODO 2 - typecast pointer
    pixel* pixels_1;
    pixel* pixels_2;

    //TODO 3 - malloc
    pixel* pixels_out = (pixel *) char_pixels_1;

    //TODO 4 - loop
    //Write your loop here

    stbi_write_png("output.png", width, height, STBI_rgb_alpha, pixels_out, sizeof(pixel) * width);

    //TODO 5 - free

    return 0;
}
