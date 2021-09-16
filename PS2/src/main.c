#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>
#include <image_utils.h>
#include <argument_utils.h>
#include <mpi.h>

// NOTE TO STUDENT:
// The kernels are defined under argument_utils.h
// Take a look at this file to get a feel for how the kernels look.

// Apply convolutional kernel on image data
void applyKernel(pixel **out, pixel **in, unsigned int width, unsigned int height, int *kernel, unsigned int kernelDim, float kernelFactor) {
    unsigned int const kernelCenter = (kernelDim / 2);
    for (unsigned int imageY = 0; imageY < height; imageY++) {
        for (unsigned int imageX = 0; imageX < width; imageX++) {
            unsigned int ar = 0, ag = 0, ab = 0;
            for (unsigned int kernelY = 0; kernelY < kernelDim; kernelY++) {
                int nky = kernelDim - 1 - kernelY;
                for (unsigned int kernelX = 0; kernelX < kernelDim; kernelX++) {
                    int nkx = kernelDim - 1 - kernelX;

                    int yy = imageY + (kernelY - kernelCenter);
                    int xx = imageX + (kernelX - kernelCenter);
                    if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height) {
                        ar += in[yy][xx].r * kernel[nky * kernelDim + nkx];
                        ag += in[yy][xx].g * kernel[nky * kernelDim + nkx];
                        ab += in[yy][xx].b * kernel[nky * kernelDim + nkx];
                    }
                }
            }
            if (ar || ag || ab) {
                ar *= kernelFactor;
                ag *= kernelFactor;
                ab *= kernelFactor;
                out[imageY][imageX].r = (ar > 255) ? 255 : ar;
                out[imageY][imageX].g = (ag > 255) ? 255 : ag;
                out[imageY][imageX].b = (ab > 255) ? 255 : ab;
                out[imageY][imageX].a = 255;
            } else {
                out[imageY][imageX].r = 0;
                out[imageY][imageX].g = 0;
                out[imageY][imageX].b = 0;
                out[imageY][imageX].a = 255;
            }
        }
    }
}

int main(int argc, char **argv) {


    MPI_Init(&argc, &argv);

    int world_sz;
    int world_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &world_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    OPTIONS my_options;
    OPTIONS *options = &my_options;

    if ( world_rank == 0 ) {
        options = parse_args(argc, argv);

        if ( options == NULL )
        {
            fprintf(stderr, "Options == NULL\n");
            exit(1);
        }
    }

    MPI_Bcast(options, sizeof(OPTIONS), MPI_BYTE, 0, MPI_COMM_WORLD);

    if( world_rank > 0 ) {
        options->input = NULL;
        options->output = NULL;
    }

    image_t dummy;
    dummy.rawdata = NULL;
    dummy.data = NULL;

    image_t *image = &dummy;
    image_t *my_image;

    if( world_rank == 0 ) {
        image = loadImage(options->input);
        if (image == NULL) {
            fprintf(stderr, "Could not load bmp image '%s'!\n", options->input);
            freeImage(image);
            abort();
        }
    }

    if ( world_rank == 0 ) {
        printf("Apply kernel '%s' on image with %u x %u pixels for %u iterations\n",
                kernelNames[options->kernelIndex],
                image->width,
                image->height,
                options->iterations);
    }

    // Broadcast image information
    MPI_Bcast(image,            // Send Buffer
            sizeof(image_t),    // Send Count
            MPI_BYTE,           // Send Type
            0,                  // Root
            MPI_COMM_WORLD);    // Communicator


    //////////////////////////////////////////////////////////
    // Calculate how much of the image to send to each rank //
    //////////////////////////////////////////////////////////
    int rows_to_receive[world_sz];
    int bytes_to_transfer[world_sz];
    int displacements[world_sz];
    displacements[0] = 0;

    int rows_per_rank = image->height / world_sz;
    int remainder_rows = image->height % world_sz;

    for(int i = 0; i < world_sz; i++)
    {
        int rows_this_rank = rows_per_rank;

        if ( i < remainder_rows ) {
            rows_this_rank++;
        }

        int bytes_this_rank = rows_this_rank * image->width * sizeof(pixel);

        rows_to_receive[i] = rows_this_rank;
        bytes_to_transfer[i] = bytes_this_rank;

        if(i > 0) {
            displacements[i] = displacements[i - 1] + bytes_to_transfer[i - 1];
        }

    }


    int num_border_rows = (kernelDims[options->kernelIndex] - 1 ) / 2;
    int my_image_height = rows_to_receive[world_rank];

    // TODO: Make space for halo-exchange
    // ------------------------------------------------------------
    // This should include space for the rows that are to be exchanged both
    // at the top and at the bottom of each respective slice.
    my_image = newImage(image->width, my_image_height);



    // Ternary operator
    // Every rank other than 0 are not senders and thus
    // do not need to actually have anything in the send buffer. These
    // get their send buffer pointer set to NULL.
    pixel *image_send_buffer = world_rank == 0 ? image->rawdata : NULL;

    ///////////////////////////////////////////////////////////////////////////
    // TODO: Update the recv buffer pointer.                                 //
    //-----------------------------------------------------------------------//
    // Should point to the start of where this rank's slice of the image     //
    // starts. The topmost and bottom-most rows should not be written by the //
    // scatter operation                                                     //
    ///////////////////////////////////////////////////////////////////////////
    pixel *my_image_slice = my_image->rawdata;

    MPI_Scatterv(image_send_buffer,        // Send Buffer
            bytes_to_transfer,             // Send Counts
            displacements,                 // Displacements
            MPI_BYTE,                      // Send Type
            my_image_slice,                // Recv Buffer
            bytes_to_transfer[world_rank], // Recv Count
            MPI_BYTE,                      // Recv Type
            0,                             // Root
            MPI_COMM_WORLD);               // Communicator

    ///////////////////////////////////////////////////
    // TODO: implement time measurement from here    //
    ///////////////////////////////////////////////////
    double starttime = 0.0f;

    // Here we do the actual computation!
    // image->data is a 2-dimensional array of pixel which is accessed row
    // first ([y][x]) each pixel is a struct of 4 unsigned char for the red,
    // blue and green colour channel
    image_t *processImage = newImage(image->width, my_image->height);

    size_t bytes_to_exchange = num_border_rows * sizeof(pixel) * my_image->width;
    for (unsigned int i = 0; i < options->iterations; i ++) {


        ///////////////////////////
        // TODO: BORDER EXCHANGE //
        ///////////////////////////

        // Apply Kernel
        applyKernel(processImage->data,
                my_image->data,
                my_image->width,
                my_image->height,
                kernels[options->kernelIndex],
                kernelDims[options->kernelIndex],
                kernelFactors[options->kernelIndex]
                );

        swapImage(&processImage, &my_image);

        // Wait until all ranks have done their part before resuming
        MPI_Barrier(MPI_COMM_WORLD);
    }

    freeImage(processImage);
    /////////////////////////////////////////////////////////////////////
    // TODO: Update the "Send Buffer" pointer such that it points      //
    // to the starting location in each respective slice.              //
    /////////////////////////////////////////////////////////////////////

    MPI_Gatherv(my_image->rawdata,         // Send Buffer
            bytes_to_transfer[world_rank], // Send Count
            MPI_BYTE,                      // Send Type
            image->rawdata,                // Recv Buffer
            bytes_to_transfer,             // Recv Counts
            displacements,                 // Recv Displacements
            MPI_BYTE,                      // Recv Type
            0,                             // Root
            MPI_COMM_WORLD);               // Communicator


    //////////////////////////////////////////////
    // TODO: implement time measurement to here //
    //////////////////////////////////////////////
    double spentTime = 0.0f;
    printf("Time spent: %.3f seconds\n", spentTime);


    if ( world_rank == 0) {
        //Write the image back to disk
        if (saveImage(image, options->output) < 1) {
            fprintf(stderr, "Could not save output to '%s'!\n", options->output);
            freeImage(image);
            abort();
        };
    }

    MPI_Finalize();

graceful_exit:
    options->ret = 0;
error_exit:
    if (options->input != NULL)
        free(options->input);
    if (options->output != NULL)
        free(options->output);
    return options->ret;
};
