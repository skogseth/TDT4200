#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>
#include <image_utils.h>
#include <argument_utils.h>
#include <mpi.h>


// The kernels are defined under argument_utils.h
// Apply convolutional kernel on image data
void applyKernel(pixel **out, pixel **in, unsigned int width, unsigned int height, int* kernel, unsigned int kernelDim, float kernelFactor) {
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
    //////////////////////////////////////////////
    // Initialization of MPI and handling input //
    //////////////////////////////////////////////

    // Initialization of MPI and getting key variables
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Handling input using OPTIONS type (root reads)
    OPTIONS my_options;
    OPTIONS *options = &my_options;
    if (world_rank == 0) {
        options = parse_args(argc, argv);
        if (options == NULL) {
            fprintf(stderr, "Options == NULL\n");
            exit(1);
        }
    }

    //Broadcasting options to all processes, and removing uneccesary information for non-root processes
    MPI_Bcast(options, sizeof(OPTIONS), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (world_rank > 0) {
        options->input = NULL;
        options->output = NULL;
    }



    ////////////////////////////////////////
    // Root loads image and broadcasts it //
    ////////////////////////////////////////
    image_t dummy;
    dummy.rawdata = NULL;
    dummy.data = NULL;

    // image->data is a 2-dimensional array of pixels which is accessed row first ([row][col])
    // image->rawdata is a 1-dimensional array pf pixels which is accessed like [row*image->width+col]
    image_t *image = &dummy;
    image_t *my_image;

    // Root loads image and prints information
    if (world_rank == 0) {
        image = loadImage(options->input);
        if (image == NULL) {
            fprintf(stderr, "Could not load bmp image '%s'!\n", options->input);
            freeImage(image);
            abort();
        }
        printf("Apply kernel '%s' on image with %u x %u pixels for %u iterations\n",
                kernelNames[options->kernelIndex],
                image->width,
                image->height,
                options->iterations);
    }

    // Broadcast image to all processes
    MPI_Bcast(image,            // Send Buffer
            sizeof(image_t),    // Send Count
            MPI_BYTE,           // Send Type
            0,                  // Root
            MPI_COMM_WORLD);    // Communicator



    //////////////////////////////////////////////////////////
    // Calculate how much of the image to send to each rank //
    //////////////////////////////////////////////////////////
    int rows_to_receive[world_size];
    int transfer_size[world_size];
    int displacements[world_size];

    // the base number of rows per process and the remainder
    int rows_per_rank = image->height / world_size;
    int remainder_rows = image->height % world_size;

    // Iterate through each process
    // displacements[i] is set by constantly updating current_displacement by the transfer_size of each process
    // rows_to_receive[i] is set to rows_per_rank + 1 for the first processes until all remainder_rows has been distributed
    // the transfer_size is just equal to the number of pixels (rows * cols) multiplied by the size of a pixel
    // (because we send them as bytes)
    int current_displacement = 0;
    for (int i = 0; i < world_size; i++) {
        displacements[i] = current_displacement;
        rows_to_receive[i] = rows_per_rank + (i < remainder_rows ? 1 : 0);
        transfer_size[i] = rows_to_receive[i] * image->width * sizeof(pixel);
        current_displacement += transfer_size[i];
    }
    // I considered doing this calculation only in root and scattering the results to all processes
    // But I think the message passing overhead makes it a worse solution, so I dropped it



    ////////////////////////////////////////////////////////
    // Find border rows and allocate room for local image //
    ////////////////////////////////////////////////////////
    int num_border_rows = (kernelDims[options->kernelIndex] - 1 ) / 2;

    // If process is either the first or last it has one border, otherwise it has 2 (also, only one process => no borders)
    int num_borders;
    if (world_size == 1) num_borders = 0;
    else num_borders = (world_rank == 0 || world_rank == world_size-1) ? 1 : 2;

    // The total number of border rows for each process
    int my_border_rows =  num_borders * num_border_rows;

    // Local image: Includes core image (part of full image owned by a process) and border rows
    my_image = newImage(image->width, rows_to_receive[world_rank] + my_border_rows);

    // Image-data for root process and NULL for non-root processes (since they do not need it)
    pixel *image_data = (world_rank == 0) ? image->rawdata : NULL;

    // Pointer to beginning of core image
    pixel *my_image_slice = (world_rank == 0) ? my_image->rawdata : my_image->rawdata + num_border_rows * image->width;



    ////////////////////////////////////////////
    // Scatter full image to local core image //
    ////////////////////////////////////////////
    MPI_Scatterv(image_data,               // Send Buffer
            transfer_size,                 // Send Counts
            displacements,                 // Displacements
            MPI_BYTE,                      // Send Type
            my_image_slice,                // Recv Buffer
            transfer_size[world_rank],     // Recv Count
            MPI_BYTE,                      // Recv Type
            0,                             // Root
            MPI_COMM_WORLD);               // Communicator



    ///////////////////////////////////
    // Beginning of time measurement //
    ///////////////////////////////////
    double start_time, end_time;
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) start_time = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    // The strategy behind this way of measuring time is explained a bit more at the end of the measurement



    //////////////////////////////////////////////////////
    // Applying kernel to image a given number of times //
    //////////////////////////////////////////////////////
    image_t *process_image = newImage(image->width, my_image->height);

    // Define pointers that point to exchange rows in local image.
    pixel *upper_core_row; // upper row(s) from core image which process will send to upper neighbouring image slice
    pixel *lower_core_row; // lower row(s) from core image which process will send to lower neighbouring image slice
    pixel *upper_border_row; // upper border row(s) for image which process will receive from upper neighbouring image slice
    pixel *lower_border_row; // lower border row(s) for image which process will receive from lower neighbouring image slice

    // Define some variables needed for communication
    int exchange_size = num_border_rows * image->width * sizeof(pixel);
    MPI_Request requests[num_borders+1]; // requests for each message (for both sides), the extra element is there to avoid overflow
    MPI_Request *req_ptr; // keeps track of which request is up next

    // Run border exchange and apply kernel for a given number of iterations
    for (unsigned int i = 0; i < options->iterations; i++) {
        /////////////////////
        // Border exchange //
        /////////////////////

        // All messages are tagged with senders rank
        // All sends/recvs are non-blocking, to avoid deadlocks and uneccesary waiting

        // Request pointer set to point to first request slot
        req_ptr = requests;

        // Exchange rows with upper neighbouring image slice (not needed for root process)
        if (world_rank != 0) {
            upper_core_row = my_image->rawdata + num_border_rows * image->width;
            upper_border_row = my_image->rawdata;
            MPI_Isend(upper_core_row, exchange_size, MPI_BYTE, world_rank-1, world_rank, MPI_COMM_WORLD, req_ptr++);
            MPI_Irecv(upper_border_row, exchange_size, MPI_BYTE, world_rank-1, world_rank-1, MPI_COMM_WORLD, req_ptr++);
        }

        // Exchange rows with lower neighbouring image slice (not needed for last process)
        if (world_rank != world_size-1) {
            lower_core_row = my_image->rawdata + my_image->height * image->width - 2 * num_border_rows * image->width;
            lower_border_row = my_image->rawdata + my_image->height * image->width - num_border_rows * image->width;
            MPI_Isend(lower_core_row, exchange_size, MPI_BYTE, world_rank+1, world_rank, MPI_COMM_WORLD, req_ptr++);
            MPI_Irecv(lower_border_row, exchange_size, MPI_BYTE, world_rank+1, world_rank+1, MPI_COMM_WORLD, req_ptr++);
        }

        // Every process now makes sure that their border exchanges are finished
        for (int i = 0; i < num_borders; i++) MPI_Wait(&requests[i], MPI_STATUS_IGNORE);

        // We then wait for all processes.
        // This makes output consistently good, but I'm not entirely sure why it's needed.
        // I thought every process would be fine applying the kernel so long as their own exchanges were succesfull.
        // However, if I remove this the output __sometimes__ have symptoms of bad border exchange.
        MPI_Barrier(MPI_COMM_WORLD);


        /////////////////////
        // Applying kernel //
        /////////////////////
        applyKernel(process_image->data,
                my_image->data,
                my_image->width,
                my_image->height,
                kernels[options->kernelIndex],
                kernelDims[options->kernelIndex],
                kernelFactors[options->kernelIndex]
                );

        // Swaps the two images to "save changes"
        swapImage(&process_image, &my_image);

        // Wait until all ranks have done their part before resuming
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Process image only used for kernel computation, memory can now be freed
    freeImage(process_image);



    /////////////////////////////////////////
    // Gather image data from all processes//
    /////////////////////////////////////////
    MPI_Gatherv(my_image_slice,            // Send Buffer
            transfer_size[world_rank],     // Send Count
            MPI_BYTE,                      // Send Type
            image_data,                    // Recv Buffer
            transfer_size,                 // Recv Counts
            displacements,                 // Recv Displacements
            MPI_BYTE,                      // Recv Type
            0,                             // Root
            MPI_COMM_WORLD);               // Communicator

    // Local image is no longer needed
    freeImage(my_image);



    /////////////////////////////
    // End of time measurement //
    /////////////////////////////
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        end_time = MPI_Wtime();
        printf("Time spent: %.3f seconds\n", end_time - start_time);
    }
    /*
    This seems, in my opinion, to be the best way to measure the time the program takes:
    All processes are aligned at the start, root begins timing, everyone starts. When everyone is done root stops timing.
    A bit too much overhead, but what can you do. I allocate space for the variable before timing, hopefully that helps a bit.
    */



    /////////////////////////////////////////
    // Save image (root), end MPI and exit //
    /////////////////////////////////////////
    if (world_rank == 0) {
        //Write the image back to disk
        if (saveImage(image, options->output) < 1) {
            fprintf(stderr, "Could not save output to '%s'!\n", options->output);
            abort();
        };

        // Root has saved the image so can now free the memory
        freeImage(image);
    }

    MPI_Finalize();

graceful_exit:
    options->ret = 0;
error_exit:
    if (options->input != NULL) free(options->input);
    if (options->output != NULL) free(options->output);
    return options->ret;
};
