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
void applyKernel(pixel** out, pixel** in, unsigned int width, unsigned int height, int* kernel, unsigned int kernelDim, float kernelFactor) {
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


int main(int argc, char** argv) {
    //////////////////////////////////////////////
    // Initialization of MPI and handling input //
    //////////////////////////////////////////////

    // Initialization MPI
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Handling input using OPTIONS type (root reads)
    OPTIONS my_options;
    OPTIONS* options = &my_options;
    if(world_rank == 0){
        options = parse_args(argc, argv);
        if(options == NULL){
            fprintf(stderr, "Options == NULL\n");
            exit(1);
        }
    }

    //Broadcasting options to all processes, and removing uneccesary information for non-root processes
    MPI_Bcast(options, sizeof(OPTIONS), MPI_BYTE, 0, MPI_COMM_WORLD);
    if(world_rank > 0){
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
    image_t* image = &dummy;
    image_t* my_image;

    // Root: Load image and print information
    if(world_rank == 0){
        image = loadImage(options->input);
        if(image == NULL){
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

    // Broadcast image information
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

    int rows_per_rank = image->height / world_size;
    int remainder_rows = image->height % world_size;

    int current_displacement = 0;
    for(int i = 0; i < world_size; i++){
        displacements[i] = current_displacement;
        rows_to_receive[i] = rows_per_rank + (i < remainder_rows ? 1 : 0);
        transfer_size[i] = rows_to_receive[i] * image->width * sizeof(pixel);
        current_displacement += transfer_size[i];
    }

    int my_image_height = rows_to_receive[world_rank];



    /////////////////////////////////////////////////////////////////////////////////////
    // Allocate room for local image slice, and scatter core image (not including halo) //
    /////////////////////////////////////////////////////////////////////////////////////
    int num_border_rows = (kernelDims[options->kernelIndex] - 1 ) / 2;
    int my_border_rows;
    // If process is either the first or last it needs one border row, otherwise it needs 2 (also, only one process -> no border rows)
    if(world_size == 1) my_border_rows = 0;
    else my_border_rows = ( (world_rank == 0 || world_rank == world_size-1) ? 1 : 2 ) * num_border_rows;
    my_image = newImage(image->width, my_image_height + my_border_rows);

    // Image-data for root process and NULL for non-root processes (since they do not need it)
    pixel* image_data = (world_rank == 0) ? image->rawdata : NULL;
    // local image slice: if root -> point at first pixel, if not -> point at second row (first core row)
    pixel* my_image_slice = (world_rank == 0) ? my_image->rawdata : my_image->rawdata + num_border_rows * image->width;

    MPI_Scatterv(image_data,               // Send Buffer
            transfer_size,                 // Send Counts
            displacements,                 // Displacements
            MPI_BYTE,                      // Send Type
            my_image_slice,                // Recv Buffer
            transfer_size[world_rank],     // Recv Count
            MPI_BYTE,                      // Recv Type
            0,                             // Root
            MPI_COMM_WORLD);               // Communicator



    //////////////////////////////////////////////////////////////////////////////////
    // Taking a break to print information about the rows allocated to each process //
    //////////////////////////////////////////////////////////////////////////////////
    MPI_Barrier(MPI_COMM_WORLD);
    if(world_rank == 0){
        printf("Total rows: %d\n", image->height);
        printf("Core rows (border rows) => rows\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    wait(100);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Process %d: %d (%d) ==> %d\n", world_rank, my_image_height, my_border_rows, my_image->height);
    MPI_Barrier(MPI_COMM_WORLD);
    ///////////////////////////////////////////////////////////////////////////////////////////



    ///////////////////////////////////
    // Beginning of time measurement //
    ///////////////////////////////////
    double start_time = MPI_Wtime();


    /////////////////
    // Computation //
    /////////////////
    image_t* process_image = newImage(image->width, my_image->height);

    // Define pointers that point to exchange rows
    pixel* upper_personal_row;
    pixel* lower_personal_row;
    pixel* upper_border_row;
    pixel* lower_border_row;

    // Define some variables needed for communication
    int exchange_size = num_border_rows * image->width * sizeof(pixel);
    int num_exchanges = my_border_rows/num_border_rows;
    MPI_Request requests[num_exchanges+1]; // last element is just an overflow buffer
    MPI_Request* req_ptr;
    for(unsigned int i = 0; i < options->iterations; i++){
        // Border exchange -  0 marks exchanging up, 1 marks exchanging down
        req_ptr = requests;
        if(world_rank != 0){
            upper_personal_row = my_image->rawdata + num_border_rows * image->width;
            upper_border_row = my_image->rawdata;
            MPI_Isend(upper_personal_row, exchange_size, MPI_BYTE, world_rank-1, world_rank, MPI_COMM_WORLD, req_ptr++);
            MPI_Irecv(upper_border_row, exchange_size, MPI_BYTE, world_rank-1, world_rank-1, MPI_COMM_WORLD, req_ptr++);
        }
        if(world_rank != world_size-1){
            lower_personal_row = my_image->rawdata + my_image->height * image->width - 2 * num_border_rows * image->width;
            lower_border_row = my_image->rawdata + my_image->height * image->width - num_border_rows * image->width;
            MPI_Isend(lower_personal_row, exchange_size, MPI_BYTE, world_rank+1, world_rank, MPI_COMM_WORLD, req_ptr++);
            MPI_Irecv(lower_border_row, exchange_size, MPI_BYTE, world_rank+1, world_rank+1, MPI_COMM_WORLD, req_ptr++);
        }
        for(int i = 0; i < num_exchanges; i++) MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
        MPI_Barrier(MPI_COMM_WORLD);

        // Apply Kernel
        applyKernel(process_image->data,
                my_image->data,
                my_image->width,
                my_image->height,
                kernels[options->kernelIndex],
                kernelDims[options->kernelIndex],
                kernelFactors[options->kernelIndex]
                );

        // Save changes, basically
        swapImage(&process_image, &my_image);

        // Wait until all ranks have done their part before resuming
        MPI_Barrier(MPI_COMM_WORLD);
    }

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

    freeImage(my_image);


    /////////////////////////////
    // End of time measurement //
    /////////////////////////////
    double spent_time = MPI_Wtime() - start_time;
    if(world_rank == 0) printf("Time spent: %.3f seconds\n", spent_time);


    //////////////////////////////////
    // Save image, end MPI and exit //
    //////////////////////////////////
    if(world_rank == 0){
        //Write the image back to disk
        if(saveImage(image, options->output) < 1){
            fprintf(stderr, "Could not save output to '%s'!\n", options->output);
            freeImage(image);
            abort();
        };
    }

    MPI_Finalize();

graceful_exit:
    options->ret = 0;
error_exit:
    if(options->input != NULL) free(options->input);
    if(options->output != NULL) free(options->output);
    return options->ret;
};
