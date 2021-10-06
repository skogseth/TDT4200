#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define DEFAULT_ROWS 8
#define DEFAULT_COLS 8

// Global variables for MPI rank and size of communicator
int world_size, world_rank;


void print_matrix_d(double *matrix, int rows, int cols){
    printf("{\n");
    for (int i = 0; i < rows; i++) {
        printf("\t{");
        for (int j = 0; j < cols-1; j++) {
            printf("%f,", matrix[cols*i+j]);
        }
        printf("%f}\n", matrix[cols*i+cols-1]);
    }
    printf("}\n");
}

void divide_matrix(int rows, int cols, int process_rows[], int process_cols[]){
    double x_d = log2(world_size);
    if (floor(x_d) != x_d) {
        printf("Number of processes is not a power of 2\n");
        exit(1);
    }
    int x = (int)floor(x_d);

    int m = (int)pow(2, x/2 + x%2);
    int n = (int)pow(2, x/2);

    printf("Height will be split by %d processes\n", m);
    printf("Width will be split by %d processes\n", n);

    int rows_per_process = rows / m;
    int cols_per_process = cols / n;
    int rows_remaining = rows % m;
    int cols_remaining = cols % n;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            process_rows[i * n + j] = rows_per_process + (i < rows_remaining ? 1 : 0);
            process_cols[i * n + j] = cols_per_process + (j < cols_remaining ? 1 : 0);
        }
    }
}


int main(int argc, char **argv){
    //////////////////////////
    // Initialization of MPI//
    //////////////////////////
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if(world_rank == 0){
        printf("Booting up\n");
        printf("Number of processes: %d\n", world_size);
        printf("Number of arguments: %d\n", argc);
        printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);



    ////////////////////
    // Read arguments //
    ////////////////////
    int rows = argc > 1 ? atoi(argv[1]) : DEFAULT_ROWS;
    int cols = argc > 2 ? atoi(argv[2]) : DEFAULT_COLS;
    if (world_rank == 0) printf("rows = %d, cols = %d\n", rows, cols);


    ///////////////////
    // Divide matrix //
    ///////////////////
    int process_rows[world_size], process_cols[world_size];
    int local_rows, local_cols;

    if (world_rank == 0) divide_matrix(rows, cols, process_rows, process_cols);

    MPI_Scatter(process_rows, 1, MPI_INT, &local_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(process_cols, 1, MPI_INT, &local_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("[%d] Local rows is %d and local cols is %d\n", world_rank, local_rows, local_cols);



    //////////////////////////////////
    // Calculate data transfer size //
    //////////////////////////////////
    int send_counts[world_size];
    int displacements[world_size];
    if (world_rank == 0) {
        int current_displacement = 0;
        for (int i = 0; i < world_size; i++) {
            displacements[i] = current_displacement;
            send_counts[i] = process_rows[i] * process_cols[i];
            current_displacement += send_counts[i];
        }
    }



    ////////////////////////////
    // Allocate room for data //
    ////////////////////////////
    double *data = (world_rank == 0) ? (double*)malloc(rows*cols*sizeof(char)) : NULL;
    double *local_data = (double*)malloc(local_rows*local_cols*sizeof(char));



    //////////////////
    // Finishing up //
    //////////////////
    if (world_rank == 0) free(data);
    free(local_data);
    MPI_Finalize();
    return 0;
}
