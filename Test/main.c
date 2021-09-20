#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define DEFAULT_ROWS 4
#define DEFAULT_COLS 4


void print_matrix_c(char* matrix, int rows, int cols);

int main(int argc, char** argv){
    /* INITIALIZATION */
    int world_size, world_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if(world_rank == 0){
        printf("Booting up\n");
        printf("Number of processes: %d\n", world_size);
        printf("Number of arguments: %d\n", argc);
        printf("\n");
    }

    int rows = argc > 1 ? atoi(argv[1]) : DEFAULT_ROWS;
    int cols = argc > 2 ? atoi(argv[2]) : DEFAULT_COLS;

    int rows_per_process = rows / world_size;
    int rows_remaining = rows % world_size;
    int local_rows = rows_per_process + (world_rank < rows_remaining ? 1 : 0);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Process %d has %d local rows\n", world_rank, local_rows);
    MPI_Barrier(MPI_COMM_WORLD);


    /* CREATION AND DISTRIBUTION OF INITIAL DATA */
    char* data = (world_rank == 0) ? (char*)malloc(rows*cols*sizeof(char)) : NULL;
    char* local_data = (char*)malloc(local_rows*cols*sizeof(char));
    int send_counts[world_size];
    int displacements[world_size];

    if(world_rank == 0){
        int current_displacement = 0;
        for(int r = 0; r < world_size; r++){
            send_counts[r] = (rows_per_process + (r < rows_remaining ? 1 : 0))*cols;
            displacements[r] = current_displacement;
            current_displacement += send_counts[r];
        }

        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                data[cols*i+j] = 'A' + cols*i + j;
            }
        }
        print_matrix_c(data, rows, cols);
    }

    MPI_Scatterv(data, send_counts, displacements, MPI_CHAR, local_data, local_rows*cols, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    print_matrix_c(local_data, local_rows, cols);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Status status;
    if(world_rank == 0) MPI_Send(local_data, local_rows*cols, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    if(world_rank == 1) MPI_Recv(local_data, local_rows*cols, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);

    MPI_Barrier(MPI_COMM_WORLD);
    print_matrix_c(local_data, local_rows, cols);
    MPI_Barrier(MPI_COMM_WORLD);


    /* FINALIZATION */
    free(data);
    free(local_data);
    MPI_Finalize();
    return 0;
}


void print_matrix_c(char* matrix, int rows, int cols){
    printf("{\n");
    for(int i = 0; i < rows; i++){
        printf("\t{");
        for(int j = 0; j < cols-1; j++){
            printf("%c,",matrix[cols*i+j]);
        }
        printf("%c}\n",matrix[cols*i+cols-1]);
    }
    printf("}\n");
}
