##Part A
###Task1
Convert the functions to __global__ or __device__ functions

###Task2
Setup and transfer data where the respective TODO notes are.

###Task3
Change the functions to run correctly.

###Task4
Launch the kernel and ensure that it works

##Part B

###Task1
Take a look at the ncu-compute-workload-analysis.sh script and ensure that you understand what it does.
This uses the Occupancy section in the NCU profile report.

###Task2
run the script and describe the consequences of the output. What information does this give us to improve the program? 
Hint: Is there a significant difference between the theoretical maximum occupancy and the achieved occupancy? What can we do to change this?

###Task3
Use the CUDA Occupancy API to find a block size which gives theoretical maximum occupancy

###Task 4
Compare the results from before and after

##Part C
###Task 1
Implement static Shared memory, assuming some fixed number of feature lines. Which variables in the program can potentially be placed in shared memory to speed up the application? Why?

###Task 2
Profile the performance before and after, how has it changed?
Use the ncu-memory-workload-analysis.sh script and ensure that you understand what it does.
This uses the MemoryWorkloadAnalysis section in the NCU profile report.

###Task 3
Implement dynamic shared memory, to accept any number of feature lines.

###Task 4
Ensure that it works and compare the results.

##Part D
###Task 1
Use the nsys-profile.sh (and check the command in the script) to see the number of API calls (cudaMalloc, cudaMemcpy, etc.) and see what consumes the most time.
Is there a way to reduce the number of calls? Change the code so that you reduce the number of CUDA API calls to cudaMalloc and cudaMemcpy. 

Hint1: You may need to move code between doMorph() and main().
Hint2: Some of the API calls do not need to be run for every iteration of the loop, which ones? Can we move these to only be called once?
Hint3: Some of the data is not necessary to copy back to host. Remove the lines of code that copy back this data.

###Task 2
Profile the program and see how it changed the performance.

###Task 3
Implement CUDA Streams to asynchronous data handling.

###Task 4
Profile the program and see how it changed the performance.


