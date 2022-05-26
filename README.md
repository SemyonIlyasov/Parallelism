# Parallelism
time, ms
maxi = 1000000
time of program working

   //mpi_cuda_nccl.cu\\

Matrix_size = 128

  - 1 - 0.813
  - 2 - 1.704
  - 4 - 1.965
  - 
Matrix_size = 256
  - 1 - 2.906
  - 2 - 3.504
  - 4 - 3.291
  - 
Matrix_size = 512

  - 1 - 7.022
  - 2 - 9.718
  - 4 - 7.591
  - 
Matrix_size = 1024

  - 1 - 38.334
  - 2 - 36.744
  - 4 - 22.146

   //mpi_cuda.cu\\
  
Matrix_size = 128

  - 1 - 1.271
  - 2 - 1.740
  - 4 - 2.150

Matrix_size = 256

  - 1 - 3.701
  - 2 - 4.707
  - 4 - 5.156

Matrix_size = 512

  - 1 - 15.783
  - 2 - 15.903
  - 4 - 16.801
  - 
Matrix_size = 1024

  - 1 - 56.893
  - 2 - 55.665
  - 4 - 51.189
