#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>


// The cuda kernel
__global__ void quamsim_kernel(const float *qstates,const float *qbit_gate, float *q_out, int n_size,int q_index ) {


  int i = blockIdx.x* blockDim.x + threadIdx.x;
  int temp = 1<<q_index;
  int i_opp = i ^ (temp);

  if(i<n_size){
    if((i_opp & (temp))){
      // int src1 = (i%(1<<(q_index-1)))+((1<<q_index)*(i/(1<<(q_index-1))));
      // int src2 = src1+(1<<(q_index-1));
      // int dest1 = src1;
      // int dest2 = src2;
      q_out[i] = qbit_gate[0] * qstates[i] + qbit_gate[1] * qstates[i_opp];
      q_out[i_opp] = qbit_gate[2] * qstates[i] + qbit_gate[3] * qstates[i_opp];
    }
  }
}


int main(int argc, char *argv[]) {

  // Read the inputs from command line

  char *trace_file;
  trace_file = argv[1];

  std::ifstream file(trace_file);

  float gate1[4], gate2[4], gate3[4], gate4[4], gate5[4], gate6[4];
  char space;

  for(int i = 0; i<4; i++){
      
          file >> gate1[i];
      
  }

  // file>> space;
  for(int i = 0; i<4; i++){
      
          file >> gate2[i];
      
  }

  // file>> space;
  for(int i = 0; i<4; i++){
      
          file >> gate3[i];
      
  }
  // file>> space;
  for(int i = 0; i<4; i++){
      
          file >> gate4[i];
      
  }
  // file>> space;
  for(int i = 0; i<4; i++){
      
          file >> gate5[i];
      
  }
  // file>> space;
  for(int i = 0; i<4; i++){
      
          file >> gate6[i];
      
  }
  // file>>space;

  std::vector<float> states;
  
  float instate;
  while(file>>instate){
      states.push_back(instate);
  }

  int t1,t2,t3,t4,t5,t6;
  int t[6];
  for(int j = 5; j>=0 ; j--){
    t[j] = states.back();
    states.pop_back();
  }
  // states.pop_back();
  int vector_size = states.size();
  // for(int i = 0; i<vector_size; i++){
  //   std::cout<< std::fixed << std::setprecision(3) << states[i] << std::endl;
  // }
  // std::cout << states << std::endl;
  // std::cout << t[0] << t[1] << t[2] << t[3] <<t[4] <<t[5] << std::endl;
  t1 = t[0];
  t2 =t[1];
  t3 = t[2];
  t4 = t[3];
  t5 = t[4];
  t6 = t[5];
  

  unsigned long long size = vector_size * sizeof(float);
  unsigned long long size_gate = 4 * sizeof(float);
  // int t_size = 6 * sizeof(int);
  // Allocate/move data using cudaMalloc and cudaMemCpy

  
  // std::cout << "line 100" << std::endl;
  float *host_states = (float*)malloc(size);
  float *host_out = (float*)malloc(size);
  float *host_gate1 = (float*)malloc(size_gate);
  float *host_gate2 = (float*)malloc(size_gate);
  float *host_gate3 = (float*)malloc(size_gate);
  float *host_gate4 = (float*)malloc(size_gate);
  float *host_gate5 = (float*)malloc(size_gate);
  float *host_gate6 = (float*)malloc(size_gate);
  

  // std::cout << "line 104" << std::endl;
  for (int i = 0; i < vector_size; ++i)
  {
      host_states[i] = states[i];
  }

  for (int i = 0; i < 4; i++) 
  {
      host_gate1[i]= gate1[i];
  }
   for (int i = 0; i < 4; i++) 
  {
      host_gate2[i]= gate2[i];
  }
   for (int i = 0; i < 4; i++) 
  {
      host_gate3[i]= gate3[i];
  }
   for (int i = 0; i < 4; i++) 
  {
      host_gate4[i]= gate4[i];
  }
   for (int i = 0; i < 4; i++) 
  {
      host_gate5[i]= gate5[i];
  }
   for (int i = 0; i < 4; i++) 
  {
      host_gate6[i]= gate6[i];
  }


  // std::cout << "line 113" << std::endl;
  //allocating the memory in GPU
  
  float *d_states, *d_gate1, *d_out1;
  float  *d_gate2, *d_out2;
  float  *d_gate3, *d_out3;
  float  *d_gate4, *d_out4;
  float  *d_gate5, *d_out5;
  float  *d_gate6, *d_result;

  
  cudaMalloc((void**)&d_states, size);
  cudaMalloc((void**)&d_gate1, size_gate);
  cudaMalloc((void**)&d_out1, size);
  cudaMalloc((void**)&d_gate2, size_gate);
  cudaMalloc((void**)&d_out2, size);
  cudaMalloc((void**)&d_gate3, size_gate);
  cudaMalloc((void**)&d_out3, size);
  cudaMalloc((void**)&d_gate4, size_gate);
  cudaMalloc((void**)&d_out4, size);
  cudaMalloc((void**)&d_gate5, size_gate);
  cudaMalloc((void**)&d_out5, size);
  cudaMalloc((void**)&d_gate6, size_gate);
  cudaMalloc((void**)&d_result, size);
  

  
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // std::cout << "line 142" << std::endl;
  // cudaMemcpy(d_qubit,t, t_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_states, host_states, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_gate1, gate1, size_gate, cudaMemcpyHostToDevice);
  cudaMemcpy(d_gate2, gate2, size_gate, cudaMemcpyHostToDevice);
  cudaMemcpy(d_gate3, gate3, size_gate, cudaMemcpyHostToDevice);
  cudaMemcpy(d_gate4, gate4, size_gate, cudaMemcpyHostToDevice);
  cudaMemcpy(d_gate5, gate5, size_gate, cudaMemcpyHostToDevice);
  cudaMemcpy(d_gate6, gate6, size_gate, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_VS, &vector_size, size_VS, cudaMemcpyHostToDevice);

  // std::cout << "line 152" << std::endl;
  int threadsPerBlock = 1024;
  int blocksPerGrid = (vector_size+threadsPerBlock-1/threadsPerBlock);

  // Launch the kernel
  // cudaEventRecord(start);
  quamsim_kernel<<<blocksPerGrid, threadsPerBlock >>>(d_states,d_gate1,d_out1,vector_size,t1);
  //cudaDeviceSynchronize();
  quamsim_kernel<<<blocksPerGrid, threadsPerBlock >>>(d_out1,d_gate2,d_out2,vector_size, t2);
  //cudaDeviceSynchronize();
  quamsim_kernel<<<blocksPerGrid, threadsPerBlock >>>(d_out2,d_gate3,d_out3, vector_size, t3);
  //cudaDeviceSynchronize();
  quamsim_kernel<<<blocksPerGrid, threadsPerBlock >>>(d_out3,d_gate4,d_out4, vector_size, t4);
  //cudaDeviceSynchronize();
  quamsim_kernel<<<blocksPerGrid, threadsPerBlock >>>(d_out4,d_gate5,d_out5, vector_size, t5);
  //cudaDeviceSynchronize();
  quamsim_kernel<<<blocksPerGrid, threadsPerBlock >>>(d_out5,d_gate6,d_result, vector_size, t6);
  
  // cudaEventRecord(stop);
  // Print the output
  // std::cout << "line 173" << std::endl;
  // Copy output to host from device
  cudaMemcpy(host_out, d_result, size, cudaMemcpyDeviceToHost);
  // cudaMemcpy(host_gate, d_gate, size_gate, cudaMemcpyDeviceToHost);
  // cudaMemcpy(host_out, d_out, size, cudaMemcpyDeviceToHost);

//printing to be written

  // cudaEventSynchronize(stop);
  // float milliseconds = 0;
  // cudaEventElapsedTime(&milliseconds, start, stop);

  // std::cout<< milliseconds << std::endl;
  for(int i = 0; i<vector_size; i++){
    std::cout<< std::fixed << std::setprecision(3) << host_out[i] << std::endl;
  }
  cudaFree(d_states);
  // cudaFree(d_qubit);
  cudaFree(d_gate1);
  cudaFree(d_out1);
  cudaFree(d_gate2);
  cudaFree(d_out2);
  cudaFree(d_gate3);
  cudaFree(d_out3);
  cudaFree(d_gate4);
  cudaFree(d_out4);
  cudaFree(d_gate4);
  cudaFree(d_out4);
  cudaFree(d_gate5);
  cudaFree(d_out5);
  cudaFree(d_gate6);
  cudaFree(d_result);
  // cudaFree

  free(host_states);
  free(host_gate1);
  free(host_gate2);
  free(host_gate3);
  free(host_gate4);
  free(host_gate5);
  free(host_gate6);
  free(host_out);


  
  
  // printf("Done\n");
  return 0;
}
