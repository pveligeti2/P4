#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>




__global__ void quamsim_kernel(int *n, float *states, float *U, float *result, int *N, int *inactive_n, int inactive_qubits)
{	
	// shared memory
	__shared__ float a_shared[64];  // modify

	//LOAD DATA INTO SHARED MEMORY

	
	int inactive_mask = 0;

	for(int i=0; i<inactive_qubits; i++){
	
		inactive_mask = inactive_mask | (((blockIdx.x >> i) & 1) << inactive_n[i]);
	}

	// active mask one
	int active_mask_one = 0;
	int shared_index_one = 2 * threadIdx.x;

	for(int i=0; i<6; i++){
		//extract bit
		// int bit_i = (shared_index_one >> i) & 1;
		// position it at active index
		active_mask_one = active_mask_one | (((shared_index_one >> i) & 1) << n[i]); 
	}

	int global_index_one = active_mask_one | inactive_mask;

	int active_mask_two = 0;
	int shared_index_two = shared_index_one + 1;

	for(int i=0; i<6; i++){
		
		active_mask_two = active_mask_two | (((shared_index_two >> i) & 1) << n[i]); 
	}

	int global_index_two = active_mask_two | inactive_mask;

	

	a_shared[shared_index_one] = states[active_mask_one | inactive_mask];
	a_shared[shared_index_two] = states[active_mask_two | inactive_mask];

	__syncthreads();

	//printf("\n bId : %d tId : %d g_index_1 : %d g_index_2 : %d", blockIdx.x, threadIdx.x, global_index_one, global_index_two);
	
	// **********************************************

	int si1,si2;
	float out_1, out_2;
	int t;
	int u_index;

	for(int i=0; i<6; i++)
    {
        t = i;
        
        si1 = (threadIdx.x %  (1 <<t)) + (((1<<t)*2) * (threadIdx.x / (1<<t)));
        si2 =  si1 + (1<<t);

        out_1 = (a_shared[si1] * U[i*4]) + (a_shared[si2] * U[(i*4)+1]);
        out_2= (a_shared[si1] * U[(i*4)+2]) + (a_shared[si2] * U[(i*4)+3]);
        __syncthreads();
        a_shared[si1] = out_1;
        a_shared[si2] = out_2;
        __syncthreads();
        //printf("\n bId : %d tId : %d t : %d sh_index_1 : %d sh_index_2 : %d val1 : %0.3f val2 : %0.3f", blockIdx.x, threadIdx.x, t, si1, si2, out_1, out_2);    

    }

    // ***************** LOAD DATA INTO GLOBAL MEMORY ***********************
	result[global_index_one] = a_shared[shared_index_one];
	result[global_index_two] = a_shared[shared_index_two];	

}


int main(int argc, char *argv[])
{

	if(argc != 2)
	{
		printf("\n arguments mismatch. exiting");
		exit(0);
	}

  char *trace_file;
  trace_file = argv[1];

  std::ifstream file(trace_file);

  float gate1[4], gate2[4], gate3[4], gate4[4], gate5[4], gate6[4];
  // char space;

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
  unsigned long long size_gate = 24 * sizeof(float);
  // int t_size = 6 * sizeof(int);
  // Allocate/move data using cudaMalloc and cudaMemCpy

  
  // std::cout << "line 100" << std::endl;
  float *host_states = (float*)malloc(size);
  float *host_out = (float*)malloc(size);
  float *host_gate = (float*)malloc(size_gate);
  int *host_tqubit = (int *)(malloc(6*sizeof(int)));
  
 
  

  // std::cout << "line 104" << std::endl;
  for (int i = 0; i < vector_size; ++i)
  {
      host_states[i] = states[i];
  }

  for (int i = 0; i < 4; i++) 
  {
      host_gate[i]= gate1[i];
  }
   for (int i = 4; i < 8; i++) 
  {
      host_gate[i]= gate2[i-4];
  }
   for (int i = 8; i < 12; i++) 
  {
      host_gate[i]= gate3[i-8];
  }
   for (int i = 12; i < 16; i++) 
  {
      host_gate[i]= gate4[i-12];
  }
   for (int i = 16; i < 20; i++) 
  {
      host_gate[i]= gate5[i-16];
  }
   for (int i = 20; i < 24; i++) 
  {
      host_gate[i]= gate6[i-20];
  }
  for(int i = 0; i<6; i++){
    host_tqubit[i] = t[i];
  }



	// setup input values
	// allocate spaces for host copies 

  
  
	int inactive_N = (log(vector_size)/log(2)) - 6;
	int *inactive_n = (int *)malloc(inactive_N*sizeof(int));

	int i=0;
	int qgate_index = 0;
	bool flag = true;
	while(i < inactive_N)
	{
		for(int j=0; j< 6; j++)		
		{
			if(qgate_index == t[j])
			{
				qgate_index++;
				flag = false;
			}
		}

		if(flag)
		{
			inactive_n[i] = qgate_index;
			i++;
			qgate_index++;
		}
		flag = true;
	}

	int size_inactive_n = inactive_N * sizeof(int);
	int size_n = 6 * sizeof(int);
	int size_oprnds = vector_size * sizeof(float);
	int size_U = 24 *sizeof(float);
	int size_N = sizeof(int);
	int size_inactive_N = sizeof(int);

	// result = (float *)malloc(size_oprnds);	

	// device copies
	int *d_n, *d_N, *d_inactive_n, *d_inactive_N;
	float *d_a, *d_U, *d_result;
	
	//################### Device code #################################

	// allocate spaces for device copies 
	cudaMalloc((void **)&d_n,size_n);
	cudaMalloc((void **)&d_a,size_oprnds);
	cudaMalloc((void **)&d_result, size_oprnds);
	cudaMalloc((void **)&d_U, size_U);
	cudaMalloc((void **)&d_N, size_N);
	cudaMalloc((void **)&d_inactive_n,size_inactive_n);
	cudaMalloc((void **)&d_inactive_N,size_inactive_N);
	
	// copy inputs to device from host
	cudaMemcpy(d_n, &t, size_n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_a, host_states, size_oprnds, cudaMemcpyHostToDevice);
	cudaMemcpy(d_U, host_gate, size_U, cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, &vector_size, size_N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_inactive_N, &inactive_N, size_inactive_N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_inactive_n, inactive_n, size_inactive_n, cudaMemcpyHostToDevice);
	
	

	// launch kernel
	dim3 threadsPerBlock(32,1);	
	int blocks = vector_size/64;			
	dim3 numBlocks(blocks,1);		
	quamsim_kernel<<<numBlocks, threadsPerBlock>>>(d_n, d_a, d_U, d_result, d_N, d_inactive_n, inactive_N);

	// copy output to host from device
	cudaMemcpy(host_out, d_result, size_oprnds, cudaMemcpyDeviceToHost);
	
	//#################################################################


	// for(int i=0; i<6; i++)
  //       printf("%.3f\n", t[i]);
  // for(int i=0; i<24; i++)
  //       printf("%.3f\n", host_gate[i]);
  for(int i=0; i<vector_size; i++)
        printf("%.3f\n", host_out[i]);
  

    // cleanup
	cudaFree(d_n);
	cudaFree(d_a);
	cudaFree(d_U);
	cudaFree(d_result);
	free(host_out);
	free(host_states);
	free(host_gate);
	free(host_tqubit);
	free(inactive_n);


}