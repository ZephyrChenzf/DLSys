#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <float.h>

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

__global__ void kernel_arraySet(float* data,int n,float value){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if (idx<n){
    data[idx]=value;
  }
}

int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
  int n=1;
  const int threadNum=512;
  float* data=(float*)arr->data;
  for(int i=0;i<arr->ndim;i++){
    n*=arr->shape[i];
  }
  kernel_arraySet<<<(n+threadNum-1)/threadNum,threadNum>>>(data,n,value);
  return 0;
}

__global__ void kernel_broadCastTo(const float *input,float *output,int n,int m){
  int inpIdx=blockIdx.x*blockDim.x+threadIdx.x;
  if (inpIdx<n){
    int outIdx=inpIdx;
    while(outIdx<m){
      output[outIdx]=input[inpIdx];
      outIdx+=n;
    }
  }
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  const int threadNum=512;
  float* input_data=(float*)input->data;
  float* output_data=(float*)output->data;
  // float *dev_input,*dev_output;
  int n=1;
  for(int i=0;i<input->ndim;i++){
    n*=input->shape[i];
  }
  int m=1;
  for(int i=0;i<output->ndim;i++){
    m*=output->shape[i];
  }
  // cudaMalloc((void **)&dev_input,n*sizeof(float));
  // cudaMalloc((void **)&dev_output,m*sizeof(float));
  // cudaMemcpy(dev_input,input_data,n*sizeof(float),cudaMemcpyHostToDevice);
  
  // kernel_broadCastTo<<<(n+threadNum-1)/threadNum,threadNum>>>(dev_input,dev_output,n,m);
  kernel_broadCastTo<<<(n+threadNum-1)/threadNum,threadNum>>>(input_data,output_data,n,m);

  // cudaMemcpy(output_data,dev_output,m*sizeof(float),cudaMemcpyDeviceToHost);
  // cudaFree(dev_input);
  // cudaFree(dev_output);
  
  return 0;
}

__global__ void kernel_reduceSumAxisZero(const float *input,float *output,int n,int m){
  int outIdx=blockIdx.x*blockDim.x+threadIdx.x;
  if (outIdx<m){
    int inpIdx=outIdx;
    while (inpIdx<n){
      output[outIdx]+=input[inpIdx];
      inpIdx+=m;
    }
  }
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  const int threadNum=512;
  int n=1;
  int m=1;
  for (int i=0;i<input->ndim;i++){
    n*=input->shape[i];
  }
  for (int i=0;i<output->ndim;i++){
    m*=output->shape[i];
  }
  float *input_data=(float *)input->data;
  float *output_data=(float *)output->data;

  kernel_reduceSumAxisZero<<<(m+threadNum-1)/threadNum,threadNum>>>(input_data,output_data,n,m);
  return 0;
}

__global__ void kernel_matrixElementwiseAdd(const float *matA,const float *matB,float *output,int n){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if (idx<n){
    output[idx]=matA[idx]+matB[idx];
  }
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
  const int threadNum=512;
  int n=1;
  for (int i=0;i<matA->ndim;i++){
    n*=matA->shape[i];
  }
  kernel_matrixElementwiseAdd<<<(n+threadNum-1)/threadNum,threadNum>>>((float *)matA->data,
                                                                      (float *)matB->data,(float *)output->data,n);
  return 0;
}

__global__ void kernel_matrixElementwiseAddByConst(const float *input,float *output,float val,int n){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if (idx<n){
    output[idx]=input[idx]+val;
  }
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here */
  const int threadNum=512;
  int n=1;
  for (int i=0;i<input->ndim;i++){
    n*=input->shape[i];
  }
  kernel_matrixElementwiseAddByConst<<<(n+threadNum-1)/threadNum,threadNum>>>((float*)input->data,(float*)output->data,val,n);

  return 0;
}

__global__ void kernel_matrixElementwiseMultiply(const float *matA,const float *matB,float *output,int n){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if (idx<n){
    output[idx]=matA[idx]*matB[idx];
  }
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
  int n=1;
  const int threadNum=512;
  for (int i=0;i<matA->ndim;i++){
    n*=matA->shape[i];
  }
  kernel_matrixElementwiseMultiply<<<(n+threadNum-1)/threadNum,threadNum>>>((float*)matA->data,(float*)matB->data,(float*)output->data,n);
  return 0;
}

__global__ void kernel_matrimultiplyByConst(const float *input,float *output,float val,int n){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if (idx<n){
    output[idx]=input[idx]*val;
  }
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
  int n=1;
  const int threadNum=512;
  for (int i=0;i<input->ndim;i++){
    n*=input->shape[i];
  }

  kernel_matrimultiplyByConst<<<(n+threadNum-1)/threadNum,threadNum>>>((float*)input->data,(float*)output->data,val,n);
  return 0;
}

//normal matrixMultiply
__global__ void kernel_matrixMultiply_normal(const float *A,bool transposeA,int A_weight,int A_height,const float *B,bool transposeB,int B_weight,int B_height,float *C){
  int col=blockDim.x*blockIdx.x+threadIdx.x;
  int row=blockDim.y*blockIdx.y+threadIdx.y;
  int rowsA=A_height;
  int start_A=row*A_weight;
  int end_A=row*A_weight+A_weight;
  int step_A=1;
  if (transposeA){
    rowsA=A_weight;
    start_A=row;
    end_A=row+(A_height-1)*A_weight+1;
    step_A=A_weight;
  }
  int colsB=B_weight;
  int start_B=col;
  int step_B=B_weight;
  if (transposeB){
    colsB=B_height;
    start_B=col*B_weight;
    step_B=1;
  }
  float tempVal=0;
  if (row>=rowsA || col>=colsB) return;

  for (int i=start_A,j=start_B;i<end_A;i+=step_A,j+=step_B){
    tempVal+=A[i]*B[j];
  }
  C[row*colsB+col]=tempVal;
}

//matrixMultiply with shared memory
//有一些问题，但是没找出原因
__global__ void kernel_matirxMultiply_sm(const float *A,int A_weight,int A_height,const float *B,int B_weight,int B_height,float *C){
  float tempVal=0;
  if ((blockDim.x*blockIdx.x+threadIdx.x)>=B_weight||(blockDim.y*blockIdx.y+threadIdx.y)>=A_height) return;
  int start_A=blockDim.y*blockIdx.y*A_weight;
  int end_A=start_A+A_weight;
  int step_A=blockDim.x;

  int start_B=blockDim.x*blockIdx.x;
  int step_B=B_weight*blockDim.y;

  const int blockSize=16;
  for(int i=start_A,j=start_B;i<end_A;i+=step_A,j+=step_B){
    __shared__ float subA[blockSize][blockSize];
    __shared__ float subB[blockSize][blockSize];
    subA[threadIdx.y][threadIdx.x]=A[i+threadIdx.x+A_weight*threadIdx.y];
    subB[threadIdx.y][threadIdx.x]=B[j+threadIdx.x+B_weight*threadIdx.y];

    __syncthreads();

    for (int k=0;k<blockSize;k++){
      tempVal+=subA[threadIdx.y][k]*subB[k][threadIdx.x];
    }
    __syncthreads();
  }
  C[(blockDim.y*blockIdx.y+threadIdx.y)*B_weight+blockDim.x*blockIdx.x+threadIdx.x]=tempVal;

}


int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
  int A_height=matA->shape[0];
  int A_weight=matA->shape[1];
  int B_height=matB->shape[0];
  int B_weight=matB->shape[1];
  int C_height=matC->shape[0];
  int C_weight=matC->shape[1];
  const int blockSize=16;  
  dim3 blockDim(blockSize,blockSize);
  dim3 gridDim((C_weight+blockSize-1)/blockSize,(C_height+blockSize-1)/blockSize);
  kernel_matrixMultiply_normal<<<gridDim,blockDim>>>((float *)matA->data,transposeA,A_weight,A_height,
                                    (float *)matB->data,transposeB,B_weight,B_height,(float*)matC->data); 
  // kernel_matirxMultiply_sm<<<gridDim,blockDim>>>((float *)matA->data,A_weight,A_height,
  //                                   (float *)matB->data,B_weight,B_height,(float*)matC->data); 
  return 0;
}

__global__ void kernel_relu(const float *input,float *output,int n){
  int idx=blockDim.x*blockIdx.x+threadIdx.x;
  if (idx>=n)return;
  if (input[idx]<0){
    output[idx]=0;
  }
  else{
    output[idx]=input[idx];
  }
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int n=1;
  for(int i=0;i<input->ndim;i++){
    n*=input->shape[i];
  }
  const int threadNum=512;
  kernel_relu<<<(n+threadNum-1)/threadNum,threadNum>>>((float*)input->data,(float*)output->data,n);
  return 0;
}

//<0:-1  >0:1  =0:0
__device__ float sign_fuc(float val){
  if (val<0){
    return -1;
  }
  else if (val>0){
    return 1;
  }
  else{
    return 0;
  }
}

__global__ void kernel_reluGradient(const float *input,const float *in_grad,float *output,int n){
  int idx=blockDim.x*blockIdx.x+threadIdx.x;
  if (idx>=n)return;
  output[idx]=(sign_fuc(input[idx])+1)*0.5*in_grad[idx];
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  int n=1;
  const int threadNum=512;
  for (int i=0;i<input->ndim;i++){
    n*=input->shape[i];
  }
  kernel_reluGradient<<<(n+threadNum-1)/threadNum,threadNum>>>((float*)input->data,(float*)in_grad->data,(float*)output->data,n);
  return 0;
}

__global__ void kernel_softmax(const float *input,float *output,int rows,int cols){
  int row=blockDim.y*blockIdx.y+threadIdx.y;
  int col=blockDim.x*blockIdx.x+threadIdx.x;
  int idx=row*cols+col;
  if (row>=rows || col>=cols)return;

  float maxV=FLT_MIN;
  for(int i=0;i<cols;i++){
    maxV=max(maxV,input[row*cols+i]);
  }
  float sumV=0;
  for (int i=0;i<cols;i++){
    sumV+=exp(input[row*cols+i]-maxV);
  }
  output[idx]=exp(input[idx]-maxV)/sumV;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int rows=input->shape[0];
  int cols=input->shape[1];
  const int blockSize=16;
  dim3 blockDim(blockSize,blockSize);
  dim3 gridDim((cols+blockSize-1)/blockSize,(rows+blockSize-1)/blockSize);
  kernel_softmax<<<gridDim,blockDim>>>((float*)input->data,(float*)output->data,rows,cols);

  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
