#include <chrono>
#include <iostream>

#include <cufft.h>

using std::cout;
using std::endl;

#define XSIZE 7
#define YSIZE 128
#define ZSIZE 48

#define cudaCheckError(myerror) {checkGPU((myerror), __FILE__, __LINE__);}

inline void checkGPU(cudaError_t code, const char *file, int line) {

    if (code != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(code) << " in file " << file << ", line " << line << std::endl;
        exit(EXIT_FAILURE);
        // TODO: throw exception instead of exiting
    }

}

texture<int2, cudaTextureType3D, cudaReadModeElementType> tex;
texture<int2, cudaTextureType2D, cudaReadModeElementType> tex2;

__global__ void arrange(cufftComplex * __restrict__ out) {

    // xidx is the channel number
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;
    int2 word;

    for (int sample = 0; sample < YSIZE; sample++) {
         word = tex3D(tex, xidx, sample, yidx);
         out[(yidx * XSIZE + xidx) * YSIZE + sample].x = static_cast<float>(static_cast<short>(((word.y & 0xff000000) >> 24) | ((word.y & 0xff0000) >> 8)));
         out[(yidx * XSIZE + xidx) * YSIZE + sample].y = static_cast<float>(static_cast<short>(((word.y & 0xff00) >> 8) | ((word.y & 0xff) << 8)));
         out[XSIZE * YSIZE * ZSIZE + (yidx * XSIZE + xidx) * YSIZE + sample].x = static_cast<float>(static_cast<short>(((word.x & 0xff000000) >> 24) | ((word.x & 0xff0000) >> 8)));
         out[XSIZE * YSIZE * ZSIZE + (yidx * XSIZE + xidx) * YSIZE + sample].y = static_cast<float>(static_cast<short>(((word.x & 0xff00) >> 8) | ((word.x & 0xff) << 8)));
         //printf("%i, %i, %i, %f, %f, %f, %f\n", xidx, yidx, sample, out[(yidx * XSIZE + xidx) * YSIZE + sample].x, out[(yidx * XSIZE + xidx) * YSIZE + sample].y, out[XSIZE * YSIZE * ZSIZE + (yidx * XSIZE + xidx) * YSIZE + sample].x, out[XSIZE * YSIZE * ZSIZE + (yidx * XSIZE + xidx) * YSIZE + sample].y);
    }
}

__global__ void arrange2(cufftComplex * __restrict__ out) {
    // this is currently the ugliest solution I can think of
    // xidx is the channel number
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * 128;
    int2 word;

    for (int sample = 0; sample < YSIZE; sample++) {
         word = tex2D(tex2, xidx, yidx + sample);
         //printf("%i ", sample);
         out[xidx * 128 + 7 * yidx + sample].x = static_cast<float>(static_cast<short>(((word.y & 0xff000000) >> 24) | ((word.y & 0xff0000) >> 8)));
         out[xidx * 128 + 7 * yidx + sample].y = static_cast<float>(static_cast<short>(((word.y & 0xff00) >> 8) | ((word.y & 0xff) << 8)));
         out[336 * 128 + xidx * 128 + 7 * yidx + sample].x = static_cast<float>(static_cast<short>(((word.x & 0xff000000) >> 24) | ((word.x & 0xff0000) >> 8)));
         out[336 * 128 + xidx * 128 + 7 * yidx + sample].y = static_cast<float>(static_cast<short>(((word.x & 0xff00) >> 8) | ((word.x & 0xff) << 8)));
         //out[(yidx * XSIZE + xidx) * YSIZE + sample].x = static_cast<float>(static_cast<short>(((word.y & 0xff000000) >> 24) | ((word.y & 0xff0000) >> 8)));
         //out[(yidx * XSIZE + xidx) * YSIZE + sample].y = static_cast<float>(static_cast<short>(((word.y & 0xff00) >> 8) | ((word.y & 0xff) << 8)));
         //out[XSIZE * YSIZE * ZSIZE + (yidx * XSIZE + xidx) * YSIZE + sample].x = static_cast<float>(static_cast<short>(((word.x & 0xff000000) >> 24) | ((word.x & 0xff0000) >> 8)));
         //out[XSIZE * YSIZE * ZSIZE + (yidx * XSIZE + xidx) * YSIZE + sample].y = static_cast<float>(static_cast<short>(((word.x & 0xff00) >> 8) | ((word.x & 0xff) << 8)));
         //printf("%i, %i, %i, %f, %f \n", xidx, yidx, sample, out[xidx * 128 + 7 * yidx + sample].x, out[xidx * 128 + 7 * yidx + sample].y);
    }
}


__global__ void arrangebad(char *in, cufftComplex *out) {

    int xidx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int sample = 0; sample < YSIZE; sample++) {
        out[xidx * YSIZE + sample].x = static_cast<float>(static_cast<short>(in[blockIdx.x * 7 * 128 * 8 + sample * 7 * 8 + threadIdx.x * 8 + 7] | (in[blockIdx.x * 7 * 128 * 8 + sample * 7 * 8 + threadIdx.x * 8 + 7] << 8)));
        out[xidx * YSIZE + sample].y = static_cast<float>(static_cast<short>(in[blockIdx.x * 7 * 128 * 8 + sample * 7 * 8 + threadIdx.x * 8 + 5] | (in[blockIdx.x * 7 * 128 * 8 + sample * 7 * 8 + threadIdx.x * 8 + 4] << 8)));
        out[xidx * YSIZE + sample + XSIZE * YSIZE * ZSIZE].x = static_cast<float>(static_cast<short>(in[blockIdx.x * 7 * 128 * 8 + sample * 7 * 8 + threadIdx.x * 8 + 3] | (in[blockIdx.x * 7 * 128 * 8 + sample * 7 * 8 + threadIdx.x * 8 + 2] << 8)));
        out[xidx * YSIZE + sample + XSIZE * YSIZE * ZSIZE].x = static_cast<float>(static_cast<short>(in[blockIdx.x * 7 * 128 * 8 + sample * 7 * 8 + threadIdx.x * 8 + 1] | (in[blockIdx.x * 7 * 128 * 8 + sample * 7 * 8 + threadIdx.x * 8 + 0] << 8)));
    }
}

int main(int argc, char *argv[])
{

    float alloc_elapsed;
    cudaEvent_t alloc_start;
    cudaEvent_t alloc_end;

    cudaEventCreate(&alloc_start);
    cudaEventCreate(&alloc_end);

    char *h_in = new char[8 * XSIZE * YSIZE * ZSIZE];

    for (int ii = 0; ii < ZSIZE; ii++) {
        for (int jj = 0; jj < YSIZE; jj++) {
            for (int kk = 0; kk < 8 * XSIZE; kk++) {
                h_in[ii * XSIZE * YSIZE * 8 + jj * XSIZE * 8 + kk] = jj;
            }
        }
    }

    cufftComplex *d_out;
    cudaCheckError(cudaMalloc((void**)&d_out, 2 * XSIZE * YSIZE * ZSIZE * sizeof(cufftComplex)));

    cudaChannelFormatDesc cdesc;
    cudaExtent volume;
    cudaMemcpy3DParms params = {0};

    cdesc = cudaCreateChannelDesc<int2>();
    cudaCheckError(cudaPeekAtLastError());
    volume = make_cudaExtent(XSIZE, YSIZE, ZSIZE);
    cudaCheckError(cudaPeekAtLastError());
    cudaArray *d_array;

    cudaEventRecord(alloc_start, 0);
    cudaCheckError(cudaMalloc3DArray(&d_array, &cdesc, volume));
    cudaCheckError(cudaBindTextureToArray(tex, d_array));
    cudaEventRecord(alloc_end, 0);
    cudaEventSynchronize(alloc_end);
    cudaEventElapsedTime(&alloc_elapsed, alloc_start, alloc_end);

    cout << "3D alloc: " << alloc_elapsed << "ms" << endl;

    params.extent = volume;
    params.dstArray = d_array;
    params.kind = cudaMemcpyHostToDevice;
    params.srcPtr = make_cudaPitchedPtr((void*)h_in, XSIZE * 8, XSIZE * 8, YSIZE);

    tex.filterMode = cudaFilterModePoint;
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.addressMode[2] = cudaAddressModeClamp;

    float copy_elapsed;
    cudaEvent_t copy_start;
    cudaEvent_t copy_end;

    cudaEventCreate(&copy_start);
    cudaEventCreate(&copy_end);

    cudaEventRecord(copy_start,0);
    cudaCheckError(cudaMemcpy3D(&params));
    cudaEventRecord(copy_end, 0);
    cudaEventSynchronize(copy_end);
    cudaEventElapsedTime(&copy_elapsed, copy_start, copy_end);

    cout << "3D memcpy: " << copy_elapsed << "ms" << endl;

    float exec_elapsed;
    cudaEvent_t exec_start;
    cudaEvent_t exec_end;

    cudaEventCreate(&exec_start);
    cudaEventCreate(&exec_end);   

    dim3 nblocks(1,1,1); 
    dim3 nthreads(XSIZE, ZSIZE, 1);

    cudaEventRecord(exec_start, 0);
    arrange<<<nblocks, nthreads, 0>>>(d_out);    
    cudaCheckError(cudaPeekAtLastError());
    cudaEventRecord(exec_end, 0);
    cudaEventSynchronize(exec_end);
    cudaEventElapsedTime(&exec_elapsed, exec_start, exec_end);

    cout << "3D exec: " << exec_elapsed << "ms" << endl;

    cudaDeviceSynchronize();
    cudaUnbindTexture(tex);

    cudaFreeArray(d_array);

    // ###################
    // 2D 'IMPLEMENTATION'
    // ###################

    cudaArray *d_array2;
    cudaEventRecord(alloc_start, 0);
    cudaCheckError(cudaMallocArray(&d_array2, &cdesc, XSIZE, YSIZE * ZSIZE));
    cudaCheckError(cudaBindTextureToArray(tex2, d_array2));
    cudaEventRecord(alloc_end, 0);
    cudaEventSynchronize(alloc_end);
    cudaEventElapsedTime(&alloc_elapsed, alloc_start, alloc_end);

    cout << "2D alloc: " << alloc_elapsed << "ms" << endl;

    tex2.filterMode = cudaFilterModePoint;
    tex2.addressMode[0] = cudaAddressModeClamp;
    tex2.addressMode[1] = cudaAddressModeClamp;

    cudaEventRecord(copy_start,0);
    cudaCheckError(cudaMemcpyToArray(d_array2, 0, 0, h_in, 8 * XSIZE * YSIZE * ZSIZE, cudaMemcpyHostToDevice));
    cudaEventRecord(copy_end, 0);
    cudaEventSynchronize(copy_end);
    cudaEventElapsedTime(&copy_elapsed, copy_start, copy_end);

    cout << "2D memcpy: " << copy_elapsed << "ms" << endl;

    dim3 nblocks2(1, ZSIZE, 1);
    dim3 nthreads2(XSIZE,1,1);

    cudaEventRecord(exec_start, 0);
    arrange2<<<nblocks2, nthreads2, 0>>>(d_out);
    cudaDeviceSynchronize();
    cudaCheckError(cudaPeekAtLastError());
    cudaEventRecord(exec_end, 0);
    cudaEventSynchronize(exec_end);
    cudaEventElapsedTime(&exec_elapsed, exec_start, exec_end);

    cout << "2D exec: " << exec_elapsed << "ms" << endl;

    cufftComplex *h_out = new cufftComplex[2 * XSIZE * YSIZE * ZSIZE];
    cudaCheckError(cudaMemcpy(h_out, d_out, 2 * XSIZE * YSIZE * ZSIZE * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    //for (int ii = 0; ii < YSIZE; ii++)
    //    cout << h_out[ii].x << " + i * " << h_out[ii].y << endl;   

    cudaDeviceSynchronize();
    cudaUnbindTexture(tex2);

    cudaFreeArray(d_array2);

    char *d_in;

    cudaEventRecord(alloc_start, 0);    
    cudaCheckError(cudaMalloc((void**)&d_in, 8 * XSIZE * YSIZE * ZSIZE));
    cudaEventRecord(alloc_end, 0);
    cudaEventSynchronize(alloc_end);
    cudaEventElapsedTime(&alloc_elapsed, alloc_start, alloc_end);

    cout << "Simple device alloc: " << alloc_elapsed << "ms" << endl;

    cudaEventRecord(copy_start,0);
    cudaCheckError(cudaMemcpy(d_in, h_in, 8 * XSIZE * YSIZE * ZSIZE, cudaMemcpyHostToDevice));
    cudaEventRecord(copy_end, 0);
    cudaEventSynchronize(copy_end);
    cudaEventElapsedTime(&copy_elapsed, copy_start, copy_end);

    cout << "Simple device  memcpy: " << copy_elapsed << "ms" << endl;

    dim3 nblocks3(48, 1, 1);
    dim3 nthreads3(7, 1, 1);

    cudaEventRecord(exec_start, 0);
    arrangebad<<<nblocks3, nthreads3>>>(d_in, d_out);
    cudaCheckError(cudaPeekAtLastError());
    cudaEventRecord(exec_end, 0);
    cudaEventSynchronize(exec_end);
    cudaEventElapsedTime(&exec_elapsed, exec_start, exec_end);

    cout << "Simple exec: " << exec_elapsed << "ms" << endl;
    

    cudaFree(d_in);
    cudaFree(d_out); 
    delete [] h_in;

    return 0;
}
