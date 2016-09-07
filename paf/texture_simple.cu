#include <bitset>
#include <iostream>
#include <stdio.h>
#include <cufft.h>

using std::endl;
using std::cout;

#define SIZE 8
#define XSIZE 7
#define YSIZE 2

#define cudaCheckError(myerror) {checkGPU((myerror), __FILE__, __LINE__);}

inline void checkGPU(cudaError_t code, const char *file, int line) {

    if (code != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(code) << " in file " << file << ", line " << line << std::endl;
        exit(EXIT_FAILURE);
        // TODO: throw exception instead of exiting
    }

}

texture<int2, cudaTextureType2D, cudaReadModeElementType> tex;

__global__ void arrange(cufftComplex *out) {

    // xidx is the channel number
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int2 word;

    for (int sample = 0; sample < YSIZE; sample++) {
        word = tex2D(tex, xidx, sample);
        printf("%i, %i\n", word.x, word.y);
    }
}

__global__ void badarrange(float *in, float *out) {

    int xidx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int sample = 0; sample < 128; sample++)
        out[xidx * 128 + sample] = in[sample * 7 + xidx];

}
int main(int argc, char *argv[])
{

    char *h_in = new char[8 * XSIZE * YSIZE];

    for (int ii = 0; ii < 8 * XSIZE * YSIZE; ii++) {
        h_in[ii] = ii;
    }

    for (int ii = 0; ii < YSIZE; ii++) {
        for (int jj = 0; jj < 8 * XSIZE; jj++)
            cout << (int)h_in[ii * 8 * XSIZE + jj] << " ";
        cout << endl;
    }

    cout << endl;
    short *o = reinterpret_cast<short*>(h_in);
    cout << std::bitset<8>(h_in[0]) << endl;;
    cout << std::bitset<16>(o[0]) << " " << std::bitset<16>(o[1]) << endl;
    int *p = reinterpret_cast<int*>(h_in);
    cout << p[0] << " " << p[1] << endl;
    cout << std::bitset<32>(p[0]) << " " << std::bitset<32>(p[1]) << endl;
    cout << "printed everything out" << endl;
    cout.flush();

    cufftComplex *h_out = new cufftComplex[2 * XSIZE * YSIZE];
    cufftComplex *d_out;
    cudaCheckError(cudaMalloc((void**)&d_out, 2 * XSIZE * YSIZE * sizeof(cufftComplex)));

    cudaArray *d_array;
    cudaChannelFormatDesc cdesc = cudaCreateChannelDesc<int2>();
    cudaCheckError(cudaPeekAtLastError());

    cudaCheckError(cudaMallocArray(&d_array, &cdesc, XSIZE, YSIZE));
    cudaCheckError(cudaMemcpyToArray(d_array, 0, 0, h_in, 8 * XSIZE * YSIZE * sizeof(char), cudaMemcpyHostToDevice));

    tex.filterMode = cudaFilterModePoint;
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    cudaCheckError(cudaBindTextureToArray(tex, d_array));


    dim3 nblocks(1,1,1);
    dim3 nthreads(XSIZE,1,1);

    arrange<<<nblocks, nthreads, 0, 0>>>(d_out);
    cudaCheckError(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    cudaUnbindTexture(tex);
/*    cudaCheckError(cudaMemcpy(h_out, d_out, XSIZE * YSIZE * sizeof(float), cudaMemcpyDeviceToHost));

    cout << endl << endl;

    for (int ii = 0; ii < XSIZE; ii++) {
        for (int jj = 0; jj < YSIZE; jj++)
            cout << h_out[ii * YSIZE + jj] << " ";
        cout << endl;
    }

    float *d_in;
    cudaCheckError(cudaMalloc((void**)&d_in, XSIZE * YSIZE * sizeof(float)));
    cudaCheckError(cudaMemcpy(d_in, h_in, XSIZE * YSIZE * sizeof(float), cudaMemcpyHostToDevice));

    badarrange<<<nblocks, nthreads, 0, 0>>>(d_in, d_out);
    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaMemcpy(h_out, d_out, XSIZE * YSIZE * sizeof(float), cudaMemcpyDeviceToHost));

    cout << endl << endl;

    for (int ii = 0; ii < XSIZE; ii++) {
        for (int jj = 0; jj < YSIZE; jj++)
            cout << h_out[ii * YSIZE + jj] << " ";
        cout << endl;
    }
*/
    cudaFreeArray(d_array);
    //cudaFree(d_in);
    cudaFree(d_out);
    delete [] h_out;
    delete [] h_in;

    return 0;
}

