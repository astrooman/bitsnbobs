#include <iostream>
#include <mutex>

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <cufft.h>
#include <stdio.h>

#include "pdif.hpp"

using boost::asio::ip::udp;
using std::endl;
using std::cout;
using std::mutex;

#define SIZE 8
#define XSIZE 7
#define YSIZE 128
#define ZSIZE 48
#define BYTES_PER_WORD 8
#define HEADER 64
#define BUFLEN 7168 + 64
#define WORDS_PER_PACKET 896

#define cudaCheckError(myerror) {checkGPU((myerror), __FILE__, __LINE__);}

inline void checkGPU(cudaError_t code, const char *file, int line) {

    if (code != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(code) << " in file " << file << ", line " << line << std::endl;
        exit(EXIT_FAILURE);
        // TODO: throw exception instead of exiting
    }

}

struct obs_time {

    int start_epoch;            // reference epoch at the start of the observation
    int start_second;           // seconds from the reference epoch at the start of the observation
    int framet;                 // frame number from the start of the observation

};

texture<int2, cudaTextureType3D, cudaReadModeElementType> tex;

__global__ void arrange(cufftComplex * __restrict__ out) {

    // xidx is the channel number
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;
    int2 word;

    for (int sample = 0; sample < YSIZE; sample+=2) {
         word = tex3D(tex, xidx, sample, yidx);
         out[(yidx * XSIZE + xidx) * YSIZE + sample].x = static_cast<float>(static_cast<short>(((word.y & 0xff000000) >> 24) | ((word.y & 0xff0000) >> 8)));
         out[(yidx * XSIZE + xidx) * YSIZE + sample].y = static_cast<float>(static_cast<short>(((word.y & 0xff00) >> 8) | ((word.y & 0xff) << 8)));
         out[XSIZE * YSIZE * ZSIZE + (yidx * XSIZE + xidx) * YSIZE + sample].x = static_cast<float>(static_cast<short>(((word.x & 0xff000000) >> 24) | ((word.x & 0xff0000) >> 8)));
         out[XSIZE * YSIZE * ZSIZE + (yidx * XSIZE + xidx) * YSIZE + sample].y = static_cast<float>(static_cast<short>(((word.x & 0xff00) >> 8) | ((word.x & 0xff) << 8)));
         word = tex3D(tex, xidx, sample + 1, yidx);
         out[(yidx * XSIZE + xidx) * YSIZE + sample + 1].x = static_cast<float>(static_cast<short>(((word.y & 0xff000000) >> 24) | ((word.y & 0xff0000) >> 8)));
         out[(yidx * XSIZE + xidx) * YSIZE + sample + 1].y = static_cast<float>(static_cast<short>(((word.y & 0xff00) >> 8) | ((word.y & 0xff) << 8)));
         out[XSIZE * YSIZE * ZSIZE + (yidx * XSIZE + xidx) * YSIZE + sample + 1].x = static_cast<float>(static_cast<short>(((word.x & 0xff000000) >> 24) | ((word.x & 0xff0000) >> 8)));
         out[XSIZE * YSIZE * ZSIZE + (yidx * XSIZE + xidx) * YSIZE + sample + 1].y = static_cast<float>(static_cast<short>(((word.x & 0xff00) >> 8) | ((word.x & 0xff) << 8)));
         //printf("%i, %i, %i, %f, %f, %f, %f\n", xidx, yidx, sample, out[(yidx * XSIZE + xidx) * YSIZE + sample].x, out[(yidx * XSIZE + xidx) * YSIZE + sample].y, out[XSIZE * YSIZE * ZSIZE + (yidx * XSIZE + xidx) * YSIZE + sample].x, out[XSIZE * YSIZE * ZSIZE + (yidx * XSIZE + xidx) * YSIZE + sample].y);
    }
}

__global__ void badarrange(float *in, float *out) {

    int xidx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int sample = 0; sample < YSIZE; sample++)
        out[xidx * YSIZE + sample] = in[sample * 7 + xidx];

}

class GPUpool {
    private:
        mutex datamutex;
        size_t highest_buf;
        int highest_frame;
        bool buffer_ready[2];
        cufftComplex *h_pol;
        int d_in_size;
        short nchans = 14;
        short npol = 2;
        short fftsize = 32;
        short timesavg = 4;
        short pack_per_buf;

        int frames;
        char *h_in;
        cufftComplex *d_out;
        cudaArray *d_array;
        cudaChannelFormatDesc cdesc;
        cudaExtent volume;
        cudaMemcpy3DParms params;

    protected:

    public:
        GPUpool(void);
        ~GPUpool(void);
        //void add_data(cufftComplex *buffer, obs_time frame_time);
        void get_data(char *data, int fpga_id, obs_time start_time, header_s head);
};

GPUpool::GPUpool(void) : highest_buf(0), highest_frame(-1) {

    buffer_ready[0] = false;
    buffer_ready[1] = false;
    d_in_size = nchans * npol * fftsize * timesavg;
    pack_per_buf = nchans / 7 * 2;
    h_pol = new cufftComplex[d_in_size * 2];

    frames = 0;
    //h_in = new char[8 * XSIZE * YSIZE * ZSIZE];
    cudaCheckError(cudaHostAlloc((void**)&h_in, 8 * XSIZE * YSIZE * ZSIZE * sizeof(char), cudaHostAllocWriteCombined));

    cudaCheckError(cudaMalloc((void**)&d_out, 2 * XSIZE * YSIZE * ZSIZE * sizeof(cufftComplex)));
    cdesc = cudaCreateChannelDesc<int2>();
    cudaCheckError(cudaPeekAtLastError());
    volume = make_cudaExtent(XSIZE, YSIZE, ZSIZE);
    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaMalloc3DArray(&d_array, &cdesc, volume));
    cudaCheckError(cudaBindTextureToArray(tex, d_array));
    params.extent = volume;
    params.dstArray = d_array;
    params.kind = cudaMemcpyHostToDevice;

    tex.filterMode = cudaFilterModePoint;
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.addressMode[2] = cudaAddressModeClamp;
}

GPUpool::~GPUpool(void) {

   delete [] h_pol;

}

void GPUpool::get_data(char* data, int fpga_id, obs_time start_time, header_s head)
{
    // REMEMBER - d_in_size is the size of the single buffer (2 polarisations, 336 channels, 128 time samples)

    std::copy(data + HEADER, data + BUFLEN, h_in + frames * WORDS_PER_PACKET * 8);
    frames++;

    if (frames == 48) {
        params.srcPtr = make_cudaPitchedPtr((void*)h_in, XSIZE * 8, XSIZE * 8, YSIZE);
        cudaCheckError(cudaMemcpy3D(&params));
        //cudaCheckError(cudaBindTextureToArray(tex, d_array));
        dim3 nblocks(1,1,1);
        dim3 nthreads(XSIZE, ZSIZE, 1);
        arrange<<<nblocks, nthreads, 0>>>(d_out);
        cudaDeviceSynchronize();
        cudaCheckError(cudaPeekAtLastError());
        cudaUnbindTexture(tex);
        frames = 0;
    }

}

int main(int argc, char *argv[])
{

    try {

        GPUpool mypool;

        unsigned char *inbuf = new unsigned char[BUFLEN];

        boost::asio::io_service io_service;
        udp::endpoint sender_endpoint;

        udp::socket socket(io_service, udp::v4());

        boost::asio::socket_base::reuse_address option(true);
        boost::asio::socket_base::receive_buffer_size option2(9000);
        socket.set_option(option);
        socket.set_option(option2);
        socket.bind(udp::endpoint(boost::asio::ip::address::from_string("10.17.0.1"), atoi(argv[1])));

        //cout << "Waiting to get something on port " << atoi(argv[1]) << " ..." << endl;

        size_t len;
        int n = 0;

        header_s head;

        boost::array<unsigned char, BUFLEN> recv_buf;

        while(n < 48) {
            len = socket.receive_from(boost::asio::buffer(recv_buf), sender_endpoint);
            //cout << len << endl;

            static obs_time start_time{head.epoch, head.ref_s};
            // this is ugly, but I don't have a better solution at the moment
            int long_ip = boost::asio::ip::address_v4::from_string((sender_endpoint.address()).to_string()).to_ulong();
            int fpga = ((int)((long_ip >> 8) & 0xff) - 1) * 8 + ((int)(long_ip & 0xff) - 1) / 2;

            mypool.get_data(reinterpret_cast<char*>(recv_buf.data()), fpga, start_time, head);

            n++;
        }

        //cout << "Took " << send_elapsed.count() << " seconds to receive " << n << " buffers " << endl;
    } catch (std::exception &e) {
        cout << "Something bad happened: " << e.what() << endl;
    }
    return 0;
}

