#include <chrono>
#include <iostream>
#include <mutex>
#include <queue>
#include <utility>
#include <vector>

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <cufft.h>
#include <cuda.h>

#include "pdif.hpp"

using boost::asio::ip::udp;
using std::cout;
using std::endl;
using std::mutex;
using std::pair;
using std::queue;
using std::vector;

#define BYTES_PER_WORD 8
#define HEADER 64
#define BUFLEN 7168 + 64
#define WORDS_PER_PACKET 896

#define BSWAP_64(x)     (((uint64_t)(x) << 56) |                         \
                         (((uint64_t)(x) << 40) & 0xff000000000000ULL) | \
                         (((uint64_t)(x) << 24) & 0xff0000000000ULL) |  \
                         (((uint64_t)(x) << 8)  & 0xff00000000ULL) |    \
                         (((uint64_t)(x) >> 8)  & 0xff000000ULL) |      \
                         (((uint64_t)(x) >> 24) & 0xff0000ULL) |        \
                         (((uint64_t)(x) >> 40) & 0xff00ULL) |          \
                         ((uint64_t)(x)  >> 56))


struct obs_time {

    int start_epoch;            // reference epoch at the start of the observation
    int start_second;           // seconds from the reference epoch at the start of the observation
    int framet;                 // frame number from the start of the observation

};

__global__ void arrange(char *d_data, cufftComplex *df_data) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = (int)(idx / 7) * 896 * 8 + (idx % 7) * 8;
    for (int ii = 0; ii < 128; ii++) {
        df_data[idx * 128 + ii].x = static_cast<float>(d_data[idx2 + ii * 56 + 7] | (d_data[idx2 + ii * 56 + 6] << 8));
        df_data[idx * 128 + ii].y = static_cast<float>(d_data[idx2 + ii * 56 + 5] | (d_data[idx2 + ii * 56 + 4] << 8));
        df_data[idx * 128 + ii + 336 * 128].x = static_cast<float>(d_data[idx2 + ii * 56 + 3] | (d_data[idx2 + ii * 56 + 2] << 8));
        df_data[idx * 128 + ii + 336 * 128].y = static_cast<float>(d_data[idx2 + ii * 56 + 1] | (d_data[idx2 + ii * 56 + 0] << 8));
    }
}

class GPUpool {
    private:
        mutex datamutex;
        queue<vector<cufftComplex>> mydata;
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
        char *d_data;
        cufftComplex *df_data;
        int frames; 
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
    cudaMalloc((void**)&d_data, 896 * 8 * 48 * sizeof(char));
    cudaMalloc((void**)&df_data, 896 * 2 * 48 * sizeof(cufftComplex));
}

GPUpool::~GPUpool(void) {

   delete [] h_pol;

}

void GPUpool::get_data(char* data, int fpga_id, obs_time start_time, header_s head)
{
    // REMEMBER - d_in_size is the size of the single buffer (2 polarisations, 336 channels, 128 time samples)

    cudaMemcpy(d_data + frames * 896 * 8, data + 64, 896 * 8 * sizeof(char), cudaMemcpyHostToDevice);
    frames++;
    
    if (frames == 47) {
        arrange<<<1, 336, 0>>>(d_data, df_data);
        frames = 0;
    }
/*    for (int chan = 0; chan < 7; chan++) {
        for (int sample = 0; sample < 128; sample++) {
            idx = (sample * 7 + chan) * BYTES_PER_WORD;    // get the  start of the word in the received data array
            idx2 = chan * 128 + sample + startidx;        // get the position in the buffer
            h_pol[idx2].x = static_cast<float>(data[HEADER + idx + 7] | (data[HEADER + idx + 6] << 8));
            h_pol[idx2].y = static_cast<float>(static_cast<short>(data[HEADER + idx + 5] | (data[HEADER + idx + 4] << 8)));
            h_pol[idx2 + d_in_size / 2].x = static_cast<float>(static_cast<short>(data[HEADER + idx + 3] | (data[HEADER + idx + 2] << 8)));
            h_pol[idx2 + d_in_size / 2].y = static_cast<float>(static_cast<short>(data[HEADER + idx + 1] | (data[HEADER + idx + 0] << 8)));
        }
    }
/*    for (int sample = 0; sample < 128; sample++) {
        for (int chan = 0; chan < 7; chan++) {
            idx = (sample * 7 + chan) * BYTES_PER_WORD;
            idx2 = sample * 7 + chan;
            h_pol[idx2].x = static_cast<float>(data[HEADER + idx + 7] | (data[HEADER + idx + 6] << 8));
            h_pol[idx2].y = static_cast<float>(data[HEADER + idx + 5] | (data[HEADER + idx + 4] << 8));
            h_pol[idx2 + d_in_size / 2].x = static_cast<float>(data[HEADER + idx + 3] | (data[HEADER + idx + 2] << 8));
            h_pol[idx2 + d_in_size / 2].y = static_cast<float>(data[HEADER + idx + 1] | (data[HEADER + idx + 0] << 8));
        }
    }
*/
}

int main(int argc, char *argv[])
{
    try {

        GPUpool mypool;

        unsigned char *inbuf = new unsigned char[BUFLEN];

        std::chrono::time_point<std::chrono::system_clock> send_begin, send_end;
        std::chrono::duration<double> send_elapsed;


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
        int m = 0;
        int n = 0;

        header_s head;

	boost::array<unsigned char, BUFLEN> recv_buf;
        while (m < 10) {

            while (n < 48) {
                len = socket.receive_from(boost::asio::buffer(recv_buf), sender_endpoint);

                static obs_time start_time{head.epoch, head.ref_s};
                // this is ugly, but I don't have a better solution at the moment
                int long_ip = boost::asio::ip::address_v4::from_string((sender_endpoint.address()).to_string()).to_ulong();
                int fpga = ((int)((long_ip >> 8) & 0xff) - 1) * 8 + ((int)(long_ip & 0xff) - 1) / 2;

                send_begin = std::chrono::system_clock::now();
                mypool.get_data(reinterpret_cast<char*>(recv_buf.data()), fpga, start_time, head);
                send_end = std::chrono::system_clock::now();
                send_elapsed += send_end - send_begin;
                n++;
            }
            m++;
            cout << send_elapsed.count() / (double)n << endl;
            send_elapsed = std::chrono::seconds(0);
            n = 0;
        }

    } catch (std::exception &e) {
        cout << "Something bad happened: " << e.what() << endl;
    }
    return 0;
}
