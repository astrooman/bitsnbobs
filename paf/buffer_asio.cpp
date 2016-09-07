#include <bitset>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

#include <boost/array.hpp>
#include <boost/asio.hpp>
//#include <cufft.h>
#include "pdif_test.hpp"

using std::cout;
using std::endl;

using boost::asio::ip::udp;

#define BUFLEN 7168 + 64
#define HEADER_SIZE 64

int main(int argc, char *argv[])
{
    try {

        union {
            float input;
            int output;
        } medata;


        int tmax = std::thread::hardware_concurrency();
        cout << "Can use a maximum of " << tmax << " threads.\n";

        unsigned char *inbuf = new unsigned char[BUFLEN];

        std::chrono::time_point<std::chrono::system_clock> send_begin, send_end;
        std::chrono::duration<double> send_elapsed;


        boost::asio::io_service io_service;
        udp::endpoint sender_endpoint;

        udp::socket socket(io_service, udp::endpoint(boost::asio::ip::address::from_string("10.17.0.1"), atoi(argv[1])));

        boost::asio::socket_base::reuse_address option(true);
        boost::asio::socket_base::receive_buffer_size option2(9000);
        socket.set_option(option);
        socket.set_option(option2);
        //socket.bind(udp::endpoint(udp::v4(), atoi(argv[1])));

        cout << "Waiting to get something on port " << atoi(argv[1]) << " ..." << endl;

        size_t len;
        int n = 0;

        header_s heads;

        boost::array<unsigned char, BUFLEN> recv_buf;

        short meshort;
        float mefloat;

        while(n < 5) {
            len = socket.receive_from(boost::asio::buffer(recv_buf), sender_endpoint);
            cout << len << endl;
            get_header(recv_buf.data(), heads);
            meshort = (short)((recv_buf[HEADER + 0] << 8) | recv_buf[HEADER + 1]);
            mefloat = (float)meshort;
            cout << meshort << " = " << std::bitset<16>(meshort) << endl;
            medata.input = mefloat;
            cout << mefloat << " = " << std::bitset<32>(medata.output) << endl;

            medata.input = static_cast<float>(static_cast<short>((recv_buf[HEADER + 0] << 8) | recv_buf[HEADER + 1])); 
            cout << medata.input << " = " << std::bitset<32>(medata.output) << endl;
            cout << "Done the wrong way: " << endl;
            medata.input = (float)((recv_buf[HEADER + 0] << 8) | recv_buf[HEADER + 1]);
            cout << medata.input << " = " << std::bitset<32>(medata.output) << endl;
            n++;
        }

    } catch (std::exception &e) {
        cout << "Something bad happened: " << e.what() << endl;
    }
    return 0;
}
