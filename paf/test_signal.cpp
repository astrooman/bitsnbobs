#include <csignal>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include <unistd.h>

using std::cout;
using std::endl;
using std::mutex;
using std::thread;
using std::unique_ptr;
using std::vector;

class GPUpool;

class Oberpool
{
    private:
        
        int ngpus;

        vector<unique_ptr<GPUpool>> gpuvector;
        vector<thread> threadvector;

    protected:

    public:

        Oberpool(void) = delete;
        Oberpool(int n);
        Oberpool(const Operpool &inpool) = delete;
        Oberpool& operator=(cosnt Oberpool &inpool) = delete;
        Oberpool(Oberpool &&inpool) = delete;
        Oberpool& operator=(Oberpool &&inpool) = delete;
        ~Oberpool(void);

        void signal_handler(int signum);
};

Oberpool::Oberpool(int n) : ngpus(n)
{

    for (int ii = 0; ii < ngpus; ii++) {
        gpuvector.push_back(unique_ptr<GPUpool>(new GPUpool(ii)));
    }

    for (int ii = 0; ii < ngpus; ii++) {
        threadvector.push_back(thread(&GPUpool::execute, std::move(gpuvector[ii])));
    signal(SIGINT, Oberpool::signal_handler);
    }
}

Oberpool::~Oberpool(void)
{
    for (int ii = 0; ii < ngpus; ii++) {
        threadvector[ii].join();
    }
}

void Oberpool::signal_handler(int signum) {

    for (int ii = 0; ii < ngpus; ii++)


}

class GPUpool
{
    private:

        bool working;
        int wid;

    protected:

    public:

        GPUpool(void) = delete;
        GPUpool(int id);
        GPUpool(const GPUpool &inpool) = delete;
        GPUpool& operator=(const GPUpool &inpool) = delete;
        GPUpool(GPUpool &&inpool) = delete;
        GPUpool& operator=(const GPUpool &&inpool) = delete;
};

GPUpool::GPUpool(int id) : wid(id) {

    cout << "Starting worker " << wid << endl;

    while(working)
        sleep(1);

}

GPUpool::~GPUpool(void) {

    cout << "Destroying the worker " << wid << endl;

}

int main(int argc, char *argv[])
{

    return 0;
}


