#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "errors.hpp"

#include <assert.h>
#include "ascii_header.h"
#include "dada_client.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "futils.h"
#include "ipcio.h"
#include "multilog.h"

using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::string;
using std::thread;

struct DadaContext {
    bool verbose;
    dada_hdu_t *hdu;
    multilog_t *log;
    char *headerfile;
    char *obsheader;
    char headerwritten;
    unsigned int device;
    cudaStream_t stream;
    unsigned char *devicememory;
    uint64_t bytestransferred;
    unsigned int repwrite;
    unsigned int buffno;
};

#define NCHANS 512

void DadaCudaHostTransfer(dada_client_t *client, void *to, unsigned int buffno, size_t size, cudaStream_t stream) {
    DadaContext *tmpctx = reinterpret_cast<DadaContext*>(client->context);
    cudaCheckError(cudaMemcpyAsync(to, tmpctx->devicememory + (buffno - 1) * size, size, cudaMemcpyDeviceToHost, stream));
    cudaCheckError(cudaStreamSynchronize(stream));
}

int DadaPafOpen(dada_client_t *client) {
    cout << "I'm in the opening function!" << endl;
    assert(client != 0);
    DadaContext *tmpctx = reinterpret_cast<DadaContext*>(client->context);

    multilog(tmpctx->log, LOG_INFO, "Running DadaPafOpen\n");

    client->transfer_bytes = 0; //8 * NCHANS * 131072 * sizeof(unsigned char);
    client->optimal_bytes=0;

    tmpctx->obsheader = (char *)malloc(sizeof(char) * DADA_DEFAULT_HEADER_SIZE);
    if (!tmpctx->obsheader) {
        multilog(tmpctx->log, LOG_ERR, "Could not allocate memory fo the header!\n");
        return EXIT_FAILURE;
    }

    if (fileread(tmpctx->headerfile, tmpctx->obsheader, DADA_DEFAULT_HEADER_SIZE) < 0) {
        free (tmpctx->obsheader);
        multilog (tmpctx->log, LOG_ERR, "could not read ASCII header from %s\n", tmpctx->headerfile); 
        return (EXIT_FAILURE);
    }

    tmpctx->headerwritten = 0;
    return EXIT_SUCCESS;
}

int DadaPafClose(dada_client_t *client, uint64_t bytes_written) {
    assert(client != 0);
    DadaContext *tmpctx = reinterpret_cast<DadaContext*>(client->context);

    multilog(client->log, LOG_INFO, "Running DadaPafClose\n");
    assert(tmpctx != 0);

    free (tmpctx->obsheader);
    return 0;
}

// NOTE: used for transferring the header
int64_t DadaPafWrite(dada_client_t *client, void *buffer, uint64_t bytes) {
    cout << "I'm in the writer function" << endl;
    assert(client != 0);
    DadaContext *tmpctx = reinterpret_cast<DadaContext*>(client->context);

        if (!tmpctx->headerwritten) {
            uint64_t headersize = ipcbuf_get_bufsz(tmpctx->hdu->header_block);
            cout << "Header size: " << headersize << endl;
            char *header = ipcbuf_get_next_write(tmpctx->hdu->header_block);
            memcpy(header, tmpctx->obsheader, headersize);
        
            if (ipcbuf_mark_filled(tmpctx->hdu->header_block, headersize) < 0) {
                multilog(tmpctx->log, LOG_ERR, "Could not mark the filled Header Block!\n");
                return -1;
            }
            tmpctx->headerwritten = 1;
            cout << "Written the header" << endl;
        } else {
            memset(buffer, 0, bytes);
        }

    return bytes;
}

int64_t DadaPafWriteBlockCuda(dada_client_t *client, void *buffer, uint64_t bytes, uint64_t blockid) {
    cout << "I'm in the block writer function" << endl;
    cout << "Block ID: " << blockid << endl;
    cout << "Bytes to transfer: " << bytes << endl;    
    assert(client != 0);
    DadaContext *tmpctx = reinterpret_cast<DadaContext*>(client->context);

    while(!tmpctx -> buffno) {

    }
    cout << "Transferring buffer " << tmpctx -> buffno << " into block " << blockid << " of size " << bytes << " bytes " << endl;
    DadaCudaHostTransfer(client, buffer, tmpctx -> buffno, bytes, 0);
    tmpctx->bytestransferred  += bytes;
    tmpctx -> buffno = 0;
    return bytes;
}

class DadaClass {

    public:
        DadaClass(void) = delete;
        DadaClass(string infile, char *inheader) {
            
            client_ = 0;

            ifstream indata(infile.c_str(), ifstream::binary);

            if (!indata) {
                cerr << "Could not open the file " << infile << endl;
                cerr << "Will now quit!" << endl;
                exit(EXIT_FAILURE);
            }

            // NOTE: I'm reading the header in and treating it like data
	    indata.seekg(0, indata.end);
            size_t datasize = indata.tellg();
            indata.seekg(0, indata.beg);

            buffsize_ = NCHANS * 131072 * sizeof(unsigned char);  

            if (datasize < buffsize_) {
                cerr << "There is less data than there should be" << endl;
                cerr << "Will now quit!" << endl;
                exit(EXIT_FAILURE);
            }

            hostbuff_ = new unsigned char[buffsize_];
            cudaCheckError(cudaMalloc((void**)&devbuff_,  2 * buffsize_));

            indata.read(reinterpret_cast<char*>(hostbuff_), buffsize_);
            cudaCheckError(cudaMemcpy(devbuff_, hostbuff_, buffsize_, cudaMemcpyHostToDevice));
            cudaCheckError(cudaMemcpy(devbuff_ + buffsize_, hostbuff_, buffsize_, cudaMemcpyHostToDevice));

            dadakey_ = DADA_DEFAULT_BLOCK_KEY; 
            dcontext_.headerfile = inheader;
            dcontext_.devicememory = devbuff_;
            dcontext_.buffno = 0;
            dcontext_.log = multilog_open("PAF DADA logger\n", 0);
            multilog_add(dcontext_.log, stderr);
            dcontext_.hdu = dada_hdu_create(dcontext_.log);
            dada_hdu_set_key(dcontext_.hdu, dadakey_);

            if (dada_hdu_connect(dcontext_.hdu) < 0) {
                multilog(dcontext_.log, LOG_ERR, "Could not connect to the HDU!\n");
                exit(EXIT_FAILURE);
            }

            if (dada_hdu_lock_write(dcontext_.hdu) < 0) {
                multilog(dcontext_.log, LOG_ERR, "Could not lock write on the HDU!\n");
                exit(EXIT_FAILURE);
            }

            client_ = dada_client_create();
            client_->context = &dcontext_;
            client_->log = dcontext_.log;
            client_->data_block = dcontext_.hdu->data_block;
            client_->header_block = dcontext_.hdu->header_block;

            client_->open_function = DadaPafOpen;
            client_->io_function = DadaPafWrite;
            client_->io_block_function = DadaPafWriteBlockCuda;
            client_->close_function = DadaPafClose;
            client_->direction = dada_client_writer;

            //if (dada_client_write(client_) < 0) {
            //    multilog(dcontext_.log, LOG_ERR, "Error during transfer\n");
            //    exit(EXIT_FAILURE);
            //}

            std::thread dadathread = std::thread(&DadaClass::DadaSave, this);

//            dadathread.join();

            unsigned char ibuff = 0;

            for(;; ++ibuff) {
                dcontext_.buffno = (ibuff % 2) + 1;
                std::this_thread::sleep_for(std::chrono::seconds(7));
            }

            if (dada_hdu_unlock_write(dcontext_.hdu) < 0) {
                multilog(dcontext_.log, LOG_ERR, "could not unlock read on hdu\n");
                exit(EXIT_FAILURE);
            }

            if (dada_hdu_disconnect(dcontext_.hdu) < 0) {
                multilog(dcontext_.log, LOG_ERR, "could not disconnect from HDU\n");
                exit(EXIT_FAILURE);
            }


        }

        ~DadaClass(void) {
            delete [] hostbuff_;
            cudaCheckError(cudaFree(devbuff_));
        }

        void DadaSave(void) {
            if (dada_client_write(client_) < 0) {
                multilog(dcontext_.log, LOG_ERR, "Error during transfer\n");
                exit(EXIT_FAILURE);
            }
        }

    private:
        dada_client_t *client_;
        key_t dadakey_;
        DadaContext dcontext_;

        size_t buffsize_;

        unsigned char *devbuff_;
        unsigned char *hostbuff_;

    protected:
        
};

int main(int argc, char *argv[])
{

    if (argc < 3) {
        cerr << "Not enough arguments!" << endl;
        exit(EXIT_FAILURE);
    }   

    string filename = string(argv[1]);
    char *inputheader = strdup(argv[2]);

    DadaClass test(filename, inputheader);

    /*

    ifstream indata(filename.c_str(), ifstream::binary);

    if (!indata) {
        cerr << "Could not open the file " << filename << endl;
        exit(EXIT_FAILURE);
    }

    DadaContext dcontext;
    key_t dadakey;
    dada_client_t *client = 0;  

    // NOTE: Screw the header
    indata.seekg(0, indata.end);
    size_t datasize = indata.tellg();
    indata.seekg(0, indata.beg);
    
    size_t buffsize = NCHANS  * 131072 * sizeof(unsigned char); 

    if (datasize < buffsize) {
        cerr << "There is less data than there should be" << endl;
        exit(EXIT_FAILURE); 
    }

    unsigned char *hostbuf = new unsigned char[buffsize];
    unsigned char *devbuf;

    cudaCheckError(cudaMalloc((void**)&devbuf, buffsize));

    indata.read(reinterpret_cast<char*>(hostbuf), buffsize);

    cudaCheckError(cudaMemcpy(devbuf, hostbuf, buffsize, cudaMemcpyHostToDevice));
    cout << inputheader << endl;

    dadakey = DADA_DEFAULT_BLOCK_KEY; 
    dcontext.headerfile = inputheader;
    dcontext.devicememory = devbuf;
    dcontext.log = multilog_open("PAF DADA logger\n", 0);
    multilog_add(dcontext.log, stderr);
    dcontext.hdu = dada_hdu_create(dcontext.log);
    dada_hdu_set_key(dcontext.hdu, dadakey);
    
    if (dada_hdu_connect(dcontext.hdu) < 0) {
        multilog(dcontext.log, LOG_ERR, "Could not connect to the HDU!\n");
        exit(EXIT_FAILURE);
    }

    if (dada_hdu_lock_write(dcontext.hdu) < 0) {
        multilog(dcontext.log, LOG_ERR, "Could not lock write on the HDU!\n");
        exit(EXIT_FAILURE);
    }

    client = dada_client_create();
    client->context = &dcontext;
    client->log = dcontext.log;
    client->data_block = dcontext.hdu->data_block;
    client->header_block = dcontext.hdu->header_block;

    client->open_function = DadaPafOpen;
    client->io_function = DadaPafWrite;
    client->io_block_function = DadaPafWriteBlockCuda;
    client->close_function = DadaPafClose;
    client->direction = dada_client_writer;

    if (dada_client_write(client) < 0) {
        multilog(dcontext.log, LOG_ERR, "Error during transfer\n");
        return EXIT_FAILURE;
    }

    if (dada_hdu_unlock_write(dcontext.hdu) < 0) {
        multilog (dcontext.log, LOG_ERR, "could not unlock read on hdu\n");
        return EXIT_FAILURE;
    }

    if (dada_hdu_disconnect(dcontext.hdu) < 0) {
        multilog (dcontext.log, LOG_ERR, "could not disconnect from HDU\n");
        return EXIT_FAILURE;
    }

    */
    return 0;
}

