#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include <iostream>
#include <cstdlib>
#include <string>

#include "paf_metadata.h"

using namespace std;

#define PORTNUM 26666
#define IPADDR "130.155.182.74"  //pktos01
//#define IPADDR "127.0.0.1"   //pktos01
#define METALEN 726


int main(int argc, const char *argv[])
{
    //Connect to UDP port
    struct sockaddr_in sock_pktos;
    int ii,s,nrecv,ntot;
    socklen_t slen=sizeof(sock_pktos);
    char buf[65535];
    metadata paf_meta;

    if((s=socket(AF_INET, SOCK_DGRAM, 0)) == -1)
    {
        fprintf(stderr, "Could not create socket.\n");
        exit(1);
    }

    //Format addr struct
    fprintf(stderr, "Starting logfile...\n");
    sock_pktos.sin_family = AF_INET;
    sock_pktos.sin_port = htons(PORTNUM);
    sock_pktos.sin_addr.s_addr = htonl( INADDR_ANY);

    bind(s, (struct sockaddr *) &sock_pktos, sizeof(sock_pktos));

    //Read from UDP port
    ntot=0;
    cout << "#Timestamp\tBeamNum\tRA\tDec\tSource" << endl;
    while(ntot<10*METALEN)
    {
        nrecv=recvfrom(s, buf, 4096, 0, (struct sockaddr*) &sock_pktos, &slen);
        ntot+=nrecv;

        if(nrecv==-1)
        {
            fprintf(stderr, "UDP received failed.\n");
            exit(1);
        }
       //Parse string
       string metastr(buf);
       paf_meta.getMetaData(metastr, 0);
       cout << paf_meta.timestamp << "\t";
       cout << paf_meta.beam_num << "\t";
       cout << paf_meta.beam_ra << "\t";
       cout << paf_meta.beam_dec << "\t";
       cout << paf_meta.target_name << endl;
    }

    return 0;
}
