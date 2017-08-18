#include <cstdlib>
#include <cstring>
#include <iostream>

#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#define PORT "22222"

using std::cerr;
using std::endl;

int main(int argc, char *argv[]) {

    int sfd, numbytes, retval;

    addrinfo hints, *servinfo, *iserv;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;

    if ((retval = getaddrinfo(argv[1], PORT, &hints, &servinfo)) != 0 ) {
        cerr << "Error on getaddrinfo: " << gai_strerror(retval) << endl;
        exit(EXIT_FAILURE);
    }

    for (iserv = servinfo; iserv != NULL; iserv=iserv->ai_next) {
        if ((sfd = socket(iserv->ai_family, iserv->ai_socktype, iserv->ai_protocol)) == -1) {
            cerr << "Error on socket creation" << endl;
            continue;
        }

        break;
    }


    if (iserv == NULL) {
        cerr << "Failed to create the socket" << endl;
        exit(EXIT_FAILURE);
    }

    int broadcast = 1;
    //char broadcast = '1';

    if(setsockopt(sfd, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast)) == -1) {
        cerr << "Could not set the broadcast option" << endl;
        exit(EXIT_FAILURE);
    }

    if((numbytes = sendto(sfd, argv[2], strlen(argv[2]), 0, iserv->ai_addr, iserv->ai_addrlen)) == -1) {
        cerr << "Error on send" << endl;
        exit(EXIT_FAILURE);
    }

    freeaddrinfo(servinfo);

    close(sfd);

}
