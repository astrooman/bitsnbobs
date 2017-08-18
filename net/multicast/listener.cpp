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
using std::cout;
using std::endl;

int main(int argc, char *argv[]) {

    int sfd, numbytes, retval;
    addrinfo hints, *servinfo, *iserv;
    memset(&hints, 0, sizeof(hints));

    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;

    if (( retval = getaddrinfo(NULL, PORT, &hints, &servinfo)) != 0) {
        cerr << "Error on getaddrinfo: " << gai_strerror(retval) << endl;
        exit(EXIT_FAILURE);
    }

    for (iserv = servinfo; iserv != NULL; iserv=iserv->ai_next) {
        if((sfd = socket(iserv->ai_family, iserv->ai_socktype, iserv->ai_protocol)) == -1) {
            cerr << "Error on socket creation" << endl;
            continue;
        }

        if (bind(sfd, iserv->ai_addr, iserv->ai_addrlen) == -1) {
            cerr << "Error on bind" << endl;
            continue;
        }

        break;
    }

    if (iserv == NULL) {
        cerr << "Failed to bind the socket" << endl;
        exit(EXIT_FAILURE);
    }

    freeaddrinfo(servinfo);

    sockaddr_storage senderaddr;
    socklen_t addrlen;
    memset(&addrlen, 0, sizeof(addrlen));
    addrlen = sizeof(senderaddr);

    unsigned char *buffer = new unsigned char[128];
    if((numbytes = recvfrom(sfd, buffer, 128, 0, (struct sockaddr*)&senderaddr, &addrlen)));

    cout << buffer << endl;

    close(sfd);

}
