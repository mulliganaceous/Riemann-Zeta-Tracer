/*
 * Unused
 */

// Unix networking
#include "helper.cu"
#include "riemannzeta.cu"
#ifndef __STDC_FORMAT_MACROS
  #include <stdint.h>
  #include <stdio.h>
  #include <stdlib.h>
  #include <iostream>
#endif
#ifndef  __COMPLEX
    #include <complex>
    #include <cuComplex.h>
#endif
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/wait.h>
#include <signal.h>
#define PORT "8081"
#define BACKLOG 16
#define WIDTH 1920
#define HEIGHT 1024
#define ENTRIES (WIDTH*HEIGHT)
#define DEPTH 1024
#define CASCADE 1024
#define TERMS (DEPTH*CASCADE)
#define BATCHES 1000
#define MEMSIZE (sizeof(cuDoubleComplex) * WIDTH * HEIGHT * DEPTH)
#define IMGMEMSIZE (sizeof(unsigned char) * WIDTH * HEIGHT)

// Server code adapted from Beej's Guide to Network Programming

void sigchld_handler(int s)
{
    int saved_errno = errno;
    while (waitpid(-1, NULL, WNOHANG) > 0)
        ;
    errno = saved_errno;
}

void *get_in_addr(struct sockaddr *sa)
{
    switch (sa->sa_family)
    {
    case AF_INET:
        return &(((struct sockaddr_in *)sa)->sin_addr);
    case AF_INET6:
        return &(((struct sockaddr_in6 *)sa)->sin6_addr);
    default:
        throw std::runtime_error("Unknown address family");
    }
}

int server() {
    int sockfd, new_fd;
    struct addrinfo hints, *servinfo, *p;
    struct sockaddr_storage their_addr;
    socklen_t sin_size;
    struct sigaction sa;
    int yes = 1;
    char s[INET6_ADDRSTRLEN];
    int rv;

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    if ((rv = getaddrinfo(NULL, PORT, &hints, &servinfo)) != 0)
    {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
        return EXIT_FAILURE;
    }

    for (p = servinfo; p != NULL; p = p->ai_next)
    {
        if ((sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1)
        {
            perror("server: socket");
            continue;
        }
        if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1)
        {
            perror("setsockopt");
            return EXIT_FAILURE;
        }
        if (bind(sockfd, p->ai_addr, p->ai_addrlen) == -1)
        {
            close(sockfd);
            perror("server: bind");
            continue;
        }
        break;
    }

    freeaddrinfo(servinfo);
    if (p == NULL)
    {
        fprintf(stderr, "server: failed to bind\n");
        return EXIT_FAILURE;
    }
    if (listen(sockfd, BACKLOG) == -1)
    {
        perror("listen");
        return EXIT_FAILURE;
    }
    sa.sa_handler = sigchld_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    if (sigaction(SIGCHLD, &sa, NULL) == -1)
    {
        perror("sigaction");
        return EXIT_FAILURE;
    }

    printf("server: waiting for connections...\n");
    while (1)
    {
        sin_size = sizeof their_addr;
        new_fd = accept(sockfd, (struct sockaddr *)&their_addr, &sin_size);
        if (new_fd == -1)
        {
            perror("accept");
            continue;
        }
        printf("server: the new fd is %d\n", new_fd);
        inet_ntop(their_addr.ss_family, get_in_addr((struct sockaddr *)&their_addr), s, sizeof s);
        printf("server: got connection from %s\n", s);
        // Compute the Riemann zeta function
        while (1)
        {
            close(sockfd);
            // Allocate host memory for the plot
            cuDoubleComplex *h_plot;
            getStatus(cudaMallocHost(&h_plot, MEMSIZE/DEPTH), "Failed to allocate cudaMallocHost! ");
            cuDoubleComplex *h_input;
            getStatus(cudaMallocHost(&h_input, MEMSIZE/DEPTH), "Failed to allocate cudaMallocHost! ");
            
            cudaZeta(h_plot, 0, 0, 16, 16, h_input);
            for (int k = 0; k < BATCHES; k++)
            {
                long long bytes_to_send = sizeof(cuDoubleComplex) * WIDTH * HEIGHT;
                long long interval = bytes_to_send / BATCHES;
                printf("Preparing to send %lld bytes\n", bytes_to_send);
                ssize_t sent_bytes = send(new_fd, h_plot + (k * interval)/sizeof(cuDoubleComplex), interval, 0);
                sleep(0.1);
                if (sent_bytes == -1)
                {
                    printf("neg one\n");
                }
                else
                {
                    printf("Sent %ld bytes\n", sent_bytes);
                }
                sleep(1);
            }
        }
        close(new_fd);
    }
    return EXIT_SUCCESS;
}