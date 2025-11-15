#ifndef ShmemCommunicator_H_
#define ShmemCommunicator_H_

#include "packet_buffer.h"
#include "hsim_packet.h"
#include "configurations.h"

#include <time.h>
#include <sys/mman.h>
#include <iostream>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sstream>
#include <fstream>
#include <assert.h>

extern Configurations cfgs;

class ShmemCommunicator {
  public:
 
	ShmemCommunicator() {
		ts.tv_sec = 0;
		ts.tv_nsec = 1;
	};
    ~ShmemCommunicator();

    bool is_empty();
    int send_packet(Packet* pPacket);
    int irecv_packet(Packet* pPacket);
    int recv_packet(Packet* pPacket);
	void *establish_shm_segment(char *name, int size);
	bool prepare_connection(const char *name, int packet_size);
	bool wait_connection();

  private:
	PacketBuffer *send_buffer;
	char *bi_name;
    PacketBuffer *recv_buffer;
	char *ib_name;

	struct timespec ts;
};

#endif /* ShmemCommunicator_H_ */
