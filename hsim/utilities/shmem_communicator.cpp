#include "shmem_communicator.h"

ShmemCommunicator::~ShmemCommunicator() {
    int ret;

    munmap(send_buffer, sizeof(PacketBuffer));
    munmap(recv_buffer, sizeof(PacketBuffer));

    shm_unlink(ib_name);
    shm_unlink(bi_name);

    std::string del_file = "rm -rf /dev/shm/" + std::string(ib_name);

    ret = system(del_file.c_str());
    if (ret < 0)
        std::cout << "Failed to delete " << ib_name << std::endl;

    del_file = "rm -rf /dev/shm/" + std::string(bi_name);
    ret = system(del_file.c_str());
    if (ret < 0)
        std::cout << "Failed to delete " << bi_name << std::endl;

    delete[] ib_name;
    delete[] bi_name;
}

bool ShmemCommunicator::is_empty() {
    return pb_is_empty(recv_buffer);
}

int ShmemCommunicator::send_packet(Packet * pPacket) {
    while (pb_is_full(send_buffer)){nanosleep(&ts, NULL); };
    pb_write(send_buffer, pPacket);

	return sizeof(pPacket);
}

int ShmemCommunicator::irecv_packet(Packet * pPacket) {
    if (pb_is_empty(recv_buffer))
        return 0;
    else
        return pb_read(recv_buffer, pPacket);
}

int ShmemCommunicator::recv_packet(Packet * pPacket) {
    while (pb_is_empty(recv_buffer)) { nanosleep(&ts, NULL); };
    pb_read(recv_buffer, pPacket);

	return sizeof(pPacket);
}

bool ShmemCommunicator::prepare_connection(const char *name, int packet_size) {
    struct passwd *pd = getpwuid(getuid());

    ib_name = new char[64];
	bi_name = new char[64];

	srand((unsigned int) time(0));

	int rand_val = rand();
    sprintf(ib_name, "/%s_shmem_ib_%d_%d",pd->pw_name, 1, rand_val);
    sprintf(bi_name, "/%s_shmem_bi_%d_%d",pd->pw_name, 1, rand_val);

    std::stringstream filePath;
    filePath << ROOT_PATH << "/shared/.args.shmem.dat_1";

    std::ofstream ofs(filePath.str());
    assert(ofs.is_open());
    ofs << name << "\n" << ib_name << "\n" << bi_name << "\n";

    ofs << "relaxed_sync 1 \n";
    ofs << "sim_cache false \n";
    ofs << "buffer_size " << PKT_BUFFER_SIZE << "\n";
    ofs << "packet_sizes " << packet_size << "\n";

    ofs.close();

    send_buffer = (PacketBuffer*)establish_shm_segment(bi_name, sizeof(PacketBuffer));
    memset(send_buffer, 0x0, sizeof(PacketBuffer));
    recv_buffer = (PacketBuffer*)establish_shm_segment(ib_name, sizeof(PacketBuffer));
    memset(recv_buffer, 0x0, sizeof(PacketBuffer));

    pb_init(send_buffer);
    pb_init(recv_buffer);

    if (!send_buffer || !recv_buffer) {
        return false;
    }

    return true;
}

bool ShmemCommunicator::wait_connection(){
    std::cout << "Wait for the reponse from ISS...\n";
    Packet p;
    recv_packet(&p);
    std::cout << "Shmem is established.\n";
    return true;
}

void *ShmemCommunicator::establish_shm_segment(char *name, int size) {
    int fd;

    fd = shm_open(name, O_RDWR | O_CREAT, 0600);
    if (fd < 0)
        std::cerr << "shm_open fails with " << name << "\n";
    if (ftruncate(fd, size) < 0)
        std::cerr << "ftruncate() shared memory segment\n";
    void *segment =	(void *) mmap(NULL, size, PROT_READ | PROT_WRITE,	MAP_SHARED, fd, 0);
    if (segment == MAP_FAILED)
        std::cerr << "mapping shared memory segment\n";
    close(fd);

    return segment;
}
