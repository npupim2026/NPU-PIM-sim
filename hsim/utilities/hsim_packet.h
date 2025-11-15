#ifndef __HSIM_PACKET_H
#define __HSIM_PACKET_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>

typedef enum {
    packet_read,        /* blocking */
    packet_write,       /* blocking */
    packet_bar_wait,
    packet_bar_signal,
    packet_elapsed,
    packet_terminated,
} PacketType;

typedef enum {
    flag_high_prior = 0,
    flag_simcache_hit = 1,
} PacketFlagBits;

#define MAX_DATA_SIZE   (2048)

typedef struct _Packet{
    PacketType type;
    uint32_t size;
    uint64_t cycle;
    uint32_t address;
    uint32_t flags;
    uint8_t data[MAX_DATA_SIZE];
} Packet;

#ifdef __cplusplus
}
#endif

#endif
