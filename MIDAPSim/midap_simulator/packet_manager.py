import os
import mmap
import numpy as np
import json
import pathlib

from enum import Flag

class PacketFlag(Flag):
    high_prior = 0
    simcache_hit = 1

class PacketManager(object):

    #typedef struct {
    #   volatile int start;			/* index of oldest element              */
    #   volatile int end;			/* index at which to write new element  */
    #   int capacity;
    #   int size;
    #   Packet elems[PKT_BUFFER_SIZE+1];		/* vector of elements                   */
    #} PacketBuffer;
    data_info_type = np.dtype([('start', 'u4'), ('end', 'u4'), ('capacity', 'u4'), ('size', 'u4')])

    def __init__(self, filename, quantized = False):
        self._infoPath = str(pathlib.Path(__file__).parent.resolve()) + "/../../shared/" + filename

        self.use_relaxed_sync = False
        self.use_sim_cache = False

        self._reqReadSize = 0
        self._reqReadCycle = 0

        self._lastCycle = 0

        self._quantized = quantized

        self._pType = self.enum('read', 'write', 'bar_wait', 'bar_signal', 'elapsed', 'terminated')
        
        #typedef struct _Packet{
        #   PacketType type;
        #   uint32_t size;
        #   uint64_t cycle;
        #   uint32_t address;
        #   uint8_t data[8];
        #   uint32_t flags;
        #} Packet;

        f = open(self._infoPath, 'r')
        name = f.readline()
        ib_name = f.readline()
        bi_name = f.readline()
        relaxed_sync_flag = f.readline().split(" ")
        sim_cache_flag = f.readline().split(" ")
        buffer_size = f.readline().split(" ")
        data_size = f.readline().split(" ")
        f.close()

        self.use_relaxed_sync = bool(relaxed_sync_flag[1])
        
        if sim_cache_flag[1] == "true":
            self.use_sim_cache = True

        self._bufferSize = int(buffer_size[1])
        self._dataSize = int(data_size[1])

        if quantized == False:
            self._dataSize = self._dataSize / 4

        self._packetSize = 24 + self._dataSize
        self._dataMaxSize = 2048
        self._packetMaxSize = 24 + self._dataMaxSize

        self.data_type = np.dtype([('type', 'u4'), ('size', 'u4'), ('cycle', 'u8'), ('address', 'u4'), ('flags', 'u4'), ('data', 'u1', (self._dataSize))])

        ibFile = open('/dev/shm' + ib_name.rstrip('\n'), 'r+')
        self._sendBuffer = mmap.mmap(ibFile.fileno(), 0, mmap.PROT_READ | mmap.PROT_WRITE)
        ibFile.close()

        biFile = open('/dev/shm' + bi_name.rstrip('\n'), 'r+')
        self._receiveBuffer = mmap.mmap(biFile.fileno(), 0, mmap.PROT_READ | mmap.PROT_WRITE)
        biFile.close()

        # Check if the connection iss established.
        self.writeRequest(0x0, 4, 0, 0)

    def enum(self, *sequential, **named):
        enums = dict(zip(sequential, range(len(sequential))), **named)
        return type('Enum', (), enums)

    def isEmpty(self, buffer):
        start, end, _, _ = self.readBufInfo(buffer)
        return start == end

    def isFull(self, buffer):
        start, end, _, _ = self.readBufInfo(buffer)
        return (end + 1) % self._bufferSize == start

    def readBufInfo(self, buffer):
        buffer.seek(0)
        data_info = np.array(np.frombuffer(buffer.read(16), dtype=self.data_info_type), dtype=self.data_info_type)

        return data_info['start'], data_info['end'], data_info['capacity'], data_info['size']

    def readPacket(self):
        buffer = self._receiveBuffer
        while self.isEmpty(buffer) == True:
            pass#sleep(0.000000001)

        start, end, capacity, size = self.readBufInfo(self._receiveBuffer)

        buffer.seek(16 + int(start) * self._packetMaxSize)
        data = np.array(np.frombuffer(buffer.read(self._packetSize), dtype=self.data_type), dtype=self.data_type)

        # Increase the read index (start)
        start = (start + 1) % self._bufferSize
        buffer.seek(0)
        buffer.write(start.tobytes())

        return data

    def readPacketIfExist(self):
        buffer = self._receiveBuffer
        if self.isEmpty(buffer) == False:
            start, end, capacity, size = self.readBufInfo(self._receiveBuffer)
            buffer.seek(16 + int(start) * self._packetMaxSize)
            data = np.array(np.frombuffer(buffer.read(self._packetSize), dtype=self.data_type), dtype=self.data_type)
            # Increase the read index (start)
            start = (start + 1) % self._bufferSize
            buffer.seek(0)
            buffer.write(start.tobytes())
            return True, data
        else:
            return False, None

    def writePacket(self, packet):
        buffer = self._sendBuffer

        while self.isFull(buffer) == True:
            pass#sleep(0.000000001)

        start, end, capacity, size = self.readBufInfo(buffer)

        data = np.array(packet, dtype=self.data_type)
        buffer.seek(16 + int(end) * self._packetMaxSize)
        buffer.write(data.tobytes())

        # Increase the write index (end)
        end = (end + 1) % self._bufferSize
        buffer.seek(4)
        buffer.write(end.tobytes())
        buffer.flush()

    def readResponse(self, size):
        # packet = self.readPacket()
        # return packet['data'], packet['cycle']
        received, packet = self.readPacketIfExist()
        if received == True:
            data = packet['data']
            data = np.resize(data, int(self._reqReadSize))

            if self.use_relaxed_sync == False:
                packet['cycle'] = packet['cycle'] + self._reqReadCycle

            self._reqReadSize = 0
            self._reqReadCycle = 0
            return data, packet['cycle'], packet['address']
        else:
            return None, -1, -1


    def readRequest(self, addr, size, cycle, high_prior = True):
        cycle_info = cycle

        if self.use_relaxed_sync == False:
            if cycle > self._lastCycle:
                cycle_info = cycle - self._lastCycle
            else:
                cycle_info = 0

        if self._quantized == False:
            size = size * 4

        packet = np.array((self._pType.read, size, cycle_info, addr, 0, 0), dtype=self.data_type)

        if high_prior == True:
            packet['flags'] = 1 << PacketFlag.high_prior.value

        self.writePacket(packet)

        self._reqReadSize = size
        self._reqReadCycle = cycle

        if cycle > self._lastCycle:
            self._lastCycle = cycle

    def writeRequest(self, addr, size, data, cycle):
        cycle_info = cycle

        if self.use_relaxed_sync == False:
            if cycle > self._lastCycle:
                cycle_info = cycle - self._lastCycle
            else:
                cycle_info = 0

        if self._quantized == False:
            size = size * 4

        packet = np.array((self._pType.write, size, cycle_info, addr, 0, np.resize(data, self._dataSize)), dtype=self.data_type)

        self.writePacket(packet)
        #packet = self.readPacket()

        if cycle > self._lastCycle:
            self._lastCycle = cycle

    def waitRequest(self, wait_id, cycle, cycle_backplane, layer_id):
        cycle_info = cycle
        if self.use_relaxed_sync == False:
            if cycle > self._lastCycle:
                cycle_info = cycle - self._lastCycle
            else:
                cycle_info = 0

        packet = np.array((self._pType.bar_wait, wait_id[1], cycle_backplane, wait_id[0], layer_id, 0), dtype=self.data_type)
        self.writePacket(packet)

        if self.use_relaxed_sync == True:
            packet = self.readPacket()
        else:
            packet['cycle'] = 0

        if cycle > self._lastCycle:
            self._lastCycle = cycle

        return packet['cycle']

    def signalRequest(self, sync_id, cycle, dram_delay, layer_id):
        cycle_info = cycle
        if self.use_relaxed_sync == False:
            if cycle > self._lastCycle:
                cycle_info = cycle - self._lastCycle
            else:
                cycle_info = 0
        packet = np.array((self._pType.bar_signal, sync_id[1], cycle_info, dram_delay, layer_id, 0), dtype=self.data_type)
        self.writePacket(packet)

    def elapsedRequest(self, cycle):
        if cycle >= self._lastCycle + 100:
            cycle_info = cycle
            if self.use_relaxed_sync == False:
                 cycle_info = cycle - self._lastCycle

            packet = np.array((self._pType.elapsed, 0, cycle_info, 0, 0, 0), dtype=self.data_type)
            self.writePacket(packet)

            if cycle > self._lastCycle:
                self._lastCycle = cycle

    def terminatedRequest(self, cycle):
        cycle_info = cycle

        if self.use_relaxed_sync == False:
            cycle_info = cycle - self._lastCycle
        packet = np.array((self._pType.terminated, 0, cycle_info, 0, 0, 0), dtype=self.data_type)
        self.writePacket(packet)



