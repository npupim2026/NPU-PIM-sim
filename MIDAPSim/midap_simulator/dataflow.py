def generate_dataflow_info(
        phase = 0,
        loc = None,
        filter_idx = None,
        out_x = 0,
        out_y = 0,
        out_z = 0,
        fmem_idx = 0,
        fmem_row = -1,
        wmem_row = -1,
        broadcast_offset = 0,
        delete_foffset = 0,
        delete_boffset = 0,
        reset = True,
        last = False,
        junk = False,
        fmem_col_broadcast = None,
        ):
    if filter_idx is not None:
        out_z = filter_idx
    if loc is not None:
        if len(loc) == 2:
            out_x, out_y = loc
        elif len(loc) == 3:
            out_x, out_y, out_z = loc

    return Dataflow(
            phase,
            out_x,
            out_y,
            out_z,
            fmem_idx,
            fmem_row,
            wmem_row,
            broadcast_offset,
            delete_foffset,
            delete_boffset,
            reset,
            last,
            junk,
            fmem_col_broadcast=fmem_col_broadcast,
            )

class Dataflow():
    def __init__(
            self,
            phase,
            out_x, # Deprecated 
            out_y, # Deprecated
            out_z, # Instead of (out_x, out_y, out_z) --> use bmem_row, channel_idx, write_info(fmem_addr, tmem_addr)
            fmem_idx,
            fmem_row,
            wmem_row,
            broadcast_offset,
            delete_foffset,
            delete_boffset,
            reset,
            last,
            junk,
            write_fmem_addr = (-1, -1),
            write_tmem_addr = -1,
            transfer_info = None,
            fmem_col_broadcast = None,
            ):

        self.phase = phase
        self.out_x = out_x
        self.out_y = out_y
        self.out_z = out_z
        self.fmem_idx = fmem_idx
        self.fmem_row = fmem_row
        self.wmem_row = wmem_row
        self.broadcast_offset = broadcast_offset
        self.delete_foffset = delete_foffset
        self.delete_boffset = delete_boffset
        self.reset = reset
        self.last = last
        self.junk = junk
        self.first = out_x == 0 and out_y == 0
        self.channel_idx = out_z
        self.write_fmem_addr = write_fmem_addr
        self.write_tmem_addr = write_tmem_addr
        self.transfer_info = transfer_info
        self.fmem_col_broadcast = fmem_col_broadcast

    @property
    def out_loc(self):
        return (self.out_x, self.out_y, self.out_z)
    
    @out_loc.setter
    def out_loc(self, value):
        self.out_x, self.out_y, self.out_z = value

    def __repr__(self):
        phase = None
        p = self.phase
        if p == 0:
            phase = 'None'
        elif p == 1:
            phase = 'Main'
        elif p == 2:
            phase = 'Reduction'
        elif p == 3:
            phase = 'End'
        elif p == 4:
            phase = 'Transfer TMEM'
        else:
           raise ValueError(f"Unknown Phase: {p}")
        fmem_addr = [self.fmem_idx, self.fmem_row]
        wmem_addr = self.wmem_row
        alignment_info = [self.broadcast_offset, self.delete_foffset, self.delete_boffset]
        etc = [self.reset, self.junk, self.last]
        etc = [1 if x else 0 for x in etc]
        out_str = '[Phase: ' + str(phase)
        if p in [0, 3]:
            return out_str + ']'
        if p == 1:
            out_str += ', faddr: {}, waddr: {}, align: {}, del(l,r): {}, '.format(fmem_addr, wmem_addr, alignment_info[0], alignment_info[1:])
        out_str += 'reset: {}, ignore: {}, last: {}, wfaddr: {}, wtaddr: {}'.format(*etc, self.write_fmem_addr, self.write_tmem_addr)
        if p == 2:
           return out_str + ']'
        return out_str
