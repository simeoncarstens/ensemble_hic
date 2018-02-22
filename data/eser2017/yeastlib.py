import numpy as np

rDNA_to_left = 450000
## from USCS Genome Browser, SacCer3 April 2011 assembly
chrom_lengths = np.array([[      1,  230218],
                          [      2,  813184],
                          [      3,  316620],
                          [      4, 1531933],
                          [      5,  576874],
                          [      6,  270161],
                          [      7, 1090940],
                          [      8,  562643],
                          [      9,  439888],
                          [     10,  745751],
                          [     11,  666816],
                          [     12, 1078177],
                          [     13,  924431],
                          [     14,  784333],
                          [     15, 1091291],
                          [     16,  948066]])

centromeres = np.array([[151465,151582],
                        [238207,238323],
                        [114385,114501],  
                        [449711,449821],  
                        [151987,152104],  
                        [148510,148627],  
                        [496920,497038],  
                        [105586,105703],  
                        [355629,355745],  
                        [436307,436425],  
                        [440129,440246],  
                        [150828,150947],  
                        [268031,268149],  
                        [628758,628875],  
                        [326584,326702],  
                        [555957,556073]  
                        ])

def determine_chr_from_pos(pos):
    cl_cumsums = np.cumsum(chrom_lengths)
    return np.where(cl_cumsums < pos)[0][-1] + 1

def map_chr_pos_to_bead(chrom, pos, bead_lims):
    lower =  np.where(bead_lims[chrom - 1] <= pos)[0]
    if len(lower) < len(bead_lims[chrom - 1]):
        bead_in_chrom = lower[-1]
        return bead_in_chrom
    else:
        last_bl = np.ceil(2 * bead_lims[chrom-1][-1] - bead_lims[chrom-1][-2])
        if not chrom == 16:
            print pos, last_bl
            if pos <= last_bl:
                bead_in_chrom = len(bead_lims[chrom-1]) - 1
                return bead_in_chrom
            else:
                raise ValueError("Position {} not found in chr{}!".format(pos,
                                                                          chrom))
        else:
            if pos <= last_bl:
                bead_in_chrom = len(bead_lims[chrom-1])
                return bead_in_chrom
            else:
                raise ValueError("Position {} not found in chr{}!".format(pos,
                                                                          chrom))

def map_pos_to_bead(pos, bead_lims):

    chrom = determine_chr_from_pos(pos)
    return map_chr_pos_to_bead(chrom,
                               pos - np.insert(np.cumsum(chrom_lengths),
                                               0, 0)[chrom - 1],
                               bead_lims)
