import numpy as np
import os

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
            if pos <= last_bl:
                bead_in_chrom = len(bead_lims[chrom-1]) - 2
                return bead_in_chrom
            else:
                raise ValueError("Position {} not found in chr{}!".format(pos,
                                                                          chrom))
        else:
            if pos <= last_bl:
                bead_in_chrom = len(bead_lims[chrom-1]) - 2
                return bead_in_chrom
            else:
                raise ValueError("Position {} not found in chr{}!".format(pos,
                                                                          chrom))

def map_chr_pos_to_cont_bead(chrom, pos, bead_lims):

    bead = map_chr_pos_to_bead(chrom, pos, bead_lims)
    n_beads_cumsum = np.cumsum(np.array(map(len, bead_lims)) - 1)

    return bead + np.insert(n_beads_cumsum, 0, 0)[chrom - 1]

def map_pos_to_bead(pos, bead_lims):

    chrom = determine_chr_from_pos(pos)
    print chrom
    return map_chr_pos_to_bead(chrom,
                               pos - np.insert(np.cumsum(chrom_lengths),
                                               0, 0)[chrom - 1],
                               bead_lims)

def write_single_chr_data(matrix, fname):

    n_beads = len(matrix)
    with open(fname, 'w') as opf:
        for i in range(n_beads):
            for j in range(i + 1, n_beads):
                if np.isnan(m[i,j]):
                    continue
                opf.write('{}\t{}\t{}\n'.format(i, j, int(matrix[i,j])))

def is_trans(i, j, n_beads):
    result = True
    tmp = np.insert(n_beads, 0, 0)
    for k in range(len(tmp) - 1):
        if i >= tmp[k] and i < tmp[k+1] and j >= tmp[k] and j < tmp[k+1]:
            result = False

    return result

def write_whole_genome_data(matrix, n_rDNA_beads, n_beads, bead_lims, trans_min, fname):

    rDNA_start_bead = map_chr_pos_to_bead(12, rDNA_to_left, bead_lims)
    rDNA_start_bead += np.cumsum(n_beads)[10]
    print rDNA_start_bead
    n_normal_beads = len(matrix)
    with open(fname, 'w') as opf:
        for i in range(n_normal_beads):
            for j in range(i + 1, n_normal_beads):
                if np.isnan(matrix[i,j]):
                    continue
                if is_trans(i, j, n_beads) and matrix[i,j] < trans_min:
                    continue
                i_shifted = i
                j_shifted = j
                if i > rDNA_start_bead:
                    i_shifted += n_rDNA_beads
                if j > rDNA_start_bead:
                    j_shifted += n_rDNA_beads    
                opf.write('{}\t{}\t{}\n'.format(i_shifted, j_shifted,
                                                int(matrix[i,j])))


class CGRep(object):

    def __init__(self, bin_size=10000, n_rDNA_beads=23, rDNA_locus_length=1500000):

        this_path = os.path.expanduser('~/projects/ensemble_hic/data/eser2017/')
        self.intra_ifs = self.parse_eser_file(this_path + 'AFac_intra.txt')
        self.inter_ifs = self.parse_eser_file(this_path + 'AFac_inter.txt')
        self.cl_cumsums = np.cumsum(chrom_lengths[:,1])
        self.all_ifs = np.concatenate((self.intra_ifs, self.inter_ifs))
        self.all_cont_ifs = self.make_cont_ifs()
        self.bin_size = 10000
        self.n_rDNA_beads = n_rDNA_beads
        self.rDNA_locus_length = rDNA_locus_length
        self.n_beads = self.calculate_n_beads()
        self.bead_lims = self.calculate_bead_lims()
        self.cont_bead_lims = self.calculate_cont_bead_lims()
        self.chrom_ranges = self.calculate_chrom_ranges()
        self.centromere_beads = self.calculate_centromere_beads()
        self.telomere_beads = self.calculate_telomere_beads()
        
    def parse_eser_file(self, fname):

        data = np.loadtxt(fname, dtype=str)

        return np.array([[x[0][3:], x[1], x[2][3:], x[3], x[4]]
                         for x in data]).astype(float).astype(int)

    def make_cont_ifs(self):

        ifs = self.all_ifs.copy()
        for i in range(2, len(chrom_lengths) + 1):
            ifs[ifs[:,0] == i, 1] += self.cl_cumsums[i-2]
            ifs[ifs[:,2] == i, 3] += self.cl_cumsums[i-2]

        return ifs

    def calculate_n_beads(self):

        n_beads = np.ceil((chrom_lengths[:,1].astype(float)) / self.bin_size).astype(int)
        n_beads[11] += self.n_rDNA_beads
        
        return n_beads

    def calculate_bead_lims(self):

        bead_lims = []
        for i, cl in chrom_lengths:
            if i == 12:
                lims = np.linspace(0, cl, self.n_beads[11]+1-self.n_rDNA_beads,
                                   endpoint=False)
                rDNA_start_bead = np.where(lims > rDNA_to_left)[0][0]
                lims = np.concatenate((lims[:rDNA_start_bead - 1],
                                       [rDNA_to_left] * (self.n_rDNA_beads + 1),
                                       lims[rDNA_start_bead:]))
                bead_lims.append(lims)
            else:
                bead_lims.append(np.linspace(0, cl, self.n_beads[i-1]+1,
                                             endpoint=False))
                
        return bead_lims
        
    def calculate_cont_bead_lims(self):

        return self.bead_lims[:1] + [self.bead_lims[i] + self.cl_cumsums[i-1]
                                     for i in range(1, len(chrom_lengths))]

    def calculate_chrom_ranges(self):

        return np.insert(np.cumsum(self.n_beads), 0, 0)

    def calculate_centromere_beads(self):

        cm_beads_b = np.array([map_chr_pos_to_bead(i, centromeres[i-1,0],
                                                   self.bead_lims)
                               for i in range(1,17)])
        cm_beads_b = np.array(cm_beads_b) + np.insert(self.n_beads.cumsum(),
                                                      0, 0)[:-1]
        cm_beads_e = np.array([map_chr_pos_to_bead(i, centromeres[i-1,1],
                                                   self.bead_lims)
                               for i in range(1,17)])
        cm_beads_e = np.array(cm_beads_e) + np.insert(self.n_beads.cumsum(),
                                                      0, 0)[:-1]

        return np.array(zip(cm_beads_b, cm_beads_e))

    def calculate_telomere_beads(self):

        n_beads_cs = self.n_beads.cumsum()

        res = np.hstack(([0],
                         np.array([(n_beads_cs[i]-1, n_beads_cs[i])
                                   for i in range(len(n_beads_cs) - 2)]).ravel(),
                         [n_beads_cs[-1]]))
        res[-1] = n_beads_cs[-1] - 1
        
        return res



