import numpy as np

# start of Tsix TAD
coords_min = 100378306
# end of Xist TAD
coords_max = 101298738


'''
1) ChIP-seq peaks from https://www.encodeproject.org/experiments/ENCSR362VNF/
'''
def load_and_filter(filename):
    encode_peaks_bed = np.loadtxt(filename, dtype=str)

    # filter for genomic region of itnerest
    encode_peaks = np.array([(int(x[1]), int(x[2])) 
                             for x in encode_peaks_bed if x[0] == 'chrX' 
                             and int(x[1]) >= coords_min 
                             and int(x[2]) <= coords_max
                             ])
    return encode_peaks

encode_peaks_repl1 = load_and_filter('ENCFF001ZQW.broadPeak.bed')
encode_peaks_repl2 = load_and_filter('ENCFF001ZQX.broadPeak.bed')
encode_peaks = np.concatenate((encode_peaks_repl1, encode_peaks_repl2))

'''
2) ChIP-seq peaks from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM699165
("CTCF-Mediated Functional Chromatin Interactome in Pluripotent Cells", Handoko et al., Nat. Gen. (2012)")

Steps before this script:
1) Download and extract GSM699165_CME016.peak.txt.gz
2) run "tail -n +4 GSM699165_CME016.peak.txt | grep chrX | awk '{print $1 " " $3 " " $4}' > GSM699165_CME016.peak.chrX.bed"
3) run "liftover GSM699165_CME016.peak.chrX.bed mm8ToMm9.over.chain GSM699165_CME016.peak.chrX.mm9.bed unMapped"
'''

handoko_peaks = np.loadtxt('GSM699165_CME016.peak.chrX.mm9.bed', dtype=int, usecols=(1,2))
handoko_peaks = handoko_peaks[np.logical_and(handoko_peaks[:,0] >= coords_min,
                                             handoko_peaks[:,1] <= coords_max)]

'''
3) ChIP-seq enrichments from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM560352
("Mediator and Cohesin Connect Gene Expression and Chromatin Architecture", Kagey et al., Nature (2010)")
'''
from pyliftover import LiftOver
lo = LiftOver('mm8', 'mm9')

kagey_enrichments = open('MEF_CTCF_min0.5.WIG').readlines()
start_chrX = kagey_enrichments.index('variableStep chrom=chrX span=25\n') + 1
for i, line in enumerate(kagey_enrichments[start_chrX + 1:]):
    if 'chr' in line:
        break
end_chrX = start_chrX + i + 1
kagey_enrichments = kagey_enrichments[start_chrX:end_chrX]
kagey_enrichments = np.array([map(float, x.strip().split('\t')) for x in kagey_enrichments])
kagey_enrichments_lifted = []
for coord, enrichment in kagey_enrichments:
    lifted_coord = lo.convert_coordinate('chrX', coord)
    if len(lifted_coord) == 1:
        lifted_coord = lifted_coord[0][1]
        if coords_min <= lifted_coord <= coords_max:
            kagey_enrichments_lifted.append((lifted_coord, enrichment))
kagey_enrichments_lifted = np.array(kagey_enrichments_lifted)


import matplotlib.pyplot as plt
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.bar(encode_peaks[:,0], np.ones(len(encode_peaks)), encode_peaks[:,1] - encode_peaks[:,0],
        label="ENCODE (2014)")
ax2.bar(handoko_peaks[:,0], np.ones(len(handoko_peaks)), handoko_peaks[:,1] - handoko_peaks[:,0],
        label="Handoko et al. (2011)")
ax3.bar(kagey_enrichments_lifted[:,0], kagey_enrichments_lifted[:,1], label="Kagey et al. (2010)")

ax1.legend()
ax2.legend()
ax3.legend()
ax3.set_xlabel('genomic coordinate')
ax1.set_yticks(())
ax2.set_yticks(())
ax3.set_ylabel('enrichment')
for ax in (ax1, ax2, ax3):
    ax.axvline(100699670, color='red', linewidth=5, alpha=0.5)

plt.show()
