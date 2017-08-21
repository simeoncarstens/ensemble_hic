import numpy
import matplotlib.pyplot as plt

noise = 'junier'
noise = 'poisson'

n_beads = 45
third = n_beads / 3
numpy.zeros((n_beads, n_beads))

s1 = slice(0, third)
s2 = slice(0, 2 * third)
s3 = slice(0, 3 * third)
s4 = slice(third, 2 * third)
s5 = slice(third, 3 * third)
s6 = slice(2 * third, 3 * third)

frequencies = numpy.array([0.2, 0.15, 0.05, 0.2, 0, 0.4])
slices = [s1, s2, s3, s4, s5, s6]
lengths = numpy.array([third, 2 * third, 3 * third, third, 2 * third, third])

contributions = []
for s, l, f in zip(slices, lengths, frequencies):
    c = numpy.zeros((n_beads, n_beads))
    if noise == 'junier':
        c[s, s] = (1 + numpy.random.uniform(low=-10, high=10, size=(l, l))) * f / l
        c[c<0] = 0.0
    if noise == 'poisson':
        c[s, s] = f
    contributions.append(c)

if noise == 'poisson':
    noisy_contributions = numpy.random.poisson(contributions * 10)
if noise == 'junier':
    noisy_contributions = contributions
result = numpy.sum(noisy_contributions, 0)

plt.matshow(result, cmap=plt.cm.jet)
plt.colorbar()
plt.show()

if True:
    with open(os.path.expanduser('~/projects/hic/py/hicisd2/ensemble_scripts/bau5C_test/mockTAD_data_{}beads_{}noise.txt'.format(n_beads, noise)), 'w') as opf:
        for i in range(n_beads):
            for j in range(i + 1, n_beads):
                opf.write('{}\t{}\t{}\n'.format(i, j, result[i,j]))
    
