import matplotlib.pyplot as plt
import numpy as np

# create data
x = np.arange(4)
randoms = [0.25, 0.25, 0.33, 0.33]
baselines = [
    0.7838926174496643,
    0.7179487179487178,
    0.9419777341191882,
    0.889409984871407
]
TSTR = [
    0.668456375838926,
    0.6452991452991453,
    0.8749836280288147,
    0.8015128593040848,
]
TRTS = [
    0.6734899328859061,
    0.641025641025641,
    0.9354289456450557,
    0.837216338880484,
]

width = 0.2

fig = plt.figure()
plt.bar(x - width * 1.5, np.array(randoms) * 100, width, color='grey')
plt.bar(x - width * 0.5, np.array(baselines) * 100, width, color='green')
plt.bar(x + width * 0.5, np.array(TSTR) * 100, width, color='blue')
plt.bar(x + width * 1.5, np.array(TRTS) * 100, width, color='red')

plt.xticks(x, [
    'Loads\nWeekday',
    'Loads\nWeekend',
    'Car\nWeekday',
    'Car\nWeekend',
])
plt.ylabel("Accuracy [%]")
plt.legend(
    [
        'Random classifier',
        'Baseline: Trained on real training data, tested on real testing data',
        'TSTR: Trained on synthetic data, tested on real testing data',
        'TRTS: Trained on real testing data, tested on synthetic data',
    ]
)
fig.savefig(
    "data/other_outputs/accuracy_GANS.pdf", bbox_inches='tight', format='pdf', dpi=1200
)
plt.close('all')
