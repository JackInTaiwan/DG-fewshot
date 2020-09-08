# This code is refered to https://blog.csdn.net/wuzlun/article/details/80053277

import numpy as np
import matplotlib.pyplot as plt



depth = ['Conv-2', 'Conv-3', 'Conv-4', 'Conv-5', 'Conv-6']
pts = [
    [92.33, 94.74, 94.17, 92.33, 91.1],
    [55.41, 60.11, 61.38, 60.47, 58.92],
    [52.73, 57.55, 58.59, 57.78, 56.52]
]
marks = ['^', 's', 'D']
pacs = [62.88, 64.55, 64.78, 64.92, 65.62]


f, (ax, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 7]})

# plot the same data on both axes
for i, (pt, m) in enumerate(zip(pts,marks)):
    ax.plot(depth, pt, '-o', linewidth=4, markersize=10, marker=m)
    ax2.plot(depth, pt, '-o', linewidth=4, markersize=10, marker=m)

ax.plot(depth, pacs, '--o', linewidth=2, markersize=7)
ax2.plot(depth, pacs, '--o', linewidth=2, markersize=7)

# zoom-in / limit the view to different portions of the data
ax.set_ylim(90.5, 96)  # outliers only
ax2.set_ylim(51.5, 66)  # most of the data

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax2.xaxis.tick_bottom()


ax2.tick_params(axis='x', labelsize=19, rotation=30)
ax.tick_params(axis='y', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)

ax.yaxis.set_ticks([91, 95])
ax2.yaxis.set_ticks([52, 62])

ax.set_ylabel("             Accuracy (%)", y=1.1, rotation=0, fontsize=16)

ax.legend(labels=["Omniglot", "Mini-ImageNet", "CUB", "PACS"], loc="best", fontsize=13)
plt.subplots_adjust(hspace=0.1)


d = .008  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


plt.show()
