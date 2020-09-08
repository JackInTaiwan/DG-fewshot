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

# # Now let's make two outlier points which are far away from everything.
# pts[[3, 14]] += .8

# If we were to simply plot pts, we'd lose most of the interesting
# details due to the outliers. So let's 'break' or 'cut-out' the y-axis
# into two portions - use the top (ax) for the outliers, and the bottom
# (ax2) for the details of the majority of our data
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
# ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

ax2.tick_params(axis='x', labelsize=19, rotation=30)

ax.tick_params(axis='y', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)

ax.yaxis.set_ticks([91, 95])
ax2.yaxis.set_ticks([52, 62])

# This looks pretty good, and was fairly painless, but you can get that
# cut-out diagonal lines look with just a bit more work. The important
# thing to know here is that in axes coordinates, which are always
# between 0-1, spine endpoints are at these locations (0,0), (0,1),
# (1,0), and (1,1).  Thus, we just need to put the diagonals in the
# appropriate corners of each of our axes, and so long as we use the
# right transform and disable clipping.

d = .008  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# What's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'

ax.set_ylabel("             Accuracy (%)", y=1.1, rotation=0, fontsize=16)

ax.legend(labels=["Omniglot", "Mini-ImageNet", "CUB", "PACS"], loc="best", fontsize=13)
plt.subplots_adjust(hspace=0.1)

plt.show()
