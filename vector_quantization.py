import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

class CustomizedClustering():

    def __init__(self, x, y, k, initial_centroids, ftype='euclidean', fstat='mean', ncolors='rgbcmyk'):
        """
        x                   :   xdata; type <array>
        y                   :   ydata; type <array>
        k                   :   number of clusters; type <int>
        initial_centroids   :   location of initial centroid guess; type <dict>
                                --> {ith centroid : [xloc, yloc]}
        ftype               :   error metric; type <str>
                                --> 'euclidean', 'euclidean square', 'manhattan'
        fstat               :   error statistic; type <str>
                                --> 'mean', 'median'
        ncolors             :   colors by index; type <str>
                                --> ith color corresponds to ith centroid
        """
        self.data = {'x' : x, 'y' : y}
        self.k = k
        self.centroids = initial_centroids
        self.ftype = ftype
        self.cmap = {i+1 : ncolors[i] for i in range(len(ncolors))}
        self.fstat = {'mean' : np.mean, 'median' : np.median}[fstat]
        self.prev_centroids = {}

    @staticmethod
    def select_distance_metric(ftype):
        """
        ftype   :   type <str>
        """
        if ftype == 'euclidean':
            f = lambda data, centroid : np.sqrt((data['x'] - centroid[0])**2 + (data['y'] - centroid[1])**2)
        elif ftype == 'euclidean square':
            f = lambda data, centroid : (data['x'] - centroid[0])**2 + (data['y'] - centroid[1])**2
        elif ftype == 'manhattan':
            f = lambda data, centroid : np.abs(data['x'] - centroid[0]) + np.abs(data['y'] - centroid[1])
        else:
            raise ValueError("ftype = 'euclidean', 'euclidean square', or 'manhattan'")
        return f

    def compute_centroid_errors(self, f):
        """
        f   :   type <function>
        Color-coordinate data to smallest error (closest centroid).
        """
        mod_data = np.array([f(self.data, self.centroids[idx]) for idx in self.centroids.keys()])
        self.data['closest'] = np.argmin(mod_data, axis=0)+1
        self.data['color'] = [self.cmap[i] for i in self.data['closest']]
        self.data['error'] = mod_data

    def update_centroids(self):
        """
        Updates centroids from error values.
        """
        self.prev_centroids = copy.deepcopy(self.centroids)
        for i in self.centroids.keys():
            xloc = np.where(self.data['closest'] == i)
            yloc = np.where(self.data['closest'] == i)
            self.centroids[i][0] = self.fstat(self.data['x'][xloc])
            self.centroids[i][1] = self.fstat(self.data['y'][yloc])

    def autoupdate_centroids(self, f):
        """
        f   :   type <function>
        Repeat updating centroids until previous iteration of centroids
        is identical to current iteration of centroids.
        """
        while True:
            closest = copy.deepcopy(self.data['closest'])
            self.update_centroids()
            self.compute_centroid_errors(f)
            DIF = np.diff(np.array([closest, self.data['closest']]), axis=0)
            print("\nCLOSEST CENTROIDS PREV:\n{}\n".format(closest))
            print("\nCLOSEST CENTROIDS POST:\n{}\n".format(self.data['closest']))
            print("\nDIFFERENCES:\n{}\n".format(DIF))
            if np.all(DIF == 0):
                break

    def view(self, xlim, ylim, step, chronos='final', figsize=None):
        """
        chronos :   type <str>
        """
        fig, ax = plt.subplots(figsize=figsize)
        if chronos == 'initial':
            mhandle = ax.scatter(self.data['x'], self.data['y'], color='k')
        elif chronos == 'final':
            mhandle = ax.scatter(self.data['x'], self.data['y'], color=self.data['color'], alpha=0.5, edgecolor='k')
        else:
            raise ValueError("chronos = 'initial' or 'final'")
        for i in self.centroids.keys():
            chandle = ax.scatter(*self.centroids[i], color=self.cmap[i])
        xticks = np.arange(xlim[0], xlim[-1]+1, step, dtype=int)
        yticks = np.arange(ylim[0], ylim[-1]+1, step, dtype=int)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xlim([xticks[0], xticks[-1]])
        ax.set_ylim([yticks[0], yticks[-1]])
        ax.set_xlabel('X-Data')
        ax.set_ylabel('Y-Data')
        ax.grid(color='k', linestyle=':', alpha=0.3)
        fig.legend(loc='lower center', handles=[mhandle, chandle], labels=['Data', 'Centroids'], fontsize=7.5, mode='expand', ncol=2, scatterpoints=5)
        plt.show()
        plt.close(fig)

    def execute(self, view_final=True, view_initial=False, xlim=[0, 100], ylim=[0, 100], step=10, figsize=(7, 7)):
        """ """
        if view_initial is True:
            self.view(xlim, ylim, step, chronos='initial', figsize=figsize)
        f = self.select_distance_metric(self.ftype)
        self.compute_centroid_errors(f)
        self.update_centroids()
        self.autoupdate_centroids(f)
        if view_final is True:
            self.view(xlim, ylim, step, chronos='final', figsize=figsize)
        elif view_final is not False:
            raise ValueError("view_final = True or False")

################################################################################
#                              EXAMPLE 1                                       #
################################################################################
x = np.random.normal(50, 10, 100).astype(int)
y = np.random.normal(40, 10, 100).astype(int)
initial_centroids = {1 : [25, 75], 2 : [25, 25], 3 : [75, 75], 4 : [75, 25]}
KMEANS = CustomizedClustering(x, y, k=4, initial_centroids=initial_centroids)
KMEANS.execute(view_initial=True)

################################################################################
#                              EXAMPLE 2                                       #
################################################################################
x = np.random.normal(50, 10, 1000).astype(int)
y = np.random.normal(50, 10, 1000).astype(int)
extra_centroids = {5 : [50, 50], 6 : [65, 35]}
initial_centroids.update(extra_centroids)
KMEDIANS = CustomizedClustering(x, y, k=6, initial_centroids=initial_centroids, fstat='median', ftype='manhattan')
KMEDIANS.execute(view_initial=True)

################################################################################
#                              EXAMPLE 3                                       #
################################################################################
xmu, ymu = 45, 55
xsig, ysig = 15, 20
size = 5000
x = np.random.normal(xmu, xsig, size)
y = np.random.normal(ymu, ysig, size)
k = np.random.randint(low=2, high=25)
cmap = plt.get_cmap('plasma',10)
norm = colors.Normalize(vmin=0, vmax=k)
smap = cmx.ScalarMappable(norm=norm, cmap=cmap)
ncolors = [smap.to_rgba(i) for i in range(k)]
initial_centroids = {}
for i in range(k):
    initial_centroids[i+1] = [np.random.randint(low=10, high=90), np.random.randint(low=10, high=90)]
for fstat, ftype in zip(('median', 'mean'), ('euclidean', 'manhattan')):
    KCLUSTER = CustomizedClustering(x, y, k=k, initial_centroids=initial_centroids, ncolors=ncolors, fstat=fstat, ftype=ftype)
    KCLUSTER.execute(view_initial=True)





##
