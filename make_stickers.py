import matplotlib
matplotlib.use('Agg')  # no displayed figures -- need to call before loading pylab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from collections import defaultdict
import numpy as np
import re
import matplotlib.patheffects as PathEffects
import numpy.lib.recfunctions as recfunctions
from scipy.optimize import fmin_l_bfgs_b

inner_colors = np.random.permutation(matplotlib.colors.cnames.keys())
outer_colors = np.random.permutation(matplotlib.colors.cnames.keys())

figsize = (15,1.8)
shape_names = ['circle', 'square', 'arrow', 'pacman', 'half circle', 'triangle', 'pentagon', 'radiation', 'bullseye', 'checkers', 'star', 'equal']
# shape_names = shape_names[-1:]

fontsize_big = 80
fontsize_small = 32

def new_node_attributes():
	global inner_colors
	global outer_colors
	scl = 1.

	inner_colors = np.roll(inner_colors, 1)
	outer_colors = np.roll(outer_colors, -1)

	return {
		# 'color outer':[np.random.rand()*scl, np.random.rand()*scl, np.random.rand()*scl],
		# 'color inner':[np.random.rand()*scl, np.random.rand()*scl, np.random.rand()*scl],
		# 'color inner':inner_colors[0],
		'color':outer_colors[0],
		'shape':shape_names[np.random.randint(len(shape_names))]
		}
node_attributes = defaultdict(new_node_attributes)

def extract_name_number(text):
	p = re.compile('([a-z]+)[\s\.]+(\d+)[\s\.]+([a-z]+)')
	m = p.match(text)
	name = m.group(1)
	order = int(m.group(2))
	inout = m.group(3)

	# print name
	# print order
	# print inout

	# name = 'emu'
	# order = 6
	# inout = 'in'
	return name, order, inout


def extract_nodes(text):
	p = re.compile('([a-z]+)[\s\.]+(\d+)[\s\.]+([a-z]+)\s*\=\s*([a-z]+)[\s\.]+(\d+)[\s\.]+([a-z]+)')
	m = p.match(text)
	name1 = m.group(1).upper()
	order1 = int(m.group(2))
	inout1 = m.group(3).upper()
	name2 = m.group(4).upper()
	order2 = int(m.group(5))
	inout2 = m.group(6).upper()

	# print name
	# print order
	# print inout

	# name = 'emu'
	# order = 6
	# inout = 'in'
	return name1, order1, inout1, name2, order2, inout2


def extract_step_number(text):
	p = re.compile('step\s+(\d+)\s+.*')
	m = p.match(text)
	step = m.group(1)
	return step


def add_shape(ax, shapename, color):
	if shapename == 'circle':
		shape = mpatches.Circle([0.5,0.5], 0.25, facecolor=color)
		ax.add_patch(shape)
	elif shapename == 'square':
		shape = mpatches.Rectangle([0.25,0.25], 0.5, 0.5, facecolor=color)
		ax.add_patch(shape)
	elif shapename == 'equal':
		shape = mpatches.Rectangle([0.25,0.25], 0.16, 0.5, facecolor=color)
		ax.add_patch(shape)
		shape = mpatches.Rectangle([0.75-0.16,0.25], 0.16, 0.5, facecolor=color)
		ax.add_patch(shape)
	elif shapename == 'checkers':
		shape = mpatches.Rectangle([0.25,0.25], 0.25, 0.25, facecolor=color)
		ax.add_patch(shape)
		shape = mpatches.Rectangle([0.5,0.5], 0.25, 0.25, facecolor=color)
		ax.add_patch(shape)
	elif shapename == 'arrow':
		shape = mpatches.Arrow(0.25,0.5, 0.5,0, width=1, facecolor=color)
		ax.add_patch(shape)
	elif shapename == 'triangle':
		shape = mpatches.RegularPolygon([0.5, 0.5], 3, 0.25, facecolor=color)
		ax.add_patch(shape)
	elif shapename == 'star':
		shape = mpatches.RegularPolygon([0.5, 0.5], 3, 0.25, orientation=0, facecolor=color)
		ax.add_patch(shape)
		shape = mpatches.RegularPolygon([0.5, 0.5], 3, 0.25, orientation=45, facecolor=color)
		ax.add_patch(shape)
	elif shapename == 'pentagon':
		shape = mpatches.RegularPolygon([0.5, 0.5], 5, 0.25, facecolor=color)
		ax.add_patch(shape)
	elif shapename == 'pacman':
		shape = mpatches.Wedge([0.5, 0.5], 0.25, 20, 340, facecolor=color)
		ax.add_patch(shape)
	elif shapename == 'half circle':
		shape = mpatches.Wedge([0.5, 0.5], 0.25, 0, 180, facecolor=color)
		ax.add_patch(shape)
	elif shapename == 'radiation':
		shape = mpatches.Wedge([0.5, 0.5], 0.25, 0, 60, facecolor=color)
		ax.add_patch(shape)
		shape = mpatches.Wedge([0.5, 0.5], 0.25, 120, 180, facecolor=color)
		ax.add_patch(shape)
		shape = mpatches.Wedge([0.5, 0.5], 0.25, 240, 300, facecolor=color)
		ax.add_patch(shape)
	elif shapename == 'bullseye':
		shape = mpatches.Circle([0.5,0.5], 0.25, facecolor=color)
		ax.add_patch(shape)
		shape = mpatches.Circle([0.5,0.5], 0.2, facecolor=color)
		ax.add_patch(shape)
		shape = mpatches.Circle([0.5,0.5], 0.15, facecolor=color)
		ax.add_patch(shape)
		shape = mpatches.Circle([0.5,0.5], 0.1, facecolor=color)
		ax.add_patch(shape)
		shape = mpatches.Circle([0.5,0.5], 0.05, facecolor=color)
		ax.add_patch(shape)



def draw_arrow_cross(ax, center, rad):
	shape = mpatches.Circle(center, rad, fill=False, linewidth=8)
	ax.add_patch(shape)
	#plt.Line2D(center[0] + rad*np.array([np.sqrt(2), -np.sqrt(2)]), center[1] + rad*np.array([np.sqrt(2), -np.sqrt(2)]), linewidth=8, color='black')
	plt.plot(center[0] + 0.86*rad*np.array([1./np.sqrt(2), -1./np.sqrt(2)]), center[1] + rad*np.array([1./np.sqrt(2), -1./np.sqrt(2)]), linewidth=8, color='black')
	plt.plot(center[0] + 0.86*rad*np.array([-1./np.sqrt(2), 1./np.sqrt(2)]), center[1] + rad*np.array([1./np.sqrt(2), -1./np.sqrt(2)]), linewidth=8, color='black')


def draw_bullseye(ax, center, rad):
	shape = mpatches.Circle(center, rad, fill=False, linewidth=8)
	ax.add_patch(shape)
	# shape = mpatches.Circle([0.5,0.5], rad*4/5, facecolor='white')
	# ax.add_patch(shape)
	shape = mpatches.Circle(center, rad/5, fill=True, linewidth=8, facecolor='black')
	ax.add_patch(shape)


def fill_color(ax, color, alpha=1.):
	#facecolor='orange'
	rect = mpatches.Rectangle([-0.3,-0.3],1.6,1.6, facecolor=color, alpha=alpha )#, ec="none")
	ax.add_patch(rect)
	# patches.append(rect)
	# #colors = [attr['color']]
	# #colors = np.linspace(0, 1, len(patches))
	# collection = PatchCollection(patches)#, cmap=plt.cm.hsv)#, alpha=0.3)
	# #collection.set_array(np.array(colors))
	# ax.add_collection(collection)

	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])


def bar_location_diagram(ax, z, ii):
	shape = mpatches.Circle([0.5, 0.5], 0.4, fill=False, linewidth=8)
	ax.add_patch(shape)
	print [z[0,ii], z[1,ii]]
	shape = mpatches.Circle([z[0,ii]*0.4+0.5, z[1,ii]*0.4+0.5], 0.1, fill=True, facecolor='black')
	ax.add_patch(shape)
	plt.axis('off')


def make_node_label(ax, num, x, max_order):
	tla = x["node %d TLA"%(num)]

	# add a rectangle
	attr = node_attributes[tla]
	fill_color(ax, 'white')
	fill_color(ax, attr['color'], alpha=0.6)

	# # add a shape
	# add_shape(ax, attr['shape'], attr['color inner'])

	# add the text
	txt_core = tla
	txt = plt.text(0.35,0.5, txt_core, fontsize=fontsize_big, horizontalalignment='center',
         verticalalignment='center' ) #,  backgroundcolor='white', color='black')

	txt_order = "%d/%d"%(x["node %d order"%(num)], max_order[tla])
	txt = plt.text(0.83,0.75, txt_order, fontsize=fontsize_small, horizontalalignment='center',
         verticalalignment='center' ) #,  backgroundcolor='white', color='black')

	center = [0.83, 0.25]
	rad = 0.11
	color = 'white'
	if x["node %d inout"%(num)] == 'IN':
		draw_arrow_cross(ax, center, rad)
	else:
		draw_bullseye(ax, center, rad)

	# TODO replace attr with bar color
	# TODO add in/out arrow symbol below order

	# txt = plt.text(0.1,0.5,text, fontsize=21, rotation=90,horizontalalignment='center',
 #         verticalalignment='center' ) #,  backgroundcolor='white', color='black')
	# # txt.set_path_effects([
 # #        PathEffects.Stroke(linewidth=1.1, foreground="w"),
 # #        PathEffects.Stroke(linewidth=1, foreground="black")])
	# plt.text(0.9,0.5,text, fontsize=21, rotation=-90,horizontalalignment='center',
 #         verticalalignment='center')
	plt.axis('off')

	return attr['color']


def single_bar_figure(X, z, ii, max_order):

	plt.figure(figsize=figsize)
	# break the sticker into 3 subplots

	# the left side subplot
	ax = plt.subplot(1,4,1)
	color1 = make_node_label(ax, 1, X[ii], max_order)
	ax.axis([0,1, 0, 1])
	# the right side subplot
	ax = plt.subplot(1,4,4)
	color2 = make_node_label(ax, 2, X[ii], max_order)
	ax.axis([0,1, 0, 1])
	
	# the sticker number subplot
	ax = plt.subplot(1, 4, 2)
	fill_color(ax, 'white')
	fill_color(ax, color1, alpha=0.6)
	# fill_color(ax, center_color, alpha=0.7)
	plt.text(0.5,0.5,X[ii]['step int'], fontsize=fontsize_big,horizontalalignment='center',
         verticalalignment='center')
	plt.axis('off')
	ax.axis([0,1, 0, 1])

	# the bar location subplot
	ax = plt.subplot(1, 4, 3)
	fill_color(ax, 'white')
	fill_color(ax, color2, alpha=0.6)
	bar_location_diagram(ax, z, ii)
	ax.axis([0,1, 0, 1])

	plt.subplots_adjust(left=-0.0001, bottom=-0.0001, right=1.0001, top=1.0001, wspace=-0.0001, hspace=-0.0001)

	fname = 'bar_label_%d.pdf'%(X[ii]['step int'])
	plt.savefig(fname)
	plt.close()


def load_data(fname='Brain24ft_v0307-Jun-2014 091700_labels.csv'):

	# NOTE "nodes" are actually bars! TODO fix this

	X = np.genfromtxt(fname, delimiter=',', names=True, dtype=[('step', '|S50'), ('nodes', '|S50'), ('angles', '|S50'), ('length', '|S50')])
	X = recfunctions.append_fields(X, 'step int', np.zeros((X.shape[0],), dtype=int))
	X = recfunctions.append_fields(X, 'node 1 inout', np.zeros((X.shape[0],), dtype='|S3'))
	X = recfunctions.append_fields(X, 'node 2 inout', np.zeros((X.shape[0],), dtype='|S3'))
	X = recfunctions.append_fields(X, 'node 1 TLA', np.zeros((X.shape[0],), dtype='|S3'))
	X = recfunctions.append_fields(X, 'node 2 TLA', np.zeros((X.shape[0],), dtype='|S3'))
	X = recfunctions.append_fields(X, 'node 1 order', np.zeros((X.shape[0],), dtype=int))
	X = recfunctions.append_fields(X, 'node 2 order', np.zeros((X.shape[0],), dtype=int))
	for ii in range(X.shape[0]):
		X['step int'][ii] = extract_step_number(X['step'][ii])
		X['node 1 TLA'][ii], X['node 1 order'][ii], X['node 1 inout'][ii], \
		X['node 2 TLA'][ii], X['node 2 order'][ii], X['node 2 inout'][ii] = \
		extract_nodes(X['nodes'][ii])

	max_order = defaultdict(int)
	for ii in range(X.shape[0]):
		tla = X['node 1 TLA'][ii]
		order = X['node 1 order'][ii]
		if order > max_order[tla]:
			max_order[tla] = order
		tla = X['node 2 TLA'][ii]
		order = X['node 2 order'][ii]
		if order > max_order[tla]:
			max_order[tla] = order

	return X, max_order


def get_distances(X):
	# bar connectivity
	C = np.eye(X.shape[0])
	for ii in range(X.shape[0]):
		print "%d / %d"%(ii, X.shape[0])
		for jj in range(ii, X.shape[0]):
			if X['node 1 TLA'][ii] == X['node 1 TLA'][jj] or \
			   X['node 1 TLA'][ii] == X['node 2 TLA'][jj] or \
			   X['node 2 TLA'][ii] == X['node 1 TLA'][jj] or \
			   X['node 2 TLA'][ii] == X['node 2 TLA'][jj]:
				C[ii,jj] = 1
				C[jj,ii] = 1

	# bar distance
	D = np.ones((X.shape[0], X.shape[0]))*-1
	M = np.eye(X.shape[0])
	current_d = 0
	while np.min(D) < 0:
		D[(D<0) & (M>0)] = current_d
		current_d += 1
		M = np.dot(M, C)
		print current_d

	return D

def embed_bars(X, D):
	def f_df(z):
		z1 = z[:X.shape[0]]
		z2 = z[-X.shape[0]:]
		diff1 = z1.reshape((-1,1)) - z1.reshape((1,-1))
		diff2 = z2.reshape((-1,1)) - z2.reshape((1,-1))
		z_D = diff1**2 + diff2**2

		logdiff = np.log(z_D + np.eye(z_D.shape[0])) - np.log(D + np.eye(z_D.shape[0]))

		assert(np.sum(~np.isfinite(logdiff))==0)

		# DEBUG
		err = np.sum((z_D - D)**2 / (D+np.eye(D.shape[0]))**2)
		derrdz_D = 2.*(z_D - D) / (D + np.eye(z_D.shape[0]))**2
		# err = np.sum((z_D - D)**2 / (D+np.eye(D.shape[0])))
		# derrdz_D = 2.*(z_D - D) / (D + np.eye(z_D.shape[0]))
		# err = np.sum((z_D - D)**2)
		# derrdz_D = 2.*(z_D - D)
		# err = np.sum(logdiff**2)
		# derrdz_D = 2.*logdiff / (z_D + np.eye(z_D.shape[0]))
		# err = np.sum(logdiff**10)
		# derrdz_D = 10.*logdiff**9 / (z_D + np.eye(z_D.shape[0]))
		# err = np.sum(np.abs(logdiff))
		# derrdz_D = np.sign(logdiff) / (z_D + np.eye(z_D.shape[0]))

		derrddiff1 = derrdz_D*2.*diff1
		derrddiff2 = derrdz_D*2.*diff2
		derrdz1 = np.sum(derrddiff1, axis=1) - np.sum(derrddiff1, axis=0)
		derrdz2 = np.sum(derrddiff2, axis=1) - np.sum(derrddiff2, axis=0)

		derrdz = np.vstack((derrdz1.reshape((-1,1)), derrdz2.reshape((-1,1))))

		return err, derrdz.ravel()


	def fix_coord(name, coord, X, z):
		match = (X['node 1 TLA'] == name) | (X['node 2 TLA'] == name)
		assert(np.sum(match)>0)
		z[0,match] = coord[0] + np.random.randn(sum(match),1).ravel()/100.
		z[1,match] = coord[1] + np.random.randn(sum(match),1).ravel()/100.

	z = np.random.randn(2, X.shape[0])/100.
	for i in range(100):
		fix_coord('TAT', [1,1], X, z)
		fix_coord('LID', [1,-1], X, z)
		fix_coord('APT', [-1,-1], X, z)
		fix_coord('HAY', [-1,1], X, z)
		z, _, _ = fmin_l_bfgs_b(
				f_df,
				z.ravel(),
				disp=1,
				maxfun=5)
		z = z.reshape((2,-1))

	z, _, _ = fmin_l_bfgs_b(
			f_df,
			z.ravel(),
			disp=1,
			maxfun=2000)
	z = z.reshape((2,-1))

	z -= np.mean(z, axis=1).reshape((2,1))
	z /= np.max(z, axis=1).reshape((2,1))

	# X = recfunctions.append_fields(X, 'z embed', np.zeros((X.shape[0],), dtype=int))

	plt.figure()
	for ii in range(0, X.shape[0], 20):
		print "%d / %d"%(ii, X.shape[0])
		for jj in range(X.shape[0]):
			if D[ii,jj] == 1:
				plt.plot([z[0,ii], z[0,jj]], [z[1,ii], z[1,jj]], 'r')
	plt.plot(z[0,:], z[1,:], '.b')

	return z



def main():
	print "loading data"
	X, max_order = load_data()
	print "(inefficiently) computing pairwise distances"
	D = get_distances(X)
	print "embedding bars in 2d space"
	z = embed_bars(X, D)

	target_bars = [1,912,24,354,600,333,666,42]
	for ii in target_bars:
		single_bar_figure(X, z, ii, max_order)

if __name__ == '__main__':
	main()
