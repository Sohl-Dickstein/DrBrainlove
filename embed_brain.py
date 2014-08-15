import matplotlib
matplotlib.use('Agg')  # no displayed figures -- need to call before loading pylab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
# from mpl_toolkits.mplot3d import Axes3D

from collections import defaultdict
import numpy as np
import re
import matplotlib.patheffects as PathEffects
import numpy.lib.recfunctions as recfunctions
import fractions
from scipy.optimize import fmin_l_bfgs_b
from matplotlib import gridspec


np.random.seed(seed=1234)

## turn the crappy old node names into the new node names
homophonic_node_names 	= set(['ABS', 'ABY', 'ACE', 'ACT', 'ADD', 'ADO', 'ADS', 'AFT', 'AGE', 'AGO', 'AID', 'AIL', 'AIM', 'AIR', 'ALE', 'ALL', 'AMP', 'AND', 'ANT', 'ANY', 'APE', 'APT', 'ARC', 'ARE', 'ARK', 'ARM', 'ART', 'ASK', 'ASP', 'ASS', 'COP', 'COT', 'COW', 'COY', 'CRY', 'CUB', 'CUE', 'CUP', 'CUT', 'DAB', 'DAD', 'DAY', 'DEN', 'DEW', 'DIB', 'DID', 'DIE', 'DIG', 'DIM', 'DIN', 'DIP', 'DOE', 'DOG', 'DOT', 'DRY', 'DUB', 'DUD', 'DUE', 'DUG', 'DUH', 'GET', 'GIG', 'GIN', 'GIT', 'GNU', 'GOO', 'GOT', 'GUM', 'GUN', 'GUT', 'GUY', 'GYM', 'HAD', 'HAM', 'HAS', 'HAT', 'HAY', 'HEM', 'HEN', 'HER', 'HEW', 'HEX', 'HEY', 'HID', 'HIM', 'HIP', 'HIS', 'HIT', 'HOE', 'HOG', 'LID', 'LIE', 'LIP', 'LIT', 'LOB', 'LOG', 'LOO', 'LOP', 'LOT', 'LOW', 'LUG', 'LYE', 'MAC', 'MAD', 'MAG', 'MAN', 'MAP', 'MAR', 'MAT', 'MAW', 'MAX', 'MAY', 'MEN', 'MET', 'MIC', 'MID', 'MIT', 'MIX', 'MOB', 'MOD', 'PEN', 'PET', 'PEW', 'PIC', 'PIE', 'PIG', 'PIN', 'PIT', 'PIX', 'PLY', 'POD', 'POI', 'POP', 'POT', 'POW', 'PRO', 'PRY', 'PUB', 'PUG', 'PUN', 'PUP', 'PUT', 'QUO', 'RAD', 'RAG', 'RAM', 'RAN', 'RAP', 'RAT', 'SUP', 'TAB', 'TAD', 'TAG', 'TAN', 'TAP', 'TAR', 'TAT', 'TAX', 'TEA', 'TEE', 'TEN', 'THE', 'TIC', 'TIE', 'TIL', 'TIN', 'TIP', 'TOE', 'TOM', 'TOO', 'TOP', 'TOT', 'TOW', 'TOY', 'TRY', 'TUB', 'TUG', 'TWO',])
new_node_names 			= set(['ACE', 'ACT', 'ADO', 'AGE', 'AGO', 'AHI', 'AIM', 'AIR', 'ALA', 'ALL', 'AMP', 'ARM', 'ART', 'ASH', 'ASK', 'BAH', 'BAM', 'BAR', 'BED', 'BIO', 'BOA', 'BOX', 'BOY', 'BRO', 'BUG', 'CAB', 'CHI', 'CIS', 'COW', 'COP', 'CUP', 'COY', 'DAD', 'DAY', 'DIG', 'DOG', 'DRY', 'DUB', 'DUH', 'DUI', 'EAR', 'EGG', 'EGO', 'ELF', 'ELK', 'EON', 'ERA', 'ETA', 'EVE', 'EYE', 'FAN', 'FAX', 'FEW', 'FIG', 'FIX', 'FLU', 'FLY', 'FOG', 'FOR', 'FOX', 'FRO', 'FUN', 'GAL', 'GAS', 'GEL', 'GET', 'GIG', 'GIN', 'GOO', 'GUN', 'GUT', 'GYM', 'HAM', 'HAY', 'HEX', 'HIP', 'HIT', 'HOT', 'HOW', 'HUG', 'ICE', 'INK', 'IRE', 'IVY', 'JAM', 'JAR', 'JOB', 'JOY', 'JUG', 'KEG', 'KEY', 'KIT', 'LAB', 'LAM', 'LAW', 'LAX', 'LEG', 'LID', 'LIE', 'LUG', 'MAC', 'MAD', 'MAY', 'MIX', 'MOM', 'MOO', 'MOP', 'NAN', 'NAY', 'NIX', 'NIL', 'NEW', 'NOM', 'NOR', 'OAK', 'OAR', 'ODD', 'OFF', 'OIL', 'OLD', 'ONO', 'ORC', 'OVA', 'PAW', 'PAY', 'PHI', 'PIE', 'PIG', 'PLY', 'POP', 'PRO', 'RAD', 'RAW', 'REF', 'RIB', 'ROB', 'RUG', 'RUM', 'SAD', 'SAW', 'SAY', 'SEX', 'SHY', 'SIP', 'SIR', 'SKY', 'SOY', 'SPY', 'SUN', 'TAG', 'TAU', 'TAR', 'TAX', 'TAT', 'TOY', 'TRY', 'TUB', 'TUX', 'USE', 'VAN', 'VOW', 'WAR', 'WAS', 'WAX', 'WET', 'WHO', 'WHY', 'WIG', 'WIN', 'WIZ', 'WOW', 'YAK', 'YAY', 'YES', 'ZAP', 'ZIG', 'ZOO', 'ZZZ',])
node_name_mapping = dict()
# keep as many names as possible the same
for tla in homophonic_node_names:
	if tla in new_node_names:
		node_name_mapping[tla] = tla
		new_node_names -= set([tla])
# and now assign the rest of the node names
new_node_names = sorted(new_node_names, reverse=True)
for tla in homophonic_node_names:
	if tla not in node_name_mapping.keys():
		node_name_mapping[tla] = new_node_names.pop()
assert(len(new_node_names)==0) # make sure all names were used

def extract_nodes_length(text):
	if 'inches' in text:
		p = re.compile('([a-z]+)[\s\.]+(\d+)[\s\.]+([a-z]+)\s*\=\s*([a-z]+)[\s\.]+(\d+)[\s\.]+([a-z]+).*cut (.*) inches')
	else:
		p = re.compile('([a-z]+)[\s\.]+(\d+)[\s\.]+([a-z]+)\s*\=\s*([a-z]+)[\s\.]+(\d+)[\s\.]+([a-z]+).* (\d*\.?\d*)$')
	m = p.match(text)
	name1 = node_name_mapping[m.group(1).upper()]
	order1 = int(m.group(2))
	inout1 = m.group(3).upper()
	name2 = node_name_mapping[m.group(4).upper()]
	order2 = int(m.group(5))
	inout2 = m.group(6).upper()
	length_str = m.group(7)

	# turn english units into decimals
	length_str = length_str.strip()
	length_split = length_str.split(' ')
	bar_length = 0.
	for s in length_split:
		bar_length += float(fractions.Fraction(s))

	return name1, order1, inout1, name2, order2, inout2, bar_length

# def extract_bar_length(text):
# 	p = re.compile(',cut (.*) inches')
# 	m = p.match(text)
# 	length_str = m.group(1)
# 	return bar_length

def extract_step_number(text):
	p = re.compile('step\s+(\d+)\s+.*')
	m = p.match(text)
	step = m.group(1)
	return step

def bar_location_diagram(ax, z, ii, color):
	shape = mpatches.Circle([0.5, 0.5], 0.4, fill=False, linewidth=8)
	# plt.setp(shape, path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")])
	ax.add_patch(shape)

	# print [z[0,ii], z[1,ii]]
	shape = mpatches.Circle([z[0,ii]*0.35+0.5, z[1,ii]*0.35+0.5], 0.15, fill=True, facecolor='black')
	plt.setp(shape, path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")])
	ax.add_patch(shape)
	plt.axis('off')


def load_data(fname='Brain24ft_v0307-Jun-2014 091700_labels.csv'):

	# NOTE "nodes" is used to refer to both nodes and bars! TODO fix this naming convention

	X = np.genfromtxt(fname, delimiter=',', names=True, dtype=[('step', '|S50'), ('nodes', '|S50'), ('angles', '|S50'), ('length', '|S50')])
	X = recfunctions.append_fields(X, 'step int', np.zeros((X.shape[0],), dtype=int))
	X = recfunctions.append_fields(X, 'node 1 inout', np.zeros((X.shape[0],), dtype='|S3'))
	X = recfunctions.append_fields(X, 'node 2 inout', np.zeros((X.shape[0],), dtype='|S3'))
	X = recfunctions.append_fields(X, 'node 1 TLA', np.zeros((X.shape[0],), dtype='|S3'))
	X = recfunctions.append_fields(X, 'node 2 TLA', np.zeros((X.shape[0],), dtype='|S3'))
	X = recfunctions.append_fields(X, 'node 1 order', np.zeros((X.shape[0],), dtype=int))
	X = recfunctions.append_fields(X, 'node 2 order', np.zeros((X.shape[0],), dtype=int))
	X = recfunctions.append_fields(X, 'bar length', np.zeros((X.shape[0],), dtype=float))
	for ii in range(X.shape[0]):
		X['step int'][ii] = extract_step_number(X['step'][ii])
		X['node 1 TLA'][ii], X['node 1 order'][ii], X['node 1 inout'][ii], \
		X['node 2 TLA'][ii], X['node 2 order'][ii], X['node 2 inout'][ii], \
		X['bar length'][ii] = \
		extract_nodes_length(X['nodes'][ii] + ', ' + X['length'][ii])

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



def plot_node_positions(X, Z):
	subplot_coords = [(0, 1, 1, 'x', 'y'), (0, 2, 2, 'x', 'z'), (1, 2, 3, 'y', 'z')]

	for node_name in node_name_mapping.values():
		plt.figure(figsize=(6.5, 2.4))
		for bar in X:
			idx1 = node_name_to_number(bar['node 1 TLA'])
			idx2 = node_name_to_number(bar['node 2 TLA'])
			for ijk in subplot_coords:
				plt.subplot(1,3,ijk[2])
				plt.plot(
					[Z[ijk[0],idx1], Z[ijk[0],idx2]], 
					[Z[ijk[1],idx1], Z[ijk[1],idx2]],
					color='blue',
					alpha=0.5
					)
		idx = node_name_to_number(node_name)
		for ijk in subplot_coords:
			ax = plt.subplot(1,3,ijk[2])
			shape = mpatches.Circle(Z[ijk[:2],idx], 12, fill=True, facecolor='red', zorder=300)
			# plt.setp(shape, path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")])
			ax.add_patch(shape)
			plt.axis('equal')
			plt.xlabel(ijk[3])
			plt.ylabel(ijk[4])
			plt.title(" $\ $ ")
			if ijk[2] == 1:
				plt.gca().invert_yaxis()
		plt.suptitle("Node name: %s"%node_name)
		fname = 'node_locations/%s.pdf'%(node_name)
		try:
			plt.tight_layout()
		except:
			warnings.warn('tight_layout failed.  try running with an Agg backend.')
		plt.savefig(fname)
		plt.close()


def embed_nodes_warm_start(X, Z=None):

	def f_df(Z, do_plot=False):
		Z = Z.copy().reshape((3,-1))

		f = 0.
		df = np.zeros(Z.shape)

		for bar in X:
			idx1 = node_name_to_number(bar['node 1 TLA'])
			idx2 = node_name_to_number(bar['node 2 TLA'])
			dist = bar['bar length']
			# dist = 1. # DEBUG

			diff = Z[:,idx1] - Z[:,idx2]
			true_dist = np.sqrt(np.sum(diff**2))

			assert(true_dist > 0)

			err = (true_dist - dist)**2

			f += err

			derrdtruedist = 2.*(true_dist - dist)
			# derrdtruedist = 4.*(true_dist - dist)**3
			derrddiff = derrdtruedist/true_dist*diff
			derrdZ1 = derrddiff
			derrdZ2 = -derrddiff

			df[:,idx1] += derrdZ1
			df[:,idx2] += derrdZ2

			if do_plot:
				err_cap = np.log(err + 0.001)					
				err_cap = np.min([err_cap, 2.5])
				err_cap -= np.log(0.001)
				err_cap /= 2.5 - np.log(0.001)
				for ijk in [(0, 1, 1, 'x', 'y'), (0, 2, 2, 'x', 'z'), (1, 2, 3, 'y', 'z')]:
					plt.subplot(1,3,ijk[2])
					plt.plot(
						[Z[ijk[0],idx1], Z[ijk[0],idx2]], 
						[Z[ijk[1],idx1], Z[ijk[1],idx2]],
						color=[err_cap, 0., 1.-err_cap],
						alpha=0.5
						)
					plt.axis('equal')
					plt.xlabel(ijk[3])
					plt.ylabel(ijk[4])

		return f, df.ravel()

	if Z is None:
		Z = np.load('node_distance_positions_Z_warmstart.npz')['Z']
	Z, _, _ = fmin_l_bfgs_b(
			f_df,
			Z.copy().ravel(),
			disp=1,
			maxfun=1000)
	Z = Z.reshape((3,-1))
	Z -= np.mean(Z, axis=1).reshape((3,1))

	plt.figure()
	f_df(Z.copy(), do_plot=True)

	return Z


def embed_nodes(X):

	fig = plt.figure(figsize=[18,7])
	# ax = fig.add_subplot(111)

	global proj
	global num_active
	num_active = 1
	# reorder X using assembly order
	order = np.argsort(X['step int'])
	X = X[order]
	proj = np.random.randn(3,3)
	beta = 0.01
	global iter_i
	iter_i = 0

	def f_df(Z):
		global proj
		global num_active
		global iter_i
		iter_i += 1

		Z = Z.copy().reshape((3,-1))

		f = 0.
		df = np.zeros(Z.shape)

		# # a weak penalty to try to make it lie in plane
		# pen = lam * Z[2,:]**2

		Zp = np.dot(proj, Z)

		# do_plot = (np.random.rand() < 0.005)
		do_plot = iter_i == 150

		if do_plot:
			fig.clf()
			# for ijk in [(0, 1, 1), (0, 2, 2), (1, 2, 3)]:
			# 	plt.subplot(1,3,ijk[2])
			# 	plt.scatter(Zp[ijk[0],:], Zp[ijk[1],:])

		# proj *= np.sqrt(1.-beta)
		# proj += np.random.randn(3,3)*np.sqrt(beta)
		# proj = np.linalg.qr(proj)[0]
		proj = np.eye(3)

		# for bar in X[:90]:
		# for bar in X[91:180]:
		for bar in X[:num_active]:
			idx1 = node_name_to_number(bar['node 1 TLA'])
			idx2 = node_name_to_number(bar['node 2 TLA'])
			dist = bar['bar length']
			# dist = 1. # DEBUG

			diff = Z[:,idx1] - Z[:,idx2]
			true_dist = np.sqrt(np.sum(diff**2))

			assert(true_dist > 0)

			err = (true_dist - dist)**2
			# err = (true_dist - dist)**4

			if do_plot:
				if True: #idx1%13 == 0 or idx2 %13 == 0:
					err_cap = np.log(err + 0.001)					
					err_cap = np.min([err_cap, 2.5])
					err_cap -= np.log(0.001)
					err_cap /= 2.5 - np.log(0.001)
					for ijk in [(0, 1, 1), (0, 2, 2), (1, 2, 3)]:
						plt.subplot(1,3,ijk[2])
						plt.plot(
							[Zp[ijk[0],idx1], Zp[ijk[0],idx2]], 
							[Zp[ijk[1],idx1], Zp[ijk[1],idx2]],
							color=[err_cap, 0., 1.-err_cap],
							alpha=0.5
							)

			f += err

			derrdtruedist = 2.*(true_dist - dist)
			# derrdtruedist = 4.*(true_dist - dist)**3
			derrddiff = derrdtruedist/true_dist*diff
			derrdZ1 = derrddiff
			derrdZ2 = -derrddiff

			df[:,idx1] += derrdZ1
			df[:,idx2] += derrdZ2

		if do_plot:
			plt.draw()
			plt.show()

		return f, df.ravel()

	num_nodes = len(node_name_mapping)

	Z = np.random.randn(3, num_nodes)*1000
	# Z = np.random.randn(3, num_nodes)/100.
	# Z[2,num_active:] = -2000

	# Z[1,num_active:] = 0
	# Z[2,num_active:] = 0

	# num_active = 1000 # DEBUG

	step_length = 5

	for outer_loop in range(0, 1000):
		# start out far from the data
		num_active += step_length

		# throw any nodes that have 2 or fewer connections to the outside
		count = np.zeros((Z.shape[1]))
		for bar in X[:num_active]:
			idx1 = node_name_to_number(bar['node 1 TLA'])
			idx2 = node_name_to_number(bar['node 2 TLA'])
			count[idx1] += 1
			count[idx2] += 1
		# also throw the nodes attached to the new bar so they can escape any local minima
		for bar in X[num_active-step_length:num_active]:
			idx1 = node_name_to_number(bar['node 1 TLA'])
			idx2 = node_name_to_number(bar['node 2 TLA'])
			count[idx1] = 1 # lie about the count, so it gets pushed out
			count[idx2] = 1
		# subtract mean
		if np.sum(count>3) > 1:
			print "subtracting mean"
			Z[:,count>0] -= np.mean(Z[:,count>3], axis=1).reshape((3,1))
		rad = np.sqrt(np.sum(Z**2, axis=0))
		Z[:,count<4] = Z[:,count<4] * 300 / rad[count<4].reshape((1,-1))

		# and plop the ones that should be well defined onto the surface of a sphere
		# avoid buckles getting frozen in
		if num_active < 300:
			# project onto plane
			Z[[2],:] = 0.
			if num_active > 100:
				r2d = np.sqrt(np.sum(Z[0:2,:]**2, axis=0))
				Z[[2],:] = -(r2d - np.mean(r2d))/1000.
			 # + np.random.randn(1,Z.shape[1])/1000.
		else:
			# project onto sphere
			if np.sum(count>3) > 1:
				Z[:,count>3] = Z[:,count>3] * np.mean(rad[count>3]) / rad[count>3].reshape((1,-1))

		# force the measured nodes into a plane:
		# these are the bars around the outside edge of the brain.
		fixed_node_nums = np.asarray([356, 357, 544, 653, 675, 669, 659, 631, 616, 617, 625, 635, 663, 647, 641, 567, 377])
		idx_list = []
		for bar_num in fixed_node_nums:
			bar_idx = np.flatnonzero(X['step int']==bar_num)
			bar = X[bar_idx]
			idx1 = node_name_to_number(bar['node 1 TLA'])
			idx2 = node_name_to_number(bar['node 2 TLA'])
			idx_list.append(idx1)
			idx_list.append(idx2)
			if bar_num in [356, 616, 617]:
				Z[1,idx1] = 0.
				Z[1,idx2] = 0.
			if bar_num == 356:
				if Z[0,idx1] > 0.:
					# reflect if needed
					Z[0,:] = -Z[0,:]
				Z[0,idx1] = -np.abs(Z[0,idx1])
				Z[0,idx2] = -np.abs(Z[0,idx2])
			if bar_num in [616, 617]:
				Z[0,idx1] = np.abs(Z[0,idx1])
				Z[0,idx2] = np.abs(Z[0,idx2])
		idx_list = np.asarray(idx_list)
		if np.mean(Z[2,idx_list] > 0.):
			# reflect if needed
			Z[2,:] = -Z[2,:]
		Z[2,idx_list] = np.mean(Z[2,idx_list])


		print "mean", np.mean(Z[:,count>3], axis=1)
		print "step %d/%d, num_active %d"%(outer_loop, 700, num_active)
		# # flatten a little -- encourage this to be the z-axis
		# Z[2,:num_active]
		iter_i = 0
		Z, _, _ = fmin_l_bfgs_b(
				f_df,
				Z.copy().ravel(),
				disp=1,
				maxfun=1000)
		Z = Z.reshape((3,-1))

	Z -= np.mean(Z, axis=1).reshape((3,1))
	Z /= np.max(Z, axis=1).reshape((3,1))

	return Z


def plot_coords():
	plt.figure()
	for ii in range(0, X.shape[0], 20):
		print "%d / %d"%(ii, X.shape[0])
		for jj in range(X.shape[0]):
			if D[ii,jj] == 1:
				plt.plot([z[0,ii], z[0,jj]], [z[1,ii], z[1,jj]], 'r')
	plt.plot(z[0,:], z[1,:], '.g')
	for ii in range(0, X.shape[0]):
		if X['step int'][ii] in fixed_node_nums:
			plt.plot(z[0,ii], z[1,ii], '*b')
		if X['step int'][ii] == fixed_node_nums[0]:
			plt.plot(z[0,ii], z[1,ii], '.y')

	return z


def max_order_histogram(max_order):
	"""
	How many nodes of each cardinality -- for ashley lighting node size email.
	"""
	histogram = defaultdict(int)
	for k in max_order.keys():
		histogram[max_order[k]] += 1
	print histogram


def node_name_to_number(name):
	sorted_names = sorted(node_name_mapping.values())
	idx = sorted_names.index(name)
	return idx


def main():
	print "loading data"
	X, max_order = load_data()

	print "embedding nodes in 3d space"
	Z = embed_nodes_warm_start(X)
	# Z = embed_nodes(X)

	print "generating node position PDFs"
	plot_node_positions(X, Z)

	print "dumping new coordinates to a file"
	nodes = []
	for nm in sorted(node_name_mapping.values()):
		idx = node_name_to_number(nm)
		z_node = Z[:,idx]
		connected_nodes = []
		for bar in X:
			if bar['node 1 TLA'] == nm:
				connected_nm = bar['node 2 TLA']
			else:
				connected_nm = bar['node 1 TLA']
			idx_connected = node_name_to_number(connected_nm)
			z_connected = Z[:,idx_connected]
			connected_nodes.append({'name':connected_nm, 'distance':np.sqrt(np.sum((z_node - z_connected)**2))})
		node_dict = {'name':nm, 'location':z_node.tolist(), 'connected_nodes':connected_nodes}
		nodes.append(node_dict)
	print "dictionary prepared, writing json"
	import json
	with open('node_info.json', 'w') as outfile:
		json.dump(nodes, outfile)	


if __name__ == '__main__':
	main()
