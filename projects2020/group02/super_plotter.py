import numpy as np
import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

@click.command()
@click.option('--field', type=str, required=True, default='out', help='which field (in or out) to plot')
def main(field):
	
	##Create dictionary to store each patched up cube face/tile
	tiles={}

	for tile in range(1,7):

		##Get dimensions of tile, i.e. no. of workers along the axis 
		no_subtiles=len(glob.glob('local_%s_field_%i*.npy'%(field,tile)))
		dims=int(np.sqrt(no_subtiles))
		#print(no_subtiles,dims)
		
		##Extract all subtiles for the tile and group them in sets of size dimension
		##This allows concatenation of subtiles by stacking them along the x-axis as
		##subtile ranks are given in increasing order right to left starting from the tile's origin
		names=sorted(glob.glob('local_%s_field_%i*.npy'%(field,tile)))
		names_grouped = [ names[i:i + dims] for i in range(0, no_subtiles, dims) ]
		
		##Stack the grouped subtiles along the x-axis
		temp_hstacked=[np.dstack([np.load(names_grouped[i][j])  for j in range(dims) for i in range(dims)])]
		
		##Stack the x-axis slices along the y-axis
		tiles[tile]=np.hstack([x_tile for x_tile in temp_hstacked])
	
	##Get the unfolded cube dimensions for plotting
	nz=tiles[1].shape[0]
	ny=tiles[1].shape[1]
	nx=tiles[1].shape[2]
	field_plot = np.zeros( (nz, 3*(ny), 4*(nx)) )
	field_plot[:,:,:]=np.nan
	
	##Assign the tiles to the plotting domain making rotations where required
	field_plot[:,ny:2*ny,:nx]= np.rot90(tiles[5],axes=(1,2))
	field_plot[:,ny:2*ny,nx:2*nx]=tiles[1]
	field_plot[:,ny:2*ny,2*nx:3*nx]=tiles[2]
	field_plot[:,ny:2*ny,3*nx:4*nx]=np.rot90(tiles[4],axes=(1,2))
	field_plot[:,-ny:,nx:2*nx]=np.rot90(tiles[3],axes=(2,1))
	field_plot[:,:ny,nx:2*nx]=tiles[6]
	
	plt.imshow(field_plot[field_plot.shape[0] // 2, :, :], origin='lower',vmin=0,vmax=1)
	plt.axis('off')
	plt.colorbar()
	plt.savefig('%s_field_cube.mpi.png'%field)
	plt.close()				
    
    
    
if __name__ == '__main__':
    main()
