from pathlib import Path
import MDAnalysis as mda
import numpy as np

def get_radial_flow_profile(psf_file, dcd_files, stride=1):
    """
    Compute water radial flow profile from trajectory data from a MD trajectory

    Args:
        psf_file (str): Path to PSF file.
        dcd_files (list of str): List of paths to DCD trajectory files.
        stride (int, optional): Frame stride for analysis. Default is 1.

    Returns:
        numpy.ndarray: Radial flow profile data.######## Defining the flux numpy array    ########
                       Array strcuture: [frame, radiii, data]    ########
                       data format is:
                          Column 0: Number of particles within a radial bin.
                          Column 1: Radial component of the displacement projected onto the radial direction.
                          Column 2: Azimuthal component of the displacement projected onto the azimuthal direction.
                          Column 3: Axial component of the displacement.

    """
    
    ######## Loading the psf and dcd    ########
    u = mda.Universe(psf_file, dcd_files)
    u.trajectory[0] 


    ######## Finding the z-location of the peptide along the channel    ########
    pep_selection = "segid PEP and resid 8 and name CA"  ##### This atomselection is exactly same as VMD atomselection command
    pep_atoms = u.select_atoms(pep_selection)            
    pep_pos = pep_atoms.positions[0][2]   #### only z coordinate 
    pep_min = pep_pos - 8 * 3.5   ##### peptide has 8 residues and length of the peptide can be estimated by
    pep_max = pep_pos + 8 * 3.5   ##### the fact that each residue has 3.5A length.
    #### Note: this z region of the channel will be elimated from the calculations as the peptide can disrupt the velocity profile
    


    ######## Making a selection for water molecules    ########
    seltext = "name OH2"        ##### atomselection used for water, no hydrogens included
    sel = u.select_atoms(seltext)

    ######## Making a selection for the channel    ########
    ref = u.select_atoms("segid NM")     ###### Segname defined for the channel

    
    ######## Defining the flux numpy array    ########
    ######## Array strcuture: [frame, radiii, data]    ########
    ######## data format is:
    ########          Column 0: Number of particles within a radial bin.
    ########          Column 1: Radial component of the displacement projected onto the radial direction.
    ########          Column 2: Azimuthal component of the displacement projected onto the azimuthal direction.
    ########          Column 3: Axial component of the displacement.
    flux = np.zeros((len(range(1, len(u.trajectory)-2, stride)), len(radii)-1, 4))


    ######## Defining the flux numpy array    ########
    for i, frame in enumerate(range(1, len(u.trajectory)-2, stride)):

        ########## printing the progress   ##########
        if (frame // stride) % 100 == 0:
            print("Analyzing ({}/{})".format(frame, len(u.trajectory)))

        u.trajectory[frame]
        pbc_dims = u.dimensions[:3]
        r0 = ref.atoms.positions.mean(axis=0)
        all_r = sel.atoms.positions - r0    ####### moving the box to the center 
        ####### structure of all_r: n*3
        #######                     n: number of the atoms in the system
        #######                     colums format: (x,y,z) postion of each atom

        ######### Apply periodic boundary conditions
        all_r = apply_periodic_boundary_conditions(all_r, pbc_dims)

        ######### Filtering out water molecules within the peptide region ######
        ids = np.where((all_r[:, 2] < pep_min) | (all_r[:, 2] > pep_max))
        all_r = all_r[ids]



        ######### Loading the previous frame ######
        u.trajectory[frame-1]
        all_r0 = sel.atoms.positions[ids] ##### note that all_r0 is only for water molecules and data strucutre is same as all_r
        ######### Loading the next frame     ######
        u.trajectory[frame+1]
        all_r1 = sel.atoms.positions[ids]

        ######### measuring the radial distance of atoms from the origin
        all_r_l = np.linalg.norm(all_r[:, :2], axis=-1)  #### projected onto the xy-plane.


        zhat = np.array((0, 0, 1))       #### unit vector pointing in the positive z-direction
        rhat = np.array(all_r)        
        rhat[:, 2] = 0
        rhat = rhat / all_r_l[:, None]   #### represents the unit vector pointing radially outward from the reference point.  
        thetahat = np.cross(zhat[None, :], rhat) #### contains the azimuthal unit vectors corresponding to each radial direction

        dr = all_r1 - all_r0             ###### represents the displacement of water molecules between consecutive frames
        dr = apply_periodic_boundary_conditions(dr, pbc_dims)  #### apply PBC to the displacesments

        dr_projected = calculate_dr_projected(dr, rhat, thetahat)

        ####### Now checking if the displacements falls into bins along radius   ########
        for j, (r0, r1) in enumerate(zip(radii[:-1], radii[1:])):
            ids = (all_r_l >= r0) & (all_r_l < r1)
            if np.sum(ids) == 0:
                continue
            flux[i, j, 0] = np.sum(ids)    #### number of the atoms within the bin
            flux[i, j, 1:] = np.sum(dr_projected[ids], axis=0) / (2 * frame_step) ##### averging the displacements

    return flux




def apply_periodic_boundary_conditions(positions, dimensions):
    """
    Apply periodic boundary conditions to positions.

    Args:
        positions (numpy.ndarray): Positions array.
        dimensions (tuple): System dimensions.

    Returns:
        numpy.ndarray: Positions array with applied periodic boundary conditions.
   """
    for dim in range(3):
        ids = positions[:, dim] > dimensions[dim] * 0.5
        positions[ids, dim] -= dimensions[dim]
        ids = positions[:, dim] < -dimensions[dim] * 0.5
        positions[ids, dim] += dimensions[dim]

    return positions



def calculate_dr_projected(dr, rhat, thetahat):
    """
    Calculate projected displacement vector.

    Args:
        dr (numpy.ndarray): Displacement vector.
        rhat (numpy.ndarray): Radial unit vector.
        thetahat (numpy.ndarray): Azimuthal unit vector.

    Returns:
        numpy.ndarray: Projected displacement vector.
              Column 1: Radial component of the displacement projected onto the radial direction.
              Column 2: Azimuthal component of the displacement projected onto the azimuthal direction.
              Column 3: Axial component of the displacement.

   """
    dr_projected = np.empty(dr.shape)
    dr_projected[:, 0] = np.sum(dr * rhat, axis=-1)
    dr_projected[:, 1] = np.sum(dr * thetahat, axis=-1) 
    dr_projected[:, 2] = dr[:, 2]

    return dr_projected



def get_water_vel_radial(data):
    """
    Calculate radial velocity of water molecules from flow profile data.

    Args:
        data (numpy.ndarray): Flow profile data.

    Returns:
        numpy.ndarray: Radial velocity of water molecules.
    """
    index = 1
    y_all = data
    y = np.sum(y_all[:, :, index], axis=0) / np.sum(y_all[:, :, 0], axis=0)
    y = np.ma.array(y, mask=np.isnan(y)).data
    return y


def get_water_vel_azimuthal(data):
    """
    Calculate azimuthal velocity of water molecules from flow profile data.

    Args:
        data (numpy.ndarray): Flow profile data.

    Returns:
        numpy.ndarray: Azimuthal velocity of water molecules.
    """
    index = 2
    y_all = data
    y = np.sum(y_all[:, :, index], axis=0) / np.sum(y_all[:, :, 0], axis=0)
    y = np.ma.array(y, mask=np.isnan(y)).data
    return y


def get_water_vel_axial(data):
    """
    Calculate axial velocity of water molecules from flow profile data.

    Args:
        data (numpy.ndarray): Flow profile data.

    Returns:
        numpy.ndarray: Axial velocity of water molecules.
    """
    index = 3
    y_all = data
    y = np.sum(y_all[:, :, index], axis=0) / np.sum(y_all[:, :, 0], axis=0)
    y = np.ma.array(y, mask=np.isnan(y)).data
    return y



if __name__ == "__main__":
    radii = np.arange(0, 25.0, 3)  ###### bins of the radial slaps 
    frame_step = 5000 * 2e-6  # should be in ns 

    aa_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
               'GLN', 'GLU', 'GLY', 'HSE', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO',
               'SER', 'THR', 'TRP', 'TYR', 'VAL']

    for aa in aa_list:
        psf_file = '../../infi_pore/setup/cut_{}.psf'.format(aa)
        dcd_file = '../../force-10fN/{}/NVT_equilibration.dcd'.format(aa)
        data = get_radial_flow_profile(psf_file, [dcd_file])
        np.save("flow_data_{}.npy".format(aa), data)
        
        # Call velocity functions with 'data' parameter
        vel_axial = get_water_vel_axial(data)
        vel_radial = get_water_vel_radial(data)
        vel_azimuthal = get_water_vel_azimuthal(data)


