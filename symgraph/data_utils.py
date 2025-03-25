import os
import numpy as np
import torch
from itertools import product, combinations
from copy import deepcopy
from fractions import Fraction
import requests
from bs4 import BeautifulSoup as bs

from torch_geometric.data import Data

from pymatgen.core.structure import Structure
from pymatgen.core import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup

from symgraph.sys_utils import PROJECT_ROOT




letter_to_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8,
                   'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16,
                   'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24,
                   'z': 25}

# unit cell indices for a super cell of 3x3x3
OFFSETS = torch.tensor([
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
])


def periodic_boundary(d):
    '''
    applies periodic boundary condition to difference in coordinates
    '''
    d = d - d.round()
    return d


def get_transform_matrix(a, b, c, alpha, beta, gamma):

    alpha = alpha*np.pi/180
    beta = beta*np.pi/180
    gamma = gamma*np.pi/180
    zeta = (np.cos(alpha) - np.cos(gamma) * np.cos(beta))/np.sin(gamma)
    
    h = np.zeros((3,3))
    
    h[0,0] = a
    h[0,1] = b * np.cos(gamma)
    h[0,2] = c * np.cos(beta)

    h[1,1] = b * np.sin(gamma)
    h[1,2] = c * zeta

    h[2,2] = c * np.sqrt(1 - np.cos(beta)**2 - zeta**2)

    return h.T


def frac_to_cart(X, h):
    '''
    X: (N, 3) tensor
    h: (3, 3) tensor
    '''
    return torch.matmul(X, h)

def cart_to_frac(X, h):
    '''
    X: (N, 3) tensor
    h: (3, 3) tensor
    '''
    return torch.matmul(X, torch.pinv(h))

def get_genpos_generators(space_group_number):
    # The URL for the GENPOS program
    url = f'https://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-getgen?list=Standard/Default%20Setting&what=gen&gnum={space_group_number}'

    
    # Send a POST request with the form data
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data for space group {space_group_number}")
    
    # Parse the response HTML
    soup = bs(response.text, 'html.parser')
    
    # # Find the section with the generator information
    # generator_section = soup.find('pre')
    # if not generator_section:
    #     raise Exception(f"Could not find generator information for space group {space_group_number}")
    
    # # Extract the text from the generator section
    # generators = generator_section.text.strip()

    rows = soup.find_all('tr')

# Loop through the rows and extract the matrices

    matrices = []
    for row in rows:
        # Look for table cells containing the rotation matrix and translation vector
        pre_tag = row.find('pre')  # Rotation matrix is inside <pre> tags
        if pre_tag:
            # Get the affine matrix
            affine_matrix_str = pre_tag.get_text(strip=True)
            
            # Split the string into lines and process
            affine_matrix_lines = affine_matrix_str.split('\n')
            affine_matrix = []
            
            for line in affine_matrix_lines:
                # Split each line into elements and convert fractions to floats
                elements = line.split()
                row = [float(Fraction(e)) for e in elements]
                affine_matrix.append(row)
            
            # Convert the list of rows into a numpy array
            affine_matrix = np.array(affine_matrix)
            
            matrices.append(affine_matrix)

            # Extract the rotation matrix (first 3x3 part)
            rotation_matrix = affine_matrix[:3, :3]
            
            # Extract the translation vector (last column)
            translation_vector = affine_matrix[:3, 3]

    mats = np.array(matrices)
    mats = np.unique(mats, axis=0)

    n_ops = len(mats)

    mats = np.concatenate([mats, np.tile(np.array([0, 0, 0, 1]).reshape(1, 1, 4), (n_ops, 1, 1))], axis=1)
    
    return mats

def canonize_affine_matrix(affine_matrix):
    """
    Canonize an affine matrix by ensuring that the translation vector is in the range [0, 1)
    """
    
    affine_matrix[..., :3, 3] = affine_matrix[..., :3, 3] % 1

    return affine_matrix

def map_generators_to_sg_ops(generators, sg_ops):

    generators = canonize_affine_matrix(generators)
    generators = [SymmOp(gen) for gen in generators]

    sg_ops = [SymmOp(canonize_affine_matrix(op.affine_matrix)) for op in sg_ops]

    mapping = {sg_op.as_xyz_str(): None for sg_op in sg_ops}
    
    for gen_op in generators:
        assert gen_op.as_xyz_str() in mapping.keys()
        mapping[gen_op.as_xyz_str()] = [gen_op]

    old_combs = deepcopy(generators)
    old_combs_gens = deepcopy([[gen] for gen in generators])

    comb_size = 1
    
    while None in mapping.values():
        

        new_combs = []
        new_combs_gens = []
        

        for i, j in product(range(len(old_combs)), range(len(generators))):


            comb = old_combs[i] * generators[j]
            comb = SymmOp(canonize_affine_matrix(comb.affine_matrix))
            xyz = comb.as_xyz_str()
            if mapping.get(xyz) is None:
                mapping[xyz] = deepcopy(old_combs_gens[i]) + [generators[j]]
                new_combs.append(deepcopy(comb))
                new_combs_gens.append(deepcopy(old_combs_gens[i]) + [generators[j]])
        
        
        old_combs = deepcopy(new_combs)
        old_combs_gens = deepcopy(new_combs_gens)
        # print(comb_size)
        comb_size += 1
        if comb_size == 1000:
            break

    # print(f'Combinations of size {comb_size} were needed to map all {len(generators)} generators to space group operations')

    return mapping

def get_sympos_generators(sg_ops, generators, max_iter=15):
    """
    Get the orbit and generators needed to generate the full set of symmetry operations
    """

    generators = canonize_affine_matrix(generators)
    generators = [SymmOp(gen) for gen in generators]


    # remove identity operation
    generators = [gen for gen in generators if not np.all(gen.affine_matrix == np.eye(4))]

    sg_ops = [SymmOp(canonize_affine_matrix(op.affine_matrix)) for op in sg_ops]

    gen_out = []
    found_subset = False
    for ii in range(1, len(generators)+1):
        if found_subset:
            break
        # get all combinations of generators of size ii
        gen_combs = list(combinations(generators, ii))

        for gens in gen_combs:
            mapping = {sg_op.as_xyz_str(): None for sg_op in sg_ops}
            full_mapping = deepcopy(mapping)
            map_before = len(mapping)

            for gen_op in gens:
                if gen_op.as_xyz_str() in mapping.keys():
                    mapping[gen_op.as_xyz_str()] = [gen_op]
                else:
                    break


            old_combs = deepcopy(gens)
            old_combs_gens = deepcopy([[gen] for gen in gens])

            comb_size = 1
            
            while None in mapping.values():
                

                new_combs = []
                new_combs_gens = []
                

                for i, j in product(range(len(old_combs)), range(len(gens))):


                    comb = old_combs[i] * gens[j]
                    comb = SymmOp(canonize_affine_matrix(comb.affine_matrix))
                    xyz = comb.as_xyz_str()
                    if xyz in mapping.keys() and mapping.get(xyz) is None:
                        mapping[xyz] = deepcopy(old_combs_gens[i]) + [gens[j]]
                        full_mapping[xyz] = deepcopy(old_combs_gens[i]) + [gens[j]]
                        new_combs.append(deepcopy(comb))
                        new_combs_gens.append(deepcopy(old_combs_gens[i]) + [gens[j]])
                    elif full_mapping.get(xyz) is None:
                        full_mapping[xyz] = deepcopy(old_combs_gens[i]) + [gens[j]]
                        new_combs.append(deepcopy(comb))
                        new_combs_gens.append(deepcopy(old_combs_gens[i]) + [gens[j]])
                    
                    if xyz not in mapping.keys():
                        pass
                
                
                old_combs = deepcopy(new_combs)
                old_combs_gens = deepcopy(new_combs_gens)
                
                comb_size += 1
                if comb_size == max_iter:
                    # print('Max iterations reached')
                    break

            if len(mapping) != map_before:
                raise ValueError(f'Error in mapping: {len(mapping)} != {map_before}')
        
            # check if the combined generators generate all space group operations
            if None not in mapping.values():
                found_subset = True
                gen_out.append(gens)
                
    return gen_out


def check_pos_list_is_close(ref_pos, pos_list, tolerance=0.005):
    """
    Check if a position is close to any of the positions in a list
    """
    return any([np.all(np.abs(ref_pos - pos) < tolerance) for pos in pos_list])

def get_generators_and_orbit(pos, sg_ops, tolerance=0.005):

    pos_gens = []
    orbit = [pos]

    for op in sg_ops:

        new_pos = op.operate(pos) % 1

        if not check_pos_list_is_close(new_pos, orbit, tolerance):
            orbit.append(new_pos)
            pos_gens.append(op)
    return pos_gens, orbit

def wyckoff_per_atom(pos, wyck, wyck_dict, equi):
    '''
    Produces wyckoff positions for each atom in the unit cell based on the equivalent positions and their wyckoff letters
    '''

    wycks = torch.zeros(len(pos), dtype=int)
    mults = torch.zeros(len(pos))
    gens = [None for _ in range(len(pos))]

    for e in range(len(equi)):

        wlet = wyck[e][-1]
        wnum = wyck[e][:-1]

        wint = letter_to_index[wlet.lower()]

        wycks[equi[e]] = wint
        mults[equi[e]] = int(wnum)

        for idx in equi[e]:
            gens[idx] = wyck_dict[wyck[e]]

    return wycks, mults, gens


def edge_index_zeo(x, x_o, len_zeo, ang_zeo):

    h = torch.tensor(get_transform_matrix(*len_zeo, *ang_zeo), dtype=torch.float)

    diff = x[:,None] - x_o[None]

    # adjust diff for pbc
    diff = periodic_boundary(diff)

    diff_cart = frac_to_cart(torch.tensor(diff, dtype=torch.float), h)

    dist = diff_cart.norm(dim=-1)

    # get the indices of the 4 nearest neighbours and the distance
    idx_i, idx_j = torch.argsort(dist, axis=0)[:2]

    # symmetrize to create edge_index
    idx_1 = torch.cat([idx_i, idx_j], dim=0)
    idx_2 = torch.cat([idx_j, idx_i], dim=0)
    edge_index = torch.stack([idx_1, idx_2], dim=0)

    # get distance between edges

    i,j = edge_index

    edge_diff = torch.tensor(periodic_boundary(x[i] - x[j]), dtype=torch.float)

    edge_diff_cart = frac_to_cart(edge_diff, h)

    edge_attr = edge_diff_cart.norm(dim=-1)

    return edge_index, edge_attr

def edge_index_radius(x, len_zeo, ang_zeo, radius=5.):
    
    x = torch.tensor(x, dtype=torch.float)
    
    x_i = x
    x_j = (x[:, None] + OFFSETS[None]).reshape(-1, 3)

    j_indices = torch.arange(len(x)).reshape(-1,1).repeat(1, 27).reshape(-1)

    h = torch.tensor(get_transform_matrix(*len_zeo, *ang_zeo), dtype=torch.float)

    dist = frac_to_cart((x_i[:,None] - x_j), h).norm(dim=-1)

    mask = (dist < radius ) & (dist > 1e-5)

    i, j = torch.where(mask)
    j = j_indices[j]

    edge_index = torch.stack([i, j], dim=0)
    edge_attr = dist[mask]

    return edge_index, edge_attr

def get_edge_index(x, x_o, len_zeo, ang_zeo, edge_type='zeo', radius=5.0):
    '''
    Create edge_index based on T-O-T connections in a zeolite
    '''
    if edge_type == 'zeo':
        return edge_index_zeo(x, x_o, len_zeo, ang_zeo)
    elif edge_type == 'radius':
        return edge_index_radius(x, len_zeo, ang_zeo, radius)
    else:
        raise ValueError(f'edge_type {edge_type} not recognised')


def create_graphs_zeo(zeo, pos, atoms, lat_params, wyck, equi, hoa, hoa_err, iso_params, edge_index=None, edge_attr=None, wyck_dict=None):
    '''
    creates a graph for a zeolite structure
    '''
    lens, angs = lat_params
    h = torch.tensor(get_transform_matrix(*lens, *angs), dtype=torch.float)
    pos_frac = torch.tensor(pos, dtype=torch.float)
    pos_cart = frac_to_cart(pos_frac, h)
    atoms = torch.tensor(atoms, dtype=torch.float)
    hoa = torch.tensor(hoa, dtype=torch.float)
    hoa_err = torch.tensor(hoa_err, dtype=torch.float)

    if iso_params is not None:
        iso_params = torch.tensor(iso_params, dtype=torch.float64)
        iso = torch.tensor([True])
    else:
        iso_params = torch.zeros((len(atoms), 6), dtype=torch.float64)
        iso = torch.tensor([False])

    graphs = []

    wycks, mults, gens = wyckoff_per_atom(pos, wyck, wyck_dict, equi)

    gens, gen_index = set_generators(gens)

    for i in range(len(atoms)):

        graph = Data(x=atoms[i].unsqueeze(-1), 
                     pos=pos_cart, 
                     y=hoa[i],
                     wyck=wycks, 
                     mults=mults, 
                     zeo=zeo, 
                     edge_index=edge_index, 
                     edge_attr=edge_attr, 
                     gens=gens,
                     gen_index=gen_index,
                     hoa_err=hoa_err[i], 
                     iso_params=iso_params[i].unsqueeze(0), 
                     iso=iso)
        
        graphs.append(graph)
    
    return graphs

def set_generators(gens):
 
    gen_len = torch.tensor([len(g) for g in gens], dtype=torch.long) # number of generators for each position
    gen_idx = torch.arange(len(gens), dtype=torch.long).repeat_interleave(gen_len) # index of each generator

    gens = torch.cat([torch.tensor(g, dtype=torch.float) for g in gens])

    return gens, gen_idx


def get_wyck_dict(crystal, sga):

    sg_symbol, sg_num = sga.get_space_group_symbol(), sga.get_space_group_number()
    
    generators = get_genpos_generators(sg_num)

    sg = SpaceGroup(sg_symbol)
    sg_syms = sg.symmetry_ops

    try:
        mapping = map_generators_to_sg_ops(generators, sg_syms)
    except AssertionError as e:
        try:
            sg_syms = sga.get_symmetry_operations()
            mapping = map_generators_to_sg_ops(generators, sg_syms)
        except AssertionError as e:
            print(f'Generators not in sg_ops')
            return None
        
    if None in mapping.values():
        print(f'Error in mapping')
        return None
    
    wyck_dict = {}

    try:
        for i, wyckoff in enumerate(crystal.wyckoff_symbols):
            if wyckoff in wyck_dict:
                continue

            site = crystal.equivalent_sites[i][0].frac_coords # Get the first site in the equivalent sites list
           
            ops, _ = get_generators_and_orbit(site, sg_syms, 0.001)

            ops = [SymmOp(canonize_affine_matrix(op.affine_matrix)) for op in ops]
            
            op_gens = get_sympos_generators(ops, generators, 50)

            wyck_dict[wyckoff] = op_gens[0]
    except:

        print(f'Error in orbits')
        return None
    
    return wyck_dict


def flatten_wyck_dict(wyck_dict):
    '''
    Converts values of wyck_dict from List[SymmOp] to List[np.array]
    Where each np.array is the rotations and translations of the SymmOp flattened
    '''

    new_dict = {}

    for key, value in wyck_dict.items():
        new_dict[key] = np.array([np.concatenate([op.rotation_matrix.flatten(), op.translation_vector]) for op in value])

    return new_dict


def create_graphs(zeo_codes='all', val_split=0.0, test_split=0.0, edge_type='zeo', radius=5.0):

    data_path = str(PROJECT_ROOT / 'Data_numpy')

    if zeo_codes == 'all':
        zeo_codes = os.listdir(data_path)
    else:
        for z in zeo_codes:
            if z not in os.listdir(data_path):
                raise ValueError(f'{z} not found in Data_numpy')

    graphs = []
    val_graphs = []
    test_graphs = []

    total = 0
    bad = 0

    for zeo in zeo_codes:

        lens, angs = np.load(f'{data_path}/{zeo}/lens.npy'), np.load(f'{data_path}/{zeo}/angs.npy')
        pos = np.load(f'{data_path}/{zeo}/pos.npy')
        pos_o = np.load(f'{data_path}/{zeo}/pos_o.npy')
        ats = np.load(f'{data_path}/{zeo}/atoms.npy')
        hoa = np.load(f'{data_path}/{zeo}/hoa.npy')
        hoa_err = np.load(f'{data_path}/{zeo}/hoa_err.npy')

        # load iso_params if they exist
        if os.path.exists(f'{data_path}/{zeo}/iso_params.npy'):
            iso_params = np.load(f'{data_path}/{zeo}/iso_params.npy')
        else:
            iso_params = None
        
        edge_index, edge_attr = get_edge_index(pos, pos_o, lens, angs, edge_type, radius)

        struc = Structure(lattice=get_transform_matrix(*lens, *angs), species=['Si']*len(pos), coords=pos)

        sga = SpacegroupAnalyzer(struc, symprec=0.05)

        sym_struct = sga.get_symmetrized_structure()

        wyck = sym_struct.wyckoff_symbols
        equi = sym_struct.equivalent_indices

        wyck_dict = get_wyck_dict(sym_struct, sga)
        
        if wyck_dict is None:
            print(f'Error in {zeo}')
            continue

        wyck_dict = flatten_wyck_dict(wyck_dict)

        # print({k: v.shape for k, v in wyck_dict.items()})


        hoa_range = hoa.max() - hoa.min()

        max_err = min(0.05*hoa_range, 1.5)

        # mask for keeping only the most accurate hoa values
        mask = hoa_err < max_err



        if (len(mask) - mask.sum())/len(mask) > 0.35 and mask.sum() < 100: # skip if more than 35% of the data is inaccurate resulting in less than 100 samples
            print(f'{zeo} has too many inaccurate HOA values. Skipping.')
            continue


        total += len(mask)
        bad += len(mask) - mask.sum()

        # print how many samples are dropped and what percentage it is
        print(f'{zeo} has {len(mask) - mask.sum()} inaccurate HOA values. This is {100*(len(mask) - mask.sum())/len(mask):.2f}% of the data.')
        print(f'HOA range: {hoa_range:.3f}, Max error: {max_err:.3f}, Max HoA: {hoa.max():.3f}, Min HoA: {hoa.min():.3f}')

        new_graphs = create_graphs_zeo(zeo, pos, ats, [lens, angs], wyck, equi, hoa, hoa_err, iso_params, edge_index, edge_attr, wyck_dict)

        new_graphs = [g for i, g in enumerate(new_graphs) if mask[i]]

        if val_split > 0 or test_split > 0:
            # set seed for reproducibility
            np.random.seed(111)

            n_val = int(val_split*len(new_graphs))
            n_test = int(test_split*len(new_graphs))
            
            # shuffle the graphs
            np.random.shuffle(new_graphs)

            val_graphs.extend(new_graphs[:n_val])
            test_graphs.extend(new_graphs[n_val:n_val+n_test])

            graphs.extend(new_graphs[n_val+n_test:])
        else:
            graphs.extend(new_graphs)


    print(f'Dropped {bad} out of {total} samples. This is {100*bad/total:.2f}% of the data.')

    return graphs, val_graphs, test_graphs
