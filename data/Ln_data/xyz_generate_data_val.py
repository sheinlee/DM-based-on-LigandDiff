import torch
import os
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.ligand import ligand_breakdown
from torch_geometric.data import Data

element_to_index = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'Br': 4, 'Cl': 5, 'P': 6, 'F': 7}
LIST_OF_ELEMENT = [
    'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'
]
charges_list = {'H': 1,
           'C': 6, 
           'O': 8, 
           'N': 7, 
           'S': 16, 
           'Cl': 17, 
           'P': 15, 
           'Br': 35, 
           'F': 9,
           'Cr':24,
           'Mn':25,
           'Fe':26,
           'Co':27,
           'Ni':28,
           'Cu':29,
           'Zn':30,
           'Ru':44,
           'Pd':46,
           'La':57,'Ce': 58, 'Pr':59, 'Nd':60, 'Sm':61, 'Eu':62, 'Gd':63, 'Tb':64, 'Dy':65, 'Ho':66, 'Er':67, 'Tm':68, 'Yb':69, 'Lu':70,
            'Si':14,
            'Li': 3,
            'Na': 11,
            'B': 5,
            'Ga': 31,
            'Al': 13,
            'I': 53}

# Initialize lists to store coordinates and one-hot encoded element vectors
def extract_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    coords = []
    one_hot_vectors = []
    charges = []
    
    # Process each atom in the file
    for line in lines[2:]:  # Starting from the third line
        parts = line.split()
        element, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
        
        # Append coordinates to the coords list
        coords.append([x, y, z])
        
        # Create a one-hot vector for the current element
        if element == 'H': continue
        else:
            if element in LIST_OF_ELEMENT:
                one_hot_vector = [0] * len(element_to_index)
                # one_hot_vector[element_to_index[element]] = 1
                one_hot_vectors.append(one_hot_vector)
            else:
                one_hot_vector = [0] * len(element_to_index)
                one_hot_vector[element_to_index[element]] = 1
                one_hot_vectors.append(one_hot_vector)

        charges.append(charges_list[element])
    
    # Convert lists to PyTorch tensors
    coords_tensor = torch.tensor(coords, dtype=torch.float)
    one_hot_tensor = torch.tensor(one_hot_vectors, dtype=torch.float)
    charges_tensor = torch.tensor(charges, dtype=torch.int)
    # Graph label
    graph_label = lines[1].strip()
    return coords_tensor, one_hot_tensor, graph_label, charges_tensor

final_data = []
for ligand_f in os.listdir('val'):
    if not ligand_f.endswith('.xyz'):
        continue
    print(ligand_f)
    ligand_file = f'val/{ligand_f}'
    mol=mol3D()
    # mol.readfromxyz(ligand_file)
    try:
        # 尝试读取XYZ文件
        mol.readfromxyz(ligand_file)
    except Exception as e:
        # 如果遇到错误，则打印文件名并继续循环
        print(f"Error reading file: {ligand_file}")
        continue  # 继续处理列表中的下一个文件
    overlapping=mol.sanitycheck(silence=True)[0]
    liglist,ligdents,ligcon=ligand_breakdown(mol,silent=True,BondedOct=True)
    coords_tensor, one_hot_tensor, graph_label, charges_tensor = extract_xyz(ligand_file)

    # ligand_group_matrix = torch.zeros((one_hot_tensor.shape[0], len(liglist)))
    ligand_group_matrix = torch.zeros((one_hot_tensor.shape[0], 6))
    for col_idx, positions in enumerate(liglist):
        positions_zero_based = torch.tensor(positions) - 1
        ligand_group_matrix[positions_zero_based, col_idx] = 1
    
    for ligand_num in range(len(ligcon)):
        data = Data(pos=coords_tensor)
        data.label = graph_label
        data.context = torch.ones((one_hot_tensor.shape[0]))
        data.nuclear_charges = charges_tensor
        data.coord_site = torch.zeros((one_hot_tensor.shape[0]))
        for site in ligcon[ligand_num]:
            data.coord_site[site-1] = 1
        data.ligand_diff = torch.zeros((one_hot_tensor.shape[0]))
        data.num_atoms = torch.tensor(charges_tensor.shape[0], dtype=torch.long)
        data.one_hot = one_hot_tensor
        for diff in liglist[ligand_num]:
            data.ligand_diff[diff-1] = 1
            data.context[diff-1] = 0
        data.ligand_group = ligand_group_matrix
        final_data.append(data)


print(len(final_data))

torch.save(final_data, 'val.pt')