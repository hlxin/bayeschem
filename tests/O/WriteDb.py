#!/usr/bin/env python

import numpy as np
from ase import io
from ase.db import connect
from scipy import integrate
import subprocess as sp
import pickle
import os


def count_atoms(formula):
    """Given an atomic formula, e.g. CH2O, returns a dictionary of                                                                                                                    
    atomic counts, e.g, d = {'C':1,'H':2,'O':1}.                                                                                
    """
    d = {}
    starts = [] # indices of string formula that start a new element                                                                                                                  
    for index,letter in enumerate(formula):
        if letter.isupper():
            starts.append(index)
    starts.append(len(formula)) # (gives searchable range for last el.)                                                                                                               
    for index in range(len(starts)-1):
        name = ''
        number = ''
        for letter in formula[starts[index]:starts[index+1]]:
            if letter.isalpha():
                name = name + letter
            elif letter.isdigit():
                number = number + letter
        if len(number) == 0:
            count = 1
        else:
            count = eval(number)
        if name in d.keys():
            d[name] += count
        else:
            d[name] = count
    return d


def calculate_reference_energy(formula, ref_dict=None):
    """For a formula, such as CH2OH, counts the atoms and makes a
    reference energy (eV) based on the atomic energies in
    reference_energies."""
    reference_energies = {
        "O": -452.824762
         }
    
    if not ref_dict:
        ref_dict = reference_energies
    referenceenergy = 0.
    counts = count_atoms(formula)
    for key in counts.keys():
        referenceenergy += counts[key] * ref_dict[key]
    return referenceenergy


def get_energy(path):
#    Read the final dft energy from qn.log. 
    if os.path.exists(path + "/qn.log"):
        p1 = sp.Popen(["tail", "-n 1", path + "/qn.log"]
                      , stdout = sp.PIPE)
        p2 = sp.Popen(["awk" , "{ print $(NF-1), $NF }"],
                      stdin = p1.stdout, stdout = sp.PIPE) 
        energy, force = p2.communicate()[0].split()
    return float(energy)


def get_dos(file_pkl, atom_id, orb_id):
    """Read the DOS pickle file and return the energy, 
    total density of states,
    and the projected density of states."""
    f = open(file_pkl)
    ergy, tdos, pdos = pickle.load(f)
    f.close()
    return [ergy, pdos[atom_id][orb_id]]

ads = "O"
facet = 111
site = "atop" # atop, brg, fcc, hcp
sub_path = "/work/common/hxin_lab/zhengl/machine_learning/111/clean/M/"
ads_path = "/work/common/hxin_lab/hspillai/NewnsAnderson/O_data/ads/Atop/"
files = [f for f in os.listdir(ads_path) if os.path.isdir(f)]

vad_metal = {'Sc': 7.90, 'Ti': 4.65, 'V': 3.15, 'Cr': 2.35, 
             'Mn': 1.94, 'Fe': 1.59, 'Co': 1.34, 'Ni': 1.16, 
             'Cu': 1.00, 'Zn': 0.46, 'Y': 17.30, 'Zr': 10.90, 
             'Nb': 7.73, 'Mo': 6.62, 'Ru': 3.87, 'Rh': 3.32, 
             'Pd': 2.78, 'Ag': 2.26, 'Cd': 1.58, 'Hf': 11.90, 
             'Ta': 9.05, 'W': 7.72, 'Re': 6.04, 'Os': 5.13, 
             'Ir': 4.45, 'Pt': 3.90, 'Au': 3.35}

filling_metal = {'Fe': 0.7, 'Co': 0.8, 'Ni': 0.9, 'Cu': 1.0,
                 'Ru': 0.7, 'Rh': 0.8, 'Pd': 0.9, 'Ag': 1.0,
                 'Os': 0.7, 'Ir': 0.8, 'Pt': 0.9, 'Au': 1.0}

with connect('bayeschem.db') as db:
    for file in files:
        label = file

        pdos_ads = get_dos(ads_path + file + "/dos/dos.pickle", 16, 'p')
        pdos_sub = get_dos(sub_path + file + "/dos.pickle", 12, 'd') 

        atoms_ads = io.read(ads_path + file + "/qn.traj")
        atoms_sub = io.read(sub_path + file + "/qn.traj")

        e_ads = get_energy(ads_path + file)
        e_sub = get_energy(sub_path + file)
        e_gas = calculate_reference_energy(ads)
        de = e_ads - (e_sub + e_gas)

        db.write(atoms_ads, label = label, 
                 ads =  ads, site = site,
                 facet = facet, vad2 = vad_metal[label],
                 filling = filling_metal[label],
                 data = {"e_ads" : e_ads, 
                         "e_sub" : e_sub,
                         "de" : de,
                         "dos_ads" : pdos_ads, 
                         "dos_sub" : pdos_sub
                         })










