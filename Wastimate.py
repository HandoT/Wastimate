# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:16:40 2023

@author: hando
"""

import copy
import json
import numpy as np
import multiprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import scipy.stats as sc
import scipy.sparse as sp
import scipy.ndimage as ndimage
import scipy.sparse.linalg as sla

# OpenMC code.
def CRAM48(A, n0, dt, rad_flag):
    # CRAM48 values.
    theta_r = np.array([
        -4.465731934165702e+1, -5.284616241568964e+0,
        -8.867715667624458e+0, +3.493013124279215e+0,
        +1.564102508858634e+1, +1.742097597385893e+1,
        -2.834466755180654e+1, +1.661569367939544e+1,
        +8.011836167974721e+0, -2.056267541998229e+0,
        +1.449208170441839e+1, +1.853807176907916e+1,
        +9.932562704505182e+0, -2.244223871767187e+1,
        +8.590014121680897e-1, -1.286192925744479e+1,
        +1.164596909542055e+1, +1.806076684783089e+1,
        +5.870672154659249e+0, -3.542938819659747e+1,
        +1.901323489060250e+1, +1.885508331552577e+1,
        -1.734689708174982e+1, +1.316284237125190e+1])

    theta_i = np.array([
        +6.233225190695437e+1, +4.057499381311059e+1,
        +4.325515754166724e+1, +3.281615453173585e+1,
        +1.558061616372237e+1, +1.076629305714420e+1,
        +5.492841024648724e+1, +1.316994930024688e+1,
        +2.780232111309410e+1, +3.794824788914354e+1,
        +1.799988210051809e+1, +5.974332563100539e+0,
        +2.532823409972962e+1, +5.179633600312162e+1,
        +3.536456194294350e+1, +4.600304902833652e+1,
        +2.287153304140217e+1, +8.368200580099821e+0,
        +3.029700159040121e+1, +5.834381701800013e+1,
        +1.194282058271408e+0, +3.583428564427879e+0,
        +4.883941101108207e+1, +2.042951874827759e+1])

    c48_theta = np.array(theta_r + theta_i * 1j, dtype=np.complex128)

    alpha_r = np.array([
        +6.387380733878774e+2, +1.909896179065730e+2,
        +4.236195226571914e+2, +4.645770595258726e+2,
        +7.765163276752433e+2, +1.907115136768522e+3,
        +2.909892685603256e+3, +1.944772206620450e+2,
        +1.382799786972332e+5, +5.628442079602433e+3,
        +2.151681283794220e+2, +1.324720240514420e+3,
        +1.617548476343347e+4, +1.112729040439685e+2,
        +1.074624783191125e+2, +8.835727765158191e+1,
        +9.354078136054179e+1, +9.418142823531573e+1,
        +1.040012390717851e+2, +6.861882624343235e+1,
        +8.766654491283722e+1, +1.056007619389650e+2,
        +7.738987569039419e+1, +1.041366366475571e+2])

    alpha_i = np.array([
        -6.743912502859256e+2, -3.973203432721332e+2,
        -2.041233768918671e+3, -1.652917287299683e+3,
        -1.783617639907328e+4, -5.887068595142284e+4,
        -9.953255345514560e+3, -1.427131226068449e+3,
        -3.256885197214938e+6, -2.924284515884309e+4,
        -1.121774011188224e+3, -6.370088443140973e+4,
        -1.008798413156542e+6, -8.837109731680418e+1,
        -1.457246116408180e+2, -6.388286188419360e+1,
        -2.195424319460237e+2, -6.719055740098035e+2,
        -1.693747595553868e+2, -1.177598523430493e+1,
        -4.596464999363902e+3, -1.738294585524067e+3,
        -4.311715386228984e+1, -2.777743732451969e+2])

    c48_alpha = np.array(alpha_r + alpha_i * 1j, dtype=np.complex128)
    c48_alpha0 = 2.258038182743983e-47
    
    # Initializes the CRAM48Solver with alpha, theta, and alpha0 values.
    alpha = c48_alpha
    theta = c48_theta
    alpha0 = c48_alpha0
    
    # Performs computations using CRAM48 for solving a system.
    A = sp.csr_matrix(A * dt, dtype=np.float64)
    if A.count_nonzero() == 0 or not rad_flag:
        return n0
    else:
        y = n0.copy()
        if np.shape(y)[1] == 1:
            y = y.reshape(-1)
    
        identity = sp.eye(A.shape[0])
        for alpha, theta in zip(alpha, theta):
            y += 2*np.real(alpha*sla.spsolve(A - theta*identity, y))
        
        if len(np.shape(y)) == 1:
            y = y.reshape(-1, 1)
    
        return y * alpha0


def CRAM16(A, n0, dt, rad_flag):
    # CRAM16 values.
    c16_alpha = np.array([
                +5.464930576870210e+3 - 3.797983575308356e+4j,
                +9.045112476907548e+1 - 1.115537522430261e+3j,
                +2.344818070467641e+2 - 4.228020157070496e+2j,
                +9.453304067358312e+1 - 2.951294291446048e+2j,
                +7.283792954673409e+2 - 1.205646080220011e+5j,
                +3.648229059594851e+1 - 1.155509621409682e+2j,
                +2.547321630156819e+1 - 2.639500283021502e+1j,
                +2.394538338734709e+1 - 5.650522971778156e+0j],
                dtype=np.complex128)
    
    c16_theta = np.array([
                +3.509103608414918 + 8.436198985884374j,
                +5.948152268951177 + 3.587457362018322j,
                -5.264971343442647 + 16.22022147316793j,
                +1.419375897185666 + 10.92536348449672j,
                +6.416177699099435 + 1.194122393370139j,
                +4.993174737717997 + 5.996881713603942j,
                -1.413928462488886 + 13.49772569889275j,
                -10.84391707869699 + 19.27744616718165j],
                dtype=np.complex128)
    
    c16_alpha0 = 2.124853710495224e-16
    
    # Initializes the CRAM48Solver with alpha, theta, and alpha0 values.
    alpha = c16_alpha
    theta = c16_theta
    alpha0 = c16_alpha0
    
    # Performs computations using CRAM18 for solving a system.
    
    A = sp.csr_matrix(A * dt, dtype=np.float64)
    
    if A.count_nonzero() == 0 or not rad_flag:
        return n0
    else:
        y = n0.copy()
        if np.shape(y)[1] == 1:
            y = y.reshape(-1)
    
        identity = sp.eye(A.shape[0])
        for alpha, theta in zip(alpha, theta):
            y += 2*np.real(alpha*sla.spsolve(A - theta*identity, y))
        
        if len(np.shape(y)) == 1:
            y = y.reshape(-1, 1)
    
        return y * alpha0

class Package:
    def __init__(self, mass, inventory, mode="atoms", batches=1, radioactive=True, decay_chain=True, secular_equilibrium=None, initialize=None):
        # Initializes a Package with Mass, Inventory, and optional parameters like radioactivity, decay_chain, and initialize.
        self.Mass = mass
        self.Inventory = inventory
        
        if batches == 1:
            self.batches = 1
        else:
            self.batches = batches

        self.radioactive         = radioactive
        self.decay_chain         = decay_chain        
        
        self.InventoryStates = {}
        self.ActivityStates  = {}
        self.HeatStates      = {}
        self.HeatSumStates   = {}
        
        self.NuclideIndexDict = {}

        if initialize == None:
            AllNuclides = self.Initialize(self.Inventory.keys())
            
            if secular_equilibrium is None:
                self.secular_equilibrium = []
            elif secular_equilibrium == "All":
                self.secular_equilibrium = list(self.Inventory.keys())
            else:
                self.secular_equilibrium = secular_equilibrium
            
            for nuc in AllNuclides:
                if nuc not in self.Inventory:
                    self.Inventory[nuc] = np.array([0.0]*self.batches)   

            self.DecaySequence()

        else:
            (self.HalfLifeDict, self.BranchingDict, self.MotherDaughterDict, self.EnergyDict, self.NuclideDict,
             self.InvNuclideDict, self.InventoryStates, self.ActivityStates, self.HeatStates, self.HeatSumStates) = initialize   
        
        for key, values in self.Inventory.items():
            if isinstance(values, float) or isinstance(values, int):
                self.Inventory[key] = np.array([float(values)]*self.batches)
            elif isinstance(values, np.ndarray):
                pass
            else:
                self.Inventory[key] = values.rvs(self.batches)
                  
        if mode == "activity":
            # Convert activity to atomic content.
            for nuc in self.Inventory.keys():            
                if self.HalfLifeDict[nuc] == np.inf:
                    self.Inventory[nuc] = np.array([0]*self.batches)
                else:
                    lmbda = np.log(2) / self.HalfLifeDict[nuc]
                    self.Inventory[nuc] = self.Inventory[nuc] / lmbda

    def DecaySequence(self):
        def DecaySequenceEquilibrium(mother_nuc, inventory_dict):
            # Take mother nuclide and duplicate the inventory for its daughter nuclides.
            for didx, daughter_nuc in enumerate(self.MotherDaughterDict[mother_nuc]):
                if daughter_nuc in inventory_dict:
                    inventory_dict[daughter_nuc] += inventory_dict[mother_nuc] * self.BranchingDict[mother_nuc][didx]
                else:
                    inventory_dict[daughter_nuc] = inventory_dict[mother_nuc] * self.BranchingDict[mother_nuc][didx]
                
                if daughter_nuc in self.MotherDaughterDict:
                    return DecaySequenceEquilibrium(daughter_nuc, inventory_dict)
            
            return inventory_dict
            
            
        # Define a function that takes a mother nuclide, and converts the activity
        for mother_nuc in self.secular_equilibrium:
            TempInventory = DecaySequenceEquilibrium(mother_nuc, {mother_nuc:self.Inventory[mother_nuc]})
            del TempInventory[mother_nuc] # Do not double count the inputted mother nuclide activity.
            for nuc, val in TempInventory.items():
                self.Inventory[nuc] += val


    def Refactor(self, TempNuclideList):
        def RefactorArray(batches, TempNuclideList, InvNuclideList, InvArray):
            if TempNuclideList != InvNuclideList:
                # Given a list of nuclides (TempNuclideList), convert inventory (InvArray) into same-sized array (replace nonexisting nuclides with 0).
                Y_combined = np.zeros((len(TempNuclideList), batches))
    
                for val_x, val_y in zip(InvNuclideList, InvArray):
                    if val_x in TempNuclideList:
                        idx = TempNuclideList.index(val_x)
                        Y_combined[idx] = val_y
                
                return Y_combined
            else:
                return InvArray


        for time in self.InventoryStates.keys():
            NucState = self.NuclideDict.keys()
            InvState = self.InventoryStates[time]
            ActState = self.ActivityStates[time]
            
            batches = np.shape(InvState)[1]
            
            self.InventoryStates[time] = RefactorArray(batches, TempNuclideList, NucState, InvState)
            self.ActivityStates[time] = RefactorArray(batches, TempNuclideList, NucState, ActState)
            
        # Remake the Inventory, NuclideDict and InvNuclideDict
        NewInventory = {}
        NewNuclideDict = {}
        NewInvNuclideDict = {}
        for idx, nuclide in enumerate(TempNuclideList):
            if nuclide in self.Inventory:
                NewInventory[nuclide] = self.Inventory[nuclide]
            else:
                NewInventory[nuclide] = np.zeros(self.batches)

            NewNuclideDict[nuclide] = idx
            NewInvNuclideDict[idx] = nuclide
        
        self.Inventory = NewInventory
        self.NuclideDict = NewNuclideDict
        self.InvNuclideDict = NewInvNuclideDict

    def Initialize(self, initial_nuclides):
        # Initialize dictionaries for nuclide properties
        self.HalfLifeDict = {}
        self.BranchingDict = {}
        self.MotherDaughterDict = {}
        self.EnergyDict = {}
        
        def FindDecayProducts(root, Parent, History):
            # Find decay products for a given nuclide and populate dictionaries
            # Recursively traverse the decay tree and collect decay-related data
            CandidateList = []
            # Iterating over the XML tree to find decay products
            # Populating dictionaries with decay data
            
            # Return the decay products for the given parent nuclide
            for parent in root:
                parent_name = parent.attrib["name"]

                if parent_name == Parent:
                    if "half_life" in parent.attrib:
                        self.HalfLifeDict[parent_name] = float(parent.attrib["half_life"])
                        self.EnergyDict[parent_name] = float(parent.attrib["decay_energy"])
                        
                        if self.decay_chain:
                            for daughter in parent.iter('decay'):
                                if "target" in daughter.attrib:
                                    daughter_name = daughter.attrib["target"]
                                else:
                                    daughter_name = parent_name

                                if daughter_name != parent_name:
                                    CandidateList.append(daughter_name)
                                    
                                    if parent_name in self.MotherDaughterDict:
                                        if daughter_name not in self.MotherDaughterDict[parent_name]:
                                            self.MotherDaughterDict[parent_name].append(daughter_name)
                                            self.BranchingDict[parent_name].append(float(daughter.attrib["branching_ratio"]))
                                    else:
                                        self.MotherDaughterDict[parent_name] = [daughter_name]
                                        self.BranchingDict[parent_name] = [float(daughter.attrib["branching_ratio"])]
                            
                            for candidate in CandidateList:
                                History = FindDecayProducts(root, candidate, History)
                                History.append(candidate)
    
                            return History
                        
                        else:
                            return History
                    
                    self.HalfLifeDict[parent_name] = np.inf
                    self.EnergyDict[parent_name] = 0.0
                    return History

            # Add
            self.MotherDaughterDict[Parent] = [Parent]
            self.BranchingDict[Parent] = [0.0]
                        
            self.HalfLifeDict[Parent] = np.inf
            self.EnergyDict[Parent] = 0.0
            print(f"Warning: {Parent} was not found in decay data - initializing as stable.")
    
            return History
            
        # Tree parsing and finding decay products for initial nuclides
        tree = ET.parse("decay_chains_endfb71.xml")
        root = tree.getroot()
        
        # Collecting daughter nuclides from initial nuclides
        DaughterNuclides = []
        for nuclide in initial_nuclides:
            DaughterNuclides += FindDecayProducts(root, nuclide, list(initial_nuclides))
        
        # Generating a set of simulated nuclides based on the decay chains found (filter out duplicates)
        SimulatedNuclides = set(DaughterNuclides)
        
        return SimulatedNuclides


    def Calculate_Bateman(self):
        # Initializing matrices to store Bateman matrix and energy vectors
        self.BatemanMatrix = np.zeros((len(self.Inventory.keys()), len(self.Inventory.keys())))
        self.EnergyVector = np.zeros((len(self.Inventory.keys())))
        
        NuclideList = self.Inventory.keys()

        # Iterating through Inventory nuclides to compute Bateman matrix elements
        for nidx, nuc in enumerate(NuclideList):
            LambdaDiag = np.log(2.0) / self.HalfLifeDict[nuc] # Calculate decay constant
            self.NuclideIndexDict[nuc] = nidx # Assign index for the nuclide
            self.BatemanMatrix[nidx, nidx] = -LambdaDiag # Set diagonal element in Bateman matrix
            self.EnergyVector[nidx] = self.EnergyDict[nuc] * 1.60218e-19  # Set energy vector value
        
        # Computing off-diagonal elements in Bateman matrix based on decay relationships
        for mother, daughters in self.MotherDaughterDict.items():
            for didx in range(len(daughters)):
                LambdaBranch = np.log(2.0) / self.HalfLifeDict[mother] # Calculate decay constant for mother nuclide

                # Set off-diagonal elements in Bateman matrix using branching ratios and decay constants
                self.BatemanMatrix[self.NuclideIndexDict[daughters[didx]],
                                   self.NuclideIndexDict[mother]] = (self.BranchingDict[mother][didx]
                                                                     * LambdaBranch)
                                                                     
    
    def Calculate_States(self, stepsize, timesteps, solver="CRAM16", cont=False):
        # Calculate Bateman matrix initially
        self.Calculate_Bateman()
        
        cont = len(self.InventoryStates) != 0
        # If not continuing from previous states
        if not cont:
            N_t = np.array(list(self.Inventory.values()))

            # Initialize states at time 0
            self.InventoryStates[0] = N_t
            self.ActivityStates[0] = -np.einsum("ii,ik->ik", self.BatemanMatrix, N_t)
            self.HeatStates[0] = np.einsum("i,ij->ij", self.EnergyVector, self.ActivityStates[0])
            self.HeatSumStates[0] = self.HeatStates[0].sum(axis=0)
        else:
            last_timestep = max(list(self.InventoryStates.keys()))
            N_t = np.array(self.InventoryStates[last_timestep])
        
        # Perform iterations for given timesteps
        for step in range(timesteps):
            if solver == "CRAM48":
                N_t = CRAM48(self.BatemanMatrix, N_t, stepsize, self.radioactive)
            else:
                N_t = CRAM16(self.BatemanMatrix, N_t, stepsize, self.radioactive)
                
            ActivityVector = -np.einsum("ii,ik->ik", self.BatemanMatrix, N_t)
            HeatVector = np.einsum("i,ij->ij", self.EnergyVector, ActivityVector)
            
            # Update states based on the step and continuation status
            if not cont:
                self.InventoryStates[(step+1)*stepsize] = N_t
                self.ActivityStates[(step+1)*stepsize] = ActivityVector
                self.HeatStates[(step+1)*stepsize] = HeatVector
                self.HeatSumStates[(step+1)*stepsize] = HeatVector.sum(axis=0)
            else:
                self.InventoryStates[last_timestep+(step+1)*stepsize] = N_t
                self.ActivityStates[last_timestep+(step+1)*stepsize] = ActivityVector
                self.HeatStates[last_timestep+(step+1)*stepsize] = HeatVector
                self.HeatSumStates[last_timestep+(step+1)*stepsize] = HeatVector.sum(axis=0)
                

class CombinedPackage:
    def __init__(self, PackageStates, NuclideDict):
        self.NuclideDict = NuclideDict
        
        def compile_package_list(PackageStates):
            PackStates = []
            for PackageState, DivFactor in PackageStates:
                time, pack = PackageState
                
                if isinstance(pack, CombinedPackage):
                    compile_package_list(pack.PackageStates)
                    
                    for PackageStatez, DivFactorz in pack.PackageStates:
                        PackStates.append([[PackageStatez[0]+time, PackageStatez[1]], DivFactorz*DivFactor])

                else:
                    PackStates.append([PackageState, DivFactor])
                    
            return PackStates
        
        self.PackageStates = compile_package_list(PackageStates)
        
    def get_mass(self):
        TempMass = 0
        for (time, pack), DivFactor in self.PackageStates:
            TempMass += pack.Mass * DivFactor

        return TempMass 

    def get_inventory(self, age):
        TempInventory = 0
        for (time, pack), DivFactor in self.PackageStates:
            TempInventory += pack.InventoryStates[time+age] * DivFactor

        return TempInventory

    def get_activity(self, age):
        TempActivity = 0
        for (time, pack), DivFactor in self.PackageStates:
            TempActivity += pack.ActivityStates[time+age] * DivFactor

        return TempActivity

    def get_heat(self, age):
        TempHeat = 0
        for (time, pack), DivFactor in self.PackageStates:
            TempHeat += pack.HeatStates[time+age] * DivFactor
        return TempHeat

    def get_packages(self):
        TempPackages = []
        for i, j in self.PackageStates:
            TempPackages.append(i[1])

        return TempPackages


class Source:
    def __init__(self, awaynode, package, magnitude, rate=0):
        # Initializes a Source with AwayNode, Package, Magnitude, and an optional Rate.
        self.AwayNode = awaynode
        self.Package = package
        self.Magnitude = magnitude
        
        self.Rate = rate
        self.UntilDelivery = rate

    def Transfer(self):
        # Transfers packages from the source to the away node.
        for mag in range(self.Magnitude):
            PackageState = [0, self.Package]
            self.AwayNode.add_package(PackageState)

class Node:
    def __init__(self, packagelist=None, multiplication_factor=1):
        # Initializes a Node with an optional PackageList and multiplication factor.
        if packagelist == None:
            self.PackageList = []
            self.TempPackageList = []
        else:
            self.PackageList  = packagelist
            self.TempPackageList = packagelist
            
        MultiPackageList  = []
        for pack in self.PackageList:
            for i in range(multiplication_factor):
                MultiPackageList.append([0, pack])

        self.PackageList = MultiPackageList
        self.TempPackageList =  MultiPackageList
                    

    def add_package(self, package):
        # Adds a package to the node.
        self.TempPackageList.insert(0, package)
        
    def delete_package(self, package):
        # Deletes a package from the node.
        self.TempPackageList.remove(package)
        
    def update(self):
        # Updates the node's package list.
        self.PackageList = self.TempPackageList
        
    def get_packages(self):
        # Retrieves the list of packages in the node.
        return list(map(list, zip(*self.PackageList)))
    
    def get_mass(self):
        # Calculates and retrieves the total mass of packages in the node.
        TempMass = 0
        for pack in list(map(list, zip(*self.PackageList)))[1]:
            if isinstance(pack, CombinedPackage):
                TempMass += pack.get_mass()
            else:
                TempMass += pack.Mass
        return None, TempMass
    
    def get_heat(self):
        # Retrieves the total heat generated by packages in the node.
        TempHeat = 0
        
        for (time, pack) in self.PackageList:
            if isinstance(pack, CombinedPackage):
                TempHeat += pack.get_heat(time)
            else:
                TempHeat += pack.HeatStates[time]

        return pack.NuclideDict, TempHeat
    
    def get_inventory(self):
        # Retrieves the total inventory of nuclides in the node.
        TempInventory = 0
        for (time, pack) in self.PackageList:
            if isinstance(pack, CombinedPackage):
                TempInventory += pack.get_inventory(time)
            else:
                TempInventory += pack.InventoryStates[time]

        return pack.NuclideDict, TempInventory
    
    def get_activity(self):
        # Retrieves the total activity of nuclides in the node.
        TempActivity = 0
        for (time, pack) in self.PackageList:
            if isinstance(pack, CombinedPackage):
                TempActivity += pack.get_inventory(time)
            else:
                TempActivity += pack.ActivityStates[time]
                
        return pack.NuclideDict, TempActivity

class Order:
    def __init__(self, homenode, ordernodes, magnitude=1, mode="Transfer", remaindernode=None, rate=0, direction="old",
                 criteria=None, instruct=None, crumbs=False, centile=80):
        # Create the same way as a link. Instead of awaynode,  define a list of AwayNodes, search for packages in that list that satisfy the criteria.
        self.HomeNode = homenode
        self.OrderNodes = ordernodes
        
        self.Magnitude = magnitude # Number of packages to move OR a criteria for movement.
        self.MagnitudeSample = self.Magnitude # Storing a sample of the magnitude
        self.Rate = rate # Rate of movement
        self.Mode = mode
        self.RemainderNode = remaindernode
        self.UntilDelivery = rate # Store rate for tracking delivery
        
        self.Instruct = instruct # Modification instructions for movement
        
        self.Direction = direction # Order for movement (default: "old")
        self.Crumbs = crumbs # Toggle for recording movement details
        
        self.Centile = centile
        
        # Default criteria for movement based on inventory level of a specific nuclide
        Default_Criteria = {"region"   : "package",
                            "variable" : "inventory",
                            "criteria" : 0.1, # Default inventory criteria
                            "principle": "min"} # Default nuclide for criteria
        
        #{"sort_nuclides":["Sr90", "Co60"],
        # "sort_factors":[1, 1],
        # "sort_node":Node3}
        
        #Instruct={"package_out":3,
        #          "package_mass":10}
        
        # Handling provided or default criteria for movement
        if criteria != None:
            self.Criteria = criteria
            for Crit in criteria:
                # Update with default criteria for missing keys
                for key, value in Default_Criteria.items():
                    if key not in Crit:
                        Crit[key] = value
        else:
            self.Criteria = None # Use provided criteria or default to None if not provide
            
    
    def Transfer(self, stepsize, timesteps, UniquePackageDict, InvUniquePackageDict, centile):
        # There are multiple modes: Regular Order of Package/s, Order of a part of a package.
        # Make it possible to combine the ordered packages into one new package.
        # Make it possible to modify the package (e.g. volume/mass)

        # Sample from the Magnitude distribution, convert the distribution into a temp int value.
        if isinstance(self.Magnitude, int) or isinstance(self.Magnitude, float):
            self.MagnitudeSample = int(round(self.Magnitude))
        else:
            self.MagnitudeSample = int(self.Magnitude.rvs(1))
        
        # Go through the AwayNodes and find one that has fitting characteristics.
        # This means that it must Evaluate each Node, and rank them.
        # Ranking requires a criteria. Criteria is passed using the self.Criteria.
        # For Each criteria, there is True, False, or a number, thus returning a dict of node -> [True, True, False] e.g.
        # Pick either the list with the total highest sum, or highest sum and > the number of entires, or first in priority and > the number of entires.
        # If use ratio between criteria and value to assess the number.
        NodeCriteriaResults = {}
        for AwayNode in self.OrderNodes:
            NodeCriteriaResults[AwayNode] = self.CheckCriteria(AwayNode, centile)
        
        SuitableNodes = []
        for nodes, res in NodeCriteriaResults.items():
            if all(res):
                SuitableNodes.append(nodes)

        if len(SuitableNodes) != 0:
            indeces = []
            for suit_node in SuitableNodes:
                index = self.OrderNodes.index(suit_node)
                indeces.append(index)
            indeces.sort()
            
            TransferNodes = []
            for idx in indeces:
                TransferNodes.append(self.OrderNodes[idx])
                
                if len(self.OrderNodes[idx].PackageList) > self.MagnitudeSample or len(indeces) == 0:
                    break
            
            # last and first options while constructing the AvailablePackages
            if self.Direction == "new":
                AvailablePackages = [obj for instance in TransferNodes for obj in instance.PackageList]
            elif self.Direction == "old":
                AvailablePackages = [obj for instance in TransferNodes for obj in list(reversed(instance.PackageList))]

            NodePackageContributions = [len(nod.PackageList) - 1 for nod in TransferNodes]

            if self.Mode == "Transfer" or self.Mode == "Combine": 
                if len(AvailablePackages) >= self.MagnitudeSample: 
                    TempMagnitudeSample = self.MagnitudeSample
                elif len(AvailablePackages) < self.MagnitudeSample and self.Crumbs:
                    TempMagnitudeSample = len(AvailablePackages)
                else:
                    TempMagnitudeSample = 0
        
                # Transfer the packages between nodes.
                
                if self.Mode == "Combine":
                    UniquePackageDict, InvUniquePackageDict = self.CombinePackages(TransferNodes, AvailablePackages, NodePackageContributions, stepsize, timesteps, UniquePackageDict, InvUniquePackageDict)
                else:
                    mag_idx = 0
                    for mag in range(TempMagnitudeSample):
                        # All the filters need to go here.
                        # Select the removable package state.
                        selected_package_state = AvailablePackages[0] # Oldest packaging.
                            
                        # Remove state from old node.
                        if mag <= NodePackageContributions[mag_idx]:
                            TransferNodes[mag_idx].delete_package(selected_package_state)
                        else:
                            mag_idx += 1
                            TransferNodes[mag_idx].delete_package(selected_package_state)

                        AvailablePackages.remove(selected_package_state)
                        # Add state to new node.
                        self.HomeNode.add_package(selected_package_state.copy())

            if self.Mode == "Separate":       
                # Transfer the packages between nodes.                            
                UniquePackageDict, InvUniquePackageDict = self.Separate(TransferNodes, AvailablePackages, NodePackageContributions, stepsize, timesteps, UniquePackageDict, InvUniquePackageDict)

            if self.Mode == "Sort":
                # Instruct must have an additional parameters {"remainder":Node, "separated_nuclides":[], "Separation_factor":[]}.
                # Transfer the packages between nodes.                             
                UniquePackageDict, InvUniquePackageDict = self.Sort(TransferNodes, AvailablePackages, NodePackageContributions, stepsize, timesteps, UniquePackageDict, InvUniquePackageDict)

        return UniquePackageDict, InvUniquePackageDict


    def Separate(self, TransferNodes, AvailablePackages, NodePackageContributions, stepsize, timesteps, UniquePackageDict, InvUniquePackageDict):
        InputNumber = len(AvailablePackages)
        NumOfPacks = int(self.Instruct["package_out"])
            
        CombinationStates = []
        TotalMass = 0
        # First create a list of states to be combined.
        if InputNumber != 0:
            mag_idx = 0
            for mag in range(InputNumber):
                time, selected_package = AvailablePackages[0]
                CombinationStates.append([time, selected_package])
                
                if isinstance(selected_package, CombinedPackage):
                    TotalMass += selected_package.get_mass()
                else:
                    TotalMass += selected_package.Mass
                
                # Remove state from old node.
                if mag <= NodePackageContributions[mag_idx]:
                    TransferNodes[mag_idx].delete_package([time, selected_package])
                else:
                    mag_idx += 1
                    TransferNodes[mag_idx].delete_package([time, selected_package])

                AvailablePackages.remove([time, selected_package])
        
        # Refactor coefficients.
        TotalMass = round(TotalMass, 9)
        if TotalMass != 0:
            if self.Instruct["package_mass"] > TotalMass and self.Crumbs:
                MovedMass = TotalMass
            else:
                MovedMass = self.Instruct["package_mass"] * NumOfPacks

            Sep_Coef = min(MovedMass / TotalMass, 1)
            Rem_Coef = 1 - Sep_Coef
            
            # 1.5 Create proper package object without reading the nuclide data.
            separated_package = CombinedPackage(PackageStates=list(zip(CombinationStates, [Sep_Coef/NumOfPacks]*len(CombinationStates))), NuclideDict=selected_package.NuclideDict)
            
            if separated_package not in UniquePackageDict.values():
                if len(UniquePackageDict) != 0:
                    uni_idx = max(list(UniquePackageDict.keys())) + 1
                else:
                    uni_idx = 0
                UniquePackageDict[uni_idx] = separated_package
                InvUniquePackageDict[separated_package] = uni_idx
            
            if Rem_Coef >= 0:
                remaining_package = CombinedPackage(PackageStates=list(zip(CombinationStates, [Rem_Coef]*len(CombinationStates))), NuclideDict=selected_package.NuclideDict)
                
                if remaining_package not in UniquePackageDict.values():
                    if len(UniquePackageDict) != 0:
                        uni_idx = max(list(UniquePackageDict.keys())) + 1
                    else:
                        uni_idx = 0
                    UniquePackageDict[uni_idx] = remaining_package
                    InvUniquePackageDict[remaining_package] = uni_idx
                    
                    # Add all of the remaining into last node that was looked at.
                    # Should be changed in the future to properly account for package origins.
                    TransferNodes[mag_idx].add_package([0, remaining_package])
                
            # Convert the modified package into array for each batch.
            for i in range(NumOfPacks):
                self.HomeNode.add_package([0, separated_package])
            
        return UniquePackageDict, InvUniquePackageDict


    def Sort(self, TransferNodes, AvailablePackages, NodePackageContributions, stepsize, timesteps, UniquePackageDict, InvUniquePackageDict):
        if len(AvailablePackages) >= self.MagnitudeSample or self.Crumbs:
            if len(AvailablePackages) < self.MagnitudeSample and self.Crumbs:
                InputNumber = len(AvailablePackages)
                NumOfPacks = 1
            else:
                InputNumber = self.MagnitudeSample
                NumOfPacks = 1
            
            CombinationStates = []
            # First create a list of states to be combined.
            if InputNumber != 0:
                mag_idx = 0
                for mag in range(InputNumber):
                    time, selected_package = AvailablePackages[0]
                    CombinationStates.append([time, selected_package])
                    
                    # Remove state from old node.
                    if mag <= NodePackageContributions[mag_idx]:
                        TransferNodes[mag_idx].delete_package([time, selected_package])
                    else:
                        mag_idx += 1
                        TransferNodes[mag_idx].delete_package([time, selected_package])

                    AvailablePackages.remove([time, selected_package])
        
        # Refactor coefficients.
        Sep_Coef = self.SortFactors
        Rem_Coef = 1 - Sep_Coef

        # 1.5 Create proper package object without reading the nuclide data.
        separated_package = CombinedPackage(PackageStates=list(zip(CombinationStates, [Sep_Coef]*len(CombinationStates))), NuclideDict=selected_package.NuclideDict)
        remaining_package = CombinedPackage(PackageStates=list(zip(CombinationStates, [Rem_Coef]*len(CombinationStates))), NuclideDict=selected_package.NuclideDict)
        
        if separated_package not in UniquePackageDict.values():
            if len(UniquePackageDict) != 0:
                uni_idx = max(list(UniquePackageDict.keys())) + 1
            else:
                uni_idx = 0
            UniquePackageDict[uni_idx] = separated_package
            InvUniquePackageDict[separated_package] = uni_idx
            
        if remaining_package not in UniquePackageDict.values():
            if len(UniquePackageDict) != 0:
                uni_idx = max(list(UniquePackageDict.keys())) + 1
            else:
                uni_idx = 0
            UniquePackageDict[uni_idx] = remaining_package
            InvUniquePackageDict[remaining_package] = uni_idx
            
        # Convert the modified package into array for each batch.
        self.HomeNode.add_package([0, separated_package])
        self.Instruct["sort_node"].add_package([0, remaining_package])
            
        return UniquePackageDict, InvUniquePackageDict


    def Refactor(self, batches, TempNuclideList, InvNuclideList, InvArray):
        # Given a list of nuclides (TempNuclideList), convert inventory (InvArray) into same-sized array (replace nonexisting nuclides with 0).
        Y_combined = np.zeros((len(TempNuclideList), batches))

        for val_x, val_y in zip(InvNuclideList, InvArray):
            if val_x in TempNuclideList:
                idx = TempNuclideList.index(val_x)
                Y_combined[idx] = val_y
        
        return Y_combined


    def CombinePackages(self, TransferNodes, AvailablePackages, NodePackageContributions, stepsize, timesteps, UniquePackageDict, InvUniquePackageDict):
        if len(AvailablePackages) >= self.MagnitudeSample or self.Crumbs:
            if len(AvailablePackages) < self.MagnitudeSample and self.Crumbs:
                PackConvQuant = self.MagnitudeSample / self.Instruct["package_out"]
                Residue    = int(len(AvailablePackages) % PackConvQuant)
                NumOfPacks = int(len(AvailablePackages) // PackConvQuant) + (Residue != 0)
                InputNumber = len(AvailablePackages)
            else:
                InputNumber = self.MagnitudeSample
                NumOfPacks = int(self.Instruct["package_out"])
            
            CombinationStates = []
            # First create a list of states to be combined.
            if InputNumber != 0:
                mag_idx = 0
                for mag in range(InputNumber):
                    time, selected_package = AvailablePackages[0]
                    CombinationStates.append([time, selected_package])
                    
                    # Remove state from old node.
                    if mag <= NodePackageContributions[mag_idx]:
                        TransferNodes[mag_idx].delete_package([time, selected_package])
                    else:
                        mag_idx += 1
                        TransferNodes[mag_idx].delete_package([time, selected_package])

                    AvailablePackages.remove([time, selected_package])
    
        # 1.5 Create proper package object without reading the nuclide data.
        combined_package = CombinedPackage(PackageStates=list(zip(CombinationStates, [1/NumOfPacks]*len(CombinationStates))), NuclideDict=selected_package.NuclideDict)
        if combined_package not in UniquePackageDict.values():
            if len(UniquePackageDict) != 0:
                uni_idx = max(list(UniquePackageDict.keys())) + 1
            else:
                uni_idx = 0

            UniquePackageDict[uni_idx] = combined_package
            InvUniquePackageDict[combined_package] = uni_idx
            
        for i in range(NumOfPacks):
            # Convert the modified package into array for each batch.
            self.HomeNode.add_package([0, combined_package])

        return UniquePackageDict, InvUniquePackageDict


    def CheckCriteria(self, AwayNode, centile):    
        if self.Criteria != None:
            CriteriaResults = []
            for Crit in self.Criteria:
                CriteriaResults.append(self.CheckCrit(Crit, AwayNode, centile))
            return CriteriaResults
        else:
            CriteriaResults = []
            CriteriaResults.append(self.CheckCrit(self.Criteria, AwayNode, centile))
            return CriteriaResults


    def CheckCrit(self, Criteria, AwayNode, centile):
        # Check if there is enough packages for transfer.
        if len(AwayNode.PackageList) < self.MagnitudeSample and not self.Crumbs:
            return False
        elif len(AwayNode.PackageList) == 0:
            return False
        elif len(AwayNode.PackageList) < self.MagnitudeSample and self.Crumbs:
            # Number of available packages.
            SPidx = len(AwayNode.PackageList)
        else:
            # Number of all transfered packages.
            SPidx = self.MagnitudeSample
        
        # Check if there are any criteria present.
        if Criteria == None:
            return True
        
        # Determine the order of transfer.
        if self.Direction == "new":
            selected_package_states = AwayNode.PackageList[:SPidx]
        elif self.Direction == "old":
            selected_package_states = AwayNode.PackageList[-SPidx:]
        
        # Calculate the total mass of node packages for separation and transfer.
        sep_coef = 1
        if self.Mode == "Separate" and Criteria["region"] == "package":
            MovedMass = self.Instruct["package_mass"] * int(self.Instruct["package_out"])
            TotalMass = 0
            for (time, selected_package) in selected_package_states:    
                if isinstance(selected_package, CombinedPackage):
                    TotalMass += selected_package.get_mass()
                else:
                    TotalMass += selected_package.Mass
            TotalMass = round(TotalMass, 9)

            if TotalMass != 0:
                sep_coef = min(MovedMass / TotalMass, 1)
        
        # If packages are combined/separated, calculate variable value for combined/separated packages.
        if (self.Mode == "Combine" and Criteria["region"] == "package") or (self.Mode == "Separate" and Criteria["region"] == "package"):
            ComparedVar = 0
            for (time, selected_package) in selected_package_states:
                if isinstance(selected_package, CombinedPackage):
                    if Criteria["variable"] == "mass":
                        ComparedVar += selected_package.get_mass()
                    elif Criteria["variable"] == "inventory":
                        ComparedVar += selected_package.get_inventory(time)
                    elif Criteria["variable"] == "activity":
                        ComparedVar += selected_package.get_activity(time)
                    elif Criteria["variable"] == "heat":
                        ComparedVar += selected_package.get_heat(time)
                else:
                    if Criteria["variable"] == "mass":
                        ComparedVar += selected_package.Mass
                    elif Criteria["variable"] == "inventory":
                        ComparedVar += selected_package.InventoryStates[time] 
                    elif Criteria["variable"] == "activity":
                        ComparedVar += selected_package.ActivityStates[time]
                    elif Criteria["variable"] == "heat":
                        ComparedVar += selected_package.HeatStates[time]  

            ComparedVar = ComparedVar * sep_coef / self.Instruct["package_out"]
        
        if Criteria["region"] == "node":
            if "nuclide" not in Criteria:
                if Criteria["variable"] == "mass":
                    _, node_var = AwayNode.get_mass()
                elif Criteria["variable"] == "inventory":
                    _, node_var = AwayNode.get_inventory()
                elif Criteria["variable"] == "activity":
                    _, node_var = AwayNode.get_activity()
                elif Criteria["variable"] == "heat":
                    _, node_var = AwayNode.get_heat()
                if _ is not None:
                    node_var = node_var.sum(axis=0)

                if ((Criteria["principle"] == "max" and np.percentile(node_var, centile) < Criteria["criteria"])
                    or (Criteria["principle"] == "min" and np.percentile(node_var, centile) > Criteria["criteria"])):
                    return False
            else:
                if Criteria["variable"] == "inventory":
                    NuclideDict, node_var = AwayNode.get_inventory()
                elif Criteria["variable"] == "activity":
                    NuclideDict, node_var = AwayNode.get_activity()
                elif Criteria["variable"] == "heat":
                    NuclideDict, node_var = AwayNode.get_heat()

                for nuclide, crit in zip(Criteria["nuclide"], Criteria["criteria"]):
                    if isinstance(nuclide, str):
                        nuclides = [nuclide]
                    else:
                        nuclides = nuclide
                    ComparedVar = 0
                    for nuclide in nuclides:
                        if nuclide in NuclideDict:
                            ComparedVar += node_var[NuclideDict[nuclide]]

                    if ((Criteria["principle"] == "max" and np.percentile(ComparedVar, centile) < Criteria["criteria"])
                        or (Criteria["principle"] == "min" and np.percentile(ComparedVar, centile) > Criteria["criteria"])):
                        return False

        if Criteria["region"] == "package":
            if "nuclide" not in Criteria:
                if self.Mode == "Combine" or self.Mode == "Separate":
                    if ((Criteria["principle"] == "max" and np.percentile(ComparedVar.sum(axis=0), centile) < Criteria["criteria"])
                        or (Criteria["principle"] == "min" and np.percentile(ComparedVar.sum(axis=0), centile) > Criteria["criteria"])):
                        return False
                else:
                    for (time, selected_package) in selected_package_states:    
                        if isinstance(selected_package, CombinedPackage): 
                            if Criteria["variable"] == "mass":
                                pack_var = AwayNode.get_mass()
                            elif Criteria["variable"] == "inventory":
                                pack_var = AwayNode.get_inventory().sum(axis=0)
                            elif Criteria["variable"] == "activity":
                                pack_var = AwayNode.get_activity().sum(axis=0)
                            elif Criteria["variable"] == "heat":
                                pack_var = AwayNode.get_heat().sum(axis=0)
                        else:
                            if Criteria["variable"] == "mass":
                                pack_var = selected_package.Mass
                            elif Criteria["variable"] == "inventory":
                                pack_var = selected_package.InventoryStates[time].sum(axis=0)
                            elif Criteria["variable"] == "activity":
                                pack_var = selected_package.ActivityStates[time].sum(axis=0)
                            elif Criteria["variable"] == "heat":
                                pack_var = selected_package.HeatStates[time].sum(axis=0)
                                
                        if ((Criteria["principle"] == "max" and np.percentile(pack_var, centile) < Criteria["criteria"])
                            or (Criteria["principle"] == "min" and np.percentile(pack_var, centile) > Criteria["criteria"])):
                            return False
                        
            else:
                for nuc, crit in zip(Criteria["nuclide"], Criteria["criteria"]):
                    if self.Mode == "Combine" or self.Mode == "Separate":
                        if isinstance(nuc, str):
                            nuclides = [nuc]
                        else:
                            nuclides = nuc
                        pack_var = 0
                        for nuclide in nuclides:
                            if nuclide in selected_package.NuclideDict:
                                pack_var += ComparedVar[selected_package.NuclideDict[nuclide]]

                        if ((Criteria["principle"] == "max" and np.percentile(pack_var, centile) < crit)
                            or (Criteria["principle"] == "min" and np.percentile(pack_var, centile) > crit)):
                            return False
                    else:
                        for (time, selected_package) in selected_package_states:
                            if isinstance(nuc, str):
                                nuclides = [nuc]
                            else:
                                nuclides = nuc
                            pack_var = 0
                            for nuclide in nuclides:
                                if nuclide in selected_package.NuclideDict:
                                    if isinstance(selected_package, CombinedPackage):
                                        if Criteria["variable"] == "inventory":
                                            pack_var += selected_package.get_inventory(time)[selected_package.NuclideDict[nuclide]]
                                        elif Criteria["variable"] == "activity":
                                            pack_var += selected_package.get_activity(time)[selected_package.NuclideDict[nuclide]]
                                        elif Criteria["variable"] == "heat":
                                            pack_var += selected_package.get_heat(time)[selected_package.NuclideDict[nuclide]]
                                    else:
                                        if Criteria["variable"] == "inventory":
                                            pack_var += selected_package.InventoryStates[time][selected_package.NuclideDict[nuclide]]
                                        elif Criteria["variable"] == "activity":
                                            pack_var += selected_package.ActivityStates[time][selected_package.NuclideDict[nuclide]]
                                        elif Criteria["variable"] == "heat":
                                            pack_var += selected_package.HeatStates[time][selected_package.NuclideDict[nuclide]]

                            #print(nuc, np.percentile(pack_var, centile), crit)
                            if ((Criteria["principle"] == "max" and np.percentile(pack_var, centile) < crit)
                                or (Criteria["principle"] == "min" and np.percentile(pack_var, centile) > crit)):
                                return False

        # Return True or False whether it passes or not.
        return True

    def Initialize(self, initial_nuclides):
        def FindDecayProducts(root, Parent, History):
            # Find decay products for a given nuclide and populate dictionaries
            # Recursively traverse the decay tree and collect decay-related data
            CandidateList = []
            # Iterating over the XML tree to find decay products
            # Populating dictionaries with decay data
            
            # Return the decay products for the given parent nuclide
            for parent in root:
                parent_name = parent.attrib["name"]

                if parent_name == Parent:
                    if "half_life" in parent.attrib:

                        for daughter in parent.iter('decay'):
                            daughter_name = daughter.attrib["target"]
                            
                            if daughter_name != parent_name:
                                CandidateList.append(daughter_name)

                        
                        for candidate in CandidateList:
                            History = FindDecayProducts(root, candidate, History)
                            History.append(candidate)

                        return History
            
                    return History
            
        # Tree parsing and finding decay products for initial nuclides
        tree = ET.parse("decay_chains_endfb71.xml")
        root = tree.getroot()
        
        # Collecting daughter nuclides from initial nuclides
        DaughterNuclides = []
        for nuclide in initial_nuclides:
            DaughterNuclides += FindDecayProducts(root, nuclide, list(initial_nuclides))
        
        # Generating a set of simulated nuclides based on the decay chains found (filter out duplicates)
        SimulatedNuclides = set(DaughterNuclides)
        
        return SimulatedNuclides

class Universe:
    def __init__(self, nodes=None, orders=None, sources=None, history=None, stepsize=None):
        if nodes is None:
            self.Nodes = []
        else:
            self.Nodes = nodes
            
        if orders is None:
            self.Orders = []
        else:
            self.Orders = orders
            
        if sources is None:
            self.Sources = []
        else:
            self.Sources = sources
            
        if history is None:
            self.History = []
        else:
            self.History = history
        
        if stepsize is None:
            self.stepsize = 1*60*60*24*365
        else:
            self.stepsize = stepsize
        
        self.NodeDict = {}
        
        self.SimTimeHistory = [0]
        self.SimTime = 0
        self.batches = 0
        
        self.BasePackageList = []
        self.UniquePackageDict = {}
        self.InvUniquePackageDict = {}
        self.NodePackageHistory = []
        self.HistoryIdx = {}


    def __add__(self, Obj):
        if isinstance(Obj, Node):
            if Obj not in self.Nodes:
                self.Nodes.append(Obj)
                self.History.append(Obj)
                
        if isinstance(Obj, Order):
            if Obj not in self.Orders:
                self.Orders.append(Obj)
                
        if isinstance(Obj, Source):
            if Obj not in self.Sources:
                self.Sources.append(Obj)
        
        return self


    def __sub__(self, Obj):
        if isinstance(Obj, Node):
            self.Nodes.remove(Obj)
            self.History.remove(Obj)
        
        if isinstance(Obj, Order):
            self.Orders.remove(Obj)
            
        if isinstance(Obj, Source):
            self.Sources.remove(Obj)
        
        return self  
    
    
    def Initialize(self, timesteps, solver, num_of_cores):
        # Unique radionuclides present in the packages.
        UniqueNuclides = []
        TotalUniquePackages = []
        
        for nidx, node in enumerate(self.Nodes):
            # Number and label the nodes.
            self.NodeDict[node] = nidx
            
            if len(node.PackageList) != 0:
                UniquePackages = set(list(map(list, zip(*node.PackageList)))[1])
                TotalUniquePackages += list(UniquePackages)
            
        UniquePackages = set([source.Package for source in self.Sources])
        TotalUniquePackages += list(UniquePackages)
        
        for UniquePack in TotalUniquePackages:
            if UniquePack not in self.UniquePackageDict.values():
                if len(self.UniquePackageDict) != 0:
                    uni_idx = max(list(self.UniquePackageDict.keys())) + 1
                else:
                    uni_idx = 0
                    
                self.UniquePackageDict[uni_idx] = UniquePack
                self.InvUniquePackageDict[UniquePack] = uni_idx
                
            if not isinstance(UniquePack, CombinedPackage) and UniquePack not in self.BasePackageList:
                self.BasePackageList.append(UniquePack)

        for BasePack in self.BasePackageList:
            UniqueNuclides = UniqueNuclides + list(BasePack.Inventory.keys())
        UniqueNuclides = list(dict.fromkeys(UniqueNuclides))
        
        # Refactor the radionuclide inventory AND bateman matrix.
        # Check whether there are any normal packages.
        TotalHalfLifeDict = {}
        TotalEnergyDict = {}
        for BasePack in self.BasePackageList:
            TotalHalfLifeDict.update(BasePack.HalfLifeDict)
            TotalEnergyDict.update(BasePack.EnergyDict)
            BasePack.Refactor(UniqueNuclides)

        # Calculate radioactive decay, but first check whether there are any normal packages.
        for BasePack in self.BasePackageList:
            BasePack.HalfLifeDict = TotalHalfLifeDict
            BasePack.EnergyDict = TotalEnergyDict
                
            if len(BasePack.InventoryStates) == 0:
                BasePack = parallel_simulate(BasePack, self.stepsize, timesteps, solver, num_of_cores)
            
            elif len(BasePack.InventoryStates) != 0:
                BasePack = parallel_simulate(BasePack, self.stepsize, timesteps, solver, num_of_cores)
            
            self.batches = BasePack.batches
        
        # Go through the sorting orders and create the initial sorting matrix.
        for order in self.Orders:
            if order.Mode == "Sort":
                sort_nuclides = []
                sort_factors  = []
                for inidx, nuc in enumerate(order.Instruct["sort_nuclides"]):
                    nuc_chain = list(order.Initialize([nuc]))
                    sort_nuclides = sort_nuclides + nuc_chain
                    sort_factors = sort_factors + len(nuc_chain) * [order.Instruct["sort_factors"][inidx]]
                
                order.SortFactors = order.Refactor(self.batches, UniqueNuclides, sort_nuclides, sort_factors)
      
    
    def simulate(self, timesteps, solver="CRAM16", num_of_cores=None):
        # Calculate the radioactive properties of the packages at the very beginning.
        self.Initialize(timesteps, solver, num_of_cores)

        # Record save simulation initial history.
        if len(self.History) != 0 and len(self.NodePackageHistory) == 0:            
            TempNodePackageHistory = []
            for idx, node in enumerate(self.History):
                # Create dictionary to retrieve node index from NodePackageHistory.
                self.HistoryIdx[node] = idx
                # Add node state to temporary list.
                TempPackageList = []
                for package_state in node.PackageList:
                    time, package = package_state
                    TempPackageList.append([time, self.InvUniquePackageDict[package]])
                        
                # Add node state to temporary list.
                TempNodePackageHistory.append(TempPackageList) #node.PackageList)
            # add node states to history.
            self.NodePackageHistory.append(copy.deepcopy(TempNodePackageHistory))

        # Change the age of the node packages.
        for step in range(timesteps):   
            for order in self.Orders:
                if order.UntilDelivery <= 0:
                    (self.UniquePackageDict, self.InvUniquePackageDict) = order.Transfer(self.stepsize, timesteps, self.UniquePackageDict, self.InvUniquePackageDict, order.Centile)
                    order.UntilDelivery += order.Rate
                else:
                    order.UntilDelivery -= self.stepsize

            for source in self.Sources:
                if source.UntilDelivery <= 0:
                        source.Transfer()
                        source.UntilDelivery += source.Rate
                else:
                    source.UntilDelivery -= self.stepsize
            
            # Update the list of packages and age them.
            for node in self.Nodes:
                # Update the list.
                node.update()
                # Age node package states.
                for sidx in range(len(node.PackageList)):
                    node.PackageList[sidx][0] += self.stepsize
                    
            # Advance simulation time.
            self.SimTime += self.stepsize
            self.SimTimeHistory.append(self.SimTime)
            
            #print(self.SimTime/(60*60*24*365))
            
            # Record save simulation history.
            if len(self.History) != 0:   
                TempNodePackageHistory = []
                for idx, node in enumerate(self.History):
                    TempPackageList = []
                    for package_state in node.PackageList:
                        time, package = package_state
                        TempPackageList.append([time, self.InvUniquePackageDict[package]])
                        
                    # Add node state to temporary list.
                    TempNodePackageHistory.append(TempPackageList) #node.PackageList)
                    
                # add node states to history.
                self.NodePackageHistory.append(copy.deepcopy(TempNodePackageHistory))

                
    def plot(self, node, variable="mass", nuclide=None, filename="", colormap="jet", time=-1, time_units="s", scale=["lin", "lin"], sigma=0.2, plottype="contour", bins=50):
        # Compile a list of all packages and states present, precalculate them.
        AllStates = []
        for state in self.NodePackageHistory:
            package_states = state[self.HistoryIdx[node]]
            for package_state in package_states:
                AllStates.append(package_state)
        
        UniqueStates = [list(y) for y in set([tuple(x) for x in AllStates])]
        StateVariables = dict()
        for unique_state in UniqueStates:
            age, pack_idx = unique_state
            pack = self.UniquePackageDict[pack_idx]
    
            if isinstance(pack, CombinedPackage):
                if variable == "mass":
                    VarState = pack.get_mass()
                    
                elif variable == "heat":
                    if nuclide == None:
                        VarState = pack.get_heat(age).sum(axis=0)
                    elif nuclide != None:
                        VarState = pack.get_heat(age)[pack.NuclideDict[nuclide]]
    
                elif variable == "inventory":
                    if nuclide == None:
                        VarState = pack.get_inventory(age).sum(axis=0)
                    elif nuclide != None:
                        VarState = pack.get_inventory(age)[pack.NuclideDict[nuclide]]

                elif variable == "activity":
                    if nuclide == None:
                        VarState = pack.get_activity(age).sum(axis=0)
                    elif nuclide != None:
                        VarState = pack.get_activity(age)[pack.NuclideDict[nuclide]]
            else:
                if variable == "mass":
                    VarState = pack.Mass
                    
                elif variable == "heat":
                    if nuclide == None:
                        VarState = pack.HeatStates[age].sum(axis=0)
                    elif nuclide != None:
                        VarState = pack.HeatStates[age][pack.NuclideDict[nuclide]]
    
                elif variable == "inventory":
                    if nuclide == None:
                        VarState = pack.InventoryStates[age].sum(axis=0)
                    elif nuclide != None:
                        VarState = pack.InventoryStates[age][pack.NuclideDict[nuclide]]

                elif variable == "activity":
                    if nuclide == None:
                        VarState = pack.ActivityStates[age].sum(axis=0)
                    elif nuclide != None:
                        VarState = pack.ActivityStates[age][pack.NuclideDict[nuclide]]

            StateVariables[tuple(unique_state)] = VarState


        VariableHistory = []
        PackageDistributions = []
        for state in self.NodePackageHistory:
            package_states = state[self.HistoryIdx[node]]
            VarState = np.zeros(self.batches)
            
            # Count the occurrance of different packages in package states
            UniqueStates, UniqueCounts = np.unique(package_states, return_counts=True, axis=0)
            UniqueStates = UniqueStates.tolist()
            
            if plottype == "hist" or plottype=="mesh-hist":
                PackageDistributions.append([UniqueStates, UniqueCounts])
            
            for pidx, unique_state in enumerate(UniqueStates):
                age, pack_idx = unique_state
                pack = self.UniquePackageDict[pack_idx]
                
                # Maybe add this section as multiprocessing as well?
                # Divide the packages into num_of_cores.
                # Give each core a representative number of packages.

                VarState += StateVariables[tuple(unique_state)] * UniqueCounts[pidx]
                    

            VariableHistory.append(VarState)
        data = np.array(VariableHistory)
        
        if plottype == "mesh":
            
            x = []
            y = np.array([])
    
            for tstep, values in enumerate(data):
                x += [tstep]*len(values)
                y = np.concatenate((y, values))
            x = np.array(x)
            
            fig, ax = plt.subplots()
            formatter = mpl.ticker.ScalarFormatter(useMathText=True)
            ax.yaxis.set_major_formatter(formatter)
    
            num_fine = tstep*10
            x_values = np.array(list(range(np.shape(data)[0])))
            x_fine = np.linspace(x_values.min(), x_values.max(), num_fine)
            y_fine = np.concatenate([np.interp(x_fine, x_values, y_row) for y_row in data.T])
            x_fine = np.broadcast_to(x_fine, (np.shape(data)[1], num_fine)).ravel()
            
            colors = plt.cm.jet(np.linspace(0, 1, 256))
            colors[0, :] = [1, 1, 1, 1]  # Set the first color to white
            cmap = plt.matplotlib.colors.ListedColormap(colors)
    
            h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[tstep*10, tstep])
            
            x_edges = xedges[:-1] - (xedges[-1] - xedges[-2])/2
            y_edges = yedges[:-1] - (yedges[-1] - yedges[-2])/2
            
            Z = ndimage.gaussian_filter(h.T, sigma=sigma, order=0) + 0.1
            
            num_levels = 50  # Specify the number of levels you want
            log_min = h.T.min()
            log_max = np.log10(h.T.max())
            log_levels = np.logspace(log_min, log_max, num_levels)
    
            norm_log = mpl.colors.LogNorm(vmin=Z.min(), vmax=Z.max())
            
            # Specify the number of color bar ticks
            num_colorbar_ticks = 5
            
            # Calculate positions for the color bar ticks
            colorbar_ticks = np.geomspace(log_levels.min(), log_levels.max(), num_colorbar_ticks)
            
            # Convert log_levels back to original scale for color bar labels
            colorbar_ticklabels_actual = 10**np.linspace(log_min, log_max, num_colorbar_ticks) / 1
            
            # Create the contour plot with logarithmic color scale
            
            N  = tstep*10 // (len(self.SimTimeHistory) - 1)
            Nn = tstep*10 % (len(self.SimTimeHistory) - 1)
            
            division_list = [N]*len(self.SimTimeHistory)
            for idx, div in enumerate(division_list):
                if Nn - 1 >= 0:
                    division_list[len(division_list) - idx - 1] += 1
                    Nn -= 1
    
            t_edges = np.array([])
            for idx, time in enumerate(self.SimTimeHistory):
                if idx == 0:
                    continue
                temp_edges = np.linspace(self.SimTimeHistory[idx-1], self.SimTimeHistory[idx], division_list[idx])
                t_edges = np.concatenate((t_edges, temp_edges))
                
            if time_units in ["year", "yr", "y", "a"]:   
               t_edges = t_edges / (60 * 60 * 24 * 365)
            if time_units in ["month", "m"]:   
               t_edges = t_edges / (60 * 60 * 24 * 365 / 12)
            elif time_units in ["d", "day"]:
                t_edges = t_edges / (60 * 60 * 24)
            elif time_units in ["h", "hour"]:
                t_edges = t_edges / (60 * 60)
            
            cont = plt.contourf(t_edges, y_edges, Z, levels=log_levels, cmap=cmap, norm=norm_log)
            
            # Add color bar
            cbar = plt.colorbar(cont, ax=ax, label="Number of Data Points", pad=0, ticks=colorbar_ticks)
            cbar.set_label('Number of Data Points', rotation=270, labelpad=15)
            cbar.set_ticklabels([f'{val:.0f}' for val in np.ceil(colorbar_ticklabels_actual)])  # Format labels
            
            if scale[1] == "lin":
                plt.ylim(np.min(y_edges)*0.99, np.max(y_edges)*1.03)
                plt.yscale('linear')
                
            if scale[1] == "log":
                plt.ylim(np.min(y_edges)*0.91, np.max(y_edges)*1.1)
                plt.yscale('symlog')
                
            if scale[0] == "log":
                plt.xlim(np.min(t_edges) + 0.01, np.max(t_edges))
                plt.xscale('symlog')
            
            #fig.colorbar(pcm, ax=ax, label="# points", pad=0)
            ax.set_xlabel(f"Simulation time ({time_units})")
            if variable=="heat":
                ax.set_ylabel("Heat (W)")
            elif variable=="activity":
                ax.set_ylabel("Activity (Bq)")
            else:
                ax.set_ylabel(f"{variable.capitalize()}")
            
        elif plottype == "hist":            
            if time == -1:  
                time_idx = time
            else:
                if time_units in ["year", "yr", "y", "a"]:   
                    time = time * (60 * 60 * 24 * 365)
                if time_units in ["month", "m"]:
                    time = time * (60 * 60 * 24 * 365 / 12)
                elif time_units in ["d", "day"]:
                    time = time * (60 * 60 * 24)
                elif time_units in ["h", "hour"]:
                    time = time * (60 * 60)
                # Find the closest timestep time_idx.
                closest_value = min(self.SimTimeHistory, key=lambda x: abs(x - time))
                time_idx = self.SimTimeHistory.index(closest_value)

            PackageHistList = []
            package_states, package_count = PackageDistributions[time_idx]
            if len(package_states) > 0:
                for pidx, package_state in enumerate(package_states):
                    tuple_state = tuple(package_state)
                    PackageHistList += [StateVariables[tuple_state]] * package_count[pidx]
            
            fig, ax = plt.subplots()
            formatter = mpl.ticker.ScalarFormatter(useMathText=True)
            ax.yaxis.set_major_formatter(formatter)
            
            PackageHistList = np.array(PackageHistList).flatten()
            plt.hist(PackageHistList, bins=bins, weights=[1/self.batches]*len(PackageHistList))
            
            #fig.colorbar(pcm, ax=ax, label="# points", pad=0)
            ax.set_ylabel("Number of Packages (arb.)")
            if variable=="heat":
                ax.set_xlabel(f"Heat (W)")
            elif variable=="activity":
                ax.set_xlabel(f"Activity (Bq)")
            else:
                ax.set_xlabel(f"{variable.capitalize()}")
            
        elif plottype == "mesh-hist":
            data = []
            
            y_bounds = [None, None]
            for (package_states, package_count) in PackageDistributions:
                PackageHistList = []
                if len(package_states) == 0:
                    data.append(PackageHistList)
                else:
                    for pidx, package_state in enumerate(package_states):
                        tuple_state = tuple(package_state)
                        PackageHistList += [StateVariables[tuple_state]] * package_count[pidx]
                            
                    PackageHistList = np.array(PackageHistList).flatten()
                    data.append(np.array(PackageHistList).flatten())
                    if y_bounds[0] == None or np.min(PackageHistList) < y_bounds[0]:
                        y_bounds[0] = np.min(PackageHistList)
                    if y_bounds[1] == None or np.max(PackageHistList) > y_bounds[1]:
                        y_bounds[1] = np.max(PackageHistList)

            if scale[1] == "lin":
                Y = np.linspace(y_bounds[0], y_bounds[1], bins)
            elif scale[1] == "log":
                Y = np.geomspace(y_bounds[0], y_bounds[1], bins)
                
            X = np.linspace(0, len(data)-1, len(data))
            h = np.zeros((len(Y), len(X)))
            h = h + 0.95
            
            for tstep, values in enumerate(data):
                # Find the closest timestep time_idx.
                for value in values:
                    closest_value = min(Y, key=lambda x: abs(x-value))
                    y_idx = np.where(Y == closest_value)
                    h[y_idx, tstep] += 1 / self.batches
            
            fig, ax = plt.subplots()
            formatter = mpl.ticker.ScalarFormatter(useMathText=True)
            ax.yaxis.set_major_formatter(formatter)
            
            colors = plt.cm.jet(np.linspace(0, 1, 750))
            colors[0, :] = [1, 1, 1, 1]  # Set the first color to white
            cmap = plt.matplotlib.colors.ListedColormap(colors)
            
            num_levels = 10  # Specify the number of levels you want
            norm_log = mpl.colors.LogNorm(vmin=h.min(), vmax=h.max())
            
            # Specify the number of color bar ticks
            num_colorbar_ticks = 5
            
            # Create the pmeshcolor plot with logarithmic color scale
            N  = len(X)*1 // (len(self.SimTimeHistory) - 1)
            Nn = len(X)*1 % (len(self.SimTimeHistory) - 1)
            
            division_list = [N]*len(self.SimTimeHistory)
            for idx, div in enumerate(division_list):
                if Nn - 1 >= 0:
                    division_list[len(division_list) - idx - 1] += 1
                    Nn -= 1
    
            t_edges = np.array([])
            for idx, time in enumerate(self.SimTimeHistory):
                if idx == 0:
                    continue
                temp_edges = np.linspace(self.SimTimeHistory[idx-1], self.SimTimeHistory[idx], division_list[idx])
                t_edges = np.concatenate((t_edges, temp_edges))
                
            if time_units in ["year", "yr", "y", "a"]:   
               t_edges = t_edges / (60 * 60 * 24 * 365)
            if time_units in ["month", "m"]:   
               t_edges = t_edges / (60 * 60 * 24 * 365 / 12)
            elif time_units in ["d", "day"]:
                t_edges = t_edges / (60 * 60 * 24)
            elif time_units in ["h", "hour"]:
                t_edges = t_edges / (60 * 60)
                
            pcm = ax.pcolormesh(t_edges, Y, h, cmap=cmap, norm=norm_log,
                                rasterized=True, shading='nearest')
            
            cbar = plt.colorbar(pcm, ax=ax, label="Number of Packages", pad=0)
            cbar.set_label('Number of Packages', rotation=270, labelpad=15)
            
            if scale[1] == "lin":
                plt.ylim(np.min(Y)*0.91, np.max(Y)*1.1)
                plt.yscale('linear')
                
            if scale[1] == "log":
                plt.ylim(np.min(Y)*0.91, np.max(Y)*1.1)
                plt.yscale('log')
                
            if scale[0] == "log":
                plt.xlim(np.min(t_edges) + 0.01, np.max(t_edges))
                plt.xscale('symlog')

            ax.set_xlabel(f"Simulation time ({time_units})")
            if variable=="heat":
                ax.set_ylabel(f"Heat (W)")
            elif variable=="activity":
                ax.set_ylabel(f"Activity (Bq)")
            else:
                ax.set_ylabel(f"{variable.capitalize()}")
                
        else:
            fig, ax = plt.subplots()
            formatter = mpl.ticker.ScalarFormatter(useMathText=True)
            ax.yaxis.set_major_formatter(formatter)
            
            t_edges = np.array(self.SimTimeHistory)
            
            if time_units in ["year", "yr", "y", "a"]:   
                t_edges = t_edges / (60 * 60 * 24 * 365)
            if time_units in ["month", "m"]:
                t_edges = t_edges / (60 * 60 * 24 * 365 / 12)
            elif time_units in ["d", "day"]:
                t_edges = t_edges / (60 * 60 * 24)
            elif time_units in ["h", "hour"]:
                t_edges = t_edges / (60 * 60)
                
            if scale[1] == "lin":
                plt.ylim(np.min(data)*0.99, np.max(data)*1.03)
                plt.yscale('linear')
                
            if scale[1] == "log":
                plt.ylim(np.min(data)*0.91, np.max(data)*1.1)
                plt.yscale('symlog')
                
            if scale[0] == "log":
                plt.xlim(np.min(t_edges) + 0.01, np.max(t_edges))
                plt.xscale('symlog')
            
            colors = plt.get_cmap(colormap)(np.linspace(0, 1, np.shape(data)[1]))
            
            for i in range(np.shape(data)[1]):
                plt.plot(t_edges, data[:,i], color=colors[i])
            
            #fig.colorbar(pcm, ax=ax, label="# points", pad=0)
            ax.set_xlabel(f"Simulation time ({time_units})")
            if variable=="heat":
                ax.set_ylabel(f"Heat (W)")
            elif variable=="activity":
                ax.set_ylabel(f"Activity (Bq)")
            else:
                ax.set_ylabel(f"{variable.capitalize()}")

            # Add gridline.
            plt.grid(color='k', alpha=0.25, linestyle='-', linewidth=1)
            
        plt.grid(True, alpha=0.2)

        if scale[1] == "lin":
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        
        if filename != "":
            plt.savefig(f'{filename}.png', dpi=200)
        else:
            plt.show()
            
        if plottype=="hist":
            return PackageHistList
        elif plottype=="mesh-hist":
            return t_edges, Y, h
        else: 
            return data


def parallel_calculate_states(arguments):
    package, (start_idx, end_idx), stepsize, timesteps, solver = arguments
    # Inputs need to be initial state.
    # Calculate the Bateman Matrix.
    # Calculate new states.
    package.Calculate_States(stepsize=stepsize, timesteps=timesteps, solver=solver)
    
    return package.InventoryStates, package.ActivityStates, package.HeatStates, package.HeatSumStates, package.EnergyVector


def parallel_simulate(package, stepsize, timesteps, solver, num_of_cores):
    # Make a list of modified packages that feeds into the parallel_calculate_states function.
    if num_of_cores == None:
        process_count = min(multiprocessing.cpu_count(), package.batches)    
    else:
        process_count = min(min(multiprocessing.cpu_count(), num_of_cores), package.batches)

    individual_batches = int(package.batches / process_count)
    
    while individual_batches < 1:
        process_count = int(process_count-1)
        individual_batches = int(package.batches / process_count)

    residual_batches = package.batches - process_count * individual_batches

    list_of_arguments = []
    for i in range(process_count):
        if i == 0:
            start_idx = i * individual_batches 
            end_idx = start_idx + individual_batches + residual_batches
        else:
            start_idx = i * individual_batches + residual_batches
            end_idx = start_idx + individual_batches

        TempPackage = copy.deepcopy(package)
        
        # Division into proper-sized chunks.
        if len(TempPackage.InventoryStates) != 0:
            for time in TempPackage.InventoryStates.keys():
                TempPackage.InventoryStates[time] = TempPackage.InventoryStates[time][:,start_idx:end_idx]
                TempPackage.ActivityStates[time]  = TempPackage.ActivityStates[time][:,start_idx:end_idx]
                TempPackage.HeatStates[time]      = TempPackage.HeatStates[time][:,start_idx:end_idx]
                TempPackage.HeatSumStates[time]   = TempPackage.HeatSumStates[time][start_idx:end_idx]
                
        # Prepare the package Inventories.
        TempInvValues = np.array(list(TempPackage.Inventory.values()))[:,start_idx:end_idx]
        TempInvKeys = TempPackage.Inventory.keys()
    
        TempInv = dict(zip(TempInvKeys, TempInvValues))
        TempPackage.Inventory = TempInv    
        
        list_of_arguments.append([TempPackage, (start_idx, end_idx), stepsize, timesteps, solver])

    # Create a pool of worker processes
    if process_count == 1:
        results = []
        for arg in list_of_arguments:
            results.append(parallel_calculate_states(arg))
        
        TimeStates = results[0][0].keys()        
        IStates = np.array(list(results[0][0].values()))
        AStates = np.array(list(results[0][1].values()))
        HStates = np.array(list(results[0][2].values()))
        HSStates = np.array(list(results[0][3].values()))
        EnergyVector = results[0][4]
        
        for idx, (IS, AS, HS, HSS, EV) in enumerate(results):
            if idx == 0:
                continue
            
            IStates = np.concatenate((IStates, np.array(list(IS.values()))), axis=2)
            AStates = np.concatenate((AStates, np.array(list(AS.values()))), axis=2)
            HStates = np.concatenate((HStates, np.array(list(HS.values()))), axis=2)
            HSStates = np.concatenate((HSStates, np.array(list(HSS.values()))), axis=1)

    else:
        with multiprocessing.Pool(processes=process_count) as pool:
            # Distribute the work across the worker processes
            results = pool.map(parallel_calculate_states, list_of_arguments)
            
            TimeStates = results[0][0].keys()        
            IStates = np.array(list(results[0][0].values()))
            AStates = np.array(list(results[0][1].values()))
            HStates = np.array(list(results[0][2].values()))
            HSStates = np.array(list(results[0][3].values()))
            EnergyVector = results[0][4]
            
            for idx, (IS, AS, HS, HSS, EV) in enumerate(results):
                if idx == 0:
                    continue
                
                IStates = np.concatenate((IStates, np.array(list(IS.values()))), axis=2)
                AStates = np.concatenate((AStates, np.array(list(AS.values()))), axis=2)
                HStates = np.concatenate((HStates, np.array(list(HS.values()))), axis=2)
                HSStates = np.concatenate((HSStates, np.array(list(HSS.values()))), axis=1)
            
    # Create Proper dictionaries to finalize the state formats.
    InventoryStates = {}; ActivityStates = {}; HeatStates = {}; HeatSumStates = {}
    for idx, time in enumerate(TimeStates):
        InventoryStates[time] = IStates[idx]
        ActivityStates[time]  = AStates[idx]
        HeatStates[time]      = HStates[idx]  
        HeatSumStates[time]   = HSStates[idx]  
        
    package.InventoryStates = InventoryStates
    package.ActivityStates = ActivityStates
    package.HeatStates = HeatStates
    package.HeatSumStates = HeatSumStates
    package.EnergyVector = EnergyVector
    
    return package

def a():

    import json
    import scipy.stats as sc

    ### Define Spent Fuel Inventory ###
    ###################################
    with open("STEP3_Fuel_Burnup_50.json", "r") as json_file: # Generate new file according to the benchmark.
        TempInv = json.load(json_file)
    
    Spent_Fuel_Inventory = {}
    for nuc, value in TempInv.items():
        if value > 1e18:
            Spent_Fuel_Inventory[nuc] = value
    
    #for nuc, value in Spent_Fuel_Inventory.items():
    #    error = 0.3 # 10%
    #    Spent_Fuel_Inventory[nuc] = sc.norm(loc=value, scale=value*error)
    
    batches = 10
    
    spent_assembly = Package(Mass=1, Inventory=Spent_Fuel_Inventory, batches=batches, decay_chain=True)
    
    ### UO2 Fuel Connections ###
    ############################
    # Define burnup and calculate the amount of fuel generated each year. Use a refueling cycle of 12 months.
    Burnup           = 50 # MWd/kg
    ReactorLifetime  = 60 # years
    CapacityFactor   = 0.90 # US level # https://www.iges.or.jp/en/publication_documents/pub/issue/en/12008/20220406+IGES+Nuclear+Report.pdf
    ReactorPower     = 900 # MW # 
    ReactorElectric  = 300  # MW # Small-scale reactors max limit, which is suitable for regular countries.
    FuelMass         = 170.9 #kgU # Options of Principles of Fuel Debris Criticality Control in Fukushima Daiichi Reactors
    CycleLength      = 1 # one to two years.
    
    OperationTime       = ReactorLifetime * 365 * CapacityFactor # days
    TotalEnergyProduced = ReactorPower * OperationTime # MW * days
    SpentFuelProduced   = TotalEnergyProduced / Burnup # kg
    AssembliesProduced  = int(np.ceil(SpentFuelProduced / FuelMass))
    SingleSwap          = int(AssembliesProduced / (ReactorLifetime / CycleLength))
    
    PoolCapacity   = 8 * SingleSwap # assemblies 
    CoreAssemblies = 240
    
    ### Dry storage requirements ###
    ################################
    TotalCaskLimit    = 50000 - 10000 # 5000 W and 1000 W safety buffer
    TotalCaskCapacity = 89 # spent fuel assemblies
    
    DryAssemblyLimit = TotalCaskLimit / TotalCaskCapacity
    
    ### Packaging facility capacity ###
    ###################################
    CanisterCapacity = 12
    CanistersOut = 20 # 4000 packages over 30 years, is it sensible for this rate for smaller plants? Optimization question. https://curie.pnnl.gov/system/files/SKB_TR-01-03.pdf # You should research/optimize this value! Currently, I dont have a good guess for this value.
    PackagingHeatLimit = 1500 # 1.5 kW per canister
    
    ### Geological disposal capacity ###
    ####################################
    GeologicalRate = 100 # https://curie.pnnl.gov/system/files/SKB_TR-01-03.pdf # Should probably be optimized for smaller plants?
    
    ### Simulation SNF Setup ###
    ############################
    OS_Pool            = Node()
    OF_Dry             = Node()
    Packaging          = Node()
    Geological_Storage = Node()
    
    Reactor = Source(AwayNode=OS_Pool, Package=spent_assembly, Magnitude=SingleSwap) #SingleSwap
    
    PoolDry_Mass_Criteria = {"region":"node", "variable":"mass", "principle":"max", "criteria":PoolCapacity}
    PoolDry_Heat_Criteria = {"region":"package", "variable":"heat", "principle":"min", "criteria":DryAssemblyLimit}
    
    Link_Pool_Dry     = Order(HomeNode=OF_Dry, OrderNodes=[OS_Pool], Magnitude=TotalCaskCapacity,
                              Criteria=[PoolDry_Heat_Criteria])
    
    Packaging_Instruct = {"package_out":1} # CanisterCapacity homogenized into 1 package
    Packaging_Criteria = {"region":"package", "variable":"heat", "principle":"min", "criteria":PackagingHeatLimit} # 1.5 kW limit per package
    
    Links_Dry_Pack = []
    for i in range(CanistersOut):
        Links_Dry_Pack.append(Order(HomeNode=Packaging, OrderNodes=[OF_Dry], Magnitude=CanisterCapacity,
                              Crumbs=False, Mode="Combine", Instruct=Packaging_Instruct, Criteria=[Packaging_Criteria]))
    
    Link_Dry_Pack = Order(HomeNode=Packaging, OrderNodes=[OF_Dry], Magnitude=int(CanisterCapacity*CanistersOut), Crumbs=False, Mode="Combine", Instruct=Packaging_Instruct, Criteria=[Packaging_Criteria])
    Link_Pack_Geo = Order(HomeNode=Geological_Storage, OrderNodes=[Packaging], Magnitude=GeologicalRate, Crumbs=True)
    
    timestep_unit = 1*60*60*24*365
    SNF_verse = (Universe(stepsize=timestep_unit)
                 + OS_Pool + OF_Dry + Packaging + Geological_Storage
                 + Reactor + Link_Pool_Dry + Link_Pack_Geo)
    
    #SNF_verse += Link_Dry_Pack
    for link in Links_Dry_Pack:
        SNF_verse += link
        
    ### Run the simulation ###
    ##########################
    
    SNF_verse.simulate(timesteps=60, num_of_cores=1)
    
    SNF_verse -= Reactor
    
    SNF_verse -= Link_Pool_Dry
    Link_Pool_Dry = Order(HomeNode=OF_Dry, OrderNodes=[OS_Pool], Magnitude=TotalCaskCapacity,
                          Criteria=[PoolDry_Heat_Criteria], Crumbs=True)
    SNF_verse += Link_Pool_Dry
    
    SNF_verse.simulate(timesteps=140, num_of_cores=1)
    
    ### Visualize the SNF results ###
    #################################
    plot_type = "line"
    SNF_verse.plot(OS_Pool, variable="mass", time_units="yr", plottype=plot_type);
    SNF_verse.plot(OF_Dry, variable="mass", time_units="yr", plottype=plot_type);
    SNF_verse.plot(Packaging, variable="mass", time_units="yr", plottype=plot_type);
    SNF_verse.plot(Geological_Storage, variable="mass", time_units="yr", plottype=plot_type);
    SNF_verse.plot(Packaging, variable="heat", time_units="yr", plottype="hist", time=53);
    #SNF_verse.plot(Geological_Storage, variable="heat", time_units="yr", plottype="hist-contour");
    #SNF_verse.plot(Geological_Storage, variable="heat", time_units="yr", plottype="contour");

def b():
    from math import ceil
    import json
    import scipy.stats as sc
    
    ### Define Spent Fuel Inventory ###
    ###################################
    with open("GE-14_inventory_50.json", "r") as json_file:
        TempInv = json.load(json_file)
    
    Spent_Fuel_Inventory = {}
    for nuc, value in TempInv.items():
        if value > 1e18:
            Spent_Fuel_Inventory[nuc] = value
    
    for nuc, value in Spent_Fuel_Inventory.items():
        error = 0.3 # 10%
        Spent_Fuel_Inventory[nuc] = sc.norm(loc=value, scale=value*error)
    
    batches = 100
    
    spent_assembly = Package(Mass=1, Inventory=Spent_Fuel_Inventory, batches=batches, decay_chain=True)
    
    Node1 = Node()
    Source1 = Source(Package=spent_assembly, AwayNode=Node1, Magnitude=1)
    verse = Universe(stepsize=1*60*60*24*365) + Source1 + Node1

    verse.simulate(timesteps=100, Solver="CRAM16")
    verse.plot(Node1, variable="heat", time_units="yr")
    
def c():    
    batches = 500
    
    waste_inventory = {"Co60":1, "Sr90":1, "Cs137":1}
    
    waste_package = Package(Mass=1, Inventory=waste_inventory, batches=batches, decay_chain=False)
    
    Node1 = Node([waste_package], multiplication_factor=1)
    Node2 = Node()
    Node3 = Node()
    
    ins = {"sort_nuclides":["Sr90", "Co60"], "sort_factors":[1, 1], "sort_node":Node3}
    Order1 = Order(HomeNode=Node2, OrderNodes=[Node1], Magnitude=1, Mode="Sort", Instruct=ins)
    verse = Universe(stepsize=1*60*60*24*365) + Node1 + Node2 + Node3 + Order1
    
    verse.simulate(timesteps=100, Solver="CRAM16", num_of_cores=1)
    verse.plot(Node2, variable="inventory", time_units="yr", plottype="mesh")

def d():
    batches = 1
    
    waste_inventory = {"Co60":1}
    
    waste_package = Package(Mass=1, Inventory=waste_inventory, batches=batches, decay_chain=False)
    
    Node1 = Node([waste_package], multiplication_factor=100)
    Node2 = Node()

    ins = {"package_out":1}
    Order1 = Order(HomeNode=Node2, OrderNodes=[Node1], Magnitude=10, Mode="Combine", Instruct=ins)
    verse = Universe(stepsize=1*60*60*24*365) + Node1 + Node2 + Order1
    
    verse.simulate(timesteps=100, Solver="CRAM16", num_of_cores=1)
    verse.plot(Node2, variable="mass", time_units="yr", plottype="line")
    
def e():
    ### ILW waste source characerization ###
    ########################################
    
    # Isotopic concentration Ci per m3
    ResinActivities = {"H3"  : 1.92E-02, "C14" : 1.92E-02, "Fe55": 9.48E-01, "Ni59": 9.80E-04, "Co60": 1.59E+00, "Ni63": 2.15E-02,
                       "Nb94": 3.09E-05, "Sr90": 3.64E-03, "Tc99": 7.65E-05, "I129": 2.04E-04, "Cs135":7.65E-05, "Cs137":2.04E+00,
                       "U235": 5.33E-08, "U238": 4.20E-07, "Np237":1.02E-11, "Pu238":1.02E-11, "Pu239":5.34E-05, "Pu241":2.60E-03,
                       "Pu242":1.17E-07, "Am241":2.32E-05, "Am243":1.57E-06, "Cm243":2.70E-08, "Cm244":1.82E-05}
    
    # From Curies to Bq
    for nuc, value in ResinActivities.items():
        ResinActivities[nuc] = value * 3.7 * 10**10
        
    evaporator_sludge = Package(Mass=1, Inventory=ResinActivities, mode="activity", batches=1, decay_chain=True)
    
def f():
    import json
    import scipy.stats as sc

    ### Define Spent Fuel Inventory ###
    ###################################
    with open("GE-14_inventory_50.json", "r") as json_file: # Generate new file according to the benchmark.
        TempInv = json.load(json_file)
    
    Spent_Fuel_Inventory = {}
    for nuc, value in TempInv.items():
        if value > 1e18:
            Spent_Fuel_Inventory[nuc] = value
    
    batches = 1
    
    spent_assembly = Package(Mass=1, Inventory=Spent_Fuel_Inventory, batches=batches, decay_chain=True)
    
    
    Node1 = Node([spent_assembly], multiplication_factor=1)
    Node2 = Node()

    ins = {"package_out":1}
    Order1 = Order(HomeNode=Node2, OrderNodes=[Node1], Magnitude=1, Mode="Combine", Instruct=ins)
    verse = Universe(stepsize=1*60*60*24*365) + Node1 + Node2
    
    verse.simulate(timesteps=10, Solver="CRAM16", num_of_cores=1)
    verse += Order1
    verse.simulate(timesteps=90, Solver="CRAM16", num_of_cores=1)
    
    verse.plot(Node1, variable="heat", time_units="yr", plottype="line")
    verse.plot(Node2, variable="heat", time_units="yr", plottype="line")
    
def g():
    batches = 1
    
    waste_inventory = {"C14":1}
    
    waste_package = Package(Mass=1, Inventory=waste_inventory, batches=batches, decay_chain=False)
    
    Node1 = Node([waste_package], multiplication_factor=100)
    Node2 = Node()

    ins = {"package_out":1}

    Order1 = Order(HomeNode=Node2, OrderNodes=[Node1], Magnitude=10, Mode="Combine", Instruct=ins)
    Order2 = Order(HomeNode=Node2, OrderNodes=[Node1], Magnitude=10, Mode="Combine", Instruct=ins)

    verse = Universe(stepsize=1*60*60*24*365) + Node1 + Node2 + Order1 + Order2
    
    verse.simulate(timesteps=10, Solver="CRAM16", num_of_cores=1)
    
    verse.plot(Node1, variable="mass", time_units="yr", plottype="line")
    verse.plot(Node2, variable="mass", time_units="yr", plottype="line")
    
def h():
    ### ILW waste source characerization ###
    ########################################
    CycleLength = 1
    ReactorElectric = 460
    batches = 1
    
    ResinPerEnergyYear = 0.081 #m3/MW(e)yr # Data Base for Radioactive Waste Management, 1981 NRC
    ResinProduced = int(np.ceil(ReactorElectric * ResinPerEnergyYear * CycleLength)) 
    
    TankConditioningCapacity = 1000 # https://www-pub.iaea.org/MTCD/Publications/PDF/TE-1701_web.pdf
    ConditioningDisposalCapacity = 1 
    
    # Isotopic concentration Ci per m3
    ResinActivities = {"H3"  : 1.92E-02, "C14" : 1.92E-02, "Fe55": 9.48E-01, "Ni59": 9.80E-04, "Co60": 1.59E+00, "Ni63": 2.15E-02,
                       "Nb94": 3.09E-05, "Sr90": 3.64E-03, "Tc99": 7.65E-05, "I129": 2.04E-04, "Cs135":7.65E-05, "Cs137":2.04E+00,
                       "U235": 5.33E-08, "U238": 4.20E-07, "Np237":1.02E-11, "Pu238":1.02E-11, "Pu239":5.34E-05, "Pu241":2.60E-03,
                       "Pu242":1.17E-07, "Am241":2.32E-05, "Am243":1.57E-06, "Cm243":2.70E-08, "Cm244":1.82E-05}
    
    # From Curies to Bq
    for nuc, value in ResinActivities.items():
        ResinActivities[nuc] = value * 3.7 * 10**10
        
    ### Simulation ILW Setup ###
    ############################
    
    # Add proper inventory definition
    resin_waste = Package(Mass=1, Inventory=ResinActivities, mode="activity", batches=batches, decay_chain=True)
    
    Resin_Tank = Node()
    Resin_Conditioning = Node()
    ILW_Disposal = Node()
    
    # Add proper amount according to the 1982 report.
    Exchanger = Source(AwayNode=Resin_Tank, Package=resin_waste, Magnitude=ResinProduced)
    
    Tank_Instruct = {"package_out":1, "package_mass":1}
    Conditioning_Criteria = {"region":"package", "variable":"activity", "principle":"min",
                             "nuclide":["Cs137", "Co60", "Sr90", "Fe55"], "criteria":[0.1, 0.2, 0.3, 0.4]}
    
    Link_Tank_Cond = Order(HomeNode=Resin_Conditioning, OrderNodes=[Resin_Tank], Instruct=Tank_Instruct,
                           Magnitude=TankConditioningCapacity, Mode="Separate", Crumbs=True, Criteria=[Conditioning_Criteria])
    
    Link_Cond_Disp = Order(HomeNode=ILW_Disposal, OrderNodes=[Resin_Conditioning], Magnitude=ConditioningDisposalCapacity, Crumbs=True)
        
    ILW_verse = (Universe(stepsize=1*60*60*24*365)
             + Exchanger + Resin_Tank + Resin_Conditioning + ILW_Disposal
             + Link_Tank_Cond + Link_Cond_Disp)
    
    ### Run the simulation ###
    ##########################
    ILW_verse.simulate(timesteps=15)
    
    ILW_verse -= Exchanger
    
    ILW_verse.simulate(timesteps=20)

    ### Plot Settings ###
    #####################
    plot_type = "line"
    ILW_verse.plot(Resin_Tank, variable="mass", time_units="yr", plottype=plot_type);
    ILW_verse.plot(Resin_Conditioning, variable="mass", time_units="yr", plottype=plot_type);
    ILW_verse.plot(ILW_Disposal, variable="mass", time_units="yr", plottype=plot_type);
    
def i():
    
    ReactorElectric  = 300
    CycleLength      = 1 # one to two years.
    batches          = 1
    
    ### ILW waste source characerization ###
    ########################################
    LiquidsPerEnergyYear = 0.223 #m3/MW(e)yr # Data Base for Radioactive Waste Management, 1981 NRC
    LiquidsProduced = int(np.ceil(ReactorElectric * LiquidsPerEnergyYear * CycleLength)) 
    
    ### NEED GOOD VALUES FOR THESE or SENSITIVITY ANALYSIS ###
    TankConditioningCapacity = 50 # I THINK IT IS IMPORTANT TO OPTIMIZE THIS VALUE. https://www-pub.iaea.org/MTCD/Publications/PDF/TE-1701_web.pdf
    WasteDemandPerPackage = 0.22/1.4*8 # Data Base for Radioactive Waste Management, 1981 NRC and COST/BENEFIT SYSTEMS ANALYSIS AND COMPARISON OF SHALLOW LAND BURIAL AND GREATER CONFINEMENT DISPOSAL FOR THE FINAL DISPOSITION OF LOW-LEVEL RADIOACTIVE WASTES
    
    # Isotopic concentration Ci per m3
    LiquidActivities = {"H3":6.24E-04, "C14":3.89E-05, "Fe55":7.60E-02, "Ni59":7.85E-05, "Co60":1.27E-01, "Ni63":1.72E-03,
                       "Nb94":2.48E-06, "Sr90":1.18E-04, "Tc99":2.50E-06, "I129": 6.65E-06, "Cs135":2.50E-06, "Cs137":6.65E-02}
    
    # From Curies/m3 to Bq
    for nuc, value in LiquidActivities.items():
        LiquidActivities[nuc] = LiquidsProduced * value * 3.7 * 10**10
        
    ### Simulation ILW Setup ###
    ############################
    
    # Add proper inventory definition
    liquid_waste = Package(Mass=LiquidsProduced, Inventory=LiquidActivities, mode="activity", batches=batches, decay_chain=True)
    
    Liquids_Tank = Node()
    Liquids_Conditioning = Node()
    Liquids_Release = Node()
    
    # Add proper amount according to the 1982 report.
    Exchanger = Source(AwayNode=Liquids_Tank, Package=liquid_waste, Magnitude=1)
    
    Tank_Instruct = {"package_out":TankConditioningCapacity, "package_mass":WasteDemandPerPackage}
    
    Criteria_Nuclides = [["Cs137", "Ba137_m1"], "Co60", ["Sr90", "Y90"], "Fe55"]
    Criteria_Values = np.array([6e-1, 4e-1, 3e-1, 4e1])*1e12/8 # TBq and 8 barrels.
    Conditioning_Criteria = {"region":"package", "variable":"activity", "principle":"min",
                             "nuclide":Criteria_Nuclides, "criteria":Criteria_Values}
    
    Link_Tank_Cond = Order(HomeNode=Liquids_Conditioning, OrderNodes=[Liquids_Tank], Instruct=Tank_Instruct,
                           Magnitude=0, Mode="Separate", Crumbs=False, Criteria=[Conditioning_Criteria])
    
    Release_Nuclides = ["Cs137", "Co60", "Sr90", "Fe55"]
    Release_Values = np.array([1e4, 1e5, 1e4, 1e6]) # https://www.riigiteataja.ee/aktilisa/1270/8202/1006/KKM_m40_lisa3.pdf#
    Release_Criteria = {"region":"package", "variable":"activity", "principle":"min",
                             "nuclide":Release_Nuclides, "criteria":Release_Values}
    
    Link_Cond_Release = Order(HomeNode=Liquids_Release, OrderNodes=[Liquids_Conditioning],
                           Magnitude=10000, Crumbs=True, Criteria=[Release_Criteria])
    
    timestep_unit = 1*60*60*24*365
    ILW_verse = (Universe(stepsize=timestep_unit)
                 + Liquids_Tank + Liquids_Conditioning + Liquids_Release
                 + Exchanger + Link_Tank_Cond + Link_Cond_Release)
    
    ### Run the simulation ###
    ##########################
    ILW_verse.simulate(timesteps=60)
    
    ILW_verse -= Exchanger
    
    ILW_verse.simulate(timesteps=140)
    
    # For release levels
    ILW_verse.simulate(timesteps=500)
    
    ### Plot Settings ###
    #####################
    plot_type = "line"
    
    ### Visualize the ILW results ###
    #################################
    ILW_verse.plot(Liquids_Tank, variable="mass", time_units="yr", plottype=plot_type);
    ILW_verse.plot(Liquids_Conditioning, variable="mass", time_units="yr", plottype=plot_type);
    ILW_verse.plot(Liquids_Release, variable="mass", time_units="yr", plottype=plot_type);
    
    ILW_verse.plot(Liquids_Conditioning, variable="activity", time_units="yr", plottype="mesh");
    ILW_verse.plot(Liquids_Conditioning, variable="activity", time_units="yr", plottype="mesh-hist", scale=["lin", "log"]);


def j():
    batches = 1
    
    waste_inventory = {"Sr90":sc.norm(loc=1e9, scale=0)}
    
    waste_package = Package(Mass=1, Inventory=waste_inventory, mode="activity", batches=batches, decay_chain=True)
    
    Node2 = Node()
    source = Source(Node2, waste_package, Magnitude=1)

    verse = Universe(stepsize=1*60*60*24*365) + source + Node2
    
    verse.simulate(timesteps=60, Solver="CRAM16", num_of_cores=1)
    
    verse.plot(Node2, variable="mass", time_units="yr", plottype="line")
    verse.plot(Node2, variable="activity", time_units="yr", plottype="mesh-hist")
    verse.plot(Node2, variable="activity", time_units="yr", plottype="hist", time=60)
    
def k():
    ### ILW waste source characerization ###
    ########################################
    ReactorElectric = 300
    CycleLength = 1
    batches = 1000
    LiquidsPerEnergyYear = 0.223 #m3/MW(e)yr # Data Base for Radioactive Waste Management, 1981 NRC
    LiquidsProduced = int(np.ceil(ReactorElectric * LiquidsPerEnergyYear * CycleLength)) 
    
    ### NEED GOOD VALUES FOR THESE or SENSITIVITY ANALYSIS ###
    TankConditioningCapacity = 45 # I THINK IT IS IMPORTANT TO OPTIMIZE THIS VALUE. https://www-pub.iaea.org/MTCD/Publications/PDF/TE-1701_web.pdf
    WasteDemandPerPackage = 0.22/1.4*8 # Data Base for Radioactive Waste Management, 1981 NRC and COST/BENEFIT SYSTEMS ANALYSIS AND COMPARISON OF SHALLOW LAND BURIAL AND GREATER CONFINEMENT DISPOSAL FOR THE FINAL DISPOSITION OF LOW-LEVEL RADIOACTIVE WASTES
    
    # Isotopic concentration Ci per m3
    LiquidActivities = {"H3":6.24E-04, "C14":3.89E-05, "Fe55":7.60E-02, "Ni59":7.85E-05, "Co60":1.27E-01, "Ni63":1.72E-03,
                       "Nb94":2.48E-06, "Sr90":1.18E-04, "Tc99":2.50E-06, "I129": 6.65E-06, "Cs135":2.50E-06, "Cs137":6.65E-02}
    
    # From Curies/m3 to Bq
    for nuc, value in LiquidActivities.items():
        act_value = LiquidsProduced * value * 3.7 * 10**10
        LiquidActivities[nuc] = sc.norm(loc=act_value, scale=act_value*0.1)
        #LiquidActivities[nuc] = LiquidsProduced * value * 3.7 * 10**10
    
    liquid_waste = Package(mass=LiquidsProduced, inventory=LiquidActivities, mode="activity", batches=batches)
    
    Liquids_Tank = Node()
    Liquids_Conditioning = Node()
    Liquids_Release = Node()
    
    # Add proper amount according to the 1982 report.
    Exchanger = Source(awaynode=Liquids_Tank, package=liquid_waste, magnitude=1)
    
    Tank_Instruct = {"package_out":TankConditioningCapacity, "package_mass":WasteDemandPerPackage}
    
    Criteria_Nuclides = ["Cs137", "Co60", "Sr90", "Fe55"]
    Criteria_Values = np.array([6e-1, 4e-1, 3e-1, 4e1])*1e12*1 # TBq and 8 barrels.
    Conditioning_Criteria = {"region":"package", "variable":"activity", "principle":"min",
                             "nuclide":Criteria_Nuclides, "criteria":Criteria_Values}
    
    Link_Tank_Cond = Order(homenode=Liquids_Conditioning, ordernodes=[Liquids_Tank], instruct=Tank_Instruct,
                           magnitude=0, mode="Separate", crumbs=False, criteria=[Conditioning_Criteria])
    
    Release_Nuclides = ["Cs137", "Co60", "Sr90", "Fe55"]
    Release_Values = np.array([1e4, 1e5, 1e4, 1e6]) # https://www.riigiteataja.ee/aktilisa/1270/8202/1006/KKM_m40_lisa3.pdf#
    Release_Criteria = {"region":"package", "variable":"activity", "principle":"min",
                             "nuclide":Release_Nuclides, "criteria":Release_Values}
        
    ILW_verse = (Universe(stepsize=1*60*60*24*365)
                 + Liquids_Tank + Liquids_Conditioning + Liquids_Release
                 + Exchanger + Link_Tank_Cond)
    
    for i in range(20):
        ILW_verse += Order(homenode=Liquids_Release, ordernodes=[Liquids_Conditioning],
                               magnitude=10, crumbs=True, criteria=[Release_Criteria])
    
    ### Run the simulation ###
    ##########################
    ILW_verse.simulate(timesteps=60)
    
    ILW_verse -= Exchanger
    
    # For release levels
    ILW_verse.simulate(timesteps=140)
    
    ### Plot Settings ###
    #####################
    plot_type = "line"
    
    ### Visualize the ILW results ###
    #################################
    #ILW_verse.plot(Liquids_Tank, variable="mass", time_units="yr", plottype=plot_type);
    ILW_verse.plot(Liquids_Conditioning, variable="activity", time_units="yr", scale=("lin", "log"), plottype="mesh", filename="mesh_plot");
    #ILW_verse.plot(Liquids_Release, variable="mass", time_units="yr", plottype=plot_type);
    #kk = ILW_verse.plot(Liquids_Conditioning, nuclide="Cs137", variable="activity", time_units="yr", plottype="hist", time=550);

def l():
    import json
    import scipy.stats as sc

    ### Define Spent Fuel Inventory ###
    ###################################
    
    
    with open("STEP3_Fuel_Burnup_50.json", "r") as json_file: # Generate new file according to the benchmark.
        TempInv = json.load(json_file)
    
    Spent_Fuel_Inventory = {}
    for nuc, value in TempInv.items():
        if value > 1e18:
            Spent_Fuel_Inventory[nuc] = value
    
    print(len(Spent_Fuel_Inventory))
    
    batches = 1
    spent_assembly = Package(Mass=1, Inventory=Spent_Fuel_Inventory, batches=batches, decay_chain=True, radioactive=False)
    
    Node1 = Node([spent_assembly], multiplication_factor=1)
    verse = Universe(stepsize=1*60*60*24*365) + Node1
    
    verse.simulate(timesteps=100, Solver="CRAM48")
    verse.stepsize *= 100
    verse.simulate(timesteps=100, Solver="CRAM48")
    
    heat_results = verse.plot(Node1, variable="heat", time_units="yr", plottype="line", scale=("lin", "lin"))
    print(repr(heat_results.flatten()))
    
def m():
    batches         = 1
    LiquidsProduced = 1
    TankConditioningCapacity = 1
    
    # Isotopic concentration Ci per m3
    LiquidActivities = {"Sr90":1e12, "Cs137":1e12}
        
    ### Simulation ILW Setup ###
    ############################
    
    # Add proper inventory definition
    liquid_waste = Package(mass=LiquidsProduced, mode="activity", inventory=LiquidActivities, batches=batches, secular_equilibrium="All", decay_chain=True)
    
    
    Tank = Node([liquid_waste])
    Conditioning = Node()
    
    Evaporator = Source(awaynode=Tank, package=liquid_waste, magnitude=1)
    
    Tank_Instruct = {"package_out":1, "package_mass":1}
    
    Criteria_Nuclides = [["Cs137", "Ba137_m1"]]
    Criteria_Values = np.array([5e11]) # TBq and 8 barrels.
    Conditioning_Criteria = {"region":"package", "variable":"activity", "principle":"min",
                             "nuclide":Criteria_Nuclides, "criteria":Criteria_Values}
    
    Link_1 = Order(homenode=Conditioning, ordernodes=[Tank], instruct=Tank_Instruct,
                   magnitude=0, mode="Separate", crumbs=False, criteria=[Conditioning_Criteria])

    ILW_verse = (Universe()
                 + Tank + Conditioning
                 + Link_1)
    
    ### Run the simulation ###
    ##########################
    ILW_verse.simulate(timesteps=100)
    
    # For release levels
    #ILW_verse.simulate(timesteps=500)
    
    ### Plot Settings ###
    #####################
    plot_type = "line"
    
    ### Visualize the ILW results ###
    #################################
    ILW_verse.plot(Tank, variable="mass", time_units="yr", plottype=plot_type);
    ILW_verse.plot(Conditioning, variable="mass", time_units="yr", plottype=plot_type);
    ILW_verse.plot(Tank, variable="activity", time_units="yr", plottype="line");

if __name__ == "__main__":
    k()