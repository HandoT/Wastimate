# Contact info
For any questions about the software, contact the main developer:

Hando Tohver
hando.tohver@ut.ee 

# Wastimate 0.1.0

Wastimate is a specialized Python-based framework designed for simulating, managing, and analyzing the behavior of radioactive waste over time. It focuses on simulating complex processes such as isotopic decay, waste sorting, conditioning, and transportation within a dynamic system of interconnected nodes. This tool can be used to assess long-term waste management strategies, ensure regulatory compliance, and optimize the handling of radioactive waste in facilities like nuclear power plants or waste repositories.

The code leverages object-oriented design principles, with core classes such as Node, Package, Order, and Universe to model the flow of radioactive materials and simulate the impact of various waste handling operations over time.

NB! Wastimate requires decay chain data to run, place the OpenMC's decay chain data in the same folder as the .py. (Data that is formatted to suit Wastimate can be found here https://www.dropbox.com/scl/fi/zexn5i1fuke8u50thj7c2/decay_chains_endfb71.xml?rlkey=xsukcszkrpagt6skcv8ara47o&st=sbx4yi28&dl=0)

## Key Features
* Models radioactive waste packages containing multiple isotopes. Each package can be initialized with specific inventory data (i.e., the amount of each radionuclide) and mass. The decay chains for each isotope can be modeled, taking into account half-lives and transformations into daughter isotopes.
* Simulates activities such as waste sorting, packaging, conditioning, and transferring between different nodes (storage facilities, treatment plants, etc.). Each process is guided by defined orders that execute specific actions within the system.
* The simulation engine allows users to model radioactive decay over time by setting the timestep, which defines the simulation's granularity (e.g., hours, days, years). Users can observe changes in waste characteristics, such as activity, mass, and isotopic composition.
Includes mechanisms for applying criteria to waste processing, such as minimizing radiation dose or reducing the concentration of specific radionuclides to acceptable levels.
* The framework provides plotting tools to visualize changes in waste properties (mass, activity, and dose) over time, allowing users to better understand the progression of waste treatment and ensure compliance with safety regulations.

## Core Components

### Waste Packages
A waste package represents a discrete unit of radioactive material, with a defined mass and inventory of isotopes. For each package, the decay of radionuclides is modeled to account for changes in activity over time.
#### Attributes
* Mass: The total mass of the waste package.
* Inventory: A dictionary containing isotopes as keys and their respective quantities (e.g., in curies or becquerels) as values.
* decay_chain: A flag that indicates whether or not the decay of parent isotopes into daughter products should be considered.
Waste packages are the core objects being managed and processed throughout the simulation. Each package undergoes various transformations based on the orders assigned to the nodes it resides in.

### Nodes
A node represents a physical location or operational unit within the waste management system. Nodes can store waste packages, and they serve as the primary points where waste processing activities occur (e.g., sorting, conditioning).
#### Attributes
* multiplication_factor: A factor that can be used to multiply the radioactive packagecontents of a node (e.g., to model larger storage volumes or more intense processing rates).
Nodes act as stations in the simulation where waste packages are either stored or processed. They can represent real-world facilities such as waste treatment plants, storage tanks, or disposal repositories.

### Orders
Orders define the actions that take place within or between nodes. Each order specifies the type of operation (mode), the magnitude of the action, and any associated criteria for processing the waste.
#### Order Modes:
* Separate: This mode is used to split waste materials within a node based on specific criteria, such as radionuclide type or activity levels. It is particularly useful for scenarios where waste must be divided into different categories for regulatory or safety reasons.
* Transfer: In this mode, waste packages are moved from one node to another. This can be used to simulate the transportation of waste between facilities, such as from a storage tank to a treatment plant, without altering the composition of the waste.
* Sort: Sorting organizes waste packages or isotopes within a node according to predefined rules. For example, waste may be sorted by activity levels or isotopic composition, allowing for the prioritization of certain packages over others.
* Condition: Conditioning applies specific criteria to waste packages in order to prepare them for storage or disposal. This mode is used to ensure that waste meets safety or environmental standards, such as reducing activity or radiation dose to acceptable levels.
* Package Out: This operation removes waste packages from a node, often in preparation for further processing or disposal. For example, after sorting or conditioning, a package may be transferred out of a node for final storage.
#### Attributes
* instruct: Instructions that guide how the order is carried out (e.g., which nuclides to separate, or how much of the waste to transfer).
* criteria: Defines the standards that must be met for the order to be completed, such as maintaining radiation doses below a specific threshold or ensuring the activity of certain nuclides is minimized.
* rate: The speed or frequency at which the order is executed.

### Universe
The universe represents the overall simulation environment, where nodes and orders interact. It orchestrates the flow of time, tracks changes in waste properties, and facilitates the execution of orders.
#### Attributes
* stepsize: Defines the simulation's timestep, which controls how much time elapses between each update (e.g., daily, yearly).
The universe serves as the simulation controller, advancing the system through time and applying changes to waste packages based on the orders placed at each node.

## How It Works:

### Initial Setup
Waste packages are defined with initial inventories of isotopes (e.g., Co-60, Cs-137) and assigned to nodes. Nodes may be empty initially, or they may contain waste packages ready for processing. Orders are then assigned to nodes to dictate how the waste will be handled.

### Processing Orders
Orders are the primary mechanism for waste processing. For example, an order might instruct a node to separate waste based on specific nuclide content or activity thresholds. Alternatively, an order could condition the waste, reducing its radioactivity to meet disposal criteria.

### Time-Step Simulation
The simulation is advanced in discrete time steps, each step representing a defined period (e.g., a year). During each step, the universe simulates the isotopic decay, changes in mass and activity, and any processing actions as defined by the orders. The user can control the granularity of the simulation by adjusting the stepsize, allowing for fine-grained analysis (e.g., on a daily or hourly basis) or longer-term studies (e.g., over years or decades).

### Visualization
The system provides tools for plotting the results of the simulation. Users can visualize how the mass, activity, or dose of waste packages changes over time, offering insights into when waste can be safely handled, transported, or disposed of. The framework supports different plot types, such as line graphs and mesh plots, to help users explore the data from different perspectives.
