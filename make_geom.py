"""
This script builds a simple geometry
"""

import argparse
from math import log10
from mgga import MGGA

import numpy as np
import openmc

import sys
import json

def add_element(element,material,fraction,fraction_type):
    to_add = openmc.Element(element)
    to_add = to_add.expand(fraction,fraction_type)

    for nuclide in to_add:
        material.add_nuclide(nuclide[0],nuclide[1],percent_type=nuclide[2])

class openmc_problem():
    def __init__(self):
        self.materials = {}
        self.model = None
        self.num_x = 0
        self.num_y = 0
        self.x_bounds = []
        self.y_bounds = []
        self.x_planes = []
        self.y_planes = []
        self.cells = []
        self.tallies = {}

    def setup(self, num_x, x_bounds, num_y, y_bounds):
        self.num_x = num_x
        self.num_y = num_y

        self.x_bounds = np.linspace(x_bounds[0], x_bounds[1], num_x)
        self.y_bounds = np.linspace(y_bounds[0], y_bounds[1], num_y)

        for i,x in enumerate(self.x_bounds):
            plane = openmc.XPlane(x0=x, name = 'xplane_' + str(i))
            self.x_planes.append(plane)

        for i,y in enumerate(self.y_bounds):
            plane = openmc.YPlane(y0=y, name = 'yplane_' + str(i))
            self.y_planes.append(plane)

        self.__build_model()

    def __build_materials(self):
        breeder = openmc.Material(name='Lithium')
        breeder.set_density('g/cm3', 0.534)
        add_element('Li',breeder,100,'ao')
        self.materials[0] = breeder

        multiplier = openmc.Material(name="Lead")
        multiplier.set_density('g/cm3', 12.3)
        add_element('Pb', multiplier, 100, 'ao')
        self.materials[1] = multiplier

        be = openmc.Material(name="Beryllium")
        be.set_density('g/cm3', 1.85)
        add_element('Be', be, 100, 'ao')
        self.materials[2] = be

        lithium6 = openmc.Material(name="lithium-6")
        lithium6.set_density('g/cm3', 0.534)
        lithium6.add_nuclide('Li6',100,'ao')
        self.materials[3] = lithium6

        lithium7 = openmc.Material(name="lithium-7")
        lithium7.set_density('g/cm3', 0.534)
        lithium7.add_nuclide('Li7',100,'ao')
        self.materials[4] = lithium7

        bismuth = openmc.Material(name="Bismuth")
        bismuth.set_density('g/cm3', 9.747)
        add_element('Bi',bismuth,100,'ao')
        self.materials[5] = bismuth

    def __build_tallies(self, flux_cells, tbr_cells):

        neutron_filter = openmc.ParticleFilter('neutron', filter_id=1)
        cell_filter = openmc.CellFilter(flux_cells, filter_id=2)
        tbr_cell_filter = openmc.CellFilter(tbr_cells, filter_id=3)

        tbr_tally = openmc.Tally(tally_id = 1, name="tbr")
        tbr_tally.scores = ['(n,t)']
        tbr_tally.estimator = 'tracklength'
        tbr_tally.filters = [tbr_cell_filter, neutron_filter]

        flux_tally = openmc.Tally(tally_id = 2, name="flux")
        flux_tally.scores = ['flux']
        flux_tally.estimator = 'tracklength'
        flux_tally.filters = [cell_filter, neutron_filter]
        flux_tally.triggers = [openmc.Trigger('rel_err',0.05)]
        flux_tally.triggers[0].scores = ['flux']

        tallies = openmc.Tallies([tbr_tally,flux_tally])
        self.model.tallies = tallies

    # generate the fitness for the current generation and
    # index
    def generate_fitness(self, directory, sp_name = "statepoint.10.h5"):
        sp = openmc.StatePoint(directory + '/' + sp_name)
        tbr = sp.get_tally(name = 'tbr')
        cells = []
        [cells.append(x.id) for x in self.cells]
        tbr_data = tbr.get_slice(scores=['(n,t)'],filters=[openmc.CellFilter], filter_bins = [tuple(cells)])
        tbr_ave = tbr_data.mean

        # maximise tbr
        fitness = sum(tbr_ave)[0][0]
        sp.close()
        return fitness

    def assign_genome(self, genome):
        idx = 0
        for x in range(self.num_x-1):
            for y in range(self.num_y-1):
                # set the material given the position in genome
                mat = self.materials[genome[idx]]
                self.cells[idx].fill = mat
                idx = idx + 1

    # given the genome build the region of geometry
    # to optimise
    def build_geometry(self):

        univ = openmc.Universe(name='optimisation')
        cells = []
        idx = 0
        for x in range(self.num_x-1):
            for y in range(self.num_y-1):
                # set the material given the position in genome
                #mat = self.materials[genome[idx]]
                cell = openmc.Cell(region = +self.x_planes[x] & -self.x_planes[x+1] & +self.y_planes[y] & -self.y_planes[y+1])
                cells.append(cell)
                # increment index
                idx = idx + 1

        univ.add_cells(cells)

        self.cells = cells

        # tally the cells in the last x row
        tally_cells = cells[-self.num_y:]

        self.__build_tallies(tally_cells, cells)

        return univ

    # run specific settings
    def __set_settings(self):
        self.model.settings.run_mode = 'fixed source'
        self.model.settings.batches = 10
        self.model.settings.particles = 100000

        # make the source spatial dist
        x_dist = openmc.stats.Discrete([0.0],[1.0])
        y_dist = openmc.stats.Uniform(a=self.y_planes[0].y0, b=self.y_planes[-1].y0)
        z_dist = openmc.stats.Discrete([0.0],[1.0])
        spatial = openmc.stats.multivariate.CartesianIndependent(x_dist, y_dist, z_dist)
        # make the angular dist
        angle_dist = openmc.stats.Monodirectional(reference_uvw = [1,0,0])
        # make the energy dist
        energy_dist = openmc.stats.Discrete([14.06e6],[1.0])
        source = openmc.Source(space = spatial, angle = angle_dist, energy = energy_dist)
        self.model.settings.source = source

    # main build function
    def __build_model(self):

        self.model = openmc.model.Model()

        self.__build_materials()

        # build the region to optimise
        optimisation = self.build_geometry()

        # Create a cell filled with the lattice
        inside_boundary  = -self.y_planes[-1] & +self.y_planes[0] & -self.x_planes[-1] & +self.x_planes[0]
        outside_boundary = +self.y_planes[-1] | -self.y_planes[0] | +self.x_planes[-1] | -self.x_planes[0]
        main_cell = openmc.Cell(fill=optimisation, region=inside_boundary)
        eou = openmc.Cell(region = outside_boundary)
        # Finally, create geometry by providing a list of cells that fill the root
        # universe
        self.model.geometry = openmc.Geometry([main_cell,eou])

        self.x_planes[0].boundary_type = 'vacuum'
        self.x_planes[-1].boundary_type = 'vacuum'

        self.y_planes[0].boundary_type = 'vacuum'
        self.y_planes[-1].boundary_type = 'vacuum'

        self.__set_settings()

# using slurm array job build description
def build_slurm(generation):
    contents = []
    contents.append('#!/bin/bash')
    contents.append('#')
    contents.append('#SBATCH --job-name=mgga')
    contents.append('#SBATCH -A UKAEA-AP001-CPU')
    contents.append('#SBATCH -p cclake')
    contents.append('#SBATCH --nodes=1')
    contents.append('#SBATCH --ntasks=56')
    contents.append('#SBATCH --time=36:00:00')
    contents.append('#SBATCH --output=array_%A-%a.out')
    contents.append('#SBATCH --array=1-1000')
    contents.append(' ')
    contents.append('module purge')
    contents.append('module load rhel7/default-ccl')
    contents.append('module load openmpi/gcc/9.2/4.0.1')
    contents.append(' ')
    contents.append('cd $WORKDIR')
    contents.append('# IDX should match the folders inside generation')
    contents.append('IDX=$(($SLURM_ARRAY_TASK_ID - 1))')
    contents.append('cd ' + str(generation) + '/$IDX')
    contents.append('export OPENMC_CROSS_SECTIONS=/home/dc-davi4/openmc-data/fendl-3.1d-hdf5/cross_sections.xml')
    contents.append('openmc')
    contents.append('cd ..')

    with open('mgga_openmc.slurm','w') as f:
        f.writelines(s + '\n' for s in contents)

def write_population(population, generation):
    data = {"population": [population]}
    json_string = json.dumps(data)
    jsonfile = open("population_" + str(generation) + ".json",'w')
    jsonfile.write(json_string)
    jsonfile.close()

def read_population(generation):
    data = None

    fileObject = open("population_" + str(generation) + ".json", "r")
    jsonContent = fileObject.read()
    data = json.loads(jsonContent)
    return data["population"][0]

def main():
    # Set up command-line arguments for generating/running the model
    parser = argparse.ArgumentParser()
    #parser.add_argument('--run', action='store_true')
    parser.add_argument('--initialise', help="In this run mode, initialise the first generation population", action='store_true')
    parser.add_argument('--generate', help="In this run mode, initialise the next generation population",)
    parser.add_argument('--input', help="JSON file containing input settings", required=True)
    args = parser.parse_args()

    try:
        with open(args.input) as f:
            data = json.load(f)
            f.close()
    except FileNotFoundError:
        print("Could not find file",args.input)
        sys.exit(1)

    # MGGA class
    mgga = MGGA(data["mgga_settings"])
    mgga.initialise()

    # initialise the first generation
    if args.initialise:
        mgga.fill_population()
        openmc_problem = openmc_problem()
        openmc_problem.setup(10,[0,100],10,[0,200])
        for idx,i in enumerate(mgga.population):
            openmc_problem.assign_genome(i)
            openmc_problem.model.export_to_xml(directory='0/'+str(idx))
        build_slurm(0)
        write_population(mgga.population, 0)

    # need to write a filename dependent population file!!

    # generate the next generation
    if args.generate:
        generation = int(args.generate)
        population = read_population(generation-1)
        mgga.population = population
        genomes = mgga.population
        openmc_problem = openmc_problem()
        openmc_problem.setup(10,[0,100],10,[0,200])
        # loop over each of the genomes
        fitness = []
        for idx,i in enumerate(genomes):
            # folder for the current problem
            directory = str(generation-1) + '/' + str(idx)
            fit = openmc_problem.generate_fitness(directory)
            fitness.append(fit)
        # set the fitness
        mgga.fitness = fitness
        print('max fitness: ' + str(max(fitness)))
        print('min fitness: ' + str(min(fitness)))

        mgga.sample_population()
        for idx,i in enumerate(mgga.children):
            openmc_problem.assign_genome(i)
            openmc_problem.model.export_to_xml(directory=str(generation) + '/'+str(idx))
        write_population(mgga.children,generation)

"""
    # translate to a higher resolution
    if args.translate:
        genomes = mgga.population
"""

if __name__ == '__main__':
    main()
