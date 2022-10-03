#!/usr/bin/python3

import random
import numpy as np
import copy

class MGGA():
    def __init__(self, settings, seed = 1283123901319):

        # Override default seed if it is in settings
        if "seed" in settings.keys():
            seed = settings["seed"]
        self.seed = seed
        random.seed(self.seed)

        # Settings
        self.population_size = settings["population"]
        self.generations = settings["generations"]
        self.num_genes = settings["num_of_states"]
        self.chromosome_length = settings["chromosome_length"]
        self.copy_prob = settings["copy_prob"]
        self.crossover_prob = settings["crossover_prob"]
        self.mutation_prob = settings["mutation_prob"]
        self.percentage_worst = settings["percentage_worst"]
        self.tourament_size = settings["tournament_size"]

        self.population = []
        self.children = []
        self.fitness = []
        self.count = {}
        self.count["mutate"] = 0
        self.count["crossover"] = 0
        self.count["clone"] = 0

    def __summary(self):
        total = self.count["crossover"]
        total = total + self.count["clone"]
        total = total + self.count["mutate"]

        per_cross = 100*self.count["crossover"]/total
        per_clone = 100*self.count["clone"]/total
        per_mutate = 100*self.count["mutate"]/total

        print("crossover happened: %f (%f%%)" % (self.count["crossover"],per_cross))
        print("clone happened: %f (%f%%)" % (self.count["clone"], per_clone))
        print("mutate happened: %f (%f%%)" % (self.count["mutate"], per_mutate))

    # purely random mutation
    def __mutate(self, chromosome):
       selected_gene = random.randint(0,self.chromosome_length-1)
       random_gene = random.randint(0,self.num_genes-1)
       new_chromosome = copy.deepcopy(chromosome)
       new_chromosome[selected_gene] = random_gene
       return new_chromosome

    # blend the two chromosomes
    def __uniform_crossover(self,chromosome1):
       rand = random.randint(0,1)
       # pick the other parent randomly
       chromosome2 = self.population[rand]
       child1 = [0]*self.chromosome_length
       child2 = [0]*self.chromosome_length

       for i in range(self.chromosome_length):
           select_parent = random.randint(0,1)

           if select_parent == 0:
               child1[i] = chromosome1[i]
               child2[i] = chromosome2[i]
           else:
               child1[i] = chromosome2[i]
               child2[i] = chromosome1[i]

       return [child1,child2]

    # copy the chromosome
    def __clone(self,chromosome):
       return copy.deepcopy(chromosome)

    def __random(self):
       chromosome = [0]*self.chromosome_length
       for i in range(self.chromosome_length):
           chromosome[i] = random.randint(0,self.num_genes-1)
       return chromosome

    def __sample(self,chromosome):
       rand = random.random()
       if rand <= self.copy_prob:
           self.count["clone"] = self.count["clone"] + 1
           return self.__clone(chromosome)
       elif rand > self.copy_prob and rand <= self.copy_prob + self.mutation_prob:
           self.count["mutate"] = self.count["mutate"] + 1
           return self.__mutate(chromosome)
       else:
           self.count["crossover"] = self.count["crossover"] + 1
           return self.__uniform_crossover(chromosome)

    def fill_population(self):
       for i in range(self.population_size):
           chromosome = self.__random()
           self.population.append(chromosome)

    def sample_population(self):
        parents = []
        for i in range(self.population_size):
            tornament_competitors = [0] * self.tornament_size
            competitor_fitness = [0] * self.tornament_size
            # pick ts individuals
            for j in range(self.tornament_size):
                rand = random.randint(0,self.chromosome_length-1)
                tornament_competitors[j] = self.population[rand]
                competitor_fitness[j] = self.fitness[rand]
            rand = random.random()
            winner = 0
            if rand > self.percentage_worst:
                winner = tornament_competitors[np.argmax(competitor_fitness)]
                # choose max fitness from the competitors
            else:
                winner = tornament_competitors[np.argmin(competitor_fitness)]
                # choose the least fit
            parents.append(winner)

        # now breed
        for i in range(self.population_size):
            child = self.__sample(parents[i])
            if len(child) < self.chromosome_length:
                self.children.append(child[0])
                self.children.append(child[1])
            else:
                self.children.append(child)

        self.__summary()
