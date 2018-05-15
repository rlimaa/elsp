# author : ricardo camargo  | rcamargo@dep.ufmg.br

from uflp import *
import sys
import os
import argparse
import math
import numpy as np
import json
import pandas
from dotmap import DotMap
import matplotlib.pyplot as plt
import cplex

if __name__ == "__main__":
   args = parse_arguments()
   print('Args = ', args)
   dat = read_data(args)
   cpx = create_model(dat)
   sol = solve_model(dat,cpx)
   #print_sol(dat,sol)
   plot_sol(dat,sol)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='[datafile]', help='datafile name')
    parser.add_argument('ni',metavar='[number of items]',type=int, help='number of items in production')
    parser.add_argument('cs',metavar='[production cycle size]',type=float, help='size in months of the production cycle')
    args = parser.parse_args()
    return args

def read_data(args):
   ft = args.file[-1]
   dat = None
   if (ft == 't'):
      dat = read_dat_file(args) 
   elif (ft == 'x'):
      dat = read_excel_file(args)
   elif (ft == 'n'): 
      dat = read_json_file(args)
   else: 
      raise ValueError('datafile format unknown')
   return dat

def read_dat_file(args):
    flname = args.file
    if (os.path.isfile(flname) == False):
       raise Exception('file {:s} not found'.format(flname))

    fl = open(flname,'r')
    lines = (line.strip() for line in fl)
    lines = (line for line in lines if line)
   
    dat = DotMap()

    dat.items = args.ni
    dat.cycleSize = args.cs

    I = range(dat.items)

    data = []
    for i in I:
        Dj,Qj,hj,cj = map(int,next(lines).split())
        data.append([Dj,Qj,hj,cj])

    fl.close()

    dat.data = data

    #with open('ex.json', 'w') as fl:
    #     fl.write(json.dumps(dat.toDict()))

    dat.ro = [ math.floor( data[i][0]/data[i][1]) for i in I];
    
    dat.ak = [ math.floor( 0.5 * (data[i][3]*dat.ro[i]*\
                                  (data[i][1]-data[i][0]))) for i in I];
    
    return dat

def read_excel_file(args):
    flname = args.file
    if (os.path.isfile(flname) == False):
       raise Exception('file {:s} not found'.format(flname))
    dat = DotMap()

    dat.items = args.ni
    dat.cycleSize = args.cs

    df = pandas.read_excel(flname,sheetname='data', dtype = {'Dj': np.int32, 'Qj': np.int32, 'hj':np.int32, 'cj':np.int32} )
    data = [list(tp)[1:] for tp in df.to_records(index=False)] 

    I = range(dat.items)
    
    dat.ro = [ math.floor( data[i][0]/data[i][1]) for i in I];

    
    dat.ak = [ math.floor( 0.5 * (data[i][3]*dat.ro[i]*\
                                 (data[i][1]-data[i][0]))) for i in I];
    dat.data = data;
     
    return dat

def read_json_file(args):
    flname = args.file
    if (os.path.isfile(flname) == False):
       raise Exception('file {:s} not found'.format(flname))

    with open(flname) as fl:
         js = json.load(fl)
  
    dat = DotMap(js)

    dat.items = args.ni
    dat.cycleSize = args.cs


    data = dat.data
 
    I = range(dat.items)

    dat.ro = [ math.floor( data[i][0]/data[i][1]) for i in I];
    
    dat.ak = [ math.floor( 0.5 * (data[i][3]*dat.ro[i]*\
                                  (data[i][1]-data[i][0]))) for i in I];
    
    return dat ;

def create_model(dat):
    items = dat.items
    cycleSize = dat.cycleSize 
    sf = dat.sf 

    I = range(items)
    J = range(cycleSize)
    IJ = [(i,j) for i in I for j in J]
 
    f = [ sf * dat.fac[j][2] for j in J] 
    d = [ dat.cli[i][2] for i in I] 

    cpx = cplex.Cplex()
    
    ix = {(i,j) : idx + cycleSize for idx,(i,j) in enumerate(IJ)}
    
    nx = ["x(" + str(i) + "," + str(j) + ")" for (i,j) in IJ]
    ny = ["y(" + str(j) + ")" for j in J] 
    

    # cpx.variable.add = usar o mesmo número de vezes quanto o número de variáveis na função objetivo
    cpx.variables.add(obj= [f[j] for j in J],
                      lb = [0.0] * cycleSize,
                      ub = [1.0] * cycleSize,
                      types = ['B'] * cycleSize,
                      names = ny)
    # adiciona a variável y(j)
    # obj = qual é o coeficiente que está multiplicando a variável adicionada na função objetivo
    # lb = limite inferior 0 (variável binária) (vetor com cycleSize posições)
    # ub = limite superior 1 (variável binária)
    # types = binário
    # nome = vetor de caracteres ny

    cpx.variables.add(obj= [d[i] * dat.c[i][j] for (i,j) in IJ],
                      lb = [0.0] * items * cycleSize,
                      ub = [cplex.infiitemsty] * items * cycleSize,
                      names = nx)
    # adiciona a variável x(ij)
    # obj = qual é o coeficiente que está multiplicando a variável adicionada na função objetivo
    # lb = limite inferior 0
    # ub = limite superior infiitemsto (não há limite superior)
    # nome = vetor de caracteres nx

    [cpx.linear_constraints.add(lin_expr=[cplex.SparsePair([ix[(i,j)] for j in J], [1.0 for j in J])],
                                senses = "E", 
                                rhs = [1.0]) for i in I]
    # adiciona a restrição somatário em j de x(ij) = 1
    # lin_expr = SparsePair (a matriz adiciona é esparsa, há vários zeros na matriz, portanto quero adicionar só os valores não nulos)
    # parâmetro 1 (SparsePair) = índice dos valores
    # parâmetro 2 (SparsePair) = valores
    # senses = E (a expressão é uma igualdade)
    # rhs = (right hand side) o lado direito é sempre igual a 1 (??)

    [cpx.linear_constraints.add(lin_expr=[cplex.SparsePair([ix[(i,j)], j], [1.0,-1.0])],
                                senses = "L", 
                                rhs = [0.0]) for (i,j) in IJ]
    # adiciona restrição x(ij) - y(j) <= 0
    # lin_expr = SparsePair (a matriz adiciona é esparsa, há vários zeros na matriz, portanto quero adicionar só os valores não nulos)
    # parâmetro 1 = índice dos valores
    # parâmetro 2 = valores
    # senses = L (a expressão é uma desigualdade menor ou igual)
    # rhs = (right hand side) o lado direito é sempre igual a 0 (??)

    cpx.write("uflp.lp")
    return cpx

def solve_model(dat,cpx):
    cpx.parameters.threads.set(1)
    cpx.solve() 
    status = cpx.solution.get_status()
    statusMsg = cpx.solution.get_status_string() 

    if (status != cpx.solution.status.optimal) and\
       (status != cpx.solution.status.optimal_tolerance) and\
       (status != cpx.solution.status.MIP_optimal) and\
       (status != cpx.solution.status.MIP_time_limit_feasible) and\
       (status != cpx.solution.status.MIP_dettime_limit_feasible) and\
       (status != cpx.solution.status.MIP_abort_feasible) and\
       (status != cpx.solution.status.MIP_feasible_relaxed_sum) and\
       (status != cpx.solution.status.MIP_feasible_relaxed_inf) and\
       (status != cpx.solution.status.MIP_optimal_relaxed_inf) and\
       (status != cpx.solution.status.MIP_feasible_relaxed_quad) and\
       (status != cpx.solution.status.MIP_optimal_relaxed_sum) and\
       (status != cpx.solution.status.MIP_feasible):

       statusMsg = cpx.solution.get_status_string() 
       print(statusMsg)
       sys.exit(-1)
    else:
       cycleSize = dat.cycleSize
       I = range(dat.items)
       J = range(dat.cycleSize) 
       IJ = [(i,j) for i in I for j in J]
       ix = {(i,j) : idx + cycleSize for idx,(i,j) in enumerate(IJ)}

       of = cpx.solution.get_objective_value()
       x = cpx.solution.get_values()

       sol = DotMap()
       sol.msg = cpx.solution.get_status_string() 
       sol.of = of
       sol.y  = [j for j in J if x[j] > 0.001] 
       sol.ny = len(sol.y)
       sol.x  = [(i,j) for (i,j) in IJ if x[ix[(i,j)]] > 0.001] 
    return sol

def print_sol(dat,sol):
    I = range(dat.items)
    print("Solver status     : {:s}".format(sol.msg))
    print("Objective function: {:18,.2f}".format(sol.of))
    print("# facilities      : {:18d}".format(sol.ny))
    print("facilities        : ",end='')
    [print("{:3d}".format(j),end=' ') for j in sol.y]
    print("\nallocation        :")
    for j in sol.y:
       print("{:17d} : ".format(j), end='')
       [print("{:3d}".format(i),end=' ') for i in I if (i,j) in sol.x]
       print()

def plot_sol(dat, sol):
    I = range(dat.items)
    for j in sol.y:
            for i in I:
                if (i,j) in sol.x:
                    xc = np.array([dat.cli[i][0], dat.fac[j][0]])
                    xy = np.array([dat.cli[i][1], dat.fac[j][1]])
                    plt.plot(xc, xy, 'ko-')
    plt.show()
    return