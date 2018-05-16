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
    parser.add_argument('p',metavar='[production unit cost]',type=int, dhelp='Cost of item in production')
    parser.add_argument('q',metavar='[production setup cost]',type=int, help='Cost of setup to product the item')
    parser.add_argument('hcost',metavar='[inventory holding cost]',type=int, help='Cost of hold the product in the inventory')
    parser.add_argument('NT',metavar='[number of periods]',type=int, help='Number of periods of time')
    parser.add_argument('-sInit',metavar='[inventory initial level]',type=float, default = 0, help='size in months of the production cycle')
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

    dat.p = args.p
    dat.q = args.q
    dat.h = args.h
    dat.nt = args.NT
    dat.sInit = args.sInit

    I = range(dat.nt)

    data = []
    for i in I:
        dj = map(int,next(lines).split())
        data.append([dj])

    fl.close()

    dat.dj = data

    #with open('ex.json', 'w') as fl:
    #     fl.write(json.dumps(dat.toDict()))
    
    return dat

def read_excel_file(args):
    flname = args.file
    if (os.path.isfile(flname) == False):
       raise Exception('file {:s} not found'.format(flname))
    dat = DotMap()

    dat.p = args.p
    dat.q = args.q
    dat.h = args.h
    dat.nt = args.NT
    dat.sInit = args.sInit

    I = range(dat.nt)

    df = pandas.read_excel(flname,sheetname='data', dtype = {'dj': np.int32} )
    data = [list(tp)[1:] for tp in df.to_records(index=False)] 

    dat.dj = data;

    return dat

def read_json_file(args):
    flname = args.file
    if (os.path.isfile(flname) == False):
       raise Exception('file {:s} not found'.format(flname))

    with open(flname) as fl:
         js = json.load(fl)
  
    dat = DotMap(js)
    
    dat.p = args.p
    dat.q = args.q
    dat.h = args.h
    dat.nt = args.NT
    dat.sInit = args.sInit

    return dat ;

def create_model(dat):
    p = dat.p
    q = dat.q
    h = dat.h
    nt = dat.nt
    sInit = dat.sInit

    I = range(nt)
    J = range(1,nt)
    NT_1 = range(nt-1)
 
    dj = [ dat.ro[i] for i in I] 
  
    cpx = cplex.Cplex()
        
    yt = ["y(" + str(i) + ")" for i in I] 
    xt = ["x(" + str(i) + ")" for i in I] 
    st = ["s(" + str(i) + ")" for i in I] 

    # cpx.variable.add = usar o mesmo número de vezes quanto o número de variáveis na função objetivo
    cpx.variables.add(obj= [p for i in I], 
                      lb = [0.0] * nt,\
                      ub = [cplex.infinity] * nt,\
                      names = xt)
    # adiciona a variável x(t)
    # obj = qual é o coeficiente que está multiplicando a variável adicionada na função objetivo
    # lb = limite inferior 0 (vetor com items posições)
    # ub = limite superior infinito
    # nome = vetor de caracteres xt

    cpx.variables.add(obj= [q for i in I], 
                      lb = [0.0] * nt,\
                      ub = [1.0] * nt,\
                      types = ['B'] * nt,\
                      names = yt)
    # adiciona a variável y(t)
    # obj = qual é o coeficiente que está multiplicando a variável adicionada na função objetivo
    # lb = limite inferior 0 (vetor com items posições)
    # ub = limite superior 1 (bool)
    # nome = vetor de caracteres yt

    cpx.variables.add(obj= [h for i in NT_1],
                      lb = [0.0] * nt-1,
                      ub = [cplex.infinity] * nt-1,
                      names = st)
    # adiciona a variável s(t)
    # obj = qual é o coeficiente que está multiplicando a variável adicionada na função objetivo
    # lb = limite inferior 0
    # ub = limite superior infinito
    # nome = vetor de caracteres st

    [cpx.linear_constraints.add(lin_expr=[cplex.SparsePair([i, i+(2*nt),\
                                                            i+((2*nt)-1) if i+(2*nt-1) >= (2*nt) else sInit,\
                                                            dj[i] ], [1.0, -1.0, 1.0, -1.0])],
                                senses = "E", 
                                rhs = [0])for i in I]
    # adiciona a restrição dem_satt -> st-1 + xt = dt + st
    # lin_expr = SparsePair (a matriz adiciona é esparsa, há vários zeros na matriz, portanto quero adicionar só os valores não nulos)
    # parâmetro 1 (SparsePair) = índice dos valores
    # parâmetro 2 (SparsePair) = valores
    # senses = E (a expressão é uma igualdade)
    # rhs = (right hand side)

    [cpx.linear_constraints.add(lin_expr=[cplex.SparsePair([i+nt, i], [dj[j], -1.0]) for j in I],
                                senses = "E", 
                                rhs = [0.0]) for i in I]
    # adiciona restrição vubt: xt =< yt*somatorio_0_a_t(dk) ---->  yt*somatorio_0_a_t(dk) - xt >= 0
    # lin_expr = SparsePair (a matriz adiciona é esparsa, há vários zeros na matriz, portanto quero adicionar só os valores não nulos)
    # parâmetro 1 = índice dos valores
    # parâmetro 2 = valores
    # senses = L (a expressão é uma desigualdade menor ou igual)
    # rhs = (right hand side) o lado direito é sempre igual a 0 (??)

    cpx.write("elsp.lp")
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
       nt = dat.nt
       I = range(nt)
       NT_1 = range(1,nt)
       
       of = cpx.solution.get_objective_value()
       x = cpx.solution.get_values()

       sol = DotMap()
       sol.msg = cpx.solution.get_status_string() 
       sol.of = of
       sol.xt  = [i for i in I if x[i] > 0.001] 
       sol.yt  = [i for i in I if x[i+nt] > 0.001] 
       sol.st  = [i for i in J if x[i+(2*nt)] > 0.001] 
       sol.st[0] = dat.sInit;
       
    return sol

def print_sol(dat,sol):
    I = range(dat.nt)
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

# def plot_sol(dat, sol):
#     I = range(dat.items)
#     for j in sol.y:
#             for i in I:
#                 if (i,j) in sol.x:
#                     xc = np.array([dat.cli[i][0], dat.fac[j][0]])
#                     xy = np.array([dat.cli[i][1], dat.fac[j][1]])
#                     plt.plot(xc, xy, 'ko-')
#     plt.show()
#     return