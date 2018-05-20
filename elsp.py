
from elsp import *
import sys
import os
import argparse
import math
import numpy as np
import json
import pandas
from dotmap import DotMap

import cplex

if __name__ == "__main__":
  args = parse_arguments()
  dat = read_dat_file(args)
  cpx = create_model(dat)
  sol = solve_model(dat,cpx)
  print_sol(dat,sol)

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('file', metavar='[datafile]', help='datafile name')
  parser.add_argument('p',metavar='[production unit cost]',type=int, help='Cost of item in production')
  parser.add_argument('q',metavar='[production setup cost]',type=int, help='Cost of setup to product the item')
  parser.add_argument('hcost',metavar='[inventory holding cost]',type=int, help='Cost of hold the product in the inventory')
  parser.add_argument('NT',metavar='[number of periods]',type=int, help='Number of periods of time')
  parser.add_argument('sInit',metavar='[inventory initial level]',type=float, default = 0, help='size in months of the production cycle')
  args = parser.parse_args()
  return args

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
  dat.h = args.hcost
  dat.nt = args.NT
  dat.sInit = args.sInit

  I = range(dat.nt)

  data = []
  for i in I:
      dj = int(next(lines))
      data.append([dj])

  fl.close()

  dat.data = data
  
  return dat

def create_model(dat):
  p = dat.p
  q = dat.q
  h = dat.h
  nt = dat.nt
  sInit = dat.sInit

  nt_1 = nt-1


  I = range(nt)
  NT_1 = range(nt-1)

  s_begin = 2*nt;
  s_begin_1 = 2*nt - 1;
  
  dj = []
  [dj.append(float(dat.data[i][0])) for i in I]

  cpx = cplex.Cplex()
      
  yt = ["yt(" + str(i) + ")" for i in I] 
  xt = ["xt(" + str(i) + ")" for i in I] 
  st = ["st(" + str(i) + ")" for i in I] 

  # cpx.variable.add = usar o mesmo número de vezes quanto o número de variáveis na função objetivo
  cpx.variables.add(obj= [p] * nt, \
                    lb = [0.0] * nt,\
                    ub = [cplex.infinity] * nt,\
                    names = xt)
  # adiciona a variável x(t)
  # obj = qual é o coeficiente que está multiplicando a variável adicionada na função objetivo
  # lb = limite inferior 0 (vetor com items posições)
  # ub = limite superior infinito
  # nome = vetor de caracteres xt

  cpx.variables.add(obj= [q]* nt,\
                    lb = [0.0] * nt,\
                    ub = [1.0] * nt,\
                    types = ['B'] * nt,\
                    names = yt)
  # adiciona a variável y(t)
  # obj = qual é o coeficiente que está multiplicando a variável adicionada na função objetivo
  # lb = limite inferior 0 (vetor com items posições)
  # ub = limite superior 1 (bool)
  # nome = vetor de caracteres yt

  cpx.variables.add(obj= [h] * nt,\
                    lb = [0.0] * nt,\
                    ub = [cplex.infinity]*nt,\
                    names = st)
  # adiciona a variável s(t)
  # obj = qual é o coeficiente que está multiplicando a variável adicionada na função objetivo
  # lb = limite inferior 0
  # ub = limite superior infinito
  # nome = vetor de caracteres st

  ####################################             Restrição dem_satt            ###########################################################
  for i in I:
    if i > 0:
      cpx.linear_constraints.add(lin_expr=[cplex.SparsePair([i, (i+s_begin), (i+(s_begin_1))], [1.0, -1.0, 1.0])],\
                          senses = "E",\
                          rhs = [dj[i]])
    else:
      cpx.linear_constraints.add(lin_expr=[cplex.SparsePair([i, (i+s_begin)], [1.0, -1.0])],\
                          senses = "E",\
                          rhs = [dj[i] - sInit])
  #############################################################################################################################

  # adiciona a restrição dem_satt -> st-1 + xt = dt + st
  # lin_expr = SparsePair (a matriz adiciona é esparsa, há vários zeros na matriz, portanto quero adicionar só os valores não nulos)
  # parâmetro 1 (SparsePair) = índice dos valores
  # parâmetro 2 (SparsePair) = valores
  # senses = E (a expressão é uma igualdade)
  # rhs = (right hand side)
  sum_dj = 0
  for i in I:
    sum_dj += dj[i]

  [cpx.linear_constraints.add(lin_expr=[cplex.SparsePair([(i+nt), i], [sum_dj, -1.0])],\
                              senses = "G", \
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
    sol.xt  = [x[i] for i in I ]#if x[i] > 0.001] 
    sol.yt  = [x[i+nt] for i in I ]#if x[i+nt] > 0.001] 
    sol.st  = [x[i+((2*nt)-1)] for i in NT_1 ]#if x[i+(2*(nt-1))] > 0.001] 

    sol.st.insert(0, dat.sInit);

  return sol

def print_sol(dat,sol):
    I = range(dat.nt)
    print("Solver status          : {:s}".format(sol.msg))
    print("Objective function     : {:18,.2f}".format(sol.of))
    print("\n\tTime period\t|\tProduction batch size\t|\tProduction set-up\t|\tEnd inventory level")
    for i in I:
      print("\t  {:.2f}\t\t|{:18,.2f}\t\t|{:18,.2f}\t\t|{:18,.2f}\t".format(i, sol.xt[i], sol.yt[i], sol.st[i]))
    


      


