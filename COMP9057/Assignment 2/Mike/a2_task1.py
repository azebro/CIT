# Mike Leske
# R00183658

###############################################################################
#
# TASK 1 - 30 points
#
# Supply chain optimisation
#   
#   Factories can order suppliers from multiple suppliers and products can be 
#   delivered to customers from multiple factories.
#
#   The goal of this task is to develop and optimise a Liner Programming model 
#   that helps decide what raw material to order from which supplier, where to 
#   manufacture the products, and how to deliver the manufactured products to 
#   the customers so that the overall cost is minimised.
#
###############################################################################

from ortools.linear_solver import pywraplp

import pandas as pd
import numpy as np
import pprint

pp = pprint.PrettyPrinter(indent=4)

print('TASK 1')


###############################################################################
#
# TASK 1 A - 1 point
# 
#   1)  Load the input data from the file "Assignment_DA_2_a_data.xlsx"
#       [1 point]. 
# 
#       Note that not all fields are filled, for example Supplier C does not 
#       stock Material A. Make sure to use the data from the file in your code, 
#       please do not hardcode any values that can be read from the file.
#
###############################################################################

file = 'Assignment_DA_2_a_data.xlsx'

supplier_stock = pd.read_excel(file, sheet_name='Supplier stock', index_col=0).fillna(0)
raw_material_costs = pd.read_excel(file, sheet_name='Raw material costs', index_col=0).fillna(0)
raw_material_shipping = pd.read_excel(file, sheet_name='Raw material shipping', index_col=0).fillna(0)
product_requirements = pd.read_excel(file, sheet_name='Product requirements', index_col=0).fillna(0)
production_capacity = pd.read_excel(file, sheet_name='Production capacity', index_col=0).fillna(0)
production_cost = pd.read_excel(file, sheet_name='Production cost', index_col=0).fillna(0)
customer_demand = pd.read_excel(file, sheet_name='Customer demand', index_col=0).fillna(0)
shipping_costs = pd.read_excel(file, sheet_name='Shipping costs', index_col=0).fillna(0)

supplier_list = list(supplier_stock.index)
material_list = list(supplier_stock.columns)
factory_list  = list(raw_material_shipping.columns)
product_list  = list(product_requirements.index)
customer_list = list(customer_demand.columns)


###############################################################################
#
# TASK 1 B - 3 points
# 
#   1)  Identify and create the decision variables for the orders from the 
#       suppliers [1 point], 
#   2)  for the production volume [1 point], 
#   3)  and for the delivery to the customers [1 point] 
#       using the OR Tools wrapper of the GLOP_LINEAR_PROGRAMMING solver.
#
###############################################################################

solver = pywraplp.Solver('LPWrapper', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)


factories = { factory: {} for factory in factory_list }

for factory, data in factories.items():
    data['production'] = {}
    for product in product_list:
        name = factory.replace(' ', '') + '_' + product.replace(' ', '')
        data['production'][product] = solver.NumVar(0, production_capacity.loc[product, factory], name)

    data['supply'] = {}
    for supplier in supplier_list:
        data['supply'][supplier] = {}
        for material in material_list:
            name = factory.replace(' ', '') + '_' + supplier.replace(' ', '') + '_' + material.replace(' ', '')
            data['supply'][supplier][material] = solver.NumVar(0, supplier_stock.loc[supplier, material], name)

    data['delivery'] = {}
    for product in product_list:
        data['delivery'][product] = {}
        for customer in customer_list:
            name = factory.replace(' ', '') + '_' + product.replace(' ', '') + '_' + customer.replace(' ', '')
            data['delivery'][product][customer] = solver.NumVar(0, customer_demand.loc[product, customer], name)


#pp.pprint(factories)

###############################################################################
#
# TASK 1 C - 2 points
# 
#   1)  Define and implement the constraints that ensure factories produce more 
#       than they ship to the customers [2 points].
#
###############################################################################

'''for product in product_list:
    for customer in customer_list:
        demand = customer_demand.loc[product, customer]
        x = solver.Constraint(demand, solver.Infinity())
        for factory in factory_list:
            x.SetCoefficient(factories[factory]['delivery'][product][customer], 1)'''

for p in product_list:
    for f in factory_list:
        x = solver.Constraint(0, solver.Infinity())
        x.SetCoefficient(factories[f]['production'][p] ,1)
        for c in customer_list:
            x.SetCoefficient(factories[f]['delivery'][p][c], -1)


###############################################################################
#
# TASK 1 D - 2 points
# 
#   1)  Define and implement the constraints that ensure that customer demand 
#       is met [2 points].
#
###############################################################################

for c in customer_list:
    for p in product_list:

        x = solver.Constraint(customer_demand.loc[p,c], customer_demand.loc[p,c])
        for f in factory_list:
            x.SetCoefficient(factories[f]['delivery'][p][c], 1)

###############################################################################
#
# TASK 1 E - 2 points
# 
#   1)  Define and implement the constraints that ensure that suppliers have 
#       all ordered items in stock [2 points].
#
###############################################################################

for s in supplier_list:
    for m in material_list:
        x = solver.Constraint(0, supplier_stock.loc[s, m])
        for f in factory_list:
            #for product in product_list:
                x.SetCoefficient(factories[f]['supply'][s][m], 1)


###############################################################################
#
# TASK 1 F - 2 points
# 
#   1)  Define and implement the constraints that ensure that factories order 
#       enough material to be able to manufacture all items [2 points].
#
###############################################################################

for f in factory_list:
    for m in material_list:
        x = solver.Constraint(0, solver.infinity())
        for s in supplier_list:
            x.SetCoefficient(factories[f]['supply'][s][m], 1)
            for p in product_list:
                x.SetCoefficient(factories[f]['production'][p], -product_requirements.loc[p, m])


###############################################################################
#
# TASK 1 G - 2 points
# 
#   1)  Define and implement the constraints that ensure that the manufacturing 
#       capacities are not exceeded [2 points].
#
###############################################################################

for p in product_list:
    for f in factory_list:
        x = solver.Constraint(0, production_capacity.loc[p, f])
        x.SetCoefficient(factories[f]['production'][p], 1)


###############################################################################
#
# TASK 1 H - 6 points
# 
#   1)  Define and implement the objective function. Make sure to consider the 
#       supplier bills comprising shipping and material costs [2 points], 
#   2)  the production cost of each factory [2 points], 
#   3)  and the cost of delivery to each customer [2 points].
#
###############################################################################

objective = solver.Objective()

for f in factory_list:
    for s in supplier_list:
        for m in material_list:
            objective.SetCoefficient(factories[f]['supply'][s][m], 
                                     raw_material_costs.loc[s,m] + raw_material_shipping.loc[s,f])

    for p in product_list:
        objective.SetCoefficient(factories[f]['production'][p], production_cost.loc[p,f])

    for c in customer_list:
        for p in product_list:
            objective.SetCoefficient(factories[f]['delivery'][p][c], int(shipping_costs.loc[f,c]))


###############################################################################
#
# TASK 1 I - 1 point
# 
#   1)  Solve the linear program and determine the optimal overall cost [1 point].
#
###############################################################################

objective.SetMinimization()
status = solver.Solve()

cost = solver.Objective().Value()

if status == solver.OPTIMAL:
    print("Optimal solution found")
    print("Optimal overall cost: ", cost)


###############################################################################
#
# TASK 1 J - 1 point
# 
#   1)  Determine for each factory how much material has to be ordered from 
#       each individual supplier [1 point].
#
###############################################################################

print('\n################## Task 1J ##################')
print(':Material order per supplier per factory:\n')
for f in factory_list:
    print(f + ':')
    for s in supplier_list:
        print('\t' + s + ':')
        for m in material_list:
            print('\t\t' + m + ': ' + str(factories[f]['supply'][s][m].solution_value()))
    

###############################################################################
#
# TASK 1 K - 1 point
# 
#   1)  Determine for each factory what the supplier bill comprising material
#       cost and delivery will be for each supplier [1 point].
#
###############################################################################

print('\n################## Task 1K ##################')
print(':Supplier bill per factory:\n')
for f in factory_list:
    print(f + ':')
    for s in supplier_list:
        sup_cost = 0
        for m in material_list:
            sup_cost += factories[f]['supply'][s][m].solution_value() * raw_material_costs.loc[s,m]
            sup_cost += factories[f]['supply'][s][m].solution_value() * raw_material_shipping.loc[s,f]
        print('\t' + s + ' bill: ' + str(sup_cost))


###############################################################################
#
# TASK 1 L - 2 points
# 
#   1)  Determine for each factory how many units of each product are being 
#       manufactured [1 point]. 
#   2)  Also determine the total manufacturing cost for each individual factory 
#       [1 point].
#
###############################################################################

print('\n################## Task 1L ##################')
print(':Unit production per factory:')
print(':Production cost per factory:\n')
for f in factory_list:
    print(f + ':')
    prod_cost = 0
    for p in product_list:
        p_production = factories[f]['production'][p].solution_value()
        prod_cost += p_production * production_cost.loc[p, f]
        print('\t' + p + ': ' + str(p_production))

    print('\t' + 'Total manufacturing cost: ' + str(prod_cost))


###############################################################################
#
# TASK 1 M - 2 points
# 
#   1)  Determine for each customer how many units of each product are being 
#       shipped from each factory [1 point]. 
#   2)  Also determine the total shipping cost per customer [1 point]
#
###############################################################################

print('\n################## Task 1M ##################')
print(':Units shipped per customer per factory:')
print(':Shipping cost per customer:\n')

for c in customer_list:
    print(c + ':')
    shipping_cost = 0
    for f in factory_list:
        print('\t' + f + ':')
        for p in product_list:
            num_p = factories[f]['delivery'][p][c].solution_value()
            shipping_cost += num_p * shipping_costs.loc[f,c]
            print('\t\t' + p + ': ' + str(num_p))
    print('\t' + 'Total shipping cost: '+ str(shipping_cost))


###############################################################################
#
# TASK 1 N - 3 points
# 
#   1)  Determine for each customer the fraction of each material each factory 
#       has to order for manufacturing products delivered to that particular 
#       customer [1 point]. 
#   2)  Based on this calculate the overall unit cost of each product per 
#       customer including the raw materials used for the manufacturing of the 
#       customer's specific product, the cost of manufacturing for the specific 
#       customer and all relevant shipping costs [2 points].
#
###############################################################################

print('\n################## Task 1N ##################')
print()

fact_mat_fm = { f: {m: {} for m in material_list} for f in factory_list }
fact_mat_fmc = { f: {m: {c: {} for c in customer_list} for m in material_list} for f in factory_list }

for f in factory_list:
    print(f + ':')

    for m in material_list:
        print('\t' + m + ':')
        m_total = sum([ factories[f]['supply'][s][m].solution_value() for s in supplier_list])
        fact_mat_fm[f][m] = m_total
        print('\t\t' + 'Total order: '+ str(m_total))
        
        for c in customer_list:
            c_mat = 0
            for p in product_list:
                c_p = factories[f]['delivery'][p][c].solution_value()
                c_mat += c_p * product_requirements.loc[p,m]

            percentage = c_mat/m_total
            fact_mat_fmc[f][m][c] = percentage
            print('\t\t\t{}: {}% ({})'.format(c, round(percentage*100, 2), c_mat))

print()
all_unit_costs = 0
for f in factory_list:
    print(f + ':')
    for p in product_list:
        print('\t' + p + ':')
        for c in customer_list:

            # Note: fpc = factory_product_customer
            fpc_count = factories[f]['delivery'][p][c].solution_value()

            if fpc_count:
                
                supply_cost = 0                
                for m in material_list:
                    temp = 0
                    if product_requirements.loc[p,m]:
                        for s in supplier_list:
                            #supply_cost +=  ( fact_mat_fmc[f][m][c] * factories[f]['supply'][s][m].solution_value() * 
                            #                                (raw_material_costs.loc[s,m] + raw_material_shipping.loc[s,f]))
                            temp +=  ( factories[f]['supply'][s][m].solution_value() * 
                                                            (raw_material_costs.loc[s,m] + raw_material_shipping.loc[s,f]))

                    scale = (product_requirements.loc[p,m] * fpc_count ) / fact_mat_fm[f][m]
                    supply_cost += temp * scale 
                
                prod_cost = fpc_count * production_cost.loc[p,f]
                delivery_cost = fpc_count * shipping_costs.loc[f,c]
                all_unit_costs += (supply_cost + prod_cost + delivery_cost)

                unit_supply_cost = round(supply_cost / fpc_count, 2)
                unit_cost = unit_supply_cost + production_cost.loc[p,f] + shipping_costs.loc[f,c]

                print('\t\t{}: {} unit price ( {} units )'.format(c, unit_cost, round(fpc_count, 0)))


print('\n\nCross-check: cost == all_unit_costs:', cost==all_unit_costs)

