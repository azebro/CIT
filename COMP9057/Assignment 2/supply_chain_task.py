'''
Decision Analytics - Assignment 2, Task 1
Adam Zebrowski - R00183247


In this task you will optimise the cost of sourcing raw material from different suppliers, 
    manufacturing products in different factories and delivering these products to customers. 
The input data for this task is contained in the Excel file “Assignment_DA_2_a_data.xlsx” and can be downloaded from Canvas. 
    The file contains 8 sheets:
    - Supplier stock
    A table indicating how many units of each raw material each of the suppliers has in stock.
    - Raw material costs
    A table indicating how much each of the suppliers is charging per unit for each of the raw materials.
    - Raw material shipping
    A table indicating the shipping costs per unit of raw material (the units for each material are the same) from each supplier to each factory
    - Product requirements
    A table indicating the amount of raw material required to manufacture one unit of each of the products.
    - Production capacity
    A table indicating how many units of each product each of the factories is able to manufacture.
    - Production cost
    A table indicating the cost of manufacturing a unit of each product in each of the factories.
    - Customer demand
    A table indicating the number of units of each product that have been ordered by the customers
    - Shipping costs
    A table indicating the shipping costs per unit for delivering a product to the customer.
Factories can order suppliers from multiple suppliers and products can be delivered to customers from multiple factories.
The goal of this task is to develop and optimise a Liner Programming model that helps decide what raw material to order from which supplier, 
    where to manufacture the products, and how to deliver the manufactured products to the customers so that the overall cost is minimised.

'''



from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np


# A. Loading the input data
'''
Load the input data from the file “Assignment_DA_2_a_data.xlsx” [1 point]. 
Note that not all fields are filled, for example Supplier C does not stock Material A.
Make sure to use the data from the file in your code, please do not hardcode any values that can be read from the file.
'''
sup_stock = pd.read_excel("Assignment_DA_2_a_data.xlsx", sheet_name="Supplier stock", index_col=0)
raw_material_cost = pd.read_excel("Assignment_DA_2_a_data.xlsx", sheet_name="Raw material costs", index_col=0)
raw_material_shipping = pd.read_excel("Assignment_DA_2_a_data.xlsx", sheet_name="Raw material shipping", index_col=0)
production_req = pd.read_excel("Assignment_DA_2_a_data.xlsx", sheet_name="Product requirements", index_col=0)
production_capacity = pd.read_excel("Assignment_DA_2_a_data.xlsx", sheet_name="Production capacity", index_col=0)
customer_demand = pd.read_excel("Assignment_DA_2_a_data.xlsx", sheet_name="Customer demand", index_col=0)
production_cost = pd.read_excel("Assignment_DA_2_a_data.xlsx", sheet_name="Production cost", index_col=0)
shipping_costs = pd.read_excel("Assignment_DA_2_a_data.xlsx", sheet_name="Shipping costs", index_col=0)

customer_demand = customer_demand.fillna(0)
production_req = production_req.fillna(0)
sup_stock = sup_stock.fillna(0)
production_capacity = production_capacity.fillna(0)
raw_material_cost = raw_material_cost.fillna(0)
production_cost = production_cost.fillna(0)

# Getting list

factories = list(raw_material_shipping.columns)
print("Factories:\n", factories)
# Getting list of materials
materials = list(raw_material_cost.columns)
print("Materials: \n", materials)
# Getting list of suppliers
suppliers = list(raw_material_cost.index)
print("Suppliers: \n", suppliers)
# Getting list of products
products = list(production_req.index)
print("Products: \n", products)
# Getting list of customers
customers = list(customer_demand.columns)
print("Customers: \n", customers)

# B. 
'''
Identify and create the decision variables for the orders from the suppliers [1 point], 
    for the production volume [1 point], 
    and for the delivery to the customers [1 point] using the OR Tools wrapper of the GLOP_LINEAR_PROGRAMMING solver.
'''
solver = pywraplp.Solver('LPWrapper',
                         pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
orders = {}
for factory in factories:
    for material in materials:
        for supplier in suppliers:
            orders[(factory, material, supplier)] = solver.NumVar(0, solver.infinity(),
                                                                  factory + "_" + material + "_" + supplier)

production_volume = {}
for factory in factories:
    for product in products:
        production_volume[(factory, product)] = solver.NumVar(0, solver.infinity(), factory + "_" + product)

delivery = {}
for factory in factories:
    for customer in customers:
        for product in products:
            delivery[(factory, customer, product)] = solver.NumVar(0, solver.infinity(),
                                                                   factory + "_" + customer + "_" + product)

# C. 
# Define and implement the constraints that ensure factories produce more than they ship to the customers [2 points].


for product in products:
    for factory in factories:
        c = solver.Constraint(0, solver.infinity())
        c.SetCoefficient(production_volume[(factory, product)], 1)
        for customer in customers:
            c.SetCoefficient(delivery[(factory, customer, product)], -1)

#D. 
# Define and implement the constraints that ensure that customer demand is met [2 points].

for customer in customers:
    for product in products:

        c = solver.Constraint(int(customer_demand.loc[product][customer]), int(customer_demand.loc[product][customer]))
        for factory in factories:
            c.SetCoefficient(delivery[(factory, customer, product)], 1)

# E. 
# Define and implement the constraints that ensure that suppliers have all ordered items in stock [2 points].


for supplier in suppliers:
    for material in materials:
        c = solver.Constraint(0, int(sup_stock.loc[supplier][material]))
        for factory in factories:
            c.SetCoefficient(orders[(factory, material, supplier)], 1)

# F. 
# Define and implement the constraints that ensure that factories order enough material to be able to manufacture all items [2 points].


for factory in factories:
    for material in materials:
        c = solver.Constraint(0, solver.infinity())
        for supplier in suppliers:
            c.SetCoefficient(orders[(factory, material, supplier)], 1)
            for product in products:
                c.SetCoefficient(production_volume[(factory, product)], - production_req.loc[product][material])

# G.
# Define and implement the constraints that ensure that the manufacturing capacities are not exceeded [2 points].

for factory in factories:
    for product in products:
        c = solver.Constraint(0, int(production_capacity.loc[product][factory]))
        c.SetCoefficient(production_volume[(factory, product)], 1)



#H.  
'''
Define and implement the objective function. 
Make sure to consider the supplier bills comprising shipping and material costs [2 points], 
    the production cost of each factory [2 points], 
    and the cost of delivery to each customer [2 points].
'''
cost = solver.Objective()
# Material Costs  + shipping costs
for factory in factories:
    for supplier in suppliers:
        for material in materials:
            cost.SetCoefficient(orders[(factory, material, supplier)],
                                raw_material_cost.loc[supplier][material] + raw_material_shipping.loc[supplier][
                                    factory])


# production cost of each factory
for factory in factories:
    for product in products:
        cost.SetCoefficient(production_volume[(factory, product)], int(production_cost.loc[product][factory]))

# shipping cost to customers
for factory in factories:
    for customer in customers:
        for product in products:
            cost.SetCoefficient(delivery[(factory, customer, product)], int(shipping_costs.loc[factory][customer]))

# I. 
# Solve the linear program and determine the optimal overall cost [1 point].

cost.SetMinimization()
status = solver.Solve()

if status == solver.OPTIMAL:
    print("Optimal Solution Found")
print("Optimal Overall Cost: ", solver.Objective().Value())

# J. and K
'''
J. Determine for each factory how much material has to be ordered from each individual supplier [1 point].
K. Determine for each factory what the supplier bill comprising material cost and delivery will be for each supplier [1 point].
'''
print("\nSupplier Bill and order quantity")
print("****************************")
for factory in factories:
    print(factory, ":")

    for supplier in suppliers:
        factory_cost = 0
        print("  ", supplier, ":")
        for material in materials:
            #J
            print("\t", material, ":", orders[(factory, material, supplier)].solution_value())

            #K
            factory_cost += orders[(factory, material, supplier)].solution_value() * raw_material_cost.loc[supplier][
                material]
            factory_cost += orders[(factory, material, supplier)].solution_value() * float(
                raw_material_shipping.loc[supplier][factory])
        print("  ", supplier, " Bill: ", factory_cost)

# L.
'''
Determine for each factory how many units of each product are being manufactured [1 point]. 
Also determine the total manufacturing cost for each individual factory [1 point].
'''
print("Production Volume:")
print("****************************")

for factory in factories:
    print(factory, ":")
    production_cost_total = 0
    for product in products:
        if production_volume[(factory, product)].solution_value() > 0:
            print("  ", product, ": ", production_volume[(factory, product)].solution_value())
            production_cost_total += production_volume[(factory, product)].solution_value() * \
                                     production_cost.loc[product][factory]
    print("   Manufacturing cost: ", production_cost_total)

# M 
'''
Determine for each customer how many units of each product are being shipped from each factory [1 point]. 
Also determine the total shipping cost per customer [1 point]
'''
print("\nShipping to Customer:")
print("****************************")

for customer in customers:
    shipping_cost = 0
    print(customer)
    for product in products:
        print("  ", product)
        for factory in factories:
            print("\t", factory, ": ", delivery[(factory, customer, product)].solution_value())
            shipping_cost += delivery[(factory, customer, product)].solution_value() * shipping_costs.loc[factory][
                customer]
    print("   Shipping Cost: ", shipping_cost)

#N
'''
Determine for each customer the fraction of each material each factory has to order for manufacturing products delivered to that particular customer [1 point]. 
Based on this calculate the overall unit cost of each product per customer including the raw materials used for the manufacturing of the customer’s specific product, 
    the cost of manufacturing for the specific customer and all relevant shipping costs [2 points].
'''
print("\nMaterial Bifurcation and Cost per unit")
print("****************************")
#

for customer in customers:
    print(customer)
    for product in products:

        unit_cost_per_product = 0
        if int(customer_demand.loc[product][customer]) > 0:
            print("  ", product)
            for factory in factories:

                if delivery[(factory, customer, product)].solution_value() > 0:
                    print("\t", factory, ": ")
                    # Calculating the Shipping cost from factory to customer based on number of products
                    shipping_cost = delivery[(factory, customer, product)].solution_value() * \
                                    shipping_costs.loc[factory][customer]
                    # Calculating the manufacturing cost
                    man_cost = delivery[(factory, customer, product)].solution_value() * production_cost.loc[product][
                        factory]
                    unit_cost_per_product += shipping_cost
                    unit_cost_per_product += man_cost
                    material_cost_to_customer = 0
                    for material in materials:
                        material_units = 0
                        material_units += delivery[(factory, customer, product)].solution_value() * \
                                          production_req.loc[product][material]

                        print("\t  ", material, ": ", material_units)
                        # raw material cost
                        material_cost = 0
                        # raw material cost
                        rshipping_cost = 0
                        material_count = 0
                        for supplier in suppliers:
                            material_cost += orders[(factory, material, supplier)].solution_value() * \
                                             raw_material_cost.loc[supplier][material]
                            rshipping_cost += orders[(factory, material, supplier)].solution_value() * \
                                              raw_material_shipping.loc[supplier][factory]
                            material_count += orders[(factory, material, supplier)].solution_value()
                        material_cost_to_customer = ((material_cost + rshipping_cost) / material_count) * material_units
                        unit_cost_per_product += material_cost_to_customer
            print("\t cost per unit : ", unit_cost_per_product / int(customer_demand.loc[product][customer]))

