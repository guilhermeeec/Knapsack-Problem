
import time as tm

class Object:
    def __init__(self, object_id, weight, value):
        self.object_id = object_id
        self.value = value
        self.weight = weight
        self.relevance = value/weight

class HeuristicSolver:

    # Initializing

    def __init__(self, itens, capacity, greedy_degree=1):
        self.build_objects_list(itens)
        self.capacity = capacity
        self.greedy_degree = greedy_degree
        self.knapsack_weigth = 0
        self.objects_list_inside_knapsack = []
        self.objective = 0

    def build_objects_list(self, itens):
        self.objects_list = []
        for item_tuple in itens:
            (object_id, weigth, value) = item_tuple
            new_object = Object(object_id, weigth, value)
            self.objects_list.append(new_object)

    # Solving

    def solve(self, method="greedy"):
        start_time = tm.time()
        self.solve_greedy()
        self.solve_time = tm.time() - start_time

    def solve_greedy(self):
        self.order_objects_list_by_relevance()
        for object in self.ordered_objects_list:
            self.insert_knapsack_if_possible(object)
        self.compute_objective()

    def order_objects_list_by_relevance(self):
        # Timsort: O(n log n) https://en.wikipedia.org/wiki/Timsort
        self.ordered_objects_list = sorted(self.objects_list, key=lambda object: -object.relevance)

    def insert_knapsack_if_possible(self, object):
        if(self.knapsack_weigth + object.weight <= self.capacity):
            self.objects_list_inside_knapsack.append(object)
            self.knapsack_weigth += object.weight

    def compute_objective(self):
        for object in self.objects_list_inside_knapsack:
            self.objective += object.value

    # Get info

    def get_execution_time(self):
        return self.solve_time
    
    def get_objective(self):
        return self.objective

    def print_status(self):
        print("-------------------")
        print("Quantity inside knapsack: ", len(self.objects_list_inside_knapsack))
        print("Total weigth: ", self.knapsack_weigth)
        print("Objective: ", self.objective)
        print("Solve time: ", self.solve_time)


def test_greedy_solution():
    # Toy problem: https://www.youtube.com/watch?v=oTTzNMHM05I
    # (Object ID, weigth, value) 
    itens = [(1,2,10),\
             (2,3,5),\
             (3,5,15),\
             (4,7,7),\
             (5,1,6),\
             (6,4,18),\
             (7,1,3)]
    capacity = 15
    solver = HeuristicSolver(itens, capacity)
    solver.solve()
    solver.print_status()

if __name__ == "__main__":
    test_greedy_solution()