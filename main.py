#Loading the required modules
from queue import PriorityQueue
import copy


class WaterBucket:
    def __init__(self, capacity, target):
        self.capacity = capacity
        self.water_level = 0
        self.target = target

    def drain(self):
        if self.water_level == 0:
            return False

        self.water_level = 0
        return True

    def fill(self):
        if self.water_level == self.capacity:
            return False

        self.water_level = self.capacity
        return True

    def pour(self, other_bucket):
        if self.capacity < other_bucket.capacity:
            return False, other_bucket.water_level

        space_in_other_bucket = other_bucket.capacity - other_bucket.water_level

        if self.water_level == 0 or space_in_other_bucket == 0:
            return False, other_bucket.water_level

        if space_in_other_bucket >= self.water_level:
            other_bucket.water_level += self.water_level
            self.water_level = 0
        else:
            self.water_level -= space_in_other_bucket
            other_bucket.water_level = other_bucket.capacity

        return True, other_bucket

    def pour_to_capacitor(self, cap):
        if self.water_level == 0 or cap + self.water_level > self.target:
            return -1

        cap += self.water_level
        self.water_level = 0

        return cap

class BucketNode:
    def __init__(self, bucket_array, children, parent_node, state_capacitor, target):
        self.bucket_array = bucket_array
        self.children = children
        self.parent_node = parent_node
        self.state_capacitor = state_capacitor
        self.heuristic = float('inf')
        self.target = target
        self.consec_pours = 0


    def update_heuristic(self, cost):
        self.heuristic = cost

    def update_consec_pours(self, cost):
        self.consec_pours = cost

    def update_state_capacitor(self, cost):
        self.state_capacitor = cost

    def __gt__(self, other):
        if isinstance(other, BucketNode):
            if self.heuristic > other.heuristic:
                return True
            if self.heuristic < other.heuristic:
                return False
            return self.state_capacitor > other.state_capacitor

    def perform_action_and_return_buckets(self, bucket_array, selected_index, action, target_index, cap):
        action_performed = False
        temp_bucket_array = copy.deepcopy(bucket_array)

        if target_index is None:
            if action == "Fill":
                action_performed = temp_bucket_array[selected_index].fill()

            elif action == "Drain":
                action_performed = temp_bucket_array[selected_index].drain()

            elif action == "Capacitor":
                action_performed = temp_bucket_array[selected_index].pour_to_capacitor(cap)
        else:
            if action == "Pour":
                action_performed, other = temp_bucket_array[selected_index].pour(
                    other_bucket=temp_bucket_array[target_index])
                if action_performed:
                    temp_bucket_array[target_index] = WaterBucket(capacity=other.capacity, target=self.target)
                    temp_bucket_array[target_index].water_level = other.water_level

        return temp_bucket_array, action_performed

    def get_children(self, node):
        children = []
        bucket_array = node.bucket_array


        max_bucket = bucket_array[len(bucket_array) - 1].capacity
        if max_bucket < self.target - node.state_capacitor:
            new_bucket_array, action = node.perform_action_and_return_buckets(bucket_array, len(bucket_array) - 1,
                                                                            "Fill", None, 0)
            if action:
                children.append(self.create_Node(node, new_bucket_array, node.state_capacitor))
            new_bucket_array, action = node.perform_action_and_return_buckets(bucket_array, len(bucket_array) - 1,
                                                                            "Capacitor", None,
                                                                            node.state_capacitor)
            if action != -1:
                children.append(self.create_Node(node, new_bucket_array, action))
            return children

        for i, bucket in enumerate(bucket_array):

            new_bucket_array, action = node.perform_action_and_return_buckets(bucket_array, i, "Fill", None, 0)
            if action:
                children.append(self.create_Node(node, new_bucket_array, node.state_capacitor))

            new_bucket_array, action = node.perform_action_and_return_buckets(bucket_array, i, "Drain", None, 0)
            if action:
                children.append(self.create_Node(node, new_bucket_array, node.state_capacitor))

            new_bucket_array, action = node.perform_action_and_return_buckets(bucket_array, i, "Capacitor", None,
                                                                            node.state_capacitor)
            if action != -1:
                children.append(self.create_Node(node, new_bucket_array, action))

            if len(bucket_array) > 1:
                rem_size = bucket_array[-1].capacity - bucket_array[-2].capacity - 2
                if (rem_size <= self.target - node.state_capacitor) or (len(bucket_array) < 4):
                    for j, bucket_j in enumerate(bucket_array):
                        if i <= j:
                            break
                        new_bucket_array, action = node.perform_action_and_return_buckets(bucket_array, i, "Pour", j, 0)
                        if action:
                            # print("Pour")
                            children.append(self.create_Node(node, new_bucket_array, node.state_capacitor))
                            cost = node.consec_pours
                            if cost < len(bucket_array):
                                children[-1].update_consec_pours(cost + 1)
                            else:
                                children.pop()
        return children
    def create_Node(self, parent_node, bucket_array, state_capacitor):
        node = BucketNode(bucket_array=bucket_array, children=[], parent_node=parent_node, state_capacitor=state_capacitor, target=self.target)
        return node


class MyPlayer:
    def run_algorithm(self, problem):
        #  function to return the path and the search_tree

        bucket_array = []
        i = 0
        while i < len(problem["size"]):
            bucket = WaterBucket(capacity=problem["size"][i], target=problem["target"])
            bucket_array.append(bucket)
            i = i + 1
        # Instantiating the root of the node
        parent_node = BucketNode(bucket_array=bucket_array, children=[], parent_node=None, state_capacitor=0, target=problem["target"])

        # Goal test in case initial state is the goal
        if problem['target'] == 0:
            return 0

        steps = self.informed_search(node=parent_node, problem=problem)
        return steps

    def check_visited_nodes(self, neighbour, visited):
        for node in visited:
            if neighbour.state_capacitor == node.state_capacitor:
                neigh_buckets = neighbour.bucket_array
                visited_node = True
                for i, bucket in enumerate(node.bucket_array):
                    if bucket.water_level != neigh_buckets[i].water_level:
                        visited_node = False
                if visited_node:
                    return True
        return False

    def construct_search_tree(self, node, target):
        #List of visited nodes
        visited = []
        # Initialize a queue
        queue = []
        visited.append(node)
        queue.append(node)
        goal = False

        # Creating loop to visit each node
        while queue:
            m = queue.pop(0)
            children = m.get_children(m)
            m.children = children

            for neighbour in children:
                if not self.check_visited_nodes(neighbour, visited):
                    visited.append(neighbour)
                    queue.append(neighbour)

            if m.state_capacitor == target:
                goal = True
                return goal, visited

        return goal, visited

    def compare_nodes(self, node_A, node_B):
        if node_A.state_capacitor == node_B.state_capacitor:
            neigh_buckets = node_A.bucket_array
            for i, bucket in enumerate(node_B.bucket_array):
                if bucket.water_level != neigh_buckets[i].water_level:
                    return False
            return True
        return False

    def heuristic(self, start_node, end_node):
        end_node.update_heuristic(0)
        cost = 1
        node = end_node
        while not self.compare_nodes(node, start_node):
            node = node.parent_node
            node.update_heuristic(cost)
            for child in node.children:
                if child.heuristic == float('inf'):
                    child.update_heuristic(cost)
                cost += 1
        return start_node

    def a_star_algorithm(self, start, goal_state):
        # A priority queue to store the nodes to be explored
        pq = PriorityQueue()
        pq.put((0, start))
        visited = []

        while not pq.empty():
            # Getting the node with the lowest cost
            cost, current_node = pq.get()
            
            # We Check if we have reached the goal state
            if current_node.state_capacitor == goal_state.state_capacitor:
                return current_node

            visited.append(current_node)
            
            # We Add the children of the current node to priority queue
            for child in current_node.children:
                if not self.check_visited_nodes(child, visited):
                    pq.put((child.heuristic + 1, child))

        return -1

    def informed_search(self, node, problem):

        goal, visited = self.construct_search_tree(node, problem["target"])

        if not goal:
            return -1

        start_node = visited[0]
        end_node = visited[len(visited) - 1]
        self.heuristic(start_node, end_node)
        end_node = self.a_star_algorithm(start_node, end_node)
        path = []
        while not self.compare_nodes(start_node, end_node):
            bucket_array = end_node.bucket_array
            buckets = []
            for bucket in bucket_array:
                buckets.append(bucket.water_level)
            buckets.append(end_node.state_capacitor)
            path.append(buckets)
            end_node = end_node.parent_node
        path.reverse
        i = 0
        for p in path:
            i += 1
            print(f"Step {i}")
            print(p)
        steps = len(path)
        return steps


if __name__ == '__main__':
    li_input = open("input.txt", "r")
    string_input = li_input.readline()
    targ_input = li_input.readline()
    targ_input = int(targ_input)
    li_values = list(string_input.split(","))
    li_values = [int(i) for i in li_values]
    li_values.sort()
    problem = {
    "size": li_values,
    "target": targ_input
    }
    player = MyPlayer()
    steps = player.run_algorithm(problem)
    print(f"Minimum number of steps required: {steps}")