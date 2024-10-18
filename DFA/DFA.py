import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from graphviz import Source

class DFA:
    '''
    A class representing a Deterministic Finite Automaton (DFA) for processing input strings.

    Attributes:
    - file_path (str): The path to the JSON file containing the DFA definition.
    - states (list): List of states in the DFA.
    - alphabet (list): List of symbols in the alphabet.
    - initial_state (str): The initial state of the DFA.
    - accept_states (list): List of accepting states in the DFA.
    - transitions (dict): Dictionary representing the transitions between states based on input symbols.

    Methods:
    - run(input_string: str) -> bool: Processes an input string and returns True if the string is accepted, False otherwise.
    - generate_random_tests(n_trials=10, min_k=1, max_k=5) -> list: Generates random input strings and tests them against the DFA.
    - bulk_run(input_strings: list) -> list: Processes a list of input strings and returns a list of results.
    - draw(file_path=None): Draws a diagram of the DFA using NetworkX and Matplotlib. Optionally saves the diagram to a file.
    '''
    def __init__(self, file_path):
        '''
        Constructs a new DFA object by parsing the JSON file at the specified file path.
        
        Args:
        - file_path (str): The path to the JSON file containing the DFA definition.
        '''
        self.file_path = file_path
        self.states, self.alphabet, self.initial_state, self.accept_states, self.transitions = self._parse_fsm(self._read_json_file())

    def _parse_fsm(self, json_data: dict) -> tuple:
        '''
        Parses the JSON data to extract the states, alphabet, initial state, accept states, and transitions.
        
        Args:
        - json_data (dict): A dictionary containing the DFA definition.
        
        Returns:
        - tuple: A tuple containing the states, alphabet, initial state, accept states, and transitions.
        '''
        required_keys = ["states", "alphabet", "initialState", "acceptStates", "transitions"]
        for key in required_keys:
            if key not in json_data:
                missing_keys = [key for key in required_keys if key not in json_data]
                raise ValueError(f"Error: missing keys in JSON file: {missing_keys}")

        states = json_data["states"]
        alphabet = json_data["alphabet"]
        initial_state = json_data["initialState"]
        accept_states = json_data["acceptStates"]
        
        transitions = {}
        for transition in json_data["transitions"]:
            from_state = transition["from"]
            input_symbol = transition["input"]
            to_state = transition["to"]
            transitions[(from_state, input_symbol)] = to_state

        return states, alphabet, initial_state, accept_states, transitions

    def _read_json_file(self) -> dict:
        '''
        Reads and loads the JSON file at the specified file path.
        
        Returns:
        - dict: A dictionary containing the JSON data from the file.        
        '''
        if not self.file_path.endswith(".json"):
            raise ValueError("Error: file must be a JSON file")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError("Error: file not found")
        with open(self.file_path) as f:
            try:
                dfa_json = json.load(f)
            except:
                raise ValueError("Error: file not valid, ensure it is a valid JSON file")
        return dfa_json

    def run(self, input_string: str) -> bool:
        '''
        Processes an input string using the DFA and returns True if the string is accepted, False otherwise.
        
        Args:
        - input_string (str): The input string to be processed.
        
        Returns:
        - bool: True if the input string is accepted by the DFA, False otherwise.
        '''
        current_string = self.initial_state
        for s in input_string:
            # unrecognized symbol
            if s not in self.alphabet:
                return False
            try:
                current_string = self.transitions[(current_string, s)]
            except:
                # undefined transition, then the string is rejected
                return False
        return current_string in self.accept_states
    
    def generate_random_tests(self, n_trials=10, min_k=1, max_k=5) -> list:
        '''
        Generates random input strings and tests them against the DFA.
        
        Args:
        - n_trials (int): The number of random input strings to generate and test.
        - min_k (int): The minimum length of the generated input strings.
        - max_k (int): The maximum length of the generated input strings.
        
        Returns:
        - list: A list of tuples containing the input strings and their corresponding results (True if accepted, False otherwise).
        '''
        results = []
        for i in range(n_trials):
            input_string = "".join(random.choices(self.alphabet, k=random.randint(min_k, max_k)))
            results.append((input_string, self.run(input_string)))
        return results
    
    def bulk_run(self, input_strings: list) -> list:
        '''
        Runs a list of input strings through the DFA and returns a list of results.
        
        Args:
        - input_strings (list): A list of input strings to be processed.
        
        Returns:
        - list: A list of tuples containing the input strings and their corresponding results (True if accepted, False otherwise).
        '''
        return [(input_string, self.run(input_string)) for input_string in input_strings]
    
    def draw(self, file_name=None):
        '''
        Draws a diagram of the DFA. Optionally saves the diagram to a file.
        
        Args:
        - file_path (str): The file path to save the diagram. If None, the diagram will be displayed but not saved.        
        '''
        # adapted from https://github.com/navin-mohan/dfa-minimization/blob/master/dfa.py
        g = nx.DiGraph()
        for x in self.states:
            g.add_node(x, shape='doublecircle' if x in self.accept_states else 'circle', fillcolor='grey' if x == self.initial_state else 'white', style='filled')
        
        temp = defaultdict(list)
        for k, v in self.transitions.items():
            temp[(k[0], v)].append(k[1])
        
        for k, v in temp.items():
            g.add_edge(k[0], k[1], label=','.join(v))
        
        # adapted to handle render and save options
        dot_representation = nx.drawing.nx_agraph.to_agraph(g).to_string()
        dfa = Source(dot_representation)
        if file_name:
            dfa.render(file_name, format='png', cleanup=True)
        else:
            dfa.view()

# Usage
dfa_handler = DFA("dfa.json")
print(dfa_handler.generate_random_tests())
print(dfa_handler.bulk_run(set(['bbaba', 'aaba', 'ab', 'babba', 'aa', 'bb', 'aaa', 'a', 'b', 'aaba', 'aaba', 'b', 'babaa', 'bbbab', 'a', 'bbbba', 'a', 'ab', 'bbbaa', 'a'])))
dfa_handler.draw(file_name="dfa_diagram")