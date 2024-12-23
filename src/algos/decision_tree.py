import numpy as np

class _Node:
    def __init__(self):
        self.attribute : int = None
        self.children : dict[int, _Node] = {}
        self.classification : int = None

class DecisionTreeClassifier:
    def __init__(self, attributes : dict[int, set[int]], range : set[int]):
        self._tree : _Node = None
        self._attributes = attributes
        self._range = range


    def fit(self, X, y):
        examples = [example for example in zip(X, y)]

        self._tree = self._build_tree(
            examples,
            set(self._attributes.keys()),
            examples
        )

        return self
    

    def predict(self, X) -> list[int]:
        prediction = []

        for input in X:
            node = self._tree
            while node.classification == None:
                value = input[node.attribute]
                node = node.children[value]
            prediction.append(node.classification)

        return prediction

    
    def _build_tree(
            self,
            examples : list[tuple[list[int], int]], 
            attributes : set[int], 
            parent_examples : list[list[int]]
        ) -> _Node:
        tree = _Node()

        example_outputs : set[int] = set()
        for _, example_output in examples:
            example_outputs.add(example_output)

        # Handle edge cases
        if len(examples) == 0:
            tree.classification = self._plurality_output(parent_examples)
            return tree
        elif len(example_outputs) == 1:
            tree.classification = example_outputs.pop()
            return tree
        elif len(attributes) == 0:
            tree.classification = self._plurality_output(parent_examples)
            return tree
        
        top_attribute = self._get_top_attribute(attributes, examples)
        tree.attribute = top_attribute

        for value in self._attributes[top_attribute]:
            examples_subset = self._get_subset(top_attribute, value, examples)
            
            attributes_subset = attributes.copy()
            attributes_subset.remove(top_attribute)

            tree.children[value] = self._build_tree(
                examples_subset, 
                attributes_subset, 
                examples
            )
            
        return tree
    

    def _plurality_output(self, examples : list[tuple[list[int], int]]) -> int:
        output_count = self._get_output_count(examples)
        
        max_output = -1
        max_count = -1
        for output, count in output_count.items():
            if count > max_count:
                max_count = count
                max_output = output

        return max_output
    

    def _get_top_attribute(self, attributes : set[int], examples : list[tuple[list[int], int]]) -> int:
        # Find the must important attribute
        top_attribute : int
        top_importance = -1
        for attribute in attributes:
            current_importance = self._importance(attribute, examples)
            if current_importance > top_importance:
                top_importance = current_importance
                top_attribute = attribute
        return top_attribute
    

    def _importance(self, attribute : int, examples : list[tuple[list[int], int]]) -> float:
        parent_entropy = self._entropy(self._get_probabilities(examples))
        children_weighted_entropy = 0

        total = len(examples)
        for value in self._attributes[attribute]:
            subset = self._get_subset(attribute, value, examples)
            probabilities = self._get_probabilities(subset)
            entropy = self._entropy(probabilities)
            children_weighted_entropy += (len(subset) / total) * entropy

        return parent_entropy - children_weighted_entropy


    def _get_probabilities(self, examples : list[tuple[list[int], int]]) -> list[float]:
        total = len(examples)
        output_count = self._get_output_count(examples)
        return [count / total for count in output_count.values()]


    def _entropy(self, probabilities : list[float]) -> float:
        sum = 0
        for p in probabilities:
            sum += p * np.log2(p)
        return -sum
    

    def _get_output_count(self, examples : list[tuple[list[int], int]]) -> dict[int, int]:
        output_count : dict[int, int] = {}
        for _, example_output in examples:
            if example_output in output_count:
                output_count[example_output] += 1
            else:
                output_count[example_output] = 1
        return output_count
    

    def _get_subset(
            self, 
            attribute : int, 
            value : int, 
            examples : list[tuple[list[int], int]]
        ) -> list[tuple[list[int], int]]:

        examples_subset : list[tuple[list[int], int]]= []
        for example in examples:
            example_input, _ = example
            if example_input[attribute] == value:
                examples_subset.append(example)
        return examples_subset