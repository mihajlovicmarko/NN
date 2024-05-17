import numpy as np
from math import factorial
import matplotlib.pyplot as plt

class ProbabilityCalculation:
    def __init__(self, k :int, n:int):
        self.k = k
        self.n = n

    def binomial_probability(self, p : float) -> float:
        n = self.n
        k = self.k
        other_divisor = n - k
        larger_divisor = np.maximum(other_divisor, k)
        smaller_divisor = np.minimum(other_divisor, k)

        comb = np.prod(np.arange(larger_divisor + 1, n + 1), dtype=np.uint64) / factorial(smaller_divisor)
        return comb * np.power(p, other_divisor) * np.power((1 - p), k)

    def calculate_distribution(self) -> list:
        probabilities = []
        for i in range(100):
            probabilities.append(self.binomial_probability(i / 100))
        self.probabilities = probabilities
        return probabilities

    def find_possible_probabilities(self, confidence_interval:float, step=0.1):
        list_of_probabilities = []
        n = self.n
        k = self.k
        not_k = n - k
        direction = 1
        expected_probability = not_k / n
        
        distance = 0
        area_under_curve = 0

        for i in range(int(1 / step)):
            area_under_curve += self.binomial_probability(i * step)
        
        cumulative_sum = 0
        iteration = 0
        
        while cumulative_sum < confidence_interval * area_under_curve:
            probability = expected_probability + distance * direction
            if 0 < probability < 1:
                cumulative_sum += self.binomial_probability(probability)
            direction *= -1
            if direction < 0:
                distance += step
            iteration += 1
            list_of_probabilities.append(probability)
        
        self.probability = list_of_probabilities[-2:]
        self.relative_area = cumulative_sum / area_under_curve
        self.distance = distance

    def display(self):
        plt.plot(self.probabilities)
        plt.show()

def main():
    obj = ProbabilityCalculation(5, 10)
    obj.calculate_distribution()
    obj.find_possible_probabilities(0.3, step=0.01)
    obj.display()
    print(obj.probability)

if __name__ == "__main__":
    main()
