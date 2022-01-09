from tools import read_json

def calculate_mean(values):
    """
    Calculate the mean of a list of values.
    """
    return sum(values) / len(values)

def calculate_squared_deviation(values, mean):
    """
    Calculate the squared deviation of a list of values.
    """
    return [(value - mean)**2 for value in values]

def calculate_variance(values):
    """
    Calculate the variance of a list of values.
    """
    mean = calculate_mean(values)
    deviations = calculate_squared_deviation(values, mean)
    return calculate_mean(deviations)

def calculate_standard_deviation(values):
    """
    Calculate the standard deviation of a list of values.
    """
    return calculate_variance(values)**0.5

def main():
    data = read_json('sums.json')
    for dataset in list(data.keys()):
        print(f'{dataset}')
        for position in list(data[dataset].keys()):
            print(f'{position}')
            standart_deviation = calculate_standard_deviation(data[dataset][position])
            print(f'{standart_deviation}')

main()