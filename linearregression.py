import pandas as pd
import argparse
import os

"""
Reading and parsing arguments passed from command line
Arguments : 
        data
        learningRate
        threshold
"""
parser = argparse.ArgumentParser()
parser.add_argument("--data")
parser.add_argument("--learningRate")
parser.add_argument("--threshold")
args = parser.parse_args()
print("data: {}".format(args.data))
print("learningRate: {}".format(args.learningRate))
print("threshold: {}".format(args.threshold))

# Parsed input data
input_file = args.data
threshold_value = float(args.threshold)
learning_rate = float(args.learningRate)

print(os.path.dirname(os.path.abspath(input_file)))

# Reading random.csv file in PyCharm
# data = pd.read_csv("random.csv", header=None)
# Reading yacht.csv file in PyCharm
# data = pd.read_csv("yacht.csv", header=None)
# Reading input file from Terminal using Pandas library
data = pd.read_csv(input_file, header=None)
# print(data)


count = 0


"""
Gradient function to calculate the derivatives of the features and calculate the gradient for 
updating the weights and linear function.
"""


def gradient_function(d, x0=1):
    dw0 = 0
    dw1 = 0
    dw2 = 0
    dw3 = 0
    dw4 = 0
    dw5 = 0
    dw6 = 0
    if len(d) == 7:
        for (x1, x2, yi) in zip(d['x1'], d['x2'], d['y']):
            dw0 += (yi-(update_linear_function(d, d['w0'], d['w1'], d['w2'], x1, x2, x3=0, x4=0, x5=0, x6=0, w3=0,
                                               w4=0, w5=0, w6=0, x0=1)))*x0
            dw1 += (yi-(update_linear_function(d, d['w0'], d['w1'], d['w2'], x1, x2, x3=0, x4=0, x5=0, x6=0, w3=0,
                                               w4=0, w5=0, w6=0, x0=1)))*x1
            dw2 += (yi-(update_linear_function(d, d['w0'], d['w1'], d['w2'], x1, x2, x3=0, x4=0, x5=0, x6=0, w3=0,
                                               w4=0, w5=0, w6=0, x0=1)))*x2
    elif len(d) == 15:
        for (x1, x2, x3, x4, x5, x6, yi) in zip(d['x1'], d['x2'], d['x3'], d['x4'], d['x5'], d['x6'], d['y']):
            dw0 += (yi - (update_linear_function(d, d['w0'], d['w1'], d['w2'], x1, x2, x3, x4, x5, x6,  d['w3'],
                                                 d['w4'], d['w5'],d['w6'], x0=1))) * x0
            dw1 += (yi - (update_linear_function(d, d['w0'], d['w1'], d['w2'], x1, x2, x3, x4, x5, x6,  d['w3'],
                                                 d['w4'], d['w5'],d['w6'], x0=1))) * x1
            dw2 += (yi - (update_linear_function(d, d['w0'], d['w1'], d['w2'], x1, x2, x3, x4, x5, x6,  d['w3'],
                                                 d['w4'], d['w5'],d['w6'], x0=1))) * x2
            dw3 += (yi - (update_linear_function(d, d['w0'], d['w1'], d['w2'], x1, x2, x3, x4, x5, x6,  d['w3'],
                                                 d['w4'], d['w5'],d['w6'], x0=1))) * x3
            dw4 += (yi - (update_linear_function(d, d['w0'], d['w1'], d['w2'], x1, x2, x3, x4, x5, x6,  d['w3'],
                                                 d['w4'], d['w5'],d['w6'], x0=1))) * x4
            dw5 += (yi - (update_linear_function(d, d['w0'], d['w1'], d['w2'], x1, x2, x3, x4, x5, x6,  d['w3'],
                                                 d['w4'], d['w5'],d['w6'], x0=1))) * x5
            dw6 += (yi - (update_linear_function(d, d['w0'], d['w1'], d['w2'], x1, x2, x3, x4, x5, x6,  d['w3'],
                                                 d['w4'], d['w5'],d['w6'], x0=1))) * x6
    return dw0, dw1, dw2, dw3, dw4, dw5, dw6


"""
Function to calculate the SSE
"""


def sum_of_squared_errors(d):
    squared_error = 0
    if len(d) == 7:
        for (x1, x2, yi) in zip(d['x1'], d['x2'], d['y']):
            squared_error += ((update_linear_function(d, d['w0'], d['w1'], d['w2'], x1, x2, x3=0, x4=0, x5=0, x6=0,
                                                      w3=0, w4=0, w5=0, w6=0, x0=1) - yi)**2)
    elif len(d) == 15:
        for (x1, x2, x3, x4, x5, x6, yi) in zip(d['x1'], d['x2'], d['x3'], d['x4'], d['x5'], d['x6'], d['y']):
            squared_error += ((update_linear_function(d, d['w0'], d['w1'], d['w2'], x1, x2, x3, x4, x5, x6,  d['w3'],
                                                      d['w4'], d['w5'], d['w6'], x0=1) - yi)**2)
    return squared_error


"""
Function to update the weights after each iterations.
input: 
    Directory d containing values provided by input file
    learning_rate provided through command line.
"""


def update_weight(d, learning_rate):
    if len(d) == 7:
        dw0, dw1, dw2, dw3, dw4, dw5, dw6 = gradient_function(d, x0=1)
        d['w0'] = d['w0'] + (learning_rate * dw0)
        d['w1'] = d['w1'] + (learning_rate * dw1)
        d['w2'] = d['w2'] + (learning_rate * dw2)
    elif len(d) == 15:
        dw0, dw1, dw2, dw3, dw4, dw5, dw6 = gradient_function(d, x0=1)
        d['w0'] = d['w0'] + (learning_rate * dw0)
        d['w1'] = d['w1'] + (learning_rate * dw1)
        d['w2'] = d['w2'] + (learning_rate * dw2)
        d['w3'] = d['w3'] + (learning_rate * dw3)
        d['w4'] = d['w4'] + (learning_rate * dw4)
        d['w5'] = d['w5'] + (learning_rate * dw5)
        d['w6'] = d['w6'] + (learning_rate * dw6)
    return d


"""
Function to update the linear function.
Input:
    Directory d containing values provided by input file
    x values
    weight w values
"""


def update_linear_function(d, w0, w1, w2, x1, x2, x3, x4, x5, x6, w3, w4, w5, w6, x0=1):
    if len(d) == 7:
        return w0*x0 + w1*x1 + w2*x2
    elif len(d) == 15:
        return w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6


"""
Function to create the output file structure to be used by df.to_csv method
"""


def create_data_frame_columns(d, iteration_no, error_value):
    if len(d) == 7:
        return {'iteration': iteration_no, 'w0': [format(d['w0'], '.4f')], 'w1': [format(d['w1'], '.4f')],
                'w2': [format(d['w2'], '.4f')], 'error': [format(error_value, '.4f')]}
    elif len(d) == 15:
        return {'iteration': iteration_no, 'w0': [format(d['w0'], '.4f')], 'w1': [format(d['w1'], '.4f')],
                'w2': [format(d['w2'], '.4f')], 'w3': [format(d['w3'], '.4f')], 'w4': [format(d['w4'], '.4f')],
                'w5': [format(d['w5'], '.4f')], 'w6': [format(d['w6'], '.4f')], 'error': [format(error_value, '.4f')]}


# Initializing directory with x0. Value of x0 is always 1
d = {'x0': 1}
# Directory for storing variable length weight values
d_weight = {}
# Count variable for creating different sizes of x, y and w present in input file
col_count = len(data.count())
for i in data:
    if i+1 == col_count:
        d['y'] = data[i]
        d_weight['w{}'.format(i)] = 0
    else:
        d['x{}'.format(i + 1)] = data[i]
        d_weight['w{}'.format(i)] = 0
    d.update(d_weight)


# Error present in previous iteration. This is used to check if the change in error is below provided threshold value
prev_error = sum_of_squared_errors(d)
itr_count = 0
if len(d) == 7:
    df_final = pd.DataFrame(create_data_frame_columns(d, itr_count, prev_error), columns=['iteration', 'w0', 'w1', 'w2', 'error'])
elif len(d) == 15:
    df_final = pd.DataFrame(create_data_frame_columns(d, itr_count, prev_error), columns=['iteration', 'w0', 'w1', 'w2', 'w3',
                                                                                  'w4', 'w5', 'w6', 'error'])
# print(df_final)
# Main loop to process all the data present in the input file.
iteration_=0
print("Gradient descent started")
while True:
    dw0, dw1, dw2, dw3, dw4, dw5, dw6 = gradient_function(d, x0=1)
    d = update_weight(d, learning_rate)
    error = sum_of_squared_errors(d)
    itr_count += 1
    if len(d) == 7:
        df = pd.DataFrame(create_data_frame_columns(d, itr_count, error), columns=['iteration', 'w0', 'w1', 'w2',
                                                                                   'error'])
    elif len(d) == 15:
        df = pd.DataFrame(create_data_frame_columns(d, itr_count, error), columns=['iteration', 'w0', 'w1', 'w2', 'w3',
                                                                                   'w4', 'w5', 'w6', 'error'])
    df_final = df_final.append(df)

    if prev_error - error < threshold_value:
        break
    prev_error = error
    iteration_+=1
    if iteration_%100==0:
        print("No of iteration completed: "+str(iteration_))

print("Iteration Completed")

# Creating csv file using DataFrame of Pandas library
df_final.to_csv("{}_output.csv".format((args.data)[:args.data.index('.')]), header=False, index=False)
print(df_final)
