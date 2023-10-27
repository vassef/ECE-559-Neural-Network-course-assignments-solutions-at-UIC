import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Funtion to return variance
def Find_VAR(lst):
    if len(lst) > 10:
        a = np.var(lst[-10:])
    else:
        a = np.var(lst)
    return a

# Function to generate random weights w0, w1, w2
def generate_random_weights():
    w0 = np.random.uniform(-1/4, 1/4)
    w1 = np.random.uniform(-1, 1)
    w2 = np.random.uniform(-1, 1)
    return w0, w1, w2

# Function to generate random data points
def generate_random_data(n):
    data = np.random.uniform(-1, 1, size=(n, 2))
    return data

# Function to classify data points into S0 and S1
def classify_data(data, w0, w1, w2):
    S0 = []
    S1 = []
    for x in data:
        if np.dot([1, x[0], x[1]], [w0, w1, w2]) >= 0:
            S1.append(x)
        else:
            S0.append(x)
    return np.array(S0), np.array(S1)

# Perceptron training algorithm
def perceptron_train(data, learning_rate, S0, S1, w_0, w_1, w_2):

    w0 = w_0
    w1 = w_1
    w2 = w_2

    misclassifications = []

    while True:
        Init = (0 if len(misclassifications) == 0 else 1) # is a flag used to enable/disable weights' updating.
        misclassified = 0
        for x in data:
            if np.dot([1, x[0], x[1]], [w0, w1, w2]) >= 0:
                target = 1 # The data is predicted as a S1 class
            else:
                target = -1 # The data is predicted as S0 class

            if x in S0:
                label = -1
            else:
                label = 1

            if target * label == -1:
                misclassified += 1
                w0 -= learning_rate * target * Init
                w1 -= learning_rate * target * x[0] * Init
                w2 -= learning_rate * target * x[1] * Init
        misclassifications.append(misclassified)
        if misclassified == 0:
            break

    return w0, w1, w2, misclassifications

# Main function
def main(number_of_points, ShowPlot = True):

    '''
    ShowPlot is the flag we used to enable/disable plotting, depending on the function being called.

    '''
    n = number_of_points  # Number of data points
    data = generate_random_data(n) # One time generated for all etas.
    w0, w1, w2 = generate_random_weights() # One time generated for all etas.
    S0, S1 = classify_data(data, w0, w1, w2) # One time generated for all etas.

    learning_rates = [1, 10, 0.1]  # Different learning rates
    w0_pr, w1_pr, w2_pr = generate_random_weights() # Generate w'0, w'1, and w'2

    if ShowPlot == False:

        print(f'A one-time setting for the initial weights, data, and two different region')
        plt.figure()
        plt.plot(S0[:, 0], S0[:, 1], 'ro', label='S0')
        plt.plot(S1[:, 0], S1[:, 1], 'bo', label='S1')
        plt.plot([-1, 1], [-(w0 + w1*(-1)) / w2, -(w0 + w1*(1)) / w2], 'k-', label='Decision Boundary')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.title(f"Problem'initialization")
        print(f'Initial weights: w0 = {w0:.4f}, w1 = {w1:.4f}, w2 = {w2:.4f} that initialize the problem in terms of S0, and S1.')
        print(f"Initial weights: w'0 = {w0_pr:.4f}, w'1 = {w1_pr:.4f}, w'2 = {w2_pr:.4f} that are used as the reference randome weights in each scenario.")
        print("--------------------------------------------\n")

    EXP_Info = {'eta=1':[],'eta=10':[],'eta=0.1':[]}
    keys = list(EXP_Info.keys())
    for i, eta in enumerate(learning_rates):

        w0_final, w1_final, w2_final, misclassifications = perceptron_train(data, eta, S0, S1, w0_pr, w1_pr, w2_pr)

        if ShowPlot == False:


            plt.figure()
            plt.plot(S0[:, 0], S0[:, 1], 'ro', label='S0')
            plt.plot(S1[:, 0], S1[:, 1], 'bo', label='S1')
            plt.plot([-1, 1], [-(w0_final + w1_final*(-1)) / w2_final, -(w0_final + w1_final*(1)) / w2_final], 'k-', label='Decision Boundary')
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.legend()
            plt.title(f'η = {eta}, Final weights')

            plt.figure()
            plt.plot(range(0, len(misclassifications)), misclassifications, color = 'b', marker = 'o', ms = 6)
            plt.xticks(range(0, len(misclassifications),((len(misclassifications)-1)//14 + 1 if len(misclassifications)>=15 else 1)))
            plt.xlabel('Epoch')
            plt.ylabel('Misclassifications')
            plt.title(f'η = {eta}')

            print(f'η = {eta}')
            print(f'Final weights: w0 = {w0_final:.4f}, w1 = {w1_final:.4f}, w2 = {w2_final:.4f}')
            print(f'Number of epochs: {len(misclassifications)-1}\n')

        EXP_Info[keys[i]] = [len(misclassifications) - 1, np.mean(np.diff(misclassifications[1:]))]
        plt.show()

    return EXP_Info

def Experiment(num_exp, num_samples):

    Epoch_Info = {'eta=1':[],'eta=10':[],'eta=0.1':[]}
    Speed_Info = {'eta=1':[],'eta=10':[],'eta=0.1':[]}

    Confirm = input(f"The following function is going to call the main function for {num_exp} times for {num_samples} samples, so it may cause computational time in your system, \
     you can find the generated output in the report. Nevertheless, do want to proceed? (Answer Y as yes, and N as no!): ")
    while Confirm not in ["Y", "N"]:
        print("Invalid input!")
        confirm = input("try again using valid input!")

    if Confirm == "Y":

        for i in range(num_exp):
            EXP_Info = main(num_samples)

            Epoch_Info['eta=1'].append(EXP_Info['eta=1'][0])
            Speed_Info['eta=1'].append(EXP_Info['eta=1'][1])

            Epoch_Info['eta=10'].append(EXP_Info['eta=10'][0])
            Speed_Info['eta=10'].append(EXP_Info['eta=10'][1])

            Epoch_Info['eta=0.1'].append(EXP_Info['eta=0.1'][0])
            Speed_Info['eta=0.1'].append(EXP_Info['eta=0.1'][1])


        plt.figure()
        plt.plot(range(1,len(Epoch_Info['eta=1'])+1), Epoch_Info['eta=1'], color = 'b', marker = 'o', ms = 6)
        plt.plot(range(1,len(Epoch_Info['eta=1'])+1), Epoch_Info['eta=10'], color = 'g', marker = 'o', ms = 6)
        plt.plot(range(1,len(Epoch_Info['eta=1'])+1), Epoch_Info['eta=0.1'], color = 'r', marker = 'o', ms = 6)
        plt.xticks(range(1,num_exp + 1, (num_exp-1)//19))
        plt.xlabel('Experiment Number')
        plt.ylabel('Epcohs')
        plt.title(f'(Num. of epochs) vs (Experiment number)')
        plt.legend(['eta=1', 'eta=10', 'eta=0.1'])

        plt.figure()
        plt.plot(range(1,len(Epoch_Info['eta=1'])+1), Speed_Info['eta=1'], color = 'b', marker = 'o', ms = 6)
        plt.plot(range(1,len(Epoch_Info['eta=1'])+1), Speed_Info['eta=10'], color = 'g', marker = 'o', ms = 6)
        plt.plot(range(1,len(Epoch_Info['eta=1'])+1), Speed_Info['eta=0.1'], color = 'r', marker = 'o', ms = 6)
        plt.xticks(range(1,num_exp + 1, int(num_exp / 20)))
        plt.xlabel('Experiment Number')
        plt.ylabel('Average decay speed')
        plt.title(f'(Speed of decaying) vs (Experiment number)')
        plt.legend(['eta=1', 'eta=10', 'eta=0.1'])

        plt.show()


        return Epoch_Info, Speed_Info

    else:

        print("Nothing executed!")
        return None, None

EXP_Info =  main(100, False)
print('\n\n\n\n\n\n----- End of one-time experiment with 100 samples -----\n\n\n\n\n\n')
EXP_Info =  main(1000, False)
print('\n\n\n\n\n\n----- End of one-time experiment with 1000 samples -----\n\n\n\n\n\n')
Epoch_Info_100, Speed_Info_100 = Experiment(100, 100)
print('\n\n\n\n\n\n----- End of 100-times experiment with 100 samples -----\n\n\n\n\n\n')
Epoch_Info_1000, Speed_Info_1000 = Experiment(100, 1000)
print('\n\n\n\n\n\n----- End of 100-times experiment with 1000 samples -----\n\n\n\n\n\n')
