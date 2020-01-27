# backpropIJK.py
# backpropIJK is an IxJxK-layer feed-forward back propagation neural net 
# (input X - hidden Z - output Y)
##
# usage: 
# whitepaper-backprop> py backpropIJK.py I J K [iterations=1, [diagnostics=False]]


# dependencies
import numpy as np
import sys



# activation sigma-function and its derivative 
# sigma f
# arg is real number x
def sigma(x):
    return 1.0/(1.0 + np.exp(-x))

# dsigma f - derivative of sigma - slope of tangent to point on sigma curve
# arg is value on sigma curve, i.e. sigma = sigma(x)
def dsigma(sigma):
    return sigma*(1.0 - sigma)



# model-building w0 weights training function 
def action(I=2, J=2, K=2, iterations=1, diag=False):
    print("action: I = " + str(I))
    print("action: J = " + str(J))
    print("action: K = " + str(K))
    print("action: iterations = " + str(iterations) + " diag = " + str(diag))

    ## variables
    # input data - constant
    i = np.zeros(I)

    # network pre-sigma input to hidden layer
    ih = np.zeros(J)

    # post-sigma hidden layer units
    h = np.zeros(J)

    # network pre-sigma input to output layer
    ho = np.zeros(K)

    # post-sigma output layer units
    o = np.zeros(K)

    # desired 'goal' output of network
    target = np.zeros(K)

    # error for each network output: E[k] = .5*(target[k] - o[k])**2
    E = np.zeros(K)

    # Total error = sum of K[k] for all k in {0,...,K-1}
    Etotal = 0.0
    
    # weights input->hidden
    wih = np.zeros((I,J))

    # weights hidden->output
    who = np.zeros((J,K))
    # previous iteration - weights hidden->output
    pwho = np.zeros((J,K))


    ## initializations:
    # set seed for np.random so random values are 'repeatable' for consistency
    np.random.seed(1)

    # learning rate alpha
    alpha = 0.5

    # inputs, biases, weights, hidden units, output units, output-targets
    if I==J==K==2:
        print("\nI==J==K=2 => @@@@ same numeric result as backprop222.py:")
        # constant input dataset and input bias term
        i[0] = .05
        i[1] = .1
        bi = .35

        # variable weights input -> hidden
        wih[0][0] = .15
        wih[1][0] = .2
        wih[0][1] = .25
        wih[1][1] = .3
        
        # constant hidden bias
        bh = .6
        # variable weights hidden -> output
        who[0][0] = .4
        who[1][0] = .45
        who[0][1] = .5
        who[1][1] = .55
    
        # constant target output 
        target[0] = .01
        target[1] = .99

    else:
        # inputs
        for _i in range(I):
            i[_i] = np.random.random()
    
        # bias to hidden units
        bi = np.random.random()
    
        # weights wih input->hidden
        for _i in range(I):
            for _j in range(J):
                wih[_i][_j] = np.random.random()
    
        # bias to output units
        bh = np.random.random()
    
        # weights who hidden->output
        for _j in range(J):
            for _k in range(K):
                who[_j][_k] = np.random.random()
    
        # target outputs - 'goal'
        for _k in range(K):
            target[_k] = np.random.random()

    
    
    # input, output (same for all training iterations) diagnostics
    print("\nconstant data input for each training iteration")
    print(i)
    print("\nbias to hidden units bi:")
    print(bi)
    print("bias to output units bh:")
    print(bh)
    print("\nweights input->hidden units wihij:")
    print(wih)
    print("\nweights hidden->output units whojk:")
    print(who)
    print("\nconstant target output")
    print(target)

    print("-------------------------------------------------------")
    

 

    # training iteration
    for cycle in range(int(iterations)):
        if diag == True:
            print("*** iteration = " + str(cycle) + "\n")
            print("@@@ forward propagation:")

        ### forward propagation
        # ih1 = i1 * wih11 + i2 * wih21 + 1 * bi
        # ih2 = i1 * wih12 + i2 * wih22 + 1 * bi
        for _j in range(J):
            ih[_j] = 1 * bi
            for _i in range(I):           
                ih[_j] += i[_i]* wih[_i][_j]

        if diag == True:
            for _i in range(I):
                print("i[" + str(_i) + "] = " + str(i[_i]))

        if diag == True:
            for _j in range(J):
                print("ih[" + str(_j) + "] = " + str(ih[_j]))

        # h1 = sigma(ih1)
        # h2 = sigma(ih2)
        for _j in range(J):
            h[_j] = sigma(ih[_j])

        if diag == True:
            for _j in range(J):
                print("h[" + str(_j) + "] = " + str(h[_j]))

        
        # ho1 = h1 * who11 + h2 * who21 + 1 * bh
        # ho2 = h1 * who12 + h2 * who22 + 1 * bh
        for _k in range(K):
            ho[_k] = 1 * bh
            for _j in range(J):           
                ho[_k] += h[_j]* who[_j][_k]

        if diag == True:
            for _k in range(K):
                print("ho[" + str(_k) + "] = " + str(ho[_k]))

        # o1 = sigma(ho1)
        # o2 = sigma(ho2)
        for _k in range(K):
            o[_k] = sigma(ho[_k])

        if diag == True:
            print("\noutput:")
            for _k in range(K):
                print("o[" + str(_k) + "] = " + str(o[_k]))
            for _k in range(K):
                print("target[" + str(_k) + "] = " + str(target[_k]))


        # E1 = .5*(target1 - o1)**2
        # E2 = .5*(target2 - o2)**2
        for _k in range(K):
            E[_k] = .5*(target[_k] - o[_k])**2
        # E = E1 + E2
        Etotal = 0.0
        for _k in range(K):
            Etotal += E[_k]

        if diag == True:
            print("\nerror:")
            for _k in range(K):
                print("E[" + str(_k) + "] = " + str(E[_k]))
            print('Etotal = ' +str(Etotal))







         
        ### back propagation
        # [1] adjust the weights who{j][k] j,k in {J,K}:
        # Let PDX_Y denote the partial derivative of X with respect to Y,
        # where X and Y may each (or only Y) have indices such as: PDX_Y[i][j]
        # meaning the partial derivative of X[i] with respect to Y[j]. 
        #
        # Our Goal is to find the direction in the space of all weights wihij
        # and the space of all weights whojk which when subtracted from each
        # reduces the errors E[k] = (o[k]-target[k]) so that the total error 
        # Etotal = E[0] + ... + E[K] is reduced and hopefully approaches zero.
        #
        # E is reduced when who[j][k] += - alpha* PDE_who[j][k] for j,k in {J,K}
        # E is also reduced when wih[i][j] += - alpha*PDE_who[i][j] i,j in {I,J}
        # PDE_whojk is also called the gradient of E in the whojk direction,
        # and PDE_wihij is called the gradient of E in the whoij direction.
        # The method of adjusting the weights described above is called
        # 'gradient descent' or 'steepest descent'
        #
        # The method updates each weight in the network so that the next forward
        # propagation produces an output [o[0],..,o[K-1]] which reduces each
        # error/cost E[k] = target[k]-o[k] and hence reduces the total 
        # error/cost Etotal = E[0] + ... + E[K-1]
        #
        # Forward propagation moved from inputs to hidden units to outputs.
        # Backward propagation moves in reverse from output errors to 
        # hidden->output weights and then to input->hidden weights.
        #
        # First we want to determine the gradients PDE_whojk and adjust the
        # hidden->output weights whojk
        if diag == True:
            print("\n\n@@@ back propagation:")
        for _j in range(J):
            for _k in range(K):
                # j=1, k=1 => gradient = (o1 - target1) * dsigma(o1) * h1
                gradient = (o[_k] - target[_k]) * dsigma(o[_k]) * h[_j]
                # pwho11 = who11
                # who11 = who11 - alpha * gradient
                # save prev. iteration who[_j][-k] for use in wih adjustment
                pwho[_j][_k] = who[_j][_k]
                who[_j][_k] += -alpha * gradient
                if diag == True:
                    print("hidden->output weight adjustments")
                    print("o["+str(_k)+"]-target["+str(_k)+"] = " + str(o[_k]-target[_k]))
                    print("o["+str(_k)+"] = " + str(o[_k]))
                    print("dsigma(o["+str(_k)+"] = " + str(dsigma(o[_k])))
                    print("h["+str(_j)+"] = " + str(h[_j]))
                    print("ho-gradient is " + str(gradient))
                    print("new who["+str(_j)+"]["+str(_k)+"] = " + str(who[_j][_k]))
                    print("\n")


        # [2] adjust the weights wih[i][j] i.j in {J,K}:
        # RECALL:
        # Let PDX_Y denote the partial derivative of X with respect to Y,
        # where X and Y may each (or only Y) have indices such as: PDX_Y[i][j]
        # meaning the partial derivative of X[i] with respect to Y[j]. 
        #
        # Our Goal is to find the direction in the space of all weights wihij
        # and the space of all weights whojk which when subtracted from each
        # reduces the errors E[k] = (o[k]-target[k]) so that the total error 
        # Etotal = E[0] + ... + E[K] is reduced and hopefully approaches zero.
        #
        # E is reduced when wih[i][j] += - alpha* PDE_who[i][j] for i,j in {I,J}
        # E is also reduced when who[j][k] += - alpha*PDE_who[j][k] j,k in {J,K}
        # PDE_whojk is also called the gradient of E in the whojk direction,
        # and PDE_wihij is called the gradient of E in the whoij direction.
        # The method of adjusting the weights described above is called
        # 'gradient descent' or 'steepest descent'
        #
        # Case [2] is more complicated than case [1] since the error at the 
        # hidden units, unlike the error at the output units in case [1],
        # is the sum of errors contributed by each output unit (o[k] k in K) 
        # In [2] the full chain rule product sequence must be computed for
        # each wih[i][j] unlike the simpler three part chain rule in part [1]
        #
        # We want to determine the gradients PDE_wih[i][j] and adjust the
        # input->hidden weights wihij = wihij - alpha * PDE_whihj i,j in IxJ
        PDE_h = np.zeros((K,J))
        PDEtotal_h = np.zeros(J)
        for _i in range(I):
            for _j in range(J):
                for _k in range(K):
                    PDE_h[_k][_j] = (o[_k]-target[_k])*dsigma(o[_k])*pwho[_j][_k] 
                    PDEtotal_h[_j] += PDE_h[_k][_j] 
                gradient = PDEtotal_h[_j] * dsigma(h[_j]) * i[_i]
                wih[_i][_j] += -alpha * gradient

        if diag == True:
            print("input->hidden weight adjustments")
            for _i in range(I):
                for _j in range(J):
                    for _k in range(K):
                        print("o["+str(_k)+"]-target["+str(_k)+"] = " + str(o[_k]-target[_k]))
                        print("dsigma(o["+str(_k)+"]) = "+str(dsigma(o[_k])))
                        print("pwho["+str(_j)+"]["+str(_k)+"] = " + str(pwho[_j][_k]))
                        print("PDE_h["+str(_k)+"]["+str(_j)+"] = " + str(PDE_h[_k][_j]))
                    print("PDEtotal_h["+str(_j)+"] = " + str(PDEtotal_h[_j]))
                    print("dsigma(h["+str(_j)+"]) = " + str(dsigma(h[_j])))
                    print("i["+str(_i)+"] = " + str(i[_i]))
                    print("ho-gradient is " + str(gradient))
                    print("new wih["+str(_i)+"]["+str(_j)+"] = " + str(wih[_i][_j]))
                    print("\n")

    
        if diag == True:
            print("-------------------------------------------------------")


    # final diagnostics
    print("\n@@@@@@final output:")
    for _k in range(K):
        print("o[" + str(_k) + "] = " + str(o[_k]))
    for _k in range(K):
        print("target[" + str(_k) + "] = " + str(target[_k]))

    print("\n final error:")
    for _k in range(K):
        print("E[" + str(_k) + "] = " + str(E[_k]))
    print('Etotal = ' +str(Etotal))
    print("-------------------------------------------------------")






if __name__ == "__main__": 
    print('\n+++++++++++ backpropIJK.py +++++++++++++++++++++')
    print("backpropIJK.py running as __main__")
    nargs = len(sys.argv) - 1
    position = 1
    iterations = 1
    diagnostics = False
    I = 2
    J = 2
    K = 2

    print("nargs = " + str(nargs))
    while nargs >= position:
        #print('backprop222: sys.argv[' + str(position) + '] = ' + str(sys.argv[position]))
        position += 1

    if nargs == 5:
        I = int(sys.argv[1])
        J = int(sys.argv[2])
        K = int(sys.argv[3])
        iterations = int(sys.argv[4])
        s = sys.argv[5].lower()
        if(s == "false" or s == "f"):
            diagnostics = False
        else:
            diagnostics = True
    elif nargs == 1:
        I = int(sys.argv[1])
        J = int(sys.argv[2])
        K = int(sys.argv[3])
        iterations = int(sys.argv[1]) 

    action(I, J, K, iterations, diagnostics)

else:
    print("backprop222.py module imported")
