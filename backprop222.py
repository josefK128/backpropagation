# backprop222.py
# backprop222 3-layer feed-forward back propagation neural net corresponding
# to 'Backpropagation Algorithm' whitepaper analysis text:
# (input X - hidden Z - output Y)
##
# usage: 
# whitepaper-backprop> py backprop222.py [iterations=4, [diagnostics=True]]
# realistic usage: 
# whitepaper-backprop> py backprop222.py 1000 False
#
# diagram:
#
#  [i1]  wih11  [h1]  who11  [o1]   target1  E1
#            
#        wih12        who12
#                                                 E
#        wih21        who21
#    
#  [i2]  wih22  [h2]  who22  [o2]   target2  E2
# 
#  [1]   bi     [1]   bh           
#
#
# variables:
# input layer units
# i1     
# i2
# bi      input bias term
#
# weights input layer -> hidden layer
# wih11   weight for i1->h1
# wih12   weight for i1->h2
# wih21   weight for i2->h1
# wih22   weight for i2->h2
#
# hidden layer units - input*weight before activation function sigma
# ih1     i1*wih11 + 12*wih21
# ih2     i1*wih12 + 12*wih22
# bh      hidden layer bias term
#
# hidden layer units - post activation function sigma
# h1      sigma(ih1)
# h2      sigma(ih2)
# bh      hidden layer bias term
# 
# weights hidden -> output
# who11   weight for h1->o1
# who12   weight for h1->o2
# who21   weight for h2->o1
# who22   weight for h2->o2
#
# output layer units - hidden*weight before activation function sigma
# ho1     h1*who11 + h2*who21
# ho2     h1*who12 + h2*who22
#
# output layer units - post activation function sigma
# o1      sigma(ho1)
# o2      sigma(ho2)
# 
# target 'goal' output values for o1, o2
# target1
# target2
#
# Error terms - E is the total error (also referred to as total 'cost')
# E1      .5*(target1 - o1)**2
# E2      .5*(target2 - o1)**2
# E       E1 + E2


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
def action(iterations=1, diag=False):
    print("action: iterations = " + str(iterations) + " diag = " + str(diag))

    # initializations:
    # constant input dataset and input bias term
    i1 = .05
    i2 = .1
    bi = .35
    # variable weights input -> hidden
    wih11 = .15
    wih21 = .2
    wih12 = .25
    wih22 = .3
    
    # constant hidden bias
    bh = .6
    # variable weights hidden -> output
    who11 = .4
    who21 = .45
    who12 = .5
    who22 = .55

    # constant target output 
    target1 = .01
    target2 = .99
    
    
    # input, output (same for all training iterations) diagnostics
    print("\nconstant data input for each training iteration) (2x1)")
    print(np.array([[i1],[i2]]))
    print("constant target output (2x1)")
    print(np.array([[target1],[target2]]))

    # set seed for np.random so random values are 'repeatable' for consistency
    np.random.seed(1)

    # learning rate alpha
    alpha = 0.5

    print("-------------------------------------------------------")
    

 

    # training iteration
    for i in range(int(iterations)):
        if diag == True:
            print("*** iteration = " + str(i) + "\n")

        ### forward propagation
        ih1 = i1 * wih11 + i2 * wih21 + 1 * bi
        ih2 = i1 * wih12 + i2 * wih22 + 1 * bi
        h1 = sigma(ih1)
        h2 = sigma(ih2)
        
        ho1 = h1 * who11 + h2 * who21 + 1 * bh
        ho2 = h1 * who12 + h2 * who22 + 1 * bh
        o1 = sigma(ho1)
        o2 = sigma(ho2)

        E1 = .5*(target1 - o1)**2
        E2 = .5*(target2 - o2)**2
        E = E1 + E2

        if diag == True:
            print("forward propagation::")
            print("input:")
            print('i1 = ' + str(i1))
            print('i2 = ' + str(i2))
            print("\nhidden:")
            print('ih1 = ' + str(ih1))
            print('ih2 = ' + str(ih2))
            print('h1 = ' + str(h1))
            print('h2 = ' + str(h2))
            print("\noutput:")
            print('ho1 = ' + str(ho1))
            print('ho2 = ' + str(ho2))
            print('o1 = ' + str(o1))
            print('o2 = ' + str(o2))
            print("\nerror:")
            print('E1 = ' +str(E1))
            print('E2 = ' +str(E2))
            print('E = ' +str(E))



         
        ### back propagation
        # [1] adjust the weights whojk j,k in {1,2}:
        # Let PDX_Y denote the partial derivative of X with respect to Y.
        #
        # Our Goal is to find the direction in the space of all weights wihij
        # and the space of all weights whojk which when subtracted from each
        # reduces the errors E1 = (o1-target1) and E2 = (o2-target2) so that
        # the total error E = E1 + E2 is reduced and hopefully approaches zero.
        # E is reduced when whojk = whojk - alpha * PDE_whojk for j,k in {1,2}
        # Also, E is reduced when wihij = wihij - alpha * PDE_whoij i,j in {1,2}
        # PDE_whojk is also called the gradient of E in the whojk direction,
        # and PDE_wihij is called the gradient of E in the whoij direction.
        # The method of adjusting the weights described above is called
        # 'gradient descent' or 'steepest descent'
        #
        # The method updates each weight in the network so that the next forward
        # propagation produces an output [o1,o2] which reduces each error/cost 
        # E1 = target1-o1 and E2 = target2-o2, and hence reduces the total 
        # error/cost E = E1 + E2
        #
        # Forward propagation moved from inputs to hidden units to outputs.
        # Backward propagation moves in reverse from output errors to 
        # hidden->output weights and then to input->hidden weights.
        #
        # First we want to determine the gradients PDE_whojk and adjust the
        # hidden->output weights whojk
        # We proceed case by case through j,k in {1,2}
        # j=1, k=1
        gradient = (o1 - target1) * dsigma(o1) * h1
        pwho11 = who11
        who11 = who11 - alpha * gradient
        if diag == True:
            print("\n**** back propagation for hidden->output weights whojk:")
            print("\no1-target1 = " + str(o1-target1))
            print("o1 = " + str(o1))
            print("dsigma(o1) = " + str(dsigma(o1)))
            print("h1 = " + str(h1))
            print("ho-gradient 1,1 is " + str(gradient))
            print("prev weight who11 is " + str(pwho11))
            print("prev weight who11 is " + str(pwho11))
            print("@@@@new weight who11 is " + str(who11))

        # j=2, k=1
        gradient = (o1 - target1) * dsigma(o1) * h1
        pwho21 = who21
        who21 = who21 - alpha * gradient
        if diag == True:
            print("\no1-target1 = " + str(o1-target1))
            print("o1 = " + str(o1))
            print("dsigma(o1) = " + str(dsigma(o1)))
            print("h1 = " + str(h1))
            print("ho-gradient 2,1 is " + str(gradient))
            print("prev weight who21 is " + str(pwho21))
            print("@@@@new weight who21 is " + str(who21))

        # j=1, k=2
        gradient = (o2 - target2) * dsigma(o2) * h2
        pwho12 = who12
        who12 = who12 - alpha * gradient
        if diag == True:
            print("\no2-target2 = " + str(o2-target2))
            print("o2 = " + str(o2))
            print("dsigma(o2) = " + str(dsigma(o2)))
            print("h2 = " + str(h2))
            print("ho-gradient 1,2 is " + str(gradient))
            print("prev weight who12 is " + str(pwho12))
            print("@@@@new weight who12 is " + str(who12))


        # j=2, k=2
        gradient = (o2 - target2) * dsigma(o2) * h2
        pwho22 = who22
        who22 = who22 - alpha * gradient
        if diag == True:
            print("\no1-target1 = " + str(o1-target1))
            print("o2 = " + str(o2))
            print("dsigma(o2) = " + str(dsigma(o2)))
            print("h2 = " + str(h2))
            print("ho-gradient 2,2 is " + str(gradient))
            print("prev weight who22 is " + str(pwho22))
            print("@@@@new weight who22 is " + str(who22))



        # [2] adjust the weights wihij i.j in {1,2}:
        # RECALL:
        # Let PDX_Y denote the partial derivative of X with respect to Y.
        #
        # Our Goal is to find the direction in the space of all weights wihij
        # and the space of all weights whojk which when subtracted from each
        # reduces the errors E1 = (o1-target1) and E2 = (o2-target2) so that
        # the total error E = E1 + E2 is reduced and hopefully approaches zero.
        # E is reduced when whojk = whojk - alpha * PDE_whojk for j,k in {1,2}
        # Also, E is reduced when wihij = wihij - alpha * PDE_whoij i,j in {1,2}
        # PDE_whojk is also called the gradient of E in the whojk direction,
        # and PDE_wihij is called the gradient of E in the whoij direction.
        # The method of adjusting the weights described above is called
        # 'gradient descent' or 'steepest descent'
        #
        # Case [2] is more complicated than case [1] since the error at the 
        # hidden units, unlike the error at the output units in case [1],
        # is the sum of errors contributed by each output unit (o1 and o2 in
        # this very simple network
        # In [2] the full chain rule product sequence must be computed for
        # each wihij unlike the simpler three part chain rule in part [1]
        #
        # We want to determine the gradients PDE_wihij and adjust the
        # hidden->output weights wihij = wihij - alpha * PDE_whihj i,j in {1,2}
        # We proceed case by case through i,j in {1,2}
        # i=1, j=1 
        # PDE_wih11 = PDE_h1 * PDh1_ih1 * PDih1_wih11 - [a]*[b]*[c]
        # [a]
        # NOTE: PDE_h1 = PDE1_h1 + PDE2_h1
        PDE1_h1 = (o1-target1)*dsigma(o1)*pwho11
        PDE2_h1 = (o2-target2)*dsigma(o2)*pwho12
        PDE_h1 = PDE1_h1 + PDE2_h1
        # [b]
        PDh1_ih1 = dsigma(h1)
        # [c]
        PDih1_wih11 = i1
        gradient = PDE_h1 * PDh1_ih1 * PDih1_wih11
        pwih11 = wih11
        wih11 = wih11 - alpha * gradient
        if diag == True:
            print("\n\n**** back propagation for input->hidden weights whiij:")
            print("\no1-target1 = " + str(o1-target1))
            print("dsigma(o1) = " + str(dsigma(o1)))
            print("pwho11 = " + str(pwho11))
            print("PDE1_h1 = " + str(PDE1_h1))
            print("\no2-target2 = " + str(o2-target2))
            print("dsigma(o2) = " + str(dsigma(ho2)))
            print("pwho12 = " + str(pwho12))
            print("PDE2_h1 = " + str(PDE2_h1))
            print("\nPDE_h1 = " + str(PDE_h1))
            print("dsigma(h1) = " + str(dsigma(h1)))
            print("i1 = " + str(i1))
            print("gradient = " + str(gradient))
            print("alpha = " + str(alpha))
            print("prev weight wih11 is " + str(pwih11))
            print("@@@@new weight wih11 is " + str(wih11))


        # 1=1 j=2
        # PDE_wih12 = PDE_h2 * PDh2_ih2 * PDih2_wih12 - [a]*[b]*[c]
        # [a]
        # NOTE: PDE_h2 = PDE1_h2 + PDE2_h2
        PDE1_h2 = (o1-target1)*dsigma(o1)*pwho21
        PDE2_h2 = (o2-target2)*dsigma(o2)*pwho22
        PDE_h2 = PDE1_h2 + PDE2_h2
        # [b]
        PDh2_ih2 = dsigma(h2)
        # [c]
        PDih2_wih12 = i1
        gradient = PDE_h2 * PDh2_ih2 * PDih2_wih12
        pwih12 = wih12
        wih12 = wih12 - alpha * gradient
        if diag == True:
            print("\n\n**** back propagation for input->hidden weights whiij:")
            print("\no1-target1 = " + str(o1-target1))
            print("dsigma(o1) = " + str(dsigma(o1)))
            print("pwho21 = " + str(pwho21))
            print("PDE1_h2 = " + str(PDE1_h2))
            print("\no2-target2 = " + str(o2-target2))
            print("dsigma(o2) = " + str(dsigma(ho2)))
            print("pwho22 = " + str(pwho22))
            print("PDE2_h2 = " + str(PDE2_h2))
            print("\nPDE_h2 = " + str(PDE_h2))
            print("dsigma(h2) = " + str(dsigma(h2)))
            print("i1 = " + str(i1))
            print("gradient = " + str(gradient))
            print("alpha = " + str(alpha))
            print("prev weight wih12 is " + str(pwih12))
            print("@@@@new weight wih12 is " + str(wih12))


        # i=2, j=1 
        # PDE_wih21 = PDE_h1 * PDh1_ih1 * PDih1_wih21 - [a]*[b]*[c]
        # [a]
        # NOTE: PDE_h1 = PDE1_h1 + PDE2_h1
        PDE1_h1 = (o1-target1)*dsigma(o1)*pwho11
        PDE2_h1 = (o2-target2)*dsigma(o2)*pwho12
        PDE_h1 = PDE1_h1 + PDE2_h1
        # [b]
        PDh1_ih1 = dsigma(h1)
        # [c]
        PDih1_wih21 = i2
        gradient = PDE_h1 * PDh1_ih1 * PDih1_wih21
        pwih21 = wih21
        wih21 = wih21 - alpha * gradient
        if diag == True:
            print("\n\n**** back propagation for input->hidden weights whiij:")
            print("\no1-target1 = " + str(o1-target1))
            print("dsigma(o1) = " + str(dsigma(o1)))
            print("pwho11 = " + str(pwho11))
            print("PDE1_h1 = " + str(PDE1_h1))
            print("\no2-target2 = " + str(o2-target2))
            print("dsigma(o2) = " + str(dsigma(ho2)))
            print("pwho12 = " + str(pwho12))
            print("PDE2_h1 = " + str(PDE2_h1))
            print("\nPDE_h1 = " + str(PDE_h1))
            print("dsigma(h1) = " + str(dsigma(h1)))
            print("i2 = " + str(i2))
            print("gradient = " + str(gradient))
            print("alpha = " + str(alpha))
            print("prev weight wih21 is " + str(pwih21))
            print("@@@@new weight wih21 is " + str(wih21))


        # i=2, j=2 
        # PDE_wih22 = PDE_h2 * PDh2_ih2 * PDih2_wih22 - [a]*[b]*[c]
        # [a]
        # NOTE: PDE_h2 = PDE1_h2 + PDE2_h2
        PDE1_h2 = (o1-target1)*dsigma(o1)*pwho12
        PDE2_h2 = (o2-target2)*dsigma(o2)*pwho22
        PDE_h2 = PDE1_h2 + PDE2_h2
        # [b]
        PDh2_ih2 = dsigma(h2)
        # [c]
        PDih2_wih22 = i2
        gradient = PDE_h2 * PDh2_ih2 * PDih2_wih22
        pwih22 = wih22
        wih22 = wih22 - alpha * gradient
        if diag == True:
            print("\n\n**** back propagation for input->hidden weights whiij:")
            print("\no1-target1 = " + str(o1-target1))
            print("dsigma(o1) = " + str(dsigma(o1)))
            print("pwho12 = " + str(pwho12))
            print("PDE1_h2 = " + str(PDE1_h2))
            print("\no2-target2 = " + str(o2-target2))
            print("dsigma(o2) = " + str(dsigma(ho2)))
            print("pwho22 = " + str(pwho22))
            print("PDE2_h2 = " + str(PDE2_h2))
            print("\nPDE_h2 = " + str(PDE_h2))
            print("dsigma(h2) = " + str(dsigma(h2)))
            print("i2 = " + str(i2))
            print("gradient = " + str(gradient))
            print("alpha = " + str(alpha))
            print("prev weight wih22 is " + str(pwih22))
            print("@@@@new weight wih22 is " + str(wih22))
            print("-------------------------------------------------------")


        # final diagnostics
        if diag == True:
            print("\n\n\nfinal output:")
            print("o1 = " + str(o1))
            print("o2 = " + str(o2))
            print("\nfinal error:")
            print("E1 = " + str(E1))
            print("E2 = " + str(E2))
            print("E = " + str(E))
            print("-------------------------------------------------------")
    
    


    # final diagnostics
    print("\n\n@@@@@@final input->hidden weights:")
    print("wih11 = " + str(wih11))
    print("wih12 = " + str(wih12))
    print("wih21 = " + str(wih21))
    print("wih22 = " + str(wih22))
    print("\nfinal hidden->output weights:")
    print("who11 = " + str(who11))
    print("who12 = " + str(who12))
    print("who21 = " + str(who21))
    print("who22 = " + str(who22))

    print("\nfinal output:")
    print("o1 = " + str(o1))
    print("o2 = " + str(o2))

    print("\nfinal error:")
    print("E1 = " + str(E1))
    print("E2 = " + str(E2))
    print("E = " + str(E))





if __name__ == "__main__": 
    print('\n+++++++++++ backprop222.py +++++++++++++++++++++')
    print("backprop222.py running as __main__")
    nargs = len(sys.argv) - 1
    position = 1
    iterations = 1
    diagnostics = False

    print("nargs = " + str(nargs))
    while nargs >= position:
        #print('backprop222: sys.argv[' + str(position) + '] = ' + str(sys.argv[position]))
        position += 1

    if nargs == 2:
        iterations = int(sys.argv[1])
        s = sys.argv[2].lower()
        if(s == "false" or s == "f"):
            diagnostics = False
        else:
            diagnostics = True
    elif nargs == 1:
        iterations = int(sys.argv[1]) 

    action(iterations, diagnostics)

else:
    print("backprop222.py module imported")
