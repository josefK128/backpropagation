__README.md__ 

* clone the repository:                                                                                                                                                   > git clone https://mark_rudolph@github.com/josefK128/backpropagation.git
  
* install python 3.X (current latest 3.8) - see 
  https://www.python.org/downloads/ - simply click download button and follow 
  defaults for installation
  
* install numpy:  > pip install numpy

* backprop222.py usage:  > py backprop.py 1 True (>log)
  This command will write out all diagnostics to the console (or to a logfile)
  
* backpropIJK.py usage:  > py backpropIJK.py I J K 1 True (>log)

* to run backpropIJK to verify correctness compare to out put of backprop222.py by running the following:  > py backpropIJK.py 2 2 2 1 True

  (Also see and compare the two log files: backprop222_diagnostics and backpropIJK-222_diagnostics)

  

* For both backprop222.py and backpropIJK.py, to view convergence of the error function E to 0 run several sessions with increasing number of iterations (best to use diagnostics=False)

  
