# GPLearn-Testing
This project tests the genetic programming library gplearn: https://gplearn.readthedocs.io/en/stable/


#
**Project Members**

[Animesh KC](https://github.com/AnimeshKC) (Black Box testing)

Emily Ingram (White Box testing lead) — Github unknown 

[Glen Rhodes](https://github.com/glnnrhodes "Github link") (White Box testing assistant)

#
**What is GPLearn?**

GPLearn implements a python genetic program toolkit specifically designed for symbolic regression problems. Symbolic regression is a form of analysis that attempts to find an expression of best fit for a given data set by combining mathematical operators from a pool of mathematical expressions. 

Genetic programming extends this by iterating for several generations in which the initial population of randomly generated possible solutions (individuals) are either mated with another individual, mutated, or both. The fitness of all individuals is then ranked and if none of the individuals achieve the desired level of fitness another generation is executed, with a biased selection so the best fit individuals make up the majority of the population for the next generation. The mating process consists of selecting two individuals and recombining them to produce two new offspring analogous to biological mating. The mutation process requires only a single individual and alters one or more gene of the individual to create a new individual. 	

#
**How is Black Box Testing Conducted for GPLearn?**

Black Box testing for GPLearn aims to look at the different types of functions that could be modelled by the library. GPLearn’s function set includes addition, subtraction, multiplication, division, sine, cosine, tangent, logarithm, square root, inverse, maximum, and minimum. To test these functions, equivalence classes were chosen as the primary method, as functions can be partitioned into these 12 categories. 

Each equivalence class is defined by its most exterior function. For example, for the function sin(max(x+y)), this will belong to the sine equivalence class, whereas max(sin(x) + sin(y)) will belong to the maximum equivalence class. This enables the equivalence classes to be mutually exclusive and collectively exhaustive for cases where one function is embedded within another. 

The test cases chosen for equivalence testing were simple ones that lie within these equivalence classes, in order to determine whether GPLearn is in general effective for mapping these functions. However, further error guessing tests were done with functions that GPLearn was more likely to have problems with modelling. To test GPLearn with the created test cases, the tester must first create a set of training and testing data that GPLearn will use to generate its function and evaluate its accuracy respectively. 

The training and testing data used were randomly generated for values that lie within the domain of the test. Each test used 50 randomly generated x and y values for both the training and test data, and for the sake of consistency, each equivalence class was tested with 2 input variables, even though some functions only require one variable. As a result, 3D plots could also be generated for every function. 

To evaluate the accuracy of the generated function compared to the testing data, GPLearn outputs an R^2 score. It is expected that valid equivalence classes will achieve an R^2 score above 0.9, as that would entail the generated function is sufficiently close to the tested function. A value of 1.0  represents that the generated function is identical to the tested function. Negative R^2 values are possible for results where the test data does not follow the trend of the generated function trend. For invalid equivalence classes, an R^2 value below 0.5 is expected, including negative values. A maximum of 20 generations were used for each function, but many functions will not require that many generations. 

#
**How is White Box Testing conducted for GPLearn?**

The white box testing, in this instance, is carried out through the use of Control Flow Graphs (CFDs). The CFD coverage criterion used was Branch Coverage. Branch Coverage makes sense in this case as GPLearn is a complex module with many interrelated parts, and due to the control flow structure of GPLearn, Statement Coverage and Branch coverage are equivalent. In fact, GPLearn is so complexly written that only specific sections of the code was considered for testing. The criteria for choosing which sections of code are tested is if they are both related to the main functionality of GPLearn and if the results of those code sections can be consistently interpreted by a test script; with more time, testing larger portions of the library is possible.  In addition to only testing the SymbolicRegressor class, only its fit, predict, and __str__ methods were tested, and as the fit method contains non-mission-critical code, sections of the fit method will also be ignored to save time. 

The actual testing was done by a Python module named Pytest. Pytest is similar to JUnit and its counterparts; however, Pytest better utilizes follows Python’s design philosophy (simpler is better). Except for the pytest.raises method, all testing in this instance is done with Python’s built-in ‘assert’ statement. 
	
As one may expect from a module that is commonly used in its applicable field, GPLearn has passed all Branch-complete tests. It’s possible that omission of non-mission-critical code has in-turn obfuscated important faults, but given that these tests focus on the testable infrastructure needed to run a basic genetic programming algorithm, any hidden faults aren’t likely to severely impact the GPLearn library.

**Key Findings**

More detail on the test findings can be found in the report folder. 

The main findings were that the GPLearn library passed black box tests for practical situations, but it understandably failed for impractically complicated equations. In terms of white box testing, the library passed all branch coverage tests, suggesting that the logic was sound. Although statement coverage was not explicitly tested, GPlearn's control flow resulted in branch coverage being near equivalent to statement coverage. 

