Project Members:
1. Jyothi Vishnu Vardhan Kolla
2. Vidya Ganesh

* Number of time-travel days used -> 3 days.
1.Operating System used: Mac OS
2.Ide Used: VS


Instructions for Running Executables:

For training models.

* Run main.py for all tasks related to training all the models.

It takes a total of 9 parametres.

first arg -> Batch size to use
second arg -> Number of epochs to train the model.
thrid arg -> if given 1 it trains the model for MNIST digit data.
fourth arg -> if given 1 trains the models for fashion MNIST model with different hyperparametres : Testing for task4
fifth arg -> if given 1 trains a tiny VGG model for classifying fashion MNIST model: Testing for Extension 2.
sixth arg -> if given 1 the models evaluates all the stored results and displays the loss and accuracy curves.
seventh arg -> if given 1 the we train a fine tuned model for the greek letters (testing for task3 and extension1).
eight arg -> we have give this parameter based on the number of classes of greek letters we are using.
9th args -> if given 1 it trains the model with by replacing first layer with gabor filter and train remaining layers. (testing for extension 4)

* Run predictions.py To test for how model is performing on custom data and plot the results using matplotlib. (Testing for task 1.f,1.g and task4 evaluations)
* Run Examine.py To test how the filters look like and how those filters work on a example image from data. (Testing for task2 and extension 3)
* Run main.py to classify digits in real time by passing it a path of the video.(Extension 5)






