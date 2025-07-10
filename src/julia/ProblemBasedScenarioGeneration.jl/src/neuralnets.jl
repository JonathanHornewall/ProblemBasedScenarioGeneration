# Define a multilayer perceptron model with input of size 3 and output of size 30
model = Chain(
    Dense(3, 3, relu),  
    Dense(3, 3, relu),   
    Dense(3, 9, relu),
    Dense(9, 18, relu),
    Dense(18, 30, relu)            
)
