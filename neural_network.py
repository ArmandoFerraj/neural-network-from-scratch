import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.w1 = np.random.randn(64, 784) * np.sqrt(2/784) 
        self.b1 = np.zeros(64)
        self.w2 = np.random.randn(10, 64) * np.sqrt(2/64)
        self.b2 = np.zeros(10)            
        self.learning_rate = .01

    def relu(self, x):
        return np.maximum(0,x)
        
    def relu_prime(self, x):
        return (x > 0) * 1.0
        
    def softmax(self, x):
        exp_vals = np.exp(x - np.max(x))
        return exp_vals / np.sum(exp_vals)
        
    def forward(self, x):
        z1 = (self.w1 @ x) + self.b1
        a1 =self.relu(z1)
        z2 = (self.w2 @ a1) + self.b2
        a2 = self.softmax(z2)
        return a2, a1, z1
        
    def error_func(self, y_prediction, y_truth):
        return -np.sum(y_truth * np.log(y_prediction + 1e-9)) #cross entropy error function. "+ 1e-9" is to avoid log(0) edge case

    def backprop(self, x, y, a2, a1, z1):
        delta_output = a2 - y
        delta_hidden = np.dot(self.w2.transpose(),delta_output) * self.relu_prime(z1)
            
        dE_dW2 = np.outer(delta_output, a1)
        dE_db2 = (delta_output)
        dE_dW1 = np.outer(delta_hidden, x)
        dE_db1 = delta_hidden
        
        return dE_dW2, dE_db2, dE_dW1, dE_db1

    def update_weights(self, dE_dW2, dE_db2, dE_dW1, dE_db1):
        self.w2 -= self.learning_rate * dE_dW2
        self.b2 -= self.learning_rate * dE_db2
        self.w1 -= self.learning_rate * dE_dW1
        self.b1 -= self.learning_rate * dE_db1
        return
        
if __name__ == "__main__":
    with np.load('data/mnist.npz') as f: # download train and test data
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    x_train = x_train.reshape(60000, 784) #flatten x data from (28,28) -> (784,)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train/255.0 #normalize x data

    y_train = np.eye(10)[y_train] # turn y data from scalar to vector (10,) to match output shape of after softmax
    y_test = np.eye(10)[y_test]

    nn = NeuralNetwork()

    print("Entering the DOJO:")
    loss_curve = []
    for epoch in range(10):
        epoch_error = []
        for x, y in zip(x_train, y_train):
            a2, a1, z1 = nn.forward(x= x)
            epoch_error.append(nn.error_func(y_prediction= a2, y_truth= y))
            dE_dW2, dE_db2, dE_dW1, dE_db1 = nn.backprop(x= x, y= y, a2= a2, a1= a1, z1= z1)
            nn.update_weights(dE_dW2= dE_dW2, dE_db2= dE_db2, dE_dW1= dE_dW1, dE_db1= dE_db1)
        loss_curve.append(np.mean(epoch_error))
        print(f"epoch {epoch}: avg loss = {loss_curve[-1]:.4f}")

    correct = 0 
    for x, y in zip(x_test, y_test): #test trained nn against unseen data
        a2, a1, z1 = nn.forward(x=x)
        predicted_digit = np.argmax(a2)      # index of highest probability = prediction
        true_digit = np.argmax(y)            # recover true digit 
        if predicted_digit == true_digit:
            correct += 1

    accuracy = correct / len(x_test)
    print(f"test accuracy: {accuracy:.4f}  ({correct}/{len(x_test)})")

    np.savez('trained_model.npz', w1=nn.w1, b1=nn.b1, w2=nn.w2, b2=nn.b2) # save the trained weights
    print("Model weights saved!")

