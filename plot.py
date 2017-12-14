import matplotlib.pyplot as plt

class Plot:

    def __init__(self, data, x_label, y_label):
        self.data = data
        self.x_label = x_label
        self.y_label = y_label
        
    def show(self):
        plt.plot(self.data)
        plt.ylabel(self.y_label)
        plt.xlabel(self.x_label)
        plt.show()
