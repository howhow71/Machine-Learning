from Perceptron import Perceptron
import pandas as pd
def main():
	print("Percepton Test File")	
	#https://archive.icu.uci.edu/ml/machine-learning-databases/iris/iris.data
    df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
	df.tail()
main()
