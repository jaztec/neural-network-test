package main

import "gonum.org/v1/gonum/mat"

type network struct {
	inputs        int
	hiddens       int
	outputs       int
	hiddenWeights *mat.Dense
	outputWeights *mat.Dense
	rate          float64
}

func createNetwork(i, h, o int, r float64) (n network) {
	n = network{
		inputs:  i,
		hiddens: h,
		outputs: o,
		rate:    r,
	}
	n.hiddenWeights = mat.NewDense(n.hiddens, n.inputs, randomArray(n.inputs*n.hiddens, float64(n.inputs)))
	n.outputWeights = mat.NewDense(n.outputs, n.hiddens, randomArray(n.hiddens*n.outputs, float64(n.hiddens)))
	return
}

func (n *network) Predict(input []float64) mat.Matrix {
	// forward propagation
	inputs := mat.NewDense(len(input), 1, input)
	hiddenInputs := dot(n.hiddenWeights, inputs)
	hiddenOutputs := apply(sigmoid, hiddenInputs)
	finalInputs := dot(n.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)
	return finalOutputs
}

func (n *network) Train(input []float64, target []float64) {
	// forward propagation
	inputs := mat.NewDense(len(input), 1, input)
	hiddenInputs := dot(n.hiddenWeights, inputs)
	hiddenOutputs := apply(sigmoid, hiddenInputs)
	finalInputs := dot(n.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)

	// find errors
	targets := mat.NewDense(len(target), 1, target)
	outputErrors := subtract(targets, finalOutputs)
	hiddenErrors := dot(n.outputWeights.T(), outputErrors)

	// backpropagate
	n.outputWeights = add(n.outputWeights,
		scale(n.rate,
			dot(multiply(outputErrors, sigmoidPrime(finalOutputs)),
				hiddenOutputs.T()))).(*mat.Dense)

	n.hiddenWeights = add(n.hiddenWeights,
		scale(n.rate,
			dot(multiply(hiddenErrors, sigmoidPrime(hiddenOutputs)),
				inputs.T()))).(*mat.Dense)
}
