package main

import (
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

func dot(m, n mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	_, cols := n.Dims()
	nM := mat.NewDense(rows, cols, nil)
	nM.Product(m, n)
	return nM
}

func apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	nM := mat.NewDense(rows, cols, nil)
	nM.Apply(fn, m)
	return nM
}

func scale(s float64, m mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	nM := mat.NewDense(rows, cols, nil)
	nM.Scale(s, m)
	return nM
}

func multiply(m, n mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	nM := mat.NewDense(rows, cols, nil)
	nM.MulElem(n, m)
	return nM
}

func add(m, n mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	nM := mat.NewDense(rows, cols, nil)
	nM.Add(n, m)
	return nM
}

func subtract(m, n mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	nM := mat.NewDense(rows, cols, nil)
	nM.Sub(m, n)
	return nM
}

func addScalar(i float64, m mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	a := make([]float64, rows*cols)
	for x := 0; x < rows*cols; x++ {
		a[x] = i
	}
	n := mat.NewDense(rows, cols, a)
	return add(m, n)
}

func randomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	data = make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}
	return
}

func sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}

func sigmoidPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return multiply(m, subtract(ones, m)) // m * (1 - m)
}

func save(net network) {
	h, err := os.Create("data/hweights.model")
	defer h.Close()
	if err == nil {
		net.hiddenWeights.MarshalBinaryTo(h)
	}
	o, err := os.Create("data/oweights.model")
	defer o.Close()
	if err == nil {
		net.outputWeights.MarshalBinaryTo(o)
	}
}

// load a neural network from file
func load(net *network) {
	h, err := os.Open("data/hweights.model")
	defer h.Close()
	if err == nil {
		net.hiddenWeights.Reset()
		net.hiddenWeights.UnmarshalBinaryFrom(h)
	}
	o, err := os.Open("data/oweights.model")
	defer o.Close()
	if err == nil {
		net.outputWeights.Reset()
		net.outputWeights.UnmarshalBinaryFrom(o)
	}
	return
}
