package main

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestDot(t *testing.T) {
	a := mat.NewDense(2, 2, []float64{
		4, 8,
		5, 6,
	})
	b := mat.NewDense(2, 2, []float64{
		8, 4,
		6, 5,
	})
	c := dot(a, b)
	if got := c.At(0, 0); got != 80.0 {
		t.Errorf("Expected to receive %f but got %f", 80.0, got)
	}
	if got := c.At(0, 1); got != 56.0 {
		t.Errorf("Expected to receive %f but got %f", 56.0, got)
	}
	if got := c.At(1, 0); got != 76.0 {
		t.Errorf("Expected to receive %f but got %f", 80.0, got)
	}
	if got := c.At(1, 1); got != 50.0 {
		t.Errorf("Expected to receive %f but got %f", 80.0, got)
	}
}

func TestMultiply(t *testing.T) {
	a := mat.NewDense(2, 2, []float64{
		4, 8,
		5, 6,
	})
	b := mat.NewDense(2, 2, []float64{
		8, 4,
		6, 5,
	})
	c := multiply(a, b)
	if got := c.At(0, 0); got != 32.0 {
		t.Errorf("Expected to receive %f but got %f", 80.0, got)
	}
	if got := c.At(0, 1); got != 32.0 {
		t.Errorf("Expected to receive %f but got %f", 56.0, got)
	}
	if got := c.At(1, 0); got != 30.0 {
		t.Errorf("Expected to receive %f but got %f", 80.0, got)
	}
	if got := c.At(1, 1); got != 30.0 {
		t.Errorf("Expected to receive %f but got %f", 80.0, got)
	}
}

func TestAddScalar(t *testing.T) {
	a := mat.NewDense(2, 2, []float64{
		4, 8,
		5, 6,
	})
	s := 2.0
	c := addScalar(s, a)
	if got := c.At(0, 0); got != 6.0 {
		t.Errorf("Expected to receive %f but got %f", 80.0, got)
	}
	if got := c.At(0, 1); got != 10.0 {
		t.Errorf("Expected to receive %f but got %f", 56.0, got)
	}
	if got := c.At(1, 0); got != 7.0 {
		t.Errorf("Expected to receive %f but got %f", 80.0, got)
	}
	if got := c.At(1, 1); got != 8.0 {
		t.Errorf("Expected to receive %f but got %f", 80.0, got)
	}
}

func TestRandomArray(t *testing.T) {
	a := randomArray(10, 1.0)
	if len(a) != 10 {
		t.Errorf("Expected slice of %d elements, got %d", 10, len(a))
	}
}

func TestApply(t *testing.T) {
	m := mat.NewDense(2, 2, []float64{
		4, 8,
		5, 6,
	})
	fn := func(i, j int, v float64) float64 {
		return 100.0
	}
	o := apply(fn, m)
	expect := 100.0
	testFull(o, expect, t)
}

func TestScale(t *testing.T) {
	m := mat.NewDense(2, 2, []float64{
		5, 5,
		5, 5,
	})
	o := scale(2.0, m)
	expect := 10.0
	testFull(o, expect, t)
}

func TestAdd(t *testing.T) {
	a := mat.NewDense(2, 2, []float64{
		5, 5,
		5, 5,
	})
	b := mat.NewDense(2, 2, []float64{
		5, 5,
		5, 5,
	})
	o := add(a, b)
	expect := 10.0
	testFull(o, expect, t)
}

func TestSubtract(t *testing.T) {
	a := mat.NewDense(2, 2, []float64{
		10, 10,
		10, 10,
	})
	b := mat.NewDense(2, 2, []float64{
		5, 5,
		5, 5,
	})
	o := subtract(a, b)
	expect := 5.0
	testFull(o, expect, t)
}

func TestSigmoid(t *testing.T) {
	o := sigmoid(0, 0, 6.84)
	if o != 0.998931040497558 {
		t.Errorf("Got %f but expected %f", o, 0.998931040497558)
	}
}

func testFull(m mat.Matrix, expect float64, t *testing.T) {
	for i := 0; i < 4; i++ {
		var j int
		switch i {
		case 0:
			j = 0
		case 3:
			j = i % 2
		default:
			j = (i + 1) % 2
		}
		if got := m.At(i%2, j); got != expect {
			t.Errorf("Expected to receive %f but got %f at %d, %d", expect, got, i%2, j)
		}
	}
}
