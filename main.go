package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"time"
)

var n = createNetwork(9, 6, 2, .25)

type set struct {
	input   []float64
	initial float64
	result  float64
}

func generateTrainingData(n int) []set {
	s := make([]set, n)

	for i := 0; i < n; i++ {
		o := set{}
		o.input = make([]float64, 8)
		o.initial = float64(rand.Intn(2))
		alive := rand.Intn(7)
		switch alive {
		case 2:
			if o.initial == 1.0 {
				o.result = 1.0
			}
		case 3:
			o.result = 1.0
		}
		m := make(map[int]bool, alive)
		fn := func(m map[int]bool) bool {
			k := rand.Intn(7)
			_, ok := m[k]
			if ok {
				return false
			}
			m[k] = true
			return true
		}
		for j := 0; j < alive; j++ {
			for !fn(m) {
			}
		}
		for x := 0; x < 8; x++ {
			if _, ok := m[x]; ok {
				o.input[x] = 1.0
			}
		}
		s[i] = o
	}

	return s
}

func train(n *network) {
	rand.Seed(int64(time.Now().Nanosecond()))
	for epochs := 0; epochs < 10; epochs++ {
		sets := generateTrainingData(10000)
		for _, s := range sets {
			inputs := make([]float64, 9)
			targets := make([]float64, 2)
			inputs[0] = s.initial
			for i, p := range s.input {
				inputs[i+1] = p
			}
			targets[int(s.result)] = 1
			n.Train(inputs, targets)
		}
	}
}

func main() {
	var doTrain bool
	var withMemory bool
	var board string
	flag.BoolVar(&doTrain, "train", false, "use to train the network")
	flag.BoolVar(&withMemory, "remember", false, "use earlier data for the training")
	flag.StringVar(&board, "board", "{\"alive\": 0, \"points\": [0,0,0,0,0,0,0,0]}", "an array of data points to show a cell and his neighbors")
	flag.Parse()

	if doTrain {
		if withMemory {
			load(&n)
		}
		train(&n)
		save(n)
	} else {
		type cell struct {
			Alive  float64   `json:"alive"`
			Points []float64 `json:"points"`
		}
		c := cell{}
		if err := json.Unmarshal([]byte(board), &c); err != nil {
			panic(err)
		}
		inputs := make([]float64, 9)
		inputs[0] = c.Alive
		for i, p := range c.Points {
			inputs[i+1] = p
		}
		load(&n)
		outputs := n.Predict(inputs)

		fmt.Println("Input")
		fmt.Println("=====")
		if c.Alive == 1.0 {
			fmt.Println("Cell is active")
		} else {
			fmt.Println("Cell is inactive")
		}
		fmt.Printf("Cell siblings are set to %v\n\n", c.Points)
		fmt.Println("Outputs")
		fmt.Println("=======")
		fmt.Printf("Death chance: %3.3f%%\n", outputs.At(0, 0))
		fmt.Printf("Alive chance: %3.3f%%\n", outputs.At(1, 0))
	}
}
