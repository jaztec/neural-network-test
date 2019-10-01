# Experimenting with neural networks

This repository is a personal introduction into the concept of machine learning and 
artificial intelligence. While trying to grasp the algebra behind it I used this great 
article from [Sau Sheong Chang](https://sausheong.github.io/posts/how-to-build-a-simple-artificial-neural-network-with-go/). All of the Go 
helper functions are direct copies from his article.

## Usage

Compile the program using regular go tooling
```bash
$ go build ./...
```

Train the dataset with the following commands (trained datasets are included in the `data` folder)
```bash
$ ./neural-network-test -train
```

It will run 10 epochs of 10000 sets each. The train command will overwrite earlier datasets, if you
want to extend the weights based on the existing datasets use:
```bash
$ ./neural-network-test -train -remember
```

To run a simulation use the following format:
```bash
$ ./neural-network-test -board="{\"alive\":1, \"points\":[0,0,1,0,1,0,1,0]}"
Input
=====
Cell is active
Cell siblings are set to [0 0 1 0 1 0 1 0]

Outputs
=======
Death chance: 0.001%
Alive chance: 0.999%

$ ./neural-network-test -board="{\"alive\":1, \"points\":[0,0,1,0,0,0,1,0]}"
Input
=====
Cell is active
Cell siblings are set to [0 0 1 0 0 0 1 0]

Outputs
=======
Death chance: 0.004%
Alive chance: 0.996%

$ ./neural-network-test -board="{\"alive\":1, \"points\":[0,0,0,0,0,0,1,0]}"
Input
=====
Cell is active
Cell siblings are set to [0 0 0 0 0 0 1 0]

Outputs
=======
Death chance: 0.993%
Alive chance: 0.007%

$ ./neural-network-test -board="{\"alive\":1, \"points\":[0,1,0,1,1,0,1,0]}"
Input
=====
Cell is active
Cell siblings are set to [0 1 0 1 1 0 1 0]

Outputs
=======
Death chance: 0.999%
Alive chance: 0.001%


```

Again, credits due for [Sau Sheong Chang](https://github.com/sausheong/gonn)