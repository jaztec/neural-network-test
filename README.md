# Experimenting with neural networks

This repository is a personal introduction into the concept of machine learning and 
artificial intelligence. While trying to grasp the algebra behind it I used this great 
article from [Sau Sheong Chang](https://sausheong.github.io/posts/how-to-build-a-simple-artificial-neural-network-with-go/). All of the Go 
helper functions are direct copies from his article.

## Usage

Compile the program using regular go tooling
```
$ go build ./...
```

Train the dataset with the following commands (trained datasets are included in the `data` folder)
```
$ ./neural-network-test -train
```

It will run 10 epochs of 10000 sets each. The train command will overwrite earlier datasets, if you
want to extend the weights based on the existing datasets use:
```
$ ./neural-network-test -train -remember
```

Again, credits due for [Sau Sheong Chang](https://github.com/sausheong/gonn)