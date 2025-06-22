I tried different versions of MNIST and settled on using torch's MNIST loading
but my Dataloader, as it's straight forward and transparent. Also gave sklearn.fetch_openml's version
a try but i got different (lets just say less expected) results. Very likely that it's just a normalization issue
but I'll go ahead now with the torch version. 

data in this directory includes:
    - dataset:Mini-MNIST_inner:GGN.json -> minimnist subsample from sklearn MNIST version (now depracated. minimnist will use torch's MNIST version.)
    - 