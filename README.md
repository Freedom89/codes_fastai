# fast ai

## Week1 - Introduction 

* [Course page](http://wiki.fast.ai/index.php/Lesson_1#Overview_of_homework_assignment)
* [Course notes](http://wiki.fast.ai/index.php/Lesson_1_Notes)
* [Notebook for lesson 1](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/lesson1.ipynb)
* [Lesson 1 Overview](https://www.youtube.com/watch?v=kzt3-FHdAeM)
* [Lesson 1: Practical Deep Learning for Coders
](https://www.youtube.com/watch?v=Th_ckFbc6bI)


### Useful commands: 
```
wget http://www.platform.ai/files/nbs/lesson1.ipynb
wget http://www.platform.ai/files/nbs/utils.zip
wget http://www.platform.ai/files/nbs/vgg16.zip
wget http://www.platform.ai/data/dogscats.zip

unzip utils.zip
unzip vgg16.zip
unzip dogscats.zip

rm dogscats.zip
rm utils.zip
rm vgg16.zip
```

## Week2 

* [Course page](http://wiki.fast.ai/index.php/Lesson_2)
* [Discussion page](http://forums.fast.ai/t/lesson-2-discussion/161/91)
* [Lesson 2: Practical Deep Learning for Coders](https://www.youtube.com/watch?v=e3aM6XTekJc&feature=youtu.be)
* [Lesson 0 - recommended to watch after lesson 2](https://www.youtube.com/watch?v=ACU-T9L4_lI&t=11s)
* [Notebooks](https://github.com/fastai/courses)
	* [convolution-intro.ipynb](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/convolution-intro.ipynb) - The convolution tutorial notebook used in the 		introductory lesson presented during the Data Institute launch
	* [lesson2.ipynb](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/lesson2.ipynb) - the main notebook for lesson 2
	* [redux.ipynb](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/dogs_cats_redux.ipynb) - how to enter the Dogs vs Cats Redux competition, and how to 		visualize your models correct and incorrect predictions
	* [sgd-intro.ipynb](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/sgd-intro.ipynb) - the simple SGD tutorial

* [Excel files used](http://www.platform.ai/files/xl/)
* [Stanford Cs231 reading](http://cs231n.github.io/)
  * [Optimization: Stochastic Gradient Descent](http://cs231n.github.io/optimization-1/)
  * [Backpropagation, Intuitions](http://cs231n.github.io/optimization-2/)
  * [Neural Networks Part 1: Setting up the Architecture](http://cs231n.github.io/neural-networks-1/)
* [Introduction to Deep Learning by Michael Nielsen](http://neuralnetworksanddeeplearning.com/)
  * [Chapter 1 - Using neural nets to recognize handwritten digits](http://neuralnetworksanddeeplearning.com/chap1.html)
  * [Chapter 2 - How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)
  * [Chapter 3 - Improving the way neural networks learn](http://neuralnetworksanddeeplearning.com/chap3.html)

### Week2 Extras 

#### Suggested Readings from notebooks
* [Chapter 4 - A visual proof that neural nets can compute any function](http://neuralnetworksanddeeplearning.com/chap4.html)

#### Errors

##### Error one
```
val_data = get_data(val_batches)
trn_data = get_data(batches)
```
should be 

```
val_data = get_data(path + 'valid')
trn_data = get_data(path + 'train')
```

More over, using get_data would cause your memory to be inefficient (on p2.x on aws), it is suggested on the [fourms](http://forums.fast.ai/search?q=memory) to use batch generator instead.

I overcame the problem by running the code as it is, save using bcolz. Afterwards i restart the notebook, and only ran the load command without get_data. 

As an aside, you can use `sys.getsizeof()` to understand the memory usage of each python object.

This [link](                        http://askubuntu.com/questions/53264/how-do-you-find-out-which-program-is-using-too-much-memory) is also a good read to understand how to monitor your instance memory stance. 

##### Error two

```
def fit_model(model, batches, val_batches, nb_epoch=1):
    model.fit_generator(batches, samples_per_epoch=batches.N, nb_epoch=nb_epoch, 
                        validation_data=val_batches, nb_val_samples=val_batches.N)

model.evaluate_generator(get_batches('valid', gen, False, batch_size*2), val_batches.N)

```

The `batches.N` and `val_batches.N` should be `batches.n` and `val_batches.n` instead based on the util functions. 

### Misc

[Kaggle cli](http://wiki.fast.ai/index.php/Kaggle_CLI)

