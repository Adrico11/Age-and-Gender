#  Age and Gender Challenge 👶👦👩 ‍

## The challenge

Design and train efficient and lightweight algorithms to estimate Age and Gender using facial crop.

## Commands to run the module :

To train a new model and test its performances, type the following in a terminal / conda prompt :
**python run.py -r train**


To predict the age and gender of a specific / random photo in a folder : 
**python run.py -r predict**

Look in the run.py file for additional options and commands !

## Next steps : 

- Add a flake8 file, a requirements.txt file and a CI file
- Make some final adjustments to the module : deal with the size issues ((64,64) or (198,198)),  add some plots at the end of the training
- Try some more in depth methods to have a more uniforme dataset : sampling techniques (subsampling, oversampling...), weighted classes, learning rate...
- Try a pretrained convnet like Xception to get even better results (especially on the age problem)
- Build a small Streamlit app to showcase the project in a more entertaining way

## Technologies

The script should be implemented in Python3

You can use machine learning libraries such as

* Tensorflow & Keras
* Pytorch
* Mxnet
* Scikit-learn

The project code should be committed to this git repository.

Please create a requirements.txt file with all the libraries and the packages
you need to run your code. 
If you set it up with a docker or a conda environement, that's even better.
You should also provide the training code and the models.

## Description

Using the UTK face Dataset, estimate the age and gender using the face.
https://www.kaggle.com/jangedoo/utkface-new

The Challenge will be 7 days long. It will end next monday, the 24th of January
At the end of the challenge, we will contact you after reviewing your code and if it's a success in order to schedule 30 minutes call to discuss more about your results.

## Results

Your results and approaches should quickly be summarized in a document (PDF). 

We will be interested by what you have tried in addition to your results.
We will also look at the quality of your code

## FAQ

1 - If you need some GPUS, you should use Google Colab.


2 - The challenge should not be publish on Github or on any pulic forum

## Questions

If you have any question please open an issue! Thank you and happy coding!
