#  Age and Gender Challenge 👶👦👩 ‍

## The challenge

Design and train efficient and lightweight algorithms to estimate Age and Gender using facial crop, based on theUTK face Dataset (https://www.kaggle.com/jangedoo/utkface-new)

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
