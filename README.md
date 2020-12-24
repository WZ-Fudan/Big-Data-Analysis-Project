## Improved Convolutional Matrix Factorization for Recommendation System

#### Overview

> The extreme sparsity of the rating data seriously affects the recommendation quality
> of the recommendation system. In order to alleviate the problem of data sparsity,
> some convolutional neural network (CNN)-based models make full use of text data
> to improve the recommendation accuracy. However, due to the inherent limitations
> of the traditional convolutional network, they have difficulty in effectively using the
> contextual information of the document, resulting in an insufficient understanding
> of the document. This paper improves the convolutional matrix factorization model
> by structural innovations. Our extensive evaluation of two real data sets shows that
> even with rating data that is extremely sparse, the performance of the improved
> model far exceeds the original model.

###### All code in the project is original, refer to: https://github.com/WZ-ZXY/NNDL_PJ

###### Our selected paper: https://dl.acm.org/doi/10.1145/2959100.2959165

#### Requirements

- Python 3.7.6
- Tensorflow 2.2.0
- Keras 2.24

#### How to Run

Note: Run `python main.py ` in bash shell. You will see the process.

#### File Explanation

+ Input File

  The input data is saved in dataset folder

+ Output File

  + The w2v folder stores the embedding information of the training model
  + The result folder stores the weights of the training model
  + The loss folder stores the loss of the output of each training model

+ Code File

  + main.py: Responsible for running and training models of the main program
  + data_preprocessing.py: Responsible for data preprocessing, return and save data in proper format
  + model.py: Responsible for saving model code
  + attention.py: Responsible for the realization of the attention module
  + configuration.py: Configuration file, which mainly stores the hyperparameters of the model
  + utils.py: Contains some other needed functions

#### File tree is shown as following:

> dataset/
>
> > 1m/
> >
> > > 0-fold
> > >
> > > ...
> >
> > 10m/
> >
> > > ml-10m_movies.dat
> > >
> > > ...
> >
> > TV_R_m2u.pkl
> >
> > ..
>
> result/
>
> w2vmodel/
>
> logs/
>
> > total_state_.log
> >
> > ...
>
> configuration.py
>
> data_preprocessing.py
>
> attention.py
>
> mode.py
>
> main.py
>
> utils.py



#### Notes

The folds are empty, if you wanna acquire these data and configuration file, please contact me: zwei19@fudan.edu.cn or wzfduwz@gmail.com.

