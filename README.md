# mSDA
Implementation and usage of marginalized stacked denoising autoencoders (mSDA), based on the "Marginalized Stacked Denoising Autoencoders for Domain Adaption" paper by Chen et. al (2012).

Details:

* Original MATLAB code is provided in the paper at http://www.cs.cornell.edu/~kilian/papers/msdadomain.pdf
* Both MATLAB and Python implementations were provided at http://www.cs.cornell.edu/~kilian/code/code.html
* This implementation of mSDA is based on both the sample code the authors provided as well as the equations in the paper.
* This Python implementation ends up being slightly more optimized than the one they provided, and it contains hopefully more explanatory variable names and comments.  

Additionally, while in the paper the authors provided literal MATLAB implementations of the main mSDA algorithm, they also described but did not give an implementation of a faster approximation for high dimensional data.
  
This project also contains an implementation of this approximation.  All this is done in msda.py.  

## Getting Started
To demonstrate the capabilities of mSDA, this project contains a simple sample application: document classification from a few categories the well-known 20 newsgroups dataset.

### Files
Included files are these:

* process_data.py - Data preprocessing, converting the raw data to a bag of words. Also contains methods to split the data into training and test sets and select the most common features (as the authors allude to doing).
* stop_words.txt - Common list of stop words.
* text_analysis.py - The actual applications, which loads in the data, classifies it with a linear SVM on bag-of-words as a baseline, and learns a "deep" representation with mSDA on which to train another linear SVM.
* msda.py - The core implementation of msda.
* md.py - Marginalized denoising with linear regression estimator.
* fetch_20newsgroups_functions.py - In case for any reason you cannot import data directly, you use this function in order to handle manually downloaded data. More info in text_analysis.py.

### How to run it
To run this program, run the command: python text_analysis.py. mSDA tends to run fairly quickly (though this is a very small problem) and produces features that lend themselves to slightly better classification accuracy than the raw bag-of-words representation.
  
This simple problem, however, is mainly a proof of concept--further work could explore more heavy-duty applications of mSDA (in particular, ones involving domain adaptation, which is what mSDA is intended for).  

### Python version
It was tested and adjusted for Python 3.6 (code was originally written for Python 2.x).

## Authors
This is not the original repository! Original repository can be found here [markheimann](https://github.com/markheimann/mSDA)

I just adjusted the scripts for Python 3.6 and for manuall data download since I wasn't able to import them directly.

## Acknowledgments
One more time, [markheimann](https://github.com/markheimann/mSDA), thank you for original implementation!

