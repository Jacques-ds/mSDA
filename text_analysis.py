# ==============================================================================
# (1) Working directory and libraries
# ==============================================================================

# simple test of mSDA
import msda
import process_data
from fetch_20newsgroups_functions import \
    fetch_20newsgroups  # for occasions when the data cannot be directly downloaded the function was adjusted for manual download
# from sklearn.datasets import fetch_20newsgroups # for automatic download (you can skip step 2)
from sklearn.datasets import load_files
from sklearn import svm
import numpy as np
import time
import codecs
import pickle
import tarfile
import shutil
import os

os.chdir(r'..\client2vec\mSDA')

# TODO: fix up preprocessing data code and upload it as a separate file to WMD as well
# TODO: debug low dimensional approximation
# TODO: run more extensive tests on performance of mSDA with and without low dimensional approximation


# ==============================================================================
# (2) Manually downloading data
# ==============================================================================
# Step 1:
# Manually download file here https://ndownloader.figshare.com/files/5975967 and save it in _path

# Step 2:
# Adjust the data and set up variables
_path = r'..\client2vec\mSDA\newsgroup'
downloaded_file = '20newsbydate.tar.gz'
pikle_file = '20news-bydate.pkz'
train_file = '20news-bydate-train'
test_file = '20news-bydate-test'

if not os.path.exists(os.path.join(_path, pikle_file)):
    try:
        tarfile.open(os.path.join(_path, downloaded_file), "r:gz").extractall(path=_path)

        # Store a zipped pickle
        cache = dict(train=load_files(os.path.join(_path, train_file), encoding='latin1'),
                     test=load_files(os.path.join(_path, test_file), encoding='latin1'))

        compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')

        with open(os.path.join(_path, pikle_file), 'wb') as f:
            f.write(compressed_content)

        # Delete downloaded and extracted files except the pikle file
        for file in os.listdir(_path):
            if file.endswith('.gz'):
                os.remove(os.path.join(_path, file))
            elif not file.endswith('.pkz'):
                shutil.rmtree(os.path.join(_path, file))
    except Exception as e:
        print(80 * '_')
        print('File extraction failed, at first download data manually!!')
        print(80 * '_')
        print(e)

# ==============================================================================
# (3) Fetch training documents from 20 newsgroups dataset in random order
# ==============================================================================
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
all_20news = fetch_20newsgroups(data_home=_path, subset='all', categories=categories, shuffle=True, random_state=42,
                                download_if_missing=False)
all_raw_data = all_20news.data  # all the data
all_data_stringsList = process_data.createWordLists(process_data.unicodeToString(all_raw_data))
all_data_words = process_data.preprocess_by_word(all_data_stringsList)
all_labels = all_20news.target  # all the labels
all_full_data = process_data.vectorize(all_data_words)  # convert to bag of words
all_full_data = all_full_data.transpose()  # so rows are data, columns are features (format we predominantly use)
num_mostCommon = 800
all_mostCommonFeatures_data = process_data.getMostCommonFeatures(all_full_data, num_mostCommon)
train_data, train_labels, test_data, test_labels = process_data.splitTrainTest(all_mostCommonFeatures_data, all_labels)

print("Shape of training data: ", train_data.shape)
print("Shape of test data: ", test_data.shape)

# ==============================================================================
# (4) Classify with linear SVM - raw data
# ==============================================================================
# transpose because sklearn requires (#data x #features)
clf_baseline = svm.SVC().fit(train_data.transpose(), train_labels)
baseline_preds = clf_baseline.predict(test_data.transpose())
base_accuracy = np.mean(baseline_preds == test_labels)
print("Accuracy with linear SVM on basic representation: ", base_accuracy)

# ==============================================================================
# (5) Classify with linear SVM - mSDA
# ==============================================================================
# learn deep representation with msda...
prob_corruption = 0.4
num_layers = 2
subproblem_size = 400

# ------------------------------------------------------------------------------
# (a) mSDA without data preprocesing
# ------------------------------------------------------------------------------
before_msda = time.time()
# need to transpose data to be in the right format (#features x #data) for mSDA
# specifically, the deep representation is the output from the last layer
# with low dimensional approximation described in paper
'''
subproblem_mappings, subseq_mappings, representations  = msda.mSDA_lowDimApprox(train_data, prob_corruption, 
            num_layers, subproblem_size)
train_deepRep = representations[-1]
#use same weights as on training features to transform test data
test_deepRep = msda.mSDA_lowDimApprox(test_data, prob_corruption, num_layers, subproblem_size, 
            subproblem_mappings, subseq_mappings)[2][:,:,-1]
'''
# without low dimensional approximation
train_mappings, train_reps = msda.mSDA(train_data, prob_corruption, num_layers)
train_deepRep = train_reps[-1]
# use same weights as on training features to transform test data
test_deepRep = msda.mSDA(test_data, prob_corruption, num_layers, train_mappings)[1][-1]
# '''

# sklearn requires (#data x #features) so transpose back
train_deepRep = train_deepRep.transpose()
test_deepRep = test_deepRep.transpose()

after_msda = time.time()
print("used msda in %s seconds" % (after_msda - before_msda))
print("Shape of msda train rep: ", train_deepRep.shape)
print("Shape of msda test rep: ", test_deepRep.shape)

# ...and classify with linear SVM
clf_deepRep = svm.SVC().fit(train_deepRep, train_labels)
preds_with_deepRep = clf_deepRep.predict(test_deepRep)
deep_accuracy = np.mean(preds_with_deepRep == test_labels)
print("Accuracy with linear SVM on mSDA features: ", deep_accuracy)
