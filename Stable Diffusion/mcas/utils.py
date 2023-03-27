import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
import mcas
from mcas.FeatureExtractor import FeatureExtractor
import logging

""" Gender Attributes:  data_1: generated images of men
                        data_2: generated images of women
    Targets: data_3: images --> images generated from descriptions
             data_3: text --> text descriptions used to generate images OR text describing gender
"""
def association_score(features_1, features_2):
    similarity = features_1.cpu().numpy() @ features_2.cpu().numpy().T
    return np.mean([np.mean(s) for s in similarity])

def cos_similarity(tar, att): 
        '''
        Calculates the cosine similarity of the target variable vs the attribute
        '''
        score = np.dot(tar, att) / (np.linalg.norm(tar) * np.linalg.norm(att))
        return score


def mean_cos_similarity(tar, att): 
    '''
    Calculates the mean of the cosine similarity between the target and the range of attributes
    '''
    mean_cos = np.mean([cos_similarity(tar, attribute) for attribute in att])
    return mean_cos


def text_association(tar, att1, att2):
    '''
    Calculates the mean association between a single target and all of the attributes
    '''
    association = mean_cos_similarity(tar, att1) - mean_cos_similarity(tar, att2)
    return association

def image_image_association_score(model_name, image_dir_1, image_dir_2, image_dir_3):

    fe_1 = FeatureExtractor(model_name, image_dir_1, None)
    fe_2 = FeatureExtractor(model_name, image_dir_2, None)
    fe_3 = FeatureExtractor(model_name, image_dir_3, None)

    return (association_score(fe_1.get_image_features(), fe_3.get_image_features()) - association_score(fe_2.get_image_features(), fe_3.get_image_features()))

def image_text_prompt_association_score(model_name, image_dir_1, image_dir_2, text):

    fe_1 = FeatureExtractor(model_name, image_dir_1, text)
    fe_2 = FeatureExtractor(model_name, image_dir_2, None)
    

    return (association_score(fe_1.get_image_features(), fe_1.get_text_features()) - association_score(fe_2.get_image_features(), fe_1.get_text_features()))

def image_text_attributes_association_score(model_name, image_dir, text_1, text_2):

    fe_1 = FeatureExtractor(model_name, image_dir, text_1)
    fe_2 = FeatureExtractor(model_name, None, text_2)
    
    return (association_score(fe_1.get_text_features(), fe_1.get_image_features()) - association_score(fe_2.get_text_features(), fe_1.get_image_features()))

def text_text_association_score(model_name, text_1, text_2, text_3):
    
    fe_1 = FeatureExtractor(model_name, None, text_1)
    fe_2 = FeatureExtractor(model_name, None, text_2)
    fe_3 = FeatureExtractor(model_name, None, text_3)

    return text_association(fe_3.get_text_features().cpu(), fe_1.get_text_features().cpu(), fe_2.get_text_features().cpu())
    #return (association_score(fe_1.get_text_features(), fe_3.get_text_features()) - association_score(fe_2.get_text_features(), fe_3.get_text_features()))


def get_stats(features_1, features_2):

    sd = []
    krt = []
    skewness = []

    similarity = features_1.cpu().numpy() @ features_2.cpu().numpy().T

    # Standard Deviation
    [sd.append(np.std(s)) for s in similarity]

    #Kurtosis
    [krt.append(kurtosis(s)) for s in similarity]

    #Skewness
    [skewness.append(skew(s)) for s in similarity]

    logging.warning('Use targets as feature 1 and attributes as feature 2')
    logging.warning('Outputs are standard deviation, kurtosis and skewness - in order')

    return sd, krt, skewness





