from utils.feature_extraction_util import test_common_words,test_total_words,test_fetch_token_features,test_fetch_fuzzy_features
import numpy as np
import pickle
def test_query_creator(q1,q2):
    test_data= []
    cv=pickle.load('model\cv.pkl')
    #Basic Features
    test_data.append(len(q1))
    test_data.append(len(q2))
    test_data.append(len(q1.split(" ")))
    test_data.append(len(q2.split(" ")))
    common_words=test_common_words(q1, q2)
    total_words=test_total_words(q1, q2)
    test_data.append(common_words)
    test_data.append(total_words)
    test_data.append(round((common_words / total_words), 2))

    ##Fuzzy features
    fuzzy_features = test_fetch_fuzzy_features(q1, q2)
    test_data.extend(fuzzy_features)

    ##Token Features
    token_features = test_fetch_token_features(q1, q2)
    test_data.extend(token_features)

    q1_bag=cv.transform([q1]).toarray()
    q2_bag=cv.transform([q2]).toarray()

    return np.hstack((np.array(test_data).reshape(1,18),q1_bag,q2_bag))

