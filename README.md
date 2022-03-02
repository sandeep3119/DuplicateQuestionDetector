# DuplicateQuestionDetector

A NLP problem where we need to find whether 2 questions are semantically similar or not.</br>

**UseCase**: To merge 2 different threads on a forum based on their similarity of the base question.</br>

**DataSet Use**: https://www.kaggle.com/c/quora-question-pairs </br>
**Sample Size:** 50000  

**Preprocessing**: Lots of Feature extraction took place here as original dataset comes with only very less parameters.  

**Types Of Feature Extracted:**  
1. Meta-Features  
    i. Length of both question  
    ii. words in both question  
    iii. words common  
    iv. total words in both  
    v. word share  
2. Token Features    
    cwc_min: This is the ratio of the number of common words to the length of the smaller question  
    cwc_max: This is the ratio of the number of common words to the length of the larger question  
    csc_min: This is the ratio of the number of common stop words to the smaller stop word count among the two questions  
    csc_max: This is the ratio of the number of common stop words to the larger stop word count among the two questions  
    ctc_min: This is the ratio of the number of common tokens to the smaller token count among the two questions  
    ctc_max: This is the ratio of the number of common tokens to the larger token count among the two questions  
    last_word_eq: 1 if the last word in the two questions is same, 0 otherwise  
    first_word_eq: 1 if the first word in the two questions is same, 0 otherwise  
3. Fuzzy Features  
    
    fuzz_ratio: fuzz_ratio score from fuzzywuzzy  
    fuzz_partial_ratio: fuzz_partial_ratio from fuzzywuzzy  
    token_sort_ratio: token_sort_ratio from fuzzywuzzy  
    token_set_ratio: token_set_ratio from fuzzywuzzy  

**Model Used**: RandomForestClassifier(Accuracy:80%)  
**Future Scope**: More Feature Engineering and use of Neural Network with bigger sample size.

