from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from scipy.sparse import csr_matrix


def LSA(data):
    vectorizer = TfidfVectorizer(use_idf=True,smooth_idf=True,sublinear_tf=True)
    dtm = vectorizer.fit_transform(data)
    
    X_sparse = csr_matrix(dtm)
    dtm = dtm.astype(float)
    lsa = TruncatedSVD(min(X_sparse.shape)-1, algorithm = 'arpack')
    dtm_lsa = lsa.fit_transform(dtm)
    dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
    return 