# ! pip install faiss-cpu
# ! pip install sentence-transformers

from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np

data = [['Where are your headquarters located?', 'location'],
['Throw my cellphone in the water', 'random'],
['Network Access Control?', 'networking'],
['Address', 'location']]

df = pd.DataFrame(data, columns = ['text', 'category'])

text = df['text']
encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
vectors = encoder.encode(text)

vector_dimension = vectors.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(vectors)
index.add(vectors)

search_text = 'where is your office?'
search_vector = encoder.encode(search_text)
_vector = np.array([search_vector])
faiss.normalize_L2(_vector)

k = 4
distances, ann = index.search(_vector, k=k)

results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})

merge = pd.merge(results,df,left_on='ann',right_index=True)

merge.head()