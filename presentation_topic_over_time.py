
# coding: utf-8

# <div style="text-align: center"><h1>Topics – Easy Topic Modeling in Python</h1></div>

# The text mining technique **Topic Modeling** has become a popular statistical method for clustering documents. This notebook introduces an user-friendly workflow, basically containing data preprocessing, an implementation of the prototypic topic model **Latent Dirichlet Allocation** (LDA) which learns the relationships between words, topics, and documents, as well as multiple visualizations to explore the trained LDA model.
#
# In this notebook, we're relying on the LDA implementation by [Andrew McCallum](https://people.cs.umass.edu/~mccallum/) called [**MALLET**](https://radimrehurek.com/gensim/).

# ## 1. Preprocessing

# Let's not pay heed to any warnings right now and execute the following cell.

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# #### Loading modules from DARIAH-Topics library
# First, we have to get access to the functionalities of the library by importing them.

# In[2]:


from dariah_topics import preprocessing
from dariah_topics import doclist
#from dariah_topics import meta
from dariah_topics import mallet
#from dariah_topics import visualization


# #### Activating inline output in Jupyter notebook
# The following line will just tell the notebook to show graphics in the output frames.

# In[3]:


#get_ipython().magic('matplotlib inline')


# ### 1.1. Reading a corpus of documents

# #### Defining the path to the corpus folder
#
# In the present example code, we are using a folder of 'txt' documents provided with the package. For using your own corpus, change the path accordingly.

# In[28]:


path = "/mnt/data/corpora/grenzboten/txt"


# #### List all documents in the folder
# We begin by creating a list of all the documents in the folder specified above. That list will tell function `pre.read_from_txt()` (see below) which text documents to read.

# In[4]:


pathdoclist = doclist.PathDocList(path)
document_list = pathdoclist.full_paths(as_str=True)


# The current list of documents looks like this:

# In[30]:


#document_list


# **Alternatively**, if we want to use other documents, or just a selction of those in the specified folder, we can define our own `doclist` by creating a list of strings containing paths to text files. For example, to use only the texts by Edgar A. Poe from the current folder, we would define the list as
#
# `
#     doclist = [ 'corpus_txt/Poe_TheMasqueoftheRedDeath.txt',
#             'corpus_txt/Poe_TheCaskofAmontillado.txt',
#             'corpus_txt/Poe_ThePurloinedLetter.txt',
#             'corpus_txt/Poe_ThePurloinedLetter.txt',
#             'corpus_txt/Poe_EurekaAProsePoem.txt']
# `

# #### Generate document labels

# In[31]:


document_labels = pathdoclist.labels()
#document_labels


# #### Optional: Accessing metadata

# In case you want a more structured overview of your corpus, execute the following cell:

# In[32]:


#import os

#metadata = meta.fn2metadata(os.path.join(path, '*.txt'))
#metadata


# #### Read listed documents from folder

# In[33]:


corpus = preprocessing.read_from_txt(document_list)


# At this point, the corpus is generator object.

# ### 1.3. Tokenize corpus
# Your text files will be tokenized. Tokenization is the task of cutting a stream of characters into linguistic units, simply words or, more precisely, tokens. The tokenize function the library provides is a simple unicode tokenizer. Depending on the corpus it might be useful to use an external tokenizer function, or even develop your own, since its efficiency varies with language, epoch and text type.

# In[34]:


tokens = [list(preprocessing.tokenize(document)) for document in list(corpus)]


# At this point, each text is represented by a list of separate token strings. If we want to look e.g. into the first text (which has the index `0` as Python starts counting at 0) and show its first 10 words/tokens (that have the indeces `0:9` accordingly) by typing:

# In[35]:


print(tokens[0][0:9])


# ### 1.4.1 Create a document-term matrix
#
# The LDA topic model is based on a [document-term matrix](https://en.wikipedia.org/wiki/Document-term_matrix) of the corpus. To improve performance in large corpora, the matrix describes the frequency of terms that occur in the collection. In a document-term matrix, rows correspond to documents in the collection and columns correspond to terms.

# In[36]:


#doc_terms = preprocessing.create_doc_term_matrix(tokens, document_labels)
#doc_terms


# ### 1.4.2 Create a sparse bag-of-words model
#
# The LDA topic model is based on a bag-of-words model of the corpus. To improve performance in large corpora, actual words and document titels are replaced by indices in the actual bag-of-words model. It is therefore necessary to create dictionaries for mapping these indices in advance.

# #### Create Dictionaries

# In[37]:


id_types = preprocessing.create_dictionary(tokens)
doc_ids = preprocessing.create_dictionary(document_labels)


# #### Create matrix market

# In[38]:


sparse_bow = preprocessing.create_sparse_bow(document_labels, tokens, id_types, doc_ids)


# ### 1.5. Feature selection and/or removal
#
# In topic modeling, it is often usefull (if not vital) to remove some types before modeling. In this example, the 100 most frequent words and the *hapax legomena* in the corpus are listed and removed. Alternatively, the 'feature_list' containing all features to be removed from the corpus can be replaced by, or combined with an external stop word list or any other list of strings containing features we want to remove.
#
# **Note**: For small/normal corpora using a **`doc_term_matrix` (1.5.1)** will be easier to handle. Using **`sparse_bow` (1.5.2)** is recommended for large corpora only.

# ### 1.5.2 Remove features from `sparse_bow`

# #### List the 10 most frequent words

# In[65]:
print("sparse_bow")

mfw3 = preprocessing.find_stopwords(sparse_bow, 3, id_types)


# These are the five most frequent words:

# In[66]:


mfw3


# #### List hapax legomena

# In[67]:


hapax_list = preprocessing.find_hapax(sparse_bow, id_types)


# #### Optional: Use external stopwordlist

# In[52]:


path_to_stopwordlist = "tutorial_supplementals/stopwords/de.txt"

extern_stopwords = [line.strip() for line in open(path_to_stopwordlist, 'r')]


# #### Combine lists

# In[53]:


features = set(mfw3 + hapax_list + extern_stopwords)


# #### Remove features from files

# In[54]:


tokens_cleaned = []

for document in tokens:
    document_clean = preprocessing.remove_features_from_file(document, list(features))
    tokens_cleaned.append(list(document_clean))


# #### Write MALLET import files

# In[55]:


preprocessing.create_mallet_import(tokens_cleaned, document_labels)


# ## 1. Setting the parameters

# #### Define path to corpus folder

# In[56]:


path_to_corpus = "tutorial_supplementals/mallet_input"


# #### Path to mallet folder
#
# Now we must tell the library where to find the local instance of mallet. If you managed to install Mallet, it is sufficient set `path_to_mallet = "mallet"`, if you just store Mallet in a local folder, you have to specify the path to the binary explictly.

# In[57]:


path_to_mallet = 'mallet'


# #### Output folder

# In[58]:


outfolder = "tutorial_supplementals/mallet_output"
binary = "tutorial_supplementals/mallet_output/binary.mallet"


# #### The Mallet corpus model
#
# Finally, we can give all these folder paths to a Mallet function that handles all the preprocessing steps and creates a Mallet-specific corpus model object.

# In[59]:


get_ipython().run_cell_magic('time', '', '\nmallet_binary = mallet.create_mallet_binary(path_to_mallet=path_to_mallet,\n                                            path_to_corpus=path_to_corpus,\n                                            output_file=binary) ')


# ## 2. Model creation

# We can define the number of topics we want to calculate as an argument (`num_topics`) in the function. Furthermore, the number of iterations (`num_iterations`) can be defined. A higher number of iterations will probably yield a better model, but also increases processing time.
#
# **Warning: this step can take quite a while!** Meaning something between some seconds and some hours depending on corpus size and the number of iterations. Our example short stories corpus should be done within a minute or two at `num_iterations=5000`.

# In[60]:


get_ipython().run_cell_magic('time', '', '\nmallet.create_mallet_model(path_to_mallet=path_to_mallet, \n                           path_to_binary=mallet_binary, \n                           folder_for_output=outfolder,\n                           num_iterations=5000,\n                           num_topics=20,\n                           output_model=True)')


# ### 2.4. Create document-topic matrix
#
# The generated model object can now be translated into a human-readable document-topic matrix (that is a actually a pandas data frame) that constitutes our principle exchange format for topic modeling results. For generating the matrix from a Gensim model, we can use the following function:

# In[61]:


doc_topic = mallet.show_doc_topic_matrix(outfolder)
doc_topic


# ## 3. Visualization

# Now we can see the topics in the model with the following function:
#
# **Hint:** Depending on the number of topics chosen in step 2, you might have to adjust *num_topics* in this step accordingly.

# In[63]:


mallet.show_topics_keys("tutorial_supplementals/mallet_output", num_topics=20)


# ### 3.1. Distribution of topics

# #### Distribution of topics over all documents
#
# The distribution of topics over all documents can now be visualized in a heat map:

# In[64]:


heatmap = visualization.doc_topic_heatmap(doc_topic)
heatmap.show()


# #### Distribution of topics in a single documents
#
# To take closer look on the topics in a single text, we can use the follwing function that shows all the topics in a text and their respective proportions. To select the document, we have to give its index to the function.

#
#     Example:
#         >>> doc_labels = ['examplefile']
#         >>> doc_tokens_cleaned = [['short', 'example', 'text']]
#         >>> create_mallet_import(doc_tokens_cleaned, doc_labels)
#         >>> outpath = os.path.join('tutorial_supplementals', 'mallet_input')
#         >>> os.path.isfile(os.path.join(outpath, 'examplefile.txt'))
#         True

# In[ ]:
