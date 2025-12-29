from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
import time
from transformers import AutoTokenizer, AutoModel
from utils import *
from data_loader import GenericDataLoader
from init_parameter import init_model
parser = init_model()
args = parser.parse_args()

dataset = args.dataset
print("************{}****************".format(dataset))
model_path = args.model_path
device = args.device
Settings.embed_model = HuggingFaceEmbedding(model_name = model_path, device = device)


texts = []
corpus_dict = {}

data_path = os.path.join(args.data, dataset)
storage_path = os.path.join(args.storage, dataset)
# Load data
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
for doc_id in corpus:
    texts.append(corpus[doc_id]["text"])
    corpus_dict[corpus[doc_id]["text"]] = doc_id
documents = [Document(text=t) for t in texts]
time_start = time.time()

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
#Build index
print("************Build index****************")
splitter = SentenceSplitter(chunk_size=1000000)
index = VectorStoreIndex.from_documents(documents,transformations=[splitter])
VIretriever = VectorIndexRetriever(index=index,similarity_top_k=100)


from collections import OrderedDict
#Initial model
print("**************Initial model**************")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)
if args.model_path == "bge-small":
    args.dim = 384
elif args.model_path == "bge-large":
    args.dim = 1024
results = {}
for query_id in queries:
    query = queries[query_id]
    print("**************Retrieve Top-K documents**************")
    bge_nodes = VIretriever.retrieve(query) #Retrieve Top-K documents
    bge_scores = {}
    doc_ids = []
    doc_embeddings  = torch.empty((0, args.dim )).to(args.device)
    doc_word_embeddings = torch.empty((0, args.dim)).to(args.device)
    for node in bge_nodes:
        document = node.text
        doc_id = corpus_dict[document]
        bge_scores[doc_id] = node.score
        doc_premise = doc2fol(document, args)
        doc_embedding,doc_word_embedding  = updated_embeddings(model, tokenizer, document, doc_premise, args) #Obtain embeddings of document in NL and FOL
        doc_embeddings = torch.cat((doc_embeddings, doc_embedding.unsqueeze(0)))
        doc_word_embeddings = torch.cat((doc_word_embeddings, doc_word_embedding.unsqueeze(0)))
        doc_ids.append(doc_id)
    query_premise = query2fol(query, args)
    query_embedding, query_word_embedding = updated_embeddings(model, tokenizer, query, query_premise, args) #Obtain embeddings of query in NL and FOL
    scores1 = ((torch.nn.CosineSimilarity(dim=1, eps=1e-6)(query_embedding, doc_embeddings) + 1)/2).cpu().numpy()
    scores2 = ((torch.nn.CosineSimilarity(dim=1, eps=1e-6)(query_word_embedding, doc_word_embeddings) + 1) / 2).cpu().numpy()

    for doc_id, score1, score2 in zip(doc_ids, scores1, scores2):
        bge_scores[doc_id] = (score1 + score2)
    results[query_id] = bge_scores
    results[query_id] = OrderedDict(sorted(bge_scores.items(), key=lambda item: item[1], reverse=True)) #rerank the candidate documents
print("**************Experimental Results**************")
k_values = [1, 3, 5, 10, 100]
from evaluate import *
ndcg, _map, recall, precision = evaluate(qrels, results, k_values)
print(ndcg)
print(_map)
print(recall)
print(precision)
time_end = time.time() #结束计时
time_c= time_end - time_start #运行所花时间
print('程序运行时间为: %s Seconds'%(time_c))