import torch
import os
import json
import torch.nn.functional as F
from ot.backend import get_backend
import ot
import openai
import re
def get_vecs(model, tokenizer, sentence, args):
    model.eval()
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {_key:encoded_input[_key].to(args.device)  for _key in encoded_input}
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        cls_vecs = model_output[0][:, 0]
    # normalize embeddings
    cls_vecs = torch.nn.functional.normalize(cls_vecs, p=2, dim=1)
    word_vecs = model_output[0]
    return cls_vecs, word_vecs
def compute_weights_uniform(s1_word_embeddigs, s2_word_embeddigs):
    s1_weights = torch.ones(s1_word_embeddigs.shape[0], dtype=torch.float64, device='cuda')
    s2_weights = torch.ones(s2_word_embeddigs.shape[0], dtype=torch.float64, device='cuda')


    # # Uniform weights to make L2 norm=1
    # s1_weights /= torch.linalg.norm(s1_weights)
    # s2_weights /= torch.linalg.norm(s2_weights)

    return s1_weights, s2_weights
def compute_distance_matrix_cosine(s1_word_embeddigs, s2_word_embeddigs, distortion_ratio):
    C = (torch.matmul(F.normalize(s1_word_embeddigs), F.normalize(s2_word_embeddigs).t()) + 1.0) / 2  # Range 0-1
    C = apply_distortion(C, distortion_ratio)
    C = min_max_scaling(C)  # Range 0-1
    C = 1.0 - C  # Convert to distance
    return C
def min_max_scaling(C):
    eps = 1e-10
    # Min-max scaling for stabilization
    nx = get_backend(C)
    C_min = nx.min(C)
    C_max = nx.max(C)
    C = (C - C_min + eps) / (C_max - C_min + eps)
    return C

def apply_distortion(sim_matrix, ratio):
    shape = sim_matrix.shape
    if (shape[0] < 2 or shape[1] < 2) or ratio == 0.0:
        return sim_matrix

    pos_x = torch.tensor([[y / float(shape[1] - 1) for y in range(shape[1])] for x in range(shape[0])],
                         device='cuda')
    pos_y = torch.tensor([[x / float(shape[0] - 1) for x in range(shape[0])] for y in range(shape[1])],
                         device='cuda')
    distortion_mask = 1.0 - ((pos_x - pos_y.T) ** 2) * ratio

    sim_matrix = torch.mul(sim_matrix, distortion_mask)

    return sim_matrix

def get_predicate_premise(args):
    c_predicates = {}
    data_path = os.path.join(args.data, args.dataset)
    with open(os.path.join(data_path, 'c_predicates.jsonl'), encoding = 'utf-8') as f:
        for line in f.readlines():
            data = json.loads(line)
            c_predicates[data['_id']] = {"title": data['title'], "text": data['text']}
    c_premises = {}
    with open(os.path.join(data_path, 'c_premises.jsonl'), encoding = 'utf-8') as f:
        for line in f.readlines():
            data = json.loads(line)
            c_premises[data['_id']] = {"title": data['title'], "text": data['text']}
    q_predicates = {}
    with open(os.path.join(data_path, 'q_predicates.jsonl'), encoding='utf-8') as f:
        for line in f.readlines():
            data = json.loads(line)
            q_predicates[data['_id']] = {"text": data['text']}
    q_premises = {}
    with open(os.path.join(data_path, 'q_premises.jsonl'), encoding='utf-8') as f:
        for line in f.readlines():
            data = json.loads(line)
            q_premises[data['_id']] = {"text": data['text']}
    return c_predicates, c_premises, q_predicates, q_premises


def comvert_to_numpy(s1_weights, s2_weights, C):
    if torch.is_tensor(s1_weights):
        s1_weights = s1_weights.cpu().numpy()
        s2_weights = s2_weights.cpu().numpy()
    if torch.is_tensor(C):
        C = C.cpu().numpy()

    return s1_weights, s2_weights, C
def updated_embeddings(model, tokenizer, text, premise, args):
    text_cls_vecs, text_word_vecs = get_vecs(model, tokenizer, text, args)
    premise_cls_vecs, premise_word_vecs = get_vecs(model, tokenizer, premise, args)
    text_word_vecs, premise_word_vecs = text_word_vecs.squeeze(0), premise_word_vecs.squeeze(0)

    C = compute_distance_matrix_cosine(text_word_vecs, premise_word_vecs, args.distotion)
    print("**************Logic Alignment****************")
    text_word_weights, premise_word_weights = compute_weights_uniform(text_word_vecs, premise_word_vecs)
    text_word_weights = text_word_weights / text_word_weights.sum()
    premise_word_weights = premise_word_weights / premise_word_weights.sum()
    text_word_weights, premise_word_weights, C = comvert_to_numpy(text_word_weights, premise_word_weights,C)
    if args.sinkhorn:
        P = ot.bregman.sinkhorn_log(text_word_weights, premise_word_weights, C, reg=args.epsilon, stopThr=args.stopThr,
                                             numItermax=args.numItermax)
    else:
        P = ot.emd(text_word_weights, premise_word_weights, C)
    P = torch.from_numpy(P).float().to(args.device)
    # query_embedding = text_word_vecs.T @ P @ premise_word_vecs @ text_cls_vecs.squeeze(0)
    query_embedding = torch.einsum("md, mn, nd, d-> d", [text_word_vecs, P, premise_word_vecs, text_cls_vecs])
    m = text_word_vecs.shape[0]
    n = premise_word_vecs.shape[0]
    d = premise_word_vecs.shape[1]
    print("**************Connective Constraint****************")
    tokens = tokenizer.tokenize(premise)
    tokens_ = []
    for token in tokens:
        if token == '¬':
            tokens_.append(-1)
        elif token == '∧' or token == '∨' or token == '⊕' or token == '→'or token == '↔'or token == '∀'or token == '∃':
            tokens_.append(1)
        else:
            tokens_.append(0)
    tokens_ = [0] + tokens_ + [0]
    tokens_ = tokens_[:512]
    tokens = torch.tensor(tokens_).to(args.device).unsqueeze(1).expand(-1, m).T
    tokens[((tokens == -1) & (P > 0)) | ((tokens == 1) & (P > 0))] = 0;
    tokens = tokens.unsqueeze(2).expand(-1, -1, d)
    text_word_vecs_ = text_word_vecs.unsqueeze(1).expand(-1, n, -1)
    premise_word_vecs_ = premise_word_vecs.unsqueeze(0).expand(m, -1, -1)
    attention = torch.einsum("nd, mnd->nm", [premise_word_vecs, text_word_vecs_ + tokens * premise_word_vecs_])
    attention = torch.softmax(attention, dim=1)
    premise_text_ = torch.einsum("nm, mnd->nd", [attention, text_word_vecs_ + tokens * premise_word_vecs_])

    return query_embedding, premise_text_.mean( dim = 0)


def get_cos_score(query, document):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    return (cos(query, document) + 1)/2
def query2fol(query, args):
    client = openai.OpenAI(
        api_key= args.api_key,
        base_url=''
    )
    with open(os.path.join('./data', 'question.txt'), 'r', encoding = 'utf-8') as f:
        prompt = f.read()
    prompt_input = prompt.replace("[[QUESTION]]", query)
    request_params = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system",
             "content": "You are a helpful assistant. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."},
            {"role": "user", "content": prompt_input},
        ],
    }
    response = client.chat.completions.create(**request_params)
    response = response.model_dump()
    generated_text = response['choices'][0]['message']['content'].strip()
    pattern = re.compile(r'(.*)(?=Predicates:)|(?<=Predicates:)(.*?)(?=Conclusion:)|(?<=Conclusion:)(.*?)(?=\Z)',
                         re.DOTALL)
    matches = pattern.findall(generated_text)
    query_premise = " ".join([q.split(" ::: ")[0] for q in matches[2][2].strip().split('\n')])
    print("************generate FOL-query************")
    return query_premise
def doc2fol(document, args):
    client = openai.OpenAI(
        api_key= args.api_key,
        base_url=''
    )
    with open(os.path.join('./data', 'problem.txt'), 'r', encoding = 'utf-8') as f:
        prompt = f.read()
    prompt_input = prompt.replace("[[CONTEXT]]", document)
    request_params = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system",
             "content": "You are a helpful assistant. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."},
            {"role": "user", "content": prompt_input},
        ],
    }
    response = client.chat.completions.create(**request_params)
    response = response.model_dump()
    generated_text = response['choices'][0]['message']['content'].strip()
    pattern = re.compile(r'(.*)(?=Predicates:)|(?<=Predicates:)(.*?)(?=Premises:)|(?<=Premises:)(.*?)(?=\Z)',
                         re.DOTALL)
    matches = pattern.findall(generated_text)
    doc_premise = " ".join([q.split(" ::: ")[0] for q in matches[2][2].strip().split('\n')])
    print("************generate FOL-document************")
    return doc_premise
