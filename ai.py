'''
The StarAI
===
'''
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import transformers
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.ops.numpy_ops import np_config 
import os
import random
import math
import re
from sentence_transformers import SentenceTransformer, util
import filecompiler as fc
np_config.enable_numpy_behavior()   

# # # Complete data collection
# # data = []
# # code = os.listdir("test_dataset2")
# # for dir in code:
# #     try:
# #         more_dirs = os.listdir(os.path.join(os.getcwd(), "test_dataset2\\"+dir))
# #         for directory in more_dirs:
# #             reader =  os.open(os.path.join(os.getcwd(), f"test_dataset2\\{dir}\\"+directory),flags=os.O_RDONLY)
# #             data.append(os.read(reader, os.path.getsize(reader)))
# #             os.close(reader)
# #     except Exception as err:
# #         reader =  os.open(os.path.join(os.getcwd(), "test_dataset2\\"+dir),flags=os.O_RDONLY)
# #         data.append(os.read(reader, os.path.getsize(reader)))
# #         os.close(reader)
# # writer = open("all_data.txt", "w")
# # [writer.write(str(x)[2:-1]+"\n\n") for x in data]


# # Demo code of executing non-eagerly
# # @tf.function
# # def compute_sum():
# #     a = tf.constant([1,2,3])
# #     b = tf.constant([4,5,6])
# #     c = a + b
# #     print(tf.executing_eagerly())
# #     return c
# # print(compute_sum())
# # quit()

# # # Create a new tokenizer
# tokenizer = Tokenizer(models.BPE())

# # # Customize pre-tokenization and decoding
# tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
# tokenizer.decoder = decoders.ByteLevel()

# # # Train the tokenizer on the dataset
# trainer = trainers.BpeTrainer(special_tokens=["[PAD]","[BOS]","[EOS]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])
# files = fc.uno_compile()  # Replace with the path to your training data
# tokenizer.train(files, trainer)

# # # Save the trained tokenizer
# tokenizer.save("code_gpt_tokenizer.json")

# # Load the tokenizer when needed
loaded_tokenizer = Tokenizer.from_file("code_gpt_tokenizer.json")

# Batched, ready-to-go dataset
# @tf.function
# def dothis():
#     compiled_tensor = fc.transformer_compile(loaded_tokenizer)
#     print("compiled")   
#     dataset = tf.data.Dataset.from_tensor_slices(compiled_tensor)
#     print("sliced")
#     dataset.save(os.path.join(os.getcwd(),"compiled_dataset"))
#     print("saved")
# print("compiling")
# dothis()
# # Tokenize text
# # text = '''
# # class ThisThing:
# #     def __init__(self,thing):
# #         self.thing = thin
# #     def do_something(self):
# #         print(self,"5")'''

# # text = "[BOS] So, how do you calculate 1 + 2? [EOS]"

# # encoded = loaded_tokenizer.encode(text)

# # Print the tokens
# # print(encoded.tokens)
# # print(len(encoded.ids))
# # arr = []
# # b = []
# # all = []
# # for y in range(5):
# #     a = 0
# #     for x in range(20):
# #         if y != 0: 
# #             arr.append(int((np.random.randint(0,5000))*b[y-1]))
# #             arr.append(0)
# #         else:
# #             arr.append(np.random.randint(0,5000))
# #             arr.append(0)
# #         [all.append(y) for y in arr[(x*20):(x+1*20)]]
# #     decoded = loaded_tokenizer.decode(arr)
# #     print(decoded)
# #     a = input("Enter a probability distribution between 0 and 1 to show how much you approve/disaprove of the previously generated message. ")
# #     b.append(float(a))
# #     arr = []

# # Decode if needed
# # decoded = loaded_tokenizer.decode(encoded.ids)

# # Print the decoded text
# # print(decoded)

class Inputs(keras.layers.Layer):
    '''
        The batched output is as follows: (batch_size, 5, max_sequence_length)\n
        Where: \n
        `batch_size` is the batch size\n
        `5` is the number of elements, all index as 0,1,2,3,4. The elements being: `sequence` (index at 0), `raw_sequence` (1), `attention_mask` (2), `full_text` (3), `mode` (4) 
        `max_sequence_length` is the maximum sequence length specified by none other but you.
    '''
    def __init__(self,max_sequence_length):
        super(Inputs,self).__init__()
        self.max_sequence_length = max_sequence_length
    def call(self,inputs):
        outputs = []
        pseudooutputs = {}
        # encoded = loaded_tokenizer.encode(inputs)
        encoded_ids = tf.constant(tf.sqrt(tf.square(inputs[0])))
        print(encoded_ids)
        quit()
        if len(encoded_ids) > self.max_sequence_length:
            sequence = [id for id in encoded_ids[-(self.max_sequence_length):-1] + encoded_ids[-1]]
        elif len(encoded_ids) < self.max_sequence_length:
            while len(encoded_ids) < self.max_sequence_length:
                encoded_ids.append(0)
            sequence = encoded_ids
        elif len(encoded_ids) == self.max_sequence_length:
            sequence = encoded_ids
        sequence = [[float(s)] for s in sequence]
        # raw_sequence = [[float(rs)] for rs in raw_sequence]
        sequence = tf.constant([sequence], dtype=tf.float32)
        # raw_sequence = tf.constant([raw_sequence],dtype=tf.float32)
        padding_mask = (sequence != 0).astype(float)
        padding_mask = tf.constant(padding_mask, dtype=tf.float32)
        padding_mask = padding_mask.reshape(sequence.shape[0],sequence.shape[1])
        pseudooutputs["sequence"] = sequence
        pseudooutputs["attention_mask"] = padding_mask
        outputs.append(pseudooutputs["sequence"])
        outputs.append(pseudooutputs["attention_mask"])
        return outputs
    
# # Test the custom `Inputs` layer
# # ===========================
# #
# # input_data = ["So, how do you calculate 1 + 2?", "How's life?","I am Iron Man."]
# # output_data = Inputs(input_data)
# #
# # =========================== 
def record_memory_unit(batch):
    with open("dcwp/memory_units.txt", "w") as dcwp:
        dcwp.write("@" + str(batch[0]) + "@%" + str(batch[1]) + "%")

def retrieve_memory_units():
    mu = open("dcwp/memory_units.txt","r")
    text = mu.read().split("\n")
    memory_unit_vals = []
    memory_unit_attn = []
    for y in text:
        pattern = re.findall('^.*([@].*[@])',y)[0][1:-1]
        memory_unit_vals.append(pattern)
    for x in text:
        pattern = re.findall('^.*([%].*[%])',x)[0][1:-1]
        memory_unit_attn.append(pattern)
    for i in range(len(memory_unit_vals)):
        memory_unit_vals[i] = memory_unit_vals[i].split(",")
        memory_unit_vals[i][0] = memory_unit_vals[i][0][1:]
        memory_unit_vals[i][-1] = memory_unit_vals[i][-1][0:-1]
    for i in range(len(memory_unit_attn)):
        memory_unit_attn[i] = memory_unit_attn[i].split(",")
        memory_unit_attn[i][0] = memory_unit_attn[i][0][1:]
        memory_unit_attn[i][-1] = memory_unit_attn[i][-1][0:-1]

    for i in range(len(memory_unit_vals)):
        for j in range(len(memory_unit_vals[i])):
            if memory_unit_vals[i][j] != "":memory_unit_vals[i][j] = float(memory_unit_vals[i][j])
            else: del memory_unit_vals[i][j]
    for i in range(len(memory_unit_attn)):
        for j in range(len(memory_unit_attn[i])):
            if memory_unit_attn[i][j] != "": memory_unit_attn[i][j] = float(memory_unit_attn[i][j])
            else: del memory_unit_attn[i][j]
    return [memory_unit_vals, memory_unit_attn]

class DCWP(keras.layers.Layer):
    '''
        `ccw_tokens` - Captured context window tokens.\n
        `current_point` - An integer indicating which epoch/user prompt it is. Starting from 1.\n
        `passthrough` - An integer specifying whether it is the passthrough the entire neural network in which text is being generated or the one in which the comprehensiveness of the model output is being assesed. Specify `1` for text generation, `2` for comprehensiveness assessment.\n
        `comprehensiveness` - A float between 0 and 1 specifying whether or not the last output was comprehensive enough. Can only be specified *after* the first epoch/user prompt.\n
        `full_text` - A string representing the entirety of the tokenized text. Can only be specified *after* the first epoch/user prompt.\n
        `current_scroll` - An integer specifying how many scrolls upwards the DCWP has already, in previous epochs/responses, went upwards in the text. Specify 0 if none.\n
        `full_attention_mask` - A string representing the entire attention mask for all the text. Can only be specified *after* the first epoch/user prompt.\n
    '''  
    def __init__(self,current_point,text_classification_model, seq_length):
        super(DCWP,self).__init__()
        self.current_point = current_point
        self.text_classification_model = text_classification_model
        self.seq_length = seq_length
        
    def call(self,ccw_tokens):
        record_memory_unit(ccw_tokens[0])
        if self.current_point > 1:
            mem_units = retrieve_memory_units()
            memory_unit_vals,memory_unit_attn = mem_units[0],mem_units[1]
            # memory_unit_keys = []
            # for x in text:
            #     pattern = re.findall("^.*[#].*[#]",x)[0][1:-1]
            #     memory_unit_keys.append(pattern)
            # for i in range(len(memory_unit_keys)):
            #     memory_unit_keys[i] = memory_unit_keys[i].split(",")
            #     memory_unit_keys[i][0] = memory_unit_keys[i][0][1:]
            #     memory_unit_keys[i][-1] = memory_unit_keys[i][-1][0:-1]

            # for i in range(len(memory_unit_keys)):
            #     for j in range(len(memory_unit_keys[i])):
            #         memory_unit_keys[i][j] = float(memory_unit_keys[i][j])
            relevance = []
            memory_units = []
            n1 = 0
            n1+=1
            for y in memory_unit_vals:
                query_emb = self.text_classification_model.encode(ccw_tokens[0])
                doc_emb = self.text_classification_model.encode(y)  
                scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
                dense = keras.layers.Dense(1, activation="softmax")
                prob = dense(scores)
                relevance.append(prob)   
                memory_units.append(y)  
            most_relevant = tf.argmax(relevance)
            if relevance[most_relevant] > 0.65:
                empty = [x for x in ccw_tokens[i][0] if x==0]
                count = len(empty)
                remove = (self.seq_length/2)-count
                if remove !=0:
                    for j in range(len(ccw_tokens[0])):
                        if j <= remove-1:
                            del ccw_tokens[0][j]
                            del ccw_tokens[1][j]
                        else: break
                    for j in range(self.seq_length-len(ccw_tokens[0])):
                        ccw_tokens[0][j+len(ccw_tokens[0])] = memory_units[most_relevant][j]
                        ccw_tokens[1][j+len(ccw_tokens[1])] = memory_unit_attn[most_relevant][j]
                if remove == 0:
                    for k in range(len(ccw_tokens[0])):
                        if ccw_tokens[0][k] == 0: 
                            del ccw_tokens[0][k]
                            del ccw_tokens[1][k]
                    for j in range(self.seq_length/2):
                        ccw_tokens[0][j+len(ccw_tokens[0])] = memory_units[most_relevant][j]
                        ccw_tokens[1][j+len(ccw_tokens[1])] = memory_unit_attn[most_relevant][j]
            return ccw_tokens
        else:
            return ccw_tokens

# # Test the custom DCWP layer:
# # ===========================
# #
# # output_dcwp = DCWP(ccw_tokens=output_data,ccw_attention_mask=output_data,current_point=1,passthrough=1)
# # print(output_dcwp[0][0].shape, output_dcwp[0][1].shape)
# # 
# # ===========================

class TransformerLayer(keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim, num_heads, vocab_size, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.mham = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)
        self.ln1 = keras.layers.LayerNormalization(epsilon=1e-3)
        self.vocab_size = vocab_size
        self.fnn = keras.Sequential([
            keras.layers.Dense(hidden_dim, activation='relu'),
            keras.layers.Dropout(dropout),
            # The point of applying dropout in ML is to keep the model from overfitting. Dropout scales any non-zero input by using the formula: 1/(1 - rate). 
            # Passing the arguement 0.1 would mean 1/(1-0.1), which is 1.11111111.
            # I assume the inputs are multiplied by the result value of the computation of the formula.
            
            keras.layers.Dense(input_dim),
            keras.layers.Dense(self.vocab_size + 1) 
            # Only two dense processing layers (excluding the 3rd and final one). This is due to the design of the transformer architecture and for the purpose of reducing
            # Computational costs.
        ])
        self.add = keras.layers.add
        self.ln2 = keras.layers.LayerNormalization(epsilon=1e-3)
        self.dropout = keras.layers.Dropout(dropout)  
        self.softmax = tf.nn.softmax     
    def call(self,x):
        '''
        `x` - the input tensor, entire sequence, including previously generated tokens.
        '''
        tr_output = []
        original = x[0]
        sequ = x[0]
        attn_mask = x[1]
        if attn_mask != None:
            attn = attn_mask
            mham_output, mham_scores = self.mham(query=sequ, value=sequ, attention_mask=attn, return_attention_scores=True)
            sequ = sequ + self.dropout(mham_output)
            sequ = self.add([original,sequ])  
            sequ = self.ln1(sequ)
            sequ1 = sequ
            fnn_output = self.fnn(mham_output) 
            sequ = sequ + self.dropout(fnn_output)
            sequ = self.add([sequ1,sequ])
            sequ = self.ln2(sequ)
            sequ = self.softmax(sequ, axis=-1)
            tr_output = sequ
        else:
            mham_output, mham_scores = self.mham(query=sequ ,value=sequ,return_attention_scores=True)
            sequ = sequ + self.dropout(mham_output)
            sequ = self.add([original,sequ])
            sequ = self.ln1(sequ)
            sequ1 = sequ
            fnn_output = self.fnn(mham_output)
            sequ = sequ + self.dropout(fnn_output)
            sequ = self.add([sequ1,sequ])
            sequ = self.ln2(sequ)
            sequ = self.softmax(sequ, axis=-1) 
            tr_output = sequ
        return tf.constant(tr_output)
    def info(self):
        return len(self.mham.weights) + len(self.fnn.weights)

def TokenGeneration(logits, seq_pos, p):        
        # An implementation of Nucleus Sampling. See: https://pubs.cs.uct.ac.za/id/eprint/1407/1/the_curious_case_of_neural_text_degeneration.pdf
        prob_seq = logits[seq_pos]
        sorted_tokens = tf.argsort(prob_seq, direction='DESCENDING')
        sorted_arr = tf.sort(prob_seq, direction='DESCENDING')
        digit = 0
        gen_seq = []
        for x, y in zip(sorted_arr, sorted_tokens):
            digit += x
            gen_seq.append(y)
            if digit >= p:
                break
        gen_token = random.choice(gen_seq)
        return [tf.constant(np.array([[[gen_token]]]), dtype=tf.float32),[[1]]]
    
# # Test the custom Transformer layer & the custom Token Generation layer:
# # ===========================
# #
# class Iteration(keras.layers.Layer):
#     def __init__(self, model, seq_length : int):
#         super(Iteration, self).__init__()
#         self.model = model
#         self.seq_length = seq_length
#     def call(self,sequ : list):
#         print("ITERATING...")
#         seq = []
#         output_seq = sequ
#         for i in range(self.seq_length):
#             logits = self.model(output_seq)
#             tok = TokenGeneration(logits, i, 0, 0.65)
#             del output_seq[0]
#             output_seq.append(tok)
#             print("Better be 512",len(output_seq))
#             seq.append(tok)
#         return tf.constant(output_seq, dtype=tf.float32)

def Iteration(model, seq_length, sequ : list):
    print("ITERATING...")
    seq = []
    output_seq = sequ
    for i in range(seq_length):
        logits = model(output_seq)
        tok = TokenGeneration(logits, i, 0.65)
        del output_seq[0]
        output_seq.append(tok)
        print("Better be 512",len(output_seq))
        seq.append(tok)
    return tf.constant(output_seq, dtype=tf.float32)   


        # concat_seq = dcwp_return
        # concat_attn = dcwp_return
        # all_tokens = []
        # n = 0
        # for batch in concat_seq:
        #     all_tokens.append([])
        #     for i in range(context_win):
        #         output_tr = self.model(concat_seq[n], 512)
        #         token_gen = TokenGeneration
        #         output_token = token_gen(output_tr, i,0,0.02)
        #         check = output_token[0].numpy().flatten()[0]
        #         if check == 30000: break
        #         else: print("Checking...", check)
        #         concat_seq[n][0] = tf.concat([concat_seq[n][0], output_token[0]], axis=-2)
        #         concat_attn[n][1] = tf.concat([concat_attn[n][1], output_token[1]], axis=-1)
        #         all_tokens[n].append(output_token[0])
        #         print("Num iter:",i)
        #     n += 1
        # return_tokens = []
        # n1 = 0 
        # for batch in all_tokens:
        #     return_tokens.append([])
        #     for tok in batch:
        #         return_tokens[n1].append(int(tok.numpy().flatten()))
        #     n1 += 1
        # return all_tokens
   
# # print(list(all_tokens))
# # print(loaded_tokenizer.decode(list(all_tokens)))

# print(''' NOTE: DO NOT FORGET TO ADD [_*<|endofprompt|>*_] AND [_*<|endofreply|>*_] TOKENS! ''')
# inputs = Inputs("So, how do you calculate 1 + 2?", 512)
# x = DCWP(inputs,current_point=1,passthrough=1)
# outputs = Iteration(x,512)(x)
# x = TransformerLayer(512, 128, num_heads=6, vocab_size=29999, dropout=0.1)(x)
# x = TransformerLayer(512, 128, num_heads=6, vocab_size=29999, dropout=0.1)(x)
# x = TransformerLayer(512, 128, num_heads=6, vocab_size=29999, dropout=0.1)(x)
# x = TransformerLayer(512, 128, num_heads=6, vocab_size=29999, dropout=0.1)(x)
# x = TransformerLayer(512, 128, num_heads=6, vocab_size=29999, dropout=0.1)(x)
# x = TransformerLayer(512, 128, num_heads=6, vocab_size=29999, dropout=0.1)(x)
# x = TransformerLayer(512, 128, num_heads=6, vocab_size=29999, dropout=0.1)(x)
# x = TransformerLayer(512, 128, num_heads=6, vocab_size=29999, dropout=0.1)(x)
# x = TransformerLayer(512, 128, num_heads=6, vocab_size=29999, dropout=0.1)(x)
# x = TransformerLayer(512, 128, num_heads=6, vocab_size=29999, dropout=0.1)(x)
# x = TransformerLayer(512, 128, num_heads=6, vocab_size=29999, dropout=0.1)(x)
# x = TransformerLayer(512, 128, num_heads=6, vocab_size=29999, dropout=0.1)(x)
# outputs = TokenGeneration(x)


# model = keras.models.Model(inputs=inputs,outputs=outputs)
# parameters = model.predict("So, how do you calculate 1 + 2?")
# print(parameters)

def loss(y_true,y_pred):
    '''
    Args:
        `real_seq` - The real sequence of shape (batch_size, sequence_length). Regular tokenized sequence. Ensure special token `30000` is included in the sequence.\n
        `pred_seq` - The model's predicted sequence of shape (batch_size, sequence_length), being, for example, (1, 512)\n
    Call args:
        `vocab_size` - The size of the tokenized vocabulary. Also happens to be the last dim value of `pred_seq` - `vocab_size`
    '''
    # one_hot = []
    # n1 = 0
    # for batch in real_seq:
    #     one_hot.append([])
    #     n1+=1
    #     for token in batch: 
    #         for i in range(vocab_size):
    #             if token == i:
    #                 one_hot.append(1)
    #             else:
    #                 one_hot.append(0)
    # one_hot = tf.constant(one_hot)
    print(y_true.shape)
    print(y_pred.shape)
    sparse = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = sparse(y_true, y_pred)
    return loss

# ''' 
# NOTE:
# `keras.optimizers.experimental.AdamW`
# is a real optimizer!
# '''
# t = 0

def get_mov_avg_minus_one(arg): 
    with open(f"optimizer-data/{arg}.txt", "r") as move:
        text = move.read().split("#")
        first_mov = text[1].split("[")[1].split(",")
        last_val_fm = first_mov[-1].split("]")[0]
        del first_mov[-1]
        first_mov.append(last_val_fm)
        second_mov = text[2].split("[")[1].split(",")
        last_val_sm = second_mov[-1].split("]")[0]
        del second_mov[-1]
        second_mov.append(last_val_sm)
        mt = []
        vt = []
        for x, y in zip(first_mov, second_mov):
            print(x,y)
            try:
                n1 = float(x)
                n2 = float(y)
                mt.append(n1)
                vt.append(n2)
            except:
                print("Not a number")
        return tf.constant(mt, dtype=tf.float32), tf.constant(vt, dtype=tf.float32)
    
# # def get_params_minus_one():
# #     with open("adamw-data/params.txt", "r") as params:
# #         text = params.read().split("#")
# #         params = text[1].split("[")[1].split(",")
# #         last_val = params[-1].split("]")[0]
# #         del params[-1]
# #         params.append(last_val)
# #         theta = [float(x) for x in params]
# #         return tf.constant(theta, dtype=tf.float32)

# # def get_loss_minus_one():
# #     with open("adamw-data/loss.txt", "r") as loss:
# #         text = loss.read().split("#")
# #         params = text[1].split("[")[1].split(",")
# #         last_val = params[-1].split("]")[0]
# #         del params[-1]
# #         params.append(last_val)
# #         loss = [float(x) for x in params]
# #         return tf.constant(loss, dtype=tf.float32)

def lr_scheduler(epoch):
    warmup_epochs = 3  
    total_epochs = 10
    base_lr = 0.001
    min_lr = 0.0001
    max_lr = 0.01

    if epoch < warmup_epochs:
        lr = (epoch / warmup_epochs) * base_lr
    else:    
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)

        # Cosine function. Creates this type of graph: https://cdn1.byjus.com/wp-content/uploads/2022/07/Cosine-Function_Artboard-3.png
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(progress * math.pi))
    return lr

def architecture_retrieve():
    with open("architecture-evo.txt", "r") as arc:
        text = arc.read().split("#")[1][1:-1].split("],")
        transformed_list = []
        n1 = 0
        for x in text:
            transformed_list.append([])
            for i in x:
                try:
                    i = float(i)
                    transformed_list[n1].append(i)
                except:
                    continue
            n1+=1
        return transformed_list

def AdamW(params: tf.Tensor, a, beta_1, beta_2, t, loss_func,inner_model,iteration,batch, lamb=0.01, epsilon=1e-7):
    '''
    `params` - model parameters. Must be 1D.\n
    `t` - timestep. Initialized as 0.\n
    ... x
    '''
    # Differing calculations - mark those which are new as tf.float64. 
    # 
    # Fill up moving averages with simple zeros if params have been removed since the previous epoch.
    # Fill up past moving averages with zeros if params have been added since the previous epoch.
    if t == 0:
        t = t + 1
        with tf.GradientTape() as g:
            y_preds = []
            y_trues = []
            for seq in batch:
                y_pred = iteration(inner_model, 512, seq[0])
                y_trues.append(seq)
                y_preds.append(y_pred)
            y = loss_func(y_trues,y_preds)
        if params == None:
            g_t = g.gradient(y, inner_model.weights)
        else: 
            g_t = g.gradient(y, params)
        g_t = tf.cast(g_t, dtype=tf.float32)
        m_t = tf.math.add(tf.matmul(beta_1, tf.zeros(params.shape)),tf.matmul((1-beta_1),g_t))
        v_t = tf.math.add(tf.matmul(beta_2, tf.zeros(params.shape)), tf.matmul((1-beta_2), tf.math.pow(g_t,2)))
        with open("optimizer-data/mov_avgs_adam.txt", "w") as move:
            move.write("#" + str(m_t) + "#" + str(v_t))
        m_t = tf.math.divide(m_t,(1-(beta_1**t)))
        v_t = tf.math.divide(v_t,(1-(beta_2**t)))
        a_t = lr_scheduler(t)
        params_t = tf.math.subtract(params, tf.math.multiply(a_t, tf.math.add(tf.math.divide(tf.math.multiply(a,m_t),tf.math.add(tf.math.sqrt(v_t),epsilon)),tf.math.multiply(lamb,params))))
        return params_t
    elif t > 0:
        t = t + 1
        with tf.GradientTape() as g:
            y_preds = []
            y_trues = []
            for seq in batch:
                y_pred = model(seq[0])
                y_trues.append(seq)
                y_preds.append(y_pred)
            y = loss_func(y_trues,y_preds)
        g_t = g.gradient(y, model.trainable_variables)
        g_t = tf.cast(g_t, dtype=tf.float32)
        past_mt, past_vt = get_mov_avg_minus_one("mov_avgs_adam")
        if params.shape[0] < past_mt.shape[0]:
            for x in range(past_mt.shape[0] - params.shape[0]):
                params = tf.concat([params, 0], axis=-1)
        elif past_mt.shape[0] < params.shape[0]:
            for x in range(params.shape[0] - past_mt.shape[0]):
                past_mt = tf.concat([past_mt, 0],axis=-1)
        m_t = tf.math.add(tf.matmul(beta_1, past_mt), tf.matmul((1-beta_1), g_t))
        v_t = tf.math.add(tf.matmul(beta_2, past_vt), tf.matmul((1-beta_2), tf.math.pow(g_t,2)))
        with open("optimizer-data/mov_avgs_adam.txt", "w") as move:
            move.write("#" + str(m_t) + "#" + str(v_t))
        params_t = tf.math.subtract(params, tf.math.divide(tf.math.multiply(a_t, m_t), tf.math.add(tf.math.sqrt(v_t), epsilon)))
        return params_t


def Evo(nn: tf.Tensor, params : tf.Tensor, a, beta_1, beta_2, t, max_layers, min_layers, loss_func=None, model=None, batch=None, epsilon=1e-7):
    '''
    `nn` - the architecture of the model.\n
    `t` - timestep. Initialized as 0.\n
    `params` - a 2D tensor of parameters, as [[params],[params],...[params]] where each tensor within the tensor is a tensor of parameters with respect to an individual transformer layer.
    '''
    if loss_func==None or model==None or batch==None:
        raise Exception("Either `loss_func`, `y_true`, `model` or `batch` is not defined (or a combination of those args are not defined), but it is a required pos arg.")
    # Differing calculations - mark those which are new as tf.float64. 
    # 
    # Fill up moving averages with simple zeros if params have been removed since the previous epoch.
    # Fill up past moving averages with zeros if params have been added since the previous epoch.
    if t ==  0:
        t = t + 1
        g_t = []
        m_t = []
        v_t = []
        flattened = tf.Variable(nn.numpy().flatten() ,dtype=tf.float32,trainable=True)   
        with tf.GradientTape() as tape:
            y_preds = []
            y_trues = []
            for seq in batch:
                y_pred = model(seq[0])
                y_trues.append(seq)
                y_preds.append(y_pred)
            y = loss_func(y_trues,y_preds)
            print("Loss shape", y)
        dL_dnn = tape.gradient(y, flattened)
        g_t = tf.reshape(tf.constant(dL_dnn,dtype=tf.float32), nn.shape)
        sum_gt = []
        for x in g_t:
            sum = 0 
            for y in x:
                sum += y
            sum_gt.append(sum)
        lowest_loss = tf.argmin(sum_gt)
        highest_loss = tf.argmax(sum_gt)
        i = 0
        for x in g_t[highest_loss]:
            g_t[highest_loss][i] = 0
            i+=1
        i = 0
        total_sum = 0
        for x in sum_gt.numpy().flatten():
            total_sum += x
        num_layers = 0
        for x in nn:
            num_layers += 1
        add = round(tf.math.divide((tf.math.sqrt(y)-(tf.math.sqrt(y)/tf.math.sqrt(a))),tf.math.sqrt(a)))
        if add >= 1 and ((add+num_layers)-1) <= max_layers and ((add+num_layers)-1) >= min_layers:
            numpy_nn = nn.numpy().tolist()  
            adjusted_nn = numpy_nn
            adjusted_params = params.numpy().tolist()
            for i in range(add):
                adjusted_nn.insert(lowest_loss+1,nn[lowest_loss].numpy().tolist())
                adjusted_params.insert(lowest_loss+1,params[lowest_loss].numpy().tolist())
            for k in range(tf.constant(adjusted_nn).shape[-1]):
                adjusted_nn[highest_loss][k] = 0 
                params[highest_loss][k] = 0 
        elif add >=1 and ((add+num_layers)-1) > max_layers:
            high_loss = highest_loss
            while True:
                add -= 1
                if ((add+num_layers)-1) <= max_layers: 
                    if add == 0:
                        adjusted_nn = nn.numpy().tolist()
                        adjusted_params = params.numpy().tolist()
                        for i in range(tf.constant(adjusted_nn).shape[-1]):
                            adjusted_nn[high_loss][i] = 0
                            params[high_loss][i] = 0
                        sum_gt[high_loss] = 0
                        high_loss = tf.argmax(sum_gt)
                    elif add >= 1:
                        adjusted_nn = nn.numpy().tolist()
                        adjusted_params = params.numpy().tolist()
                        for i in range(add):
                            adjusted_nn.insert(lowest_loss+1,nn[lowest_loss].numpy().tolist())
                            adjusted_params.insert(lowest_loss+1,params[lowest_loss].numpy().tolist())
                        for k in range(tf.constant(adjusted_nn).shape[-1]):
                            adjusted_nn[highest_loss][k] = 0
                            params[highest_loss][k] = 0 
                        break
                else: continue
        elif add >= 1 and ((add+num_layers)-1) < min_layers:
            adjusted_nn = nn.numpy().tolist()
            adjusted_params = params.numpy().tolist()
            for k in range(tf.constant(adjusted_nn).shape[-1]):
                adjusted_nn[highest_loss][k] = 0
                adjusted_params[highest_loss][k] = 0
            for i in range(add):
                adjusted_nn.insert(lowest_loss+1,nn[lowest_loss].numpy().tolist())
                adjusted_params.insert(lowest_loss+1,params[lowest_loss].numpy().tolist())
            if len(adjusted_nn) < min_layers:
                i = 0
                while True:
                    adjusted_nn.insert(lowest_loss+i,nn[lowest_loss].numpy().tolist())
                    i += 1
                    if len(adjusted_nn) >= min_layers: break
            
        elif add == 0:
            adjusted_nn = nn.numpy().tolist()
            adjusted_params = params.numpy().tolist()
            for k in range(tf.constant(adjusted_nn).shape[-1]):
                adjusted_nn[highest_loss][k] = 0
                adjusted_params[highest_loss][k] = 0
            if len(adjusted_nn) < min_layers:
                i = 0
                while True:
                    adjusted_nn.insert(lowest_loss+1,nn[lowest_loss].numpy().tolist())
                    adjusted_params.insert(lowest_loss+1,params[lowest_loss].numpy().tolist())
                    i += 1
                    if len(adjusted_nn) >= min_layers: break

        # This is defined as a breakpoint for a reason. It is the testing phase after all.
        try:
            print("The adjusted neural network is real",adjusted_nn)
        except Exception as err:
            print("Nope. The `adjusted_nn` doesn't exist after all",str(err))      
        adjusted_nn = tf.constant(adjusted_nn, dtype=tf.float32)

        # Left to do:
        # Custom `nn_params` equation. Record with past moving averages.
        m_t = tf.math.add(tf.math.multiply(beta_1, tf.zeros(nn.shape)),tf.math.multiply((1-beta_1),g_t))
        v_t = tf.math.add(tf.math.multiply(beta_2, tf.zeros(nn.shape)), tf.math.multiply((1-beta_2), tf.math.pow(g_t,2)))
        m_t = tf.math.divide(m_t, (1-(beta_1**t)))
        v_t = tf.math.divide(v_t, (1-(beta_2**t)))
        with open("optimizer-data/mov_avgs_evo.txt", "w") as move:
            move.write("#" + str(m_t) + "#" + str(v_t))
        architecture = tf.math.subtract(adjusted_nn,tf.math.divide(tf.math.multiply(m_t,a), tf.math.add(tf.math.sqrt(v_t, epsilon))))
        return architecture,adjusted_nn
    elif t > 0:
        t = t + 1
        g_t = []
        m_t = []
        v_t = []
        flattened = tf.Variable(nn.numpy().flatten() ,dtype=tf.float32,trainable=True)   
        with tf.GradientTape() as tape:
            y_preds = []
            y_trues = []
            for seq in batch:
                y_pred = model(seq[0])
                y_trues.append(seq)
                y_preds.append(y_pred)
            y = loss_func(y_trues,y_preds)
        dL_dnn = tape.gradient(y, flattened)
        g_t = tf.reshape(tf.constant(dL_dnn,dtype=tf.float32), nn.shape)
        print("gradient shape:",g_t.shape)
        sum_gt = []
        for x in g_t:
            sum = 0 
            for y in x:
                sum += y
            sum_gt.append(sum)
        lowest_loss = tf.argmin(sum_gt)
        highest_loss = tf.argmax(sum_gt)
        i = 0
        for x in g_t[highest_loss]:
            g_t[highest_loss][i] = 0
            i+=1
        i = 0
        total_sum = 0
        for x in sum_gt.numpy().flatten():
            total_sum += x
        num_layers = 0
        for x in nn:
            num_layers += 1
        add = round(tf.math.divide((tf.math.sqrt(y)-(tf.math.sqrt(y)/tf.math.sqrt(a))),tf.math.sqrt(a)))
        if add >= 1 and ((add+num_layers)-1) <= max_layers and ((add+num_layers)-1) >= min_layers:
            numpy_nn = nn.numpy().tolist()
            adjusted_nn = numpy_nn
            adjusted_params = params.numpy().tolist()
            for i in range(add):
                adjusted_nn.insert(lowest_loss+1,nn[lowest_loss].numpy().tolist())
                adjusted_params.insert(lowest_loss+1,params[lowest_loss].numpy().tolist())
            for k in range(tf.constant(adjusted_nn).shape[-1]):
                adjusted_nn[highest_loss] = 0
                adjusted_params[highest_loss][k] = 0
        elif add >=1 and ((add+num_layers)-1) > max_layers:
            high_loss = highest_loss
            while True:
                add -= 1
                if ((add+num_layers)-1) <= max_layers: 
                    if add == 0:
                        adjusted_nn = nn.numpy().tolist()
                        adjusted_params = params.numpy().tolist()
                        for i in range(tf.constant(adjusted_nn).shape[-1]):
                            adjusted_nn[high_loss] = 0
                            adjusted_params[highest_loss][i] = 0
                        sum_gt[high_loss] = 0
                        high_loss = tf.argmax(sum_gt)
                    elif add >= 1:
                        adjusted_nn = nn.numpy().tolist()
                        adjusted_params = params.numpy().tolist()
                        for i in range(add):
                            adjusted_nn = adjusted_nn.insert(lowest_loss+1,nn[lowest_loss].numpy().tolist())
                            adjusted_params.insert(lowest_loss+1,params[lowest_loss].numpy().tolist())
                        for k in range(tf.constant(adjusted_nn).shape[-1]):
                            adjusted_nn[highest_loss][k] = 0
                            adjusted_params[highest_loss][k] = 0
                        break
                else: continue
        elif add >= 1 and ((add+num_layers)-1) < min_layers:
            adjusted_nn = nn.numpy().tolist()
            adjusted_params = params.numpy().tolist()
            for j in range(tf.constant(adjusted_nn).shape[-1]):
                adjusted_nn[highest_loss][j] = 0
                adjusted_params[highest_loss][j] = 0
            for i in range(add):
                adjusted_nn.insert(lowest_loss+1,nn[lowest_loss].numpy().tolist())
                adjusted_params.insert(lowest_loss+1,params[lowest_loss].numpy().tolist())
            if len(adjusted_nn) < min_layers:
                i = 0
                while True:
                    adjusted_nn.insert(lowest_loss+1,nn[lowest_loss].numpy().tolist())
                    adjusted_params.insert(lowest_loss+1,params[lowest_loss].numpy().tolist())
                    i += 1
                    if len(adjusted_nn) >= min_layers: break
            
        elif add == 0:
            adjusted_nn = nn.numpy().tolist()
            adjusted_params = params.numpy().tolist()
            for i in range(tf.constant(adjusted_nn).shape[-1]):
                adjusted_nn[highest_loss][i] = 0
                adjusted_params[highest_loss][i] = 0
            if len(adjusted_nn) < min_layers:
                i = 0
                while True:
                    adjusted_nn.insert(lowest_loss+1,nn[lowest_loss].numpy().tolist())
                    adjusted_params.insert(lowest_loss+1,params[lowest_loss].numpy().tolist())
                    i += 1
                    if len(adjusted_nn) >= min_layers: break

        # This is defined as a breakpoint for a reason. It is the testing phase after all.
        print("Is the adjusted neural network real?",adjusted_nn)
        adjusted_nn = tf.constant(adjusted_nn, dtype=tf.float32)

        # Left to do:
        # Custom `nn_params` equation. Record with past moving averages.
        past_mt, past_vt = get_mov_avg_minus_one("mov_avgs_evo")
        m_t = tf.math.add(tf.math.multiply(beta_1,past_mt),tf.math.multiply((1-beta_1),g_t)) 
        v_t = tf.math.add(tf.math.multiply(beta_1,past_vt),tf.math.multiply((1-beta_2),tf.math.pow(g_t,2))) 
        m_t = tf.cast(tf.math.divide(m_t, (1-(beta_1**t))),dtype=tf.int32)
        v_t = tf.cast(tf.math.divide(v_t, (1-(beta_2**t))),dtype=tf.int32)
        with open("optimizer-data/mov_avgs_evo.txt", "w") as move:
            move.write("#" + str(m_t) + "#" + str(v_t))
        architecture = tf.math.subtract(adjusted_nn,tf.math.divide(tf.math.multiply(m_t,a), tf.math.add(tf.math.sqrt(v_t, epsilon))))
        return architecture,params
# 
# ===========================

def build_model(architecture,params,t):
    p = open("params.txt", "w")
    params = params.numpy().flatten()
    clear = [x!=0 for x in params]
    # p.write("#"+str(clear))
    with open("model.py", "w") as m:
        m.write("import tensorflow as tf\n")
        m.write("from tensorflow import keras\n")
        m.write("from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors\n")
        m.write("import numpy as np\nimport random\n\n")
        m.write("loaded_tokenizer = Tokenizer.from_file('code_gpt_tokenizer.json')\n")
        inputs = '''
class Inputs(keras.layers.Layer):
    def __init__(self,max_sequence_length):
        super(Inputs,self).__init__()
        self.max_sequence_length = max_sequence_length
    def call(self,inputs):
        batched_output = []
        for x in inputs:
            outputs = []
            pseudooutputs = {}
            encoded = loaded_tokenizer.encode(x)
            encoded_ids = encoded.ids
            if len(encoded_ids) > self.max_sequence_length:
                sequence = [id for id in encoded_ids[-(self.max_sequence_length):-1] + encoded_ids[-1]]
            elif len(encoded_ids) < self.max_sequence_length:
                while len(encoded_ids) < self.max_sequence_length:
                    encoded_ids.append(0)
                sequence = encoded_ids
            elif len(encoded_ids) == self.max_sequence_length:
                sequence = encoded_ids
            sequence = [[float(s)] for s in sequence]
            # raw_sequence = [[float(rs)] for rs in raw_sequence]
            sequence = tf.constant([sequence], dtype=tf.float32)
            # raw_sequence = tf.constant([raw_sequence],dtype=tf.float32)
            padding_mask = (sequence != 0).astype(float)
            padding_mask = tf.constant(padding_mask, dtype=tf.float32)
            padding_mask = padding_mask.reshape(sequence.shape[0],sequence.shape[1])
            pseudooutputs["sequence"] = sequence
            pseudooutputs["attention_mask"] = padding_mask
            outputs.append(pseudooutputs["sequence"])
            outputs.append(pseudooutputs["attention_mask"])
            batched_output.append(outputs)
        return batched_output'''
        dcwp_funcs = r'''
def record_memory_unit(batch):
    with open("dcwp/memory_units.txt", "w") as dcwp:
        dcwp.write("@" + str(batch[0]) + "@%" + str(batch[1]) + "%")

def retrieve_memory_units():
    mu = open("dcwp/memory_units.txt","r")
    text = mu.read().split("\n")
    memory_unit_vals = []
    memory_unit_attn = []
    for y in text:
        pattern = re.findall('^.*([@].*[@])',y)[0][1:-1]
        memory_unit_vals.append(pattern)
    for x in text:
        pattern = re.findall('^.*([%].*[%])',y)[0][1:-1]
        memory_unit_attn.append(pattern)
    for i in range(len(memory_unit_vals)):
        memory_unit_vals[i] = memory_unit_vals[i].split(",")
        memory_unit_vals[i][0] = memory_unit_vals[i][0][1:]
        memory_unit_vals[i][-1] = memory_unit_vals[i][-1][0:-1]
    for i in range(len(memory_unit_attn)):
        memory_unit_attn[i] = memory_unit_attn[i].split(",")
        memory_unit_attn[i][0] = memory_unit_attn[i][0][1:]
        memory_unit_attn[i][-1] = memory_unit_attn[i][-1][0:-1]

    for i in range(len(memory_unit_vals)):
        for j in range(len(memory_unit_vals[i])):
            if memory_unit_vals[i][j] != "":memory_unit_vals[i][j] = float(memory_unit_vals[i][j])
            else: del memory_unit_vals[i][j]
    for i in range(len(memory_unit_attn)):
        for j in range(len(memory_unit_attn[i])):
            if memory_unit_attn[i][j] != "": memory_unit_attn[i][j] = float(memory_unit_attn[i][j])
            else: del memory_unit_attn[i][j]
    return memory_unit_vals, memory_unit_attn'''
        dcwp = '''
    \nclass DCWP(keras.layers.Layer):
        def __init__(self,current_point,text_classification_model, seq_length):
            super(DCWP,self).__init__()
            self.current_point = current_point
            self.text_classification_model = text_classification_model
            self.seq_length = seq_length
            
        def call(self,ccw_tokens):
            for batch in range(len(ccw_tokens)):
                record_memory_unit(ccw_tokens[batch])
            if self.current_point > 1:
                memory_unit_vals,memory_unit_attn = retrieve_memory_units()
                # memory_unit_keys = []
                # for x in text:
                #     pattern = re.findall("^.*[#].*[#]",x)[0][1:-1]
                #     memory_unit_keys.append(pattern)
                # for i in range(len(memory_unit_keys)):
                #     memory_unit_keys[i] = memory_unit_keys[i].split(",")
                #     memory_unit_keys[i][0] = memory_unit_keys[i][0][1:]
                #     memory_unit_keys[i][-1] = memory_unit_keys[i][-1][0:-1]

                # for i in range(len(memory_unit_keys)):
                #     for j in range(len(memory_unit_keys[i])):
                #         memory_unit_keys[i][j] = float(memory_unit_keys[i][j])
                relevance = []
                memory_units = []
                n1 = 0
                for batch in ccw_tokens:
                    relevance.append([])
                    memory_units.append([])
                    n1+=1
                    for y in memory_unit_vals:
                        query_emb = self.text_classification_model.encode(batch[0])
                        doc_emb = self.text_classification_model.encode(y)  
                        scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
                        dense = keras.layers.Dense(1, activation="softmax")
                        prob = dense(scores)
                        relevance[n1].append(prob)
                        memory_units[n1].append(y)
                for i in range(len(ccw_tokens)):        
                    most_relevant = float(tf.argmax(tf.constant(relevance[i])))
                    if relevance[i][most_relevant] > 0.65:
                        empty = [x for x in ccw_tokens[i][0] if x==0]
                        count = len(empty)
                        remove = (self.seq_length/2)-count
                        if remove !=0:
                            for j in range(len(ccw_tokens[i])):
                                if j <= remove-1:
                                    del ccw_tokens[i][0][j]
                                else: break
                            for j in range(self.seq_length-len(ccw_tokens[i])):
                                ccw_tokens[i][0][j+len(ccw_tokens[i])] = memory_units[i][most_relevant][j]
                        if remove == 0:
                            for k in range(len(ccw_tokens[i])):
                                if ccw_tokens[i][0][k] == 0: del ccw_tokens[i][k]
                            for j in range(self.seq_length/2):
                                ccw_tokens[i][0][j+len(ccw_tokens[i])] = memory_units[i][most_relevant][j]
                return ccw_tokens
            else:
                return ccw_tokens\n'''
        token_gr = '''
    def TokenGeneration(logits, seq_pos, batch_pos, p):        
            logit = logits[batch_pos]
            prob_seq = logit[seq_pos]
            sorted_tokens = tf.argsort(prob_seq, direction='DESCENDING')
            sorted_arr = tf.sort(prob_seq, direction='DESCENDING')
            digit = 0
            gen_seq = []
            for x, y in zip(sorted_arr, sorted_tokens):
                digit += x
                gen_seq.append(y)
                if digit >= p:
                    break
            gen_token = random.choice(gen_seq)
            return [tf.constant(np.array([[[gen_token]]]), dtype=tf.float32),[[1]]]\n'''
        iter = '''
    class Iteration(keras.layers.Layer):
        def __init__(self, model, seq_length : int):
            super(Iteration, self).__init__()
            self.model = model
            self.seq_length = seq_length
        def call(self,sequ : list):
            print("ITERATING...")
            seq = []
            output_seq = sequ
            for i in range(self.seq_length):
                logits = self.model(output_seq)
                tok = TokenGeneration(logits, i, 0, 0.65)
                del output_seq[0]
                output_seq.append(tok)
                print("Better be 512",len(output_seq))
                seq.append(tok)
            return tf.constant(output_seq, dtype=tf.float32)\n'''
        token_gener = '''
class TransformerLayer(keras.layers.Layer):
	def __init__(self, input_dim, hidden_dim, num_heads, vocab_size, dropout=0.1):
		super(TransformerLayer, self).__init__()
		self.mham = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)
		self.ln1 = keras.layers.LayerNormalization(epsilon=1e-3)
		self.vocab_size = vocab_size
		self.fnn = keras.Sequential([
			keras.layers.Dense(hidden_dim, activation='relu'),
			keras.layers.Dropout(dropout),
			keras.layers.Dense(input_dim),
			keras.layers.Dense(self.vocab_size + 1)])
		self.add = keras.layers.add
		self.ln2 = keras.layers.LayerNormalization(epsilon=1e-3)
		self.dropout = keras.layers.Dropout(dropout)
		self.softmax = tf.nn.softmax
	def call(self,x,attn_mask=None):
		tr_output = []
		original = x
		sequ = x
		if attn_mask != None:
			attn = attn_mask
			mham_output, mham_scores = self.mham(query=sequ, value=sequ, attention_mask=attn, return_attention_scores=True)
			sequ = sequ + self.dropout(mham_output)
			sequ = self.add([original,sequ])  
			sequ = self.ln1(sequ)
			sequ1 = sequ
			fnn_output = self.fnn(mham_output) 
			sequ = sequ + self.dropout(fnn_output)
			sequ = self.add([sequ1,sequ])
			sequ = self.ln2(sequ)
			sequ = self.softmax(sequ, axis=-1)
			tr_output = sequ
		else:
			mham_output, mham_scores = self.mham(query=sequ ,value=sequ,return_attention_scores=True)
			sequ = sequ + self.dropout(mham_output)
			sequ = self.add([original,sequ])
			sequ = self.ln1(sequ)
			sequ1 = sequ
			fnn_output = self.fnn(mham_output)
			sequ = sequ + self.dropout(fnn_output)
			sequ = self.add([sequ1,sequ])
			sequ = self.ln2(sequ)
			sequ = self.softmax(sequ, axis=-1) 
			tr_output = sequ
		return tf.constant(tr_output)\n'''
        parameters = '''params = keras.models.load_model(\n'''
        text_classification_model = "\nmodel = SentenceTransformer('SeyedAli/Multilingual-Text-Semantic-Search-Siamese-BERT-V1')\n"
        inner_model = '''
        inner_model = keras.models.Sequential = ([
            Inputs(max_sequence_length=512),
            DCWP(1,model,512),\n
'''
        m.write(inputs)
        m.write(dcwp_funcs)
        m.write(dcwp)
        m.write(token_gr)
        m.write(iter)
        m.write(token_gener)
        m.write(parameters)
        m.write(text_classification_model)
        m.write(inner_model)
        print("Architecture shape:",architecture.shape[1])
        for i in range(architecture.shape[1]):
            if architecture[i][0] != 0 and architecture[i][1] != 0:
                m.write(f"TransformerLayer(512, {architecture[0][i][0]}, num_heads={architecture[0][i][1]}, vocab_size=30000, dropout=0.1),\n")
        m.write("])\n")
        m.write("inner_model.set_weights(params)")
        m.write("outer_model = Iteration(inner_model, 512)\n")
        m.write("inner_model = keras.models.Model(inputs=inputs,outputs=x)\ninner_model.set_weights(params)\niteration = Iteration(x,512, inner_model)\nouter_model = keras.models.Model(inputs=iteration,outputs=iteration)")
# y_true = tf.constant([[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]],dtype=tf.float32)
# y_pred = tf.constant([[1,3,2,6,4,8],[1,3,2,6,4,8],[1,3,2,6,4,8]],dtype=tf.int32)
# y_pred = tf.one_hot(y_pred, 30001,dtype=tf.float32)
# print(Evo(tf.constant([[3,45,2,5,1],[6,1,63,1,7]]),tf.constant([0.133,-0.6251,0.54,-0.9151]),0.01,0.9,0.999,0,20,5,loss_func=loss,y_true=y_true,y_pred=y_pred))


def assess_comprehensiveness(output, ccw_tokens,model):
    '''
    `output` - The generated GPT output
    `ccw_tokens` - captured context window tokens.
    Call the entire neural network with the question:
    "How comprehensive is this: `output`, with regards to this prompt: `ccw_tokens`, when measured in a float number between 0 and 1, with three decimal points (answer in one word)?
    '''
    # Note: The `ccw_tokens` should be trimmed up to fit within the 512 token context window, or padded up with zeros up to 512.
    return model(f"How comprehensive is this output: '{output}', with regards to this prompt: '{ccw_tokens}', when measured in a float number between 0 and 1, with three decimal points (answer in one word)?")

import subprocess as sp
dataset = tf.data.Dataset.load(os.path.join(os.getcwd(), "compiled_dataset"))
n = 0
for elem in dataset.as_numpy_iterator():
    n+=1
print(n)
batches = []
order = []
for k in range(10):
    order.append(int(k*(n/10)))

def return_iter(elem):
    return elem
for x in order: print(x)
# Current issue: creating 10 batches
i = 0
k = 0
for elem in dataset.as_numpy_iterator():
    print(k)
    if i in order:
        if i == 0:
            batches.append([])
        if i > 0:
            batches.append([])
            k+=1
    batches[k].append(elem)
    i+=1
iterator1 = 0
iterator2 = 1
for t in range(5):
    if t == 0: 
        model = SentenceTransformer('SeyedAli/Multilingual-Text-Semantic-Search-Siamese-BERT-V1')
        inputs_layer = Inputs(max_sequence_length=512)
        inputs = keras.Input((512,))
        x = inputs_layer(inputs)
        x = DCWP(1,model,512)(inputs_layer)
        x = TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1)(x)
        x = TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1)(x)
        x = TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1)(x)
        x = TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1)(x)
        x = TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1)(x)
        x = TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1)(x)  
        x = TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1)(x)
        x = TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1)(x)
        x = TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1)(x)
        x = TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1)(x)
        x = TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1)(x)
        x = TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1)(x)
        x = TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1)(x)
        x = TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1)(x)
        outputs = TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1)(x)
        inner_model = keras.Model(inputs=inputs, outputs=outputs)
        # inner_model = keras.models.Model([
            # Inputs(max_sequence_length=512),
            # DCWP(1,model,512),
            # TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1),
            # TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1),
            # TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1),
            # TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1),
            # TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1),
            # TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1),  
            # TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1),
            # TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1),
            # TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1),
            # TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1),
            # TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1),
            # TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1),
            # TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1),
            # TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1),
            # TransformerLayer(512, 128, num_heads=6, vocab_size=30000, dropout=0.1),
        # ])
        # CURRENT CONCERN: OUTER_MODEL DOESN'T WANT TO BUILD ITSELF
        # outer_model = keras.models.Model(inputs=Iteration(inner_model, 512),outputs=return_iter)
        # outer_model.build((None,512,))
        # outer_model.save("save-model/outer-eai-gpt.h5")
        print("building...")
        print("built")
        # inner_model.save("save-model/inner-eai-gpt.h5")
        print("outer_layer")
        iteration = Iteration
        print("outer_layer complete")
        batch = batches[iterator1]
        iterator1+=2

        params = AdamW(None, 0.01,0.9,0.999,t,loss,inner_model=inner_model,iteration=iteration,batch=batch)
        inner_model.add_update(tf.constant(params))
        batch = batches[iterator2]
        iterator2+=2
        architecture = Evo([[128,6],[128,6],[128,6],[128,6],[128,6],[128,6],[128,6],
                            [128,6],[128,6],[128,6],[128,6],[128,6],[128,6],[128,6]],
                            params, 0.01,0.9,0.999,t+1,30,5,loss,model=outer_model,batch=batch) 
        with open("optimizer-data/architecture-evo.txt", "w") as arc: 
            arc.write("#" + str(architecture.numpy().tolist()))
        build_model(architecture,params)
        sp.run("model.py")
    else:
        inner_model = keras.models.load_model("save-model/inner-eai-gpt.h5")
        outer_model = Iteration(inner_model, 512)
        batch = batches[iterator1]
        iterator1+=2
        params = AdamW(inner_model.weights, 0.01, 0.9, 0.999, t, loss, outer_model, batch=batch)
        inner_model.add_update(tf.constant(params)) 
        prev_arch = architecture_retrieve()
        batch = batches[iterator2]
        iterator2+=2
        architecture = Evo(prev_arch,params,0.01,0.9,0.999,t+1,30,5,loss_func=loss,model=outer_model,batch=batch)
        build_model(architecture,params)
        sp.run("model.py")




# Furthermore, you cannot use `model.predict`` because this returns a numpy array, 
# which basically "destroys" the computation graph chain so gradients won't be backpropagated. 
# You should simply use the model as a callable, i.e. predictions = model(pic).
