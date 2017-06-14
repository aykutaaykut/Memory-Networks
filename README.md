# Memory-Networks
This project is a reimplementation of Memory Networks (Weston, Jason, Chopra, Sumit, and Bordes, Antoine. 2015. Memory Networks. In ICLR 2015. https://arxiv.org/pdf/1410.3916.pdf) in Julia using Knet as the term project of Machine Learning class.

Memory Neural Networks (MemNNs) have hidden states and weights. This project shows the power of MemNN in the context of Question Answering. MemNNs are able to use related sentences in order to answer questions about a given story. They infer from sentences, chain different sentences, use induction and deduction in order to answer questions. MemNNs have an external memory and this memory is used as a knowledge base, and the output is textual response to the questions. MemNNs are tested with 20 different types of questions which require understanding of the sentences and the questions, and MemNNs are able to answer questions about each type.

MemNNs have 4 components:<br/>
I component takes the textual input and converts it into inner feature representation. Bag-of-Words (BoW) approach is used in this component.
G component takes the inner feature representation of a text and saves it in a memory slot in the external memory.
O component scores the match between each sentence in the memory and the question. It uses a scoring function and finally the sentence with the highest score is returned.
R component uses the same scoring function with O component and produces the final response to the question.

In this project, both margin ranking loss and softloss are used in different versions and both versions are publicly available.

MemNN-1-supporting-fact.jl: MemNN implementation using margin ranking loss for questions which require 1 supporting fact to answer.
MemNN-2-supporting-facts.jl: MemNN implementation using margin ranking loss for questions which require 2 supporting facts to answer.
MemNN-3-supporting-facts.jl: MemNN implementation using margin ranking loss for questions which require 3 supporting facts to answer.
softmax-1.jl: MemNN implementation using softloss for questions which require 1 supporting fact to answer.
softmax-2.jl: MemNN implementation using softloss for questions which require 2 supporting facts to answer.
softmax-3.jl: MemNN implementation using softloss for questions which require 3 supporting facts to answer.
softmax-n.jl: MemNN implementation using softloss for questions which require n (variable) supporting facts to answer.

The dataset which is used for training and testing in this project is different than the dataset used in Weston's paper. The dataset used in this project is the dataset of Towards AI-Complete Question Answering: A Set of Preprequisite Toy Tasks (Weston, Jason, Bordes, Antoine, Chopra, Sumit, Rush, Alexander M., Merrienboer, Bart van, Joulin Armand, and Mikolov Tomas. 2015. Towards AI-Complete Question Answering: A Set of Preprequisite Toy Tasks. In ICLR 2016. https://arxiv.org/pdf/1502.05698.pdf). This dataset is also publicly available in the dataset folder.

During the fully supervised training, Knet.Adam with learning rate of 0.001 is used as the parameter optimizer for 100 epochs. Since the story length is not constant, there is no minibatching.

My paper about this project can be found at https://drive.google.com/file/d/0B4pJdDuWybS9d2pQaUx2U3pEU1E/view?usp=sharing
My presentation of this project can be found at https://drive.google.com/file/d/0B4pJdDuWybS9X0lGSTdMUWlDbFU/view?usp=sharing
