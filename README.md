Download Link: https://assignmentchef.com/product/solved-si-630-homework-3-dependency-parsing
<br>
Despite its seeming chaos, natural language has lots of structure. We’ve already seen some of this structure in part of speech tags and how the order of parts of speech are predictive of what kinds of words might come next (via their parts of speech). In Homework 3, you’ll get a deeper view of this structure by implementing a dependency parser. We covered this topic in Week 10 of the course and it’s covered extensively in Speech &amp; Language Processing chapter 13, if you want to brush up .<sup>1 </sup>Briefly, dependency parsing identifies the syntatic relationship between word pairs to create a <em>parse tree</em>, like the one seen in Figure 1.

In Homework 3, you’ll implement the <em>shift-reduce </em>neural dependency parser of Chen and Manning [2014],<sup>2 </sup>which was one of the first neural network-based parser and is quite famous. Thankfully, its neural network is also fairly straight-forward to implement. We’ve provided the parser’s skeleton code in Python 3 that you can use to finish the implementation, with comments that outline the steps you’ll need to finish. And, importantly, we’ve provided a <em>lot </em>of boilerplate code that handles loading in the training, evaluation, and test dataset, and converting that data into a representation suitable for the network. Your part essentially boils down to two steps: (1) fill in the implementation of the neural network and (2) fill in the main training loop that processes each batch of instances and does backprop. Thankfully, <em>unlike </em>in Homeworks 1 and 2, you’ll be leveraging the miracles of modern deep learning libraries to accomplish both of these!

Homework 3 has the following learning goals:

<sup>1</sup>https://web.stanford.edu/˜jurafsky/slp3/13.pdf

<sup>2</sup>https://cs.stanford.edu/˜danqi/papers/emnlp2014.pdf

ROOT      He       has     good    control    .

PRP      VBZ        JJ          NN       .

Figure 1: An example dependency parse from the Chen and Manning [2014] paper. Note that each word is connected to another word to symbolize its syntactic relationship. Your parser will determine these edges and their types!

<ol>

 <li>Gain a working knowledge of the PyTorch library, including constructing a basic network, using layers, dropout, and loss functions.</li>

 <li>Learn how to train a network with PyTorch</li>

 <li>Learn how to use pre-trained embeddings in downstream applications</li>

 <li>Learn about the effects of changing different network hyper parameters and designs</li>

 <li>Gain a basic familiarity with dependency parsing and how a shift-reduce parser works.</li>

</ol>

You’ll notice that most of the learning goals are based on deep learning topics, which is the primary focus of this homework. The skills you learn with this homework will hopefully help you with your projects and (ideally) with any real-world situation where you’d need to build a new network. However, you’re welcome—encouraged, even!—to wade into the parsing setup and evaluation code to understand more of how this kind of model works.

In Homework 3, we’ve also included several <em>optional </em>tasks for those that feel ambitious. Please finish the regular homework first before even considering these tasks. There is no extra credit for completing any of these optional tasks, only glory and knowledge.

<h1>2           PyTorch</h1>

Homework 3 will use the PyTorch deep learning library. However, your actual implementation will use only a small part of the library’s core functionality, which should be enough to get you building networks. Rather than try to explain all of PyTorch in a mere homework write-up, we’ll refer you to the fantastic PyTorch community tutorials<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> for comprehensive coverage. Note that you <em>do not </em>need to read all these tutorials! We’re only building a feed forward network here, so there’s no need to read up on Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), or any variant thereof. Instead, as a point of departure into deep learning land, try walking through this tutorial on Logistic Regression<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a> which is the PyTorch version of what you implemented in Homework 1. That tutorial will hopefully help you see how the things you had to implement in HW1 get greatly simplified when using these deep learning libraries (e.g., compare their stochastic gradient descent code with yours!).

The biggest conceptual change for using PyTorch with this homework will be using <em>batching</em>. We talked briefly about batching during the Logistic Regression lecture, where instead of using just a single data point to compute the gradient, you use several—or a <em>batch</em>. In practice, using a batch of instances greatly speeds up the convergence of the model. Further when using a GPU, often batching is significantly more computationally efficient because you’ll be using more of the special matrix multiplication processor at once (GPUs are designed to do lots of multiplications in parallel, so batching helps “fill the capacity” of work that can be done in a single time step). In practice, we’ve already set up the code for you to be in batches so when you get an instance with <em>k </em>features, you’re really getting a tensor<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a> of size <em>b</em>×<em>k </em>where <em>b </em>is the batch size. Thanks to the magic

Softmax layer:

<em>p </em>= softmax(<em>W</em><sub>2</sub><em>h</em>)

Hidden layer:

Input layer: [<em>x<sup>w</sup>,x<sup>t</sup>,x<sup>l</sup></em>]

words                          POS tags            arc labels

Stack                                         Buffer

Figure 2: The network architecture for the Chen and Manning [2014] parser. Note that this is a feed-forward neural network, which is just one more layer than logistic regression!

of PyTorch, you can effectively treat this as a vector of size <em>k </em>and PyTorch will deal with the fact that it’s “batched” for you;<a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a> i.e., you can write your neural network in a way that ignores batching and it will just happen naturally (and more efficiently).<a href="#_ftn5" name="_ftnref5"><sup>[5]</sup></a>

<h1>3           The Parser</h1>

The Chen and Manning [2014] parser is a feed-forward neural network that encodes lexical features from the context (i.e., words on the stack and words on the buffer) as well as their parts of speech and the current dependency arcs that have been produced. Figure 2 shows a diagram of the network. The input layer consists of three pieces, <em>x<sup>w</sup></em>, <em>w<sup>t</sup></em>, and <em>x<sup>l</sup></em>, which denote the embeddings for the words, POS tags, and dependency args. Each of these embeddings is actually <em>multiple </em>embeddings concatenated together; i.e., if we represent the two words on the top if the stack, each with 50dimensional embeddings, then <em>x<sup>w </sup></em>has a total length of 50 × 2 = 100. Said another way, <em>x<sup>w </sup></em>=  where  denotes the embedding for <em>w<sub>i </sub></em>and ; is the concatenation operator. Each of the input types has separate weights <em>W</em><sub>1 </sub>for computing hidden layer raw output (before the activation function is applied), i.e.,.

Normally, we’ve talked about activation functions like sigmoid or a Rectified Linear Unit (ReLU); however, in the Chen and Manning [2014] parser, we’ll use a different activation function. Specifically, you’ll implement a <em>cubic </em>activation function that cubes the raw output value of a neuron. As a result, the activation of the hidden layer in the original paper is computed as  where <em>b</em><sub>1 </sub>is the bias term.

Chen and Manning [2014] use separate weight matrices for words, POS tags, and dependencies to make use of several optimizations which are not described here. However, we can simplify this equation and the implementation by using one weight matrix <em>W</em><sub>1 </sub>that takes in the concatenated inputs to make the activation <em>h </em>= (<em>W</em><sub>1</sub>[<em>x<sup>w</sup></em>;<em>x<sup>t</sup></em>;<em>x<sup>l</sup></em>;<em>b</em><sub>1</sub>])<sup>3</sup>. In your implementation, this means you can use a single layer to represent all three weight matrices (, and ) which should simplify your bookkeeping.

<table width="0">

 <tbody>

  <tr>

   <td width="107">Transition</td>

   <td width="126">Stack</td>

   <td width="105">Buffer</td>

   <td width="246"><em>A</em></td>

  </tr>

  <tr>

   <td width="107">SHIFT</td>

   <td width="126">[ROOT][ROOT He]</td>

   <td width="105">[He has good control .][has good control .]</td>

   <td width="246">;</td>

  </tr>

  <tr>

   <td width="107">SHIFT</td>

   <td width="126">[ROOT He has]</td>

   <td width="105">[good control .]</td>

   <td width="246"> </td>

  </tr>

  <tr>

   <td width="107">LEFT-ARC(nsubj)SHIFT</td>

   <td width="126">[ROOT has][ROOT has good]</td>

   <td width="105">[good control .][control .]</td>

   <td width="246"><em>A</em>[ nsubj(has,He)</td>

  </tr>

  <tr>

   <td width="107">SHIFT</td>

   <td width="126">[ROOT has good control]</td>

   <td width="105">[.]</td>

   <td width="246"> </td>

  </tr>

  <tr>

   <td width="107">LEFT-ARC(amod)RIGHT-ARC(dobj)…</td>

   <td width="126">[ROOT has control][ROOT has]…</td>

   <td width="105">[.][.]…</td>

   <td width="246"><em>A</em><em><sub>A</sub></em>…[[amod(control,good)dobj(has,control)</td>

  </tr>

  <tr>

   <td width="107">RIGHT-ARC(root)</td>

   <td width="126">[ROOT]</td>

   <td width="105">[]</td>

   <td width="246"><em>A </em>root(ROOT,has)</td>

  </tr>

 </tbody>

</table>

[

Figure 3: An example snippet of the parsing stack and buffer from Chen and Manning [2014] using the sentence in Figure 1. This diagram is an example of how a transition-based dependency parser works. At each step, the parser decides whether to (1) <em>shift </em>a word from the buffer onto the stack or (2) <em>reduce </em>the size of the stack by forming an edge between the top two words on the stack (further deciding which direction the edge goes).

The final outputs are computed by multiplying the hidden layer activation by the second layer’s weights and passing that through a softmax:<a href="#_ftn6" name="_ftnref6"><sup>[6]</sup></a> <em>p </em>= softmax(<em>W</em><sub>2</sub><em>h</em>).

This might all seem like a lot of math and/or implementation, but PyTorch is going to take care of most of this for you!

<h1>4              Implementation Notes</h1>

The implementation is broken up into five key files, only two of which you need to deal with:

<ul>

 <li>py<sub>line. It also contains the basic training loop, which you’ll need to implement. You can run</sub>— This file drives the whole program and is what you’ll run from the command python main.py -h to see all the options</li>

 <li>you’ll implement.model.py — This file specifies the neural network model used to do the parsing, which</li>

 <li><sub>Treebank data and turning it into training and test instances. This is where most of the</sub>py — This file contains all the gory details of reading the Penn parsing magic happens. You don’t need to read any of this file, but if you’re curious, please do!</li>

 <li>labeled Attachment Scoretestfunctions.py — This code does all the testing and, crucially, computes the(UAS) you’ll use to evaluate your parser. You don’t need to readUnthis file either.</li>

 <li>py — Random utility functions.</li>

</ul>

Figure 4: The analogy for how it <em>might </em>seem to implement this parser based on the instructions, but in reality, it’s not too hard!

We’ve provided skeleton code with TODOs for where the main parts of what you need to do are sketched out. All of your required code should be written in main.py or model.py, though you are welcome to read or adjust the other file’s code to help debug, be more verbose, or just poke around to see what it does.

<h1>5           Data</h1>

Data has already been provided for you in the data/ directory in CoNLL format. You do not need to deal with the data itself, as the featureextraction.py code can already read the data and generate training instances.

<h1>6           Task 1: Finish the implementation</h1>

In Task 1, you’ll implement the feed-forward neural network in main.py based on the description in this write-up or the original paper. Second, you’ll implement the core training loop which will

<ol>

 <li>Loop through the dataset for the specified number of epochs</li>

 <li>Sample a batch of instances</li>

 <li>Produce predictions for each instance in the batch</li>

 <li>Score those predictions using your loss function</li>

 <li>Perform backpropagation and update the parameters.</li>

</ol>

Many of these tasks are straightforward with PyTorch and none of them should require complex operations. Having a good understanding of how to implement/train logistic regression in PyTorch will go a long way.

The main.py file works with command line flags to enable quick testing and training. To train your system, run python main.py –train. Note that this will break on the released code since the model is not implemented! However, once it’s working, you should see an output that looks something like the following after one epoch:

Loading dataset for training

Loaded Train data

Loaded Dev data Loaded Test data Vocab Build Done!

embedding matrix Build Done converting data into ids..

Done!

Loading embeddings

Creating new trainable embeddings words: 39550

some hyperparameters

{’load_existing_vocab’: True, ’word_vocab_size’: 39550,

’pos_vocab_size’: 48, ’dep_vocab_size’: 42,

’word_features_types’: 18, ’pos_features_types’: 18,

’dep_features_types’: 12, ’num_features_types’: 48, ’num_classes’: 3}

Epoch: 1 [0], loss: 99.790, acc: 0.309

Epoch: 1 [50], loss: 4.170, acc: 0.786

Epoch: 1 [100], loss: 2.682, acc: 0.795

Epoch: 1 [150], loss: 1.795, acc: 0.818

Epoch: 1 [200], loss: 1.320, acc: 0.840

Epoch: 1 [250], loss: 1.046, acc: 0.837

Epoch: 1 [300], loss: 0.841, acc: 0.843

Epoch: 1 [350], loss: 0.715, acc: 0.848

Epoch: 1 [400], loss: 0.583, acc: 0.854

Epoch: 1 [450], loss: 0.507, acc: 0.864

Epoch: 1 [500], loss: 0.495, acc: 0.863

Epoch: 1 [550], loss: 0.487, acc: 0.863

Epoch: 1 [600], loss: 0.423, acc: 0.869

Epoch: 1 [650], loss: 0.386, acc: 0.867

Epoch: 1 [700], loss: 0.338, acc: 0.867

Epoch: 1 [750], loss: 0.340, acc: 0.874

Epoch: 1 [800], loss: 0.349, acc: 0.868

Epoch: 1 [850], loss: 0.320, acc: 0.873

Epoch: 1 [900], loss: 0.322, acc: 0.879

End of epoch

Saving current state of model to saved_weights/parser-epoch-1.mdl

Evaluating on valudation data after epoch 1

Validation acc: 0.341

– validation UAS: 70.42

Here, the core training loop is printing out the accuracy at each step as well as the cross-entropy loss. At the end of the epoch, the core loop scores the model on the validation data and reports the UAS, which is the score we care about. Further, the core loop will save the model after each epoch in savedweights.

<h1>7           Task 2: Score Your System</h1>

We want to build a good parser and to measure how good our parser is doing, we’ll use UAS for this assignment, which corresponds to the percentage of words that have the correct head in the dependency arc. Note that this score isn’t looking at the particular label (e.g., nsubj), just whether we’ve created the correct parsing structure. For lots more details on how to evaluate parsers, see Kubler et al. [2009] page 79.¨

For Task 2, you’ll measure the performance of your system performance relative to the number of training epochs and evaluate on the final test data. This breaks down into the following problems to solve.

Problem 2.1. Train your system for <em>at least </em>5 epochs, which should generate 5 saved models in saved-weights. For each of these saved models, compute the UAS score.<a href="#_ftn7" name="_ftnref7"><sup>[7]</sup></a> You’ll make three plots: (1) the loss during training for each epoch, (2) the accuracy for each epoch during training, and (3) the UAS score for the test data for each epoch’s model.<a href="#_ftn8" name="_ftnref8"><sup>[8]</sup></a> You can make a nice plot each run’s performance using Seaborn: http://seaborn.pydata.org/examples/ wide_data_lineplot.html

Problem 2.2. Write <em>at least </em>three sentences describing what you see in the graphs and when you would want to stop training.

<h1>8           Task 3: Try different network designs and hyperparameters</h1>

Why stop at 1 hidden layer?? And why not use a ReLU instead of Cubic for an activation function? In Task 3, you get to try out different network architectures. We suggest trying out some of the following and then repeating Task 2 to see how performance changes. For easier debugging and replicability, you should make a new class that is a copy of ParserModel (once it’s working for Task 2) and make all your modifications to that class. Some suggested modifications are:

<ol>

 <li>Add 1 or more layers to the network.</li>

 <li>Add normalization or regularization to the layers.</li>

 <li>Change to a different activation function</li>

 <li>Change the size (number of neurons) in layers</li>

 <li>Change the embedding size</li>

</ol>

How high can you get the performance to go?

Problem 3.1. Train your system for <em>at least </em>5 epochs and generate the same plots as in Problem 2.1 for this new model’s performance but include both the old model and the new model’s performances in each.

8.1        Task 4: What’s the parser doing, anyway?

A big part of the assignment is learning about how to build neural networks that solve NLP problems. However, we care about more than just a single metric! In Task 4, you’ll look at the actual shift-reduce parsing output to see how well your model is doing. We’ve already provided the functionality for you to input a sentence and have the model print out (1) the steps that the shift-reduce parser takes and (2) the resulting parse. This functionality is provided using the –parsesentence argument that takes in a string.

python main.py –parse_sentence “I eat” –load_model_file 

saved_weights/parser-epoch-5.mdl […model loading stuff…] Done!

—buffer: [’i’, ’eat’] stack: [’&lt;root&gt;’] action: shift —buffer: [’eat’] stack: [’&lt;root&gt;’, ’i’] action: shift —buffer: [] stack: [’&lt;root&gt;’, ’i’, ’eat’] action: left arc, &lt;d&gt;:compound:prt

—buffer: [] stack: [’&lt;root&gt;’, ’eat’] action: right arc, &lt;d&gt;:det:predet

&lt;root&gt;

| eat

| i

In Task 4, you’ll take a look at these outputs and determine whether they were correct.

Problem 4.1. Using one of your trained models, report the shift-reduce output for the sentence “The big dog ate my homework” and the parse tree

Problem 4.2. More than likely, the model has made a mistake somewhere. For the output, report what was the correct operation to make at each time step: shift, left-arc, right-arc (you do not need to worry about the specific dependency arc labels for this homework).

<h1>9           Optional Task</h1>

Homework 3 has lots of potential for exploration if you find parsing interesting or want to try building models. Here, we’ve listed a few different <em>fully optional </em>tasks you could try to help provide guidance. These are only for glory and will not change your score. Please please please make sure you finish the homework before trying any of these.

Optional Task 1: Measure the Effect of Pre-Trained Embeddings

In Task 2, your model learned word embeddings from scratch. However, there’s plenty of rare words in the dataset which may not have useful embeddings. Another idea is to pre-train word embeddings from a large corpus and then use those during training. This leverages the massive corpus to learn the meanings so that your model can effectively make use of the information— even for words that are rare in training. But which corpus should we use? In Optional Task 1, we’ve conveniently pre-trained 50-dimensional vectors for you from two sources: all of Wikipedia and 1B words from Twitter.

Specifically, for Optional Task 1, you will update your code to allow providing a file containing word embedding in word2vec’s binary format and use those embeddings in training <em>instead </em>of pretraining. You shouldn’t update these vectors like you would do if you were learning from scratch, so you’ll need to turn off the gradient descent for them. Part of Optional Task 1 is thinking about <em>why </em>you shouldn’t change these vectors.

Finally, once you have the vectors loaded, you’ll measure the performance just like you did in Task 2. This breaks down to the following steps:

Problem 3.1. Write code that loads in word vectors in word2vec’s binary format (see Homework 2’s code which has something like this). You’ll need to convert these vectors into PyTorch’s Embedding object to use.

Problem 3.2. Prevent the pretrained embeddings from being updated during gradient descent.

Problem 3.3. Write a few sentences about <em>why </em>we would turn off training. Be sure to describe what effect allowing the weights to change might have on future performance?<a href="#_ftn9" name="_ftnref9"><sup>[9]</sup></a>

Problem 3.4. Repeat the 5 epochs training like you did in Task 2 using the Twitter and Wikipedia embeddings and plot the performance of each on the development data (feel free to include the learned-embedding performance in this plot too). Write a few sentences describing what you see and why you think the performance is the way it is. Are you surprised?

<a href="#_ftnref1" name="_ftn1">[1]</a> https://pytorch.org/tutorials/

<a href="#_ftnref2" name="_ftn2">[2]</a> https://www.kaggle.com/negation/pytorch-logistic-regression-tutorial

<a href="#_ftnref3" name="_ftn3">[3]</a> Tensor is a fancier name for multi-dimensional data. A vector is a 1-dimensional tensor and a matrix is a 2dimensional tensor. Most of the operations for deep learning libraries will talk about “tensors” so it’s important to get used to this terminology.

<a href="#_ftnref4" name="_ftn4">[4]</a> For a useful demo of how this process works, see https://adventuresinmachinelearning.com/ pytorch-tutorial-deep-learning/

<a href="#_ftnref5" name="_ftn5">[5]</a> In later assignments where we have sequences, we’ll need to revise this statement to make things <em>even more </em>efficient!

<a href="#_ftnref6" name="_ftn6">[6]</a> Reminder: the softmax is the generalization of the sigmoid (<em>σ</em>) function for multi-class problems here.

<a href="#_ftnref7" name="_ftn7">[7]</a> The code makes it easy to do this where you can specify which saved model to use, e.g., python main.py

–test –loadmodelfile savedweights/parser-epoch-2.mdl

<a href="#_ftnref8" name="_ftn8">[8]</a> For a fun exercise, try doing this process 4-5 times and see how much variance there is.

<a href="#_ftnref9" name="_ftn9">[9]</a> If you’re struggling to write this part, try allowing their values to get updated and compare the performance difference between the development and test data. Feel free to report the scores!