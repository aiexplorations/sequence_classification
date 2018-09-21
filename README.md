# Sequence Classification using LSTMs

The objective of this repo is to explore the following problem:

**If we have a number of different time series, some of which represent a certain kind of signal, and others representing a different kind, can we solve the problem of classifying them into different categories using LSTM networks?**

Background:

1. LSTM networks are "Long-Short-Term-Memory" networks which are a class of recurrent neural networks (RNNs). RNNs are in turn a class of sequence models that have become popular in the machine learning circles for various kinds of prediction and forecasting tasks. 
2. LSTM networks help us learn the sequential or temporal structure in data, and as a result, can be powerful in their ability to distinguish and predict sequences.
3. The models we're building in this repo are simple classification models that are aimed at solving a specific set of classification problems.
4. The data that these models are trained on is synthetically generated
5. The same model topologies and approach may be used for different kinds of sequence classification tasks. While it may be non-trivial to retrain these networks for new datasets, it definitely seems possible.

Author: Rajesh S (@rexplorations)

Ideas and features to add/change: 

1. Multi-class classification (classifying sets of sequences) - with interesting use cases in areas like IoT, this seems promising
2. I love music, and I hope to learn how to classify musical sequences (melodies, and even entire songs) someday. Specifically, I would like to build a version of this that could classify Carnatic ragas. While I'd developed some Scala code that can generate such notes for specific raga definitions, would like to see if those generated tunes could be told apart by an LSTM such as this one.
