# Music-Genre-Prediction

Aim - To build a machine learning model which classifies music into its respective genre.

Introduction -
Intrigued by the audio domain, we decided to pick up this project, wherein we have to classify different songs belonging to different genres to their respective genres using deep learning models.
Before moving ahead, let's talk about the challenges to work with audio data. First and foremost , it is not possible to visualize an audio file, that is we can't represent the audio file in like a graph.

To start with processing, we need to first understand what audio signal is.
1. Produced by the vibration of an object
2. Vibrations determine oscillation of air molecules
3. Alternation of air pressure causes a wave

EDA / Visualisation
Step - 1) Analog to digital conversion : When audio signal that is inherently analog has to be processed, we need to convert it to Digital signal first. To convert it to digital , the signal is sampled at uniform time intervals. The number of times it is sampled each second is known as Sample Rate. Conversion of amplitude of the signal takes place with limited number of bits. The number of bits used for amplitude quantisation is known as Bit Depth. A typical CD album has a Sample Rate = 44,100 Hz and Bit Depth = 16.
Step - 2) FFT - Fast Fourier Transform : Fourier transform decomposes complex periodic sound into sum of sine waves oscillating at different frequencies. After fourier transform, the sound wave gets changed from time domain to frequency domain.
Step - 3) STFT - Short Time Fourier Transform : 1.Computes several FFT at different intervals
                                                2.Preserves time information
                                                3.Fixed frame size (e.g., 2048 samples)
                                                4.Gives a spectrogram (time + frequency + magnitude)
Step - 4) MFCCs - MEL FREQUENCY CEPSTRAL COEFFICIENTS : 1.Capture timbral/textural aspects of sound
                                                        2.Frequency domain feature
                                                        3.Approximate human auditory system
                                                        4.13 to 40 coefficients
                                                        5.Calculated at each frame
                                                        
About the Dataset : Link to the dataset - https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
GTZAN Dataset
10 genres
100 songs per genre
30 seconds per song

Preprocessing Library Used - Librosa : Librosa is a Python package for music and audio analysis. Librosa is basically used when we work with audio data like in music generation, Automatic Speech Recognition. It provides the building blocks necessary to create the music information retrieval systems.

FFNNs - Feed Forward Neural Networks
An FFNN consists of a large number of neurons, organized in layers: one input layer, one or more hidden layers, and one output layer. Each neuron in a layer is connected to all the neurons of the previous layer, although the connections are not all the same because they have different weights. The weights of these connections encode the knowledge of the network. Data enters at the inputs and passes through the network, layer by layer until it arrives at the outputs. During this operation, there is no feedback between layers. Therefore, these types of networks are called feed-forward neural networks.

What are Optimizers?
Optimizers are algorithms or methods used to change the attributes of the neural network such as weights and learning rate to reduce the losses. Optimizers are used to solve optimization problems by minimizing the function.

Challenges in Optimization - Saddle Points
In large N-dimensional domains, local minima are extremely rare. • Saddle points are very common in high dimensional spaces. • These saddle points make it notoriously hard for SGD to escape, as the gradient is close to zero in all dimensions.

Escaping saddle points
Somewhat counterintuitively, the best way to escape saddle points is to just move in any direction, quickly. Then we can get somewhere with more substantial curvature for a more informed update

Types of layers used : 1.Dense
                       2.Flatten
                       3.Dropout
                       4.Batch Normalization
 
CNNs - Convolutional Neural Networks
CNNs are regularized versions of multilayer perceptrons. Multilayer perceptrons usually mean fully connected networks, that is, each neuron in one layer is connected to all neurons in the next layer. The "full connectivity" of these networks make them prone to overfitting data. Typical ways of regularization, or preventing overfitting, include: penalizing parameters during training (such as weight decay) or trimming connectivity (skipped connections, dropout, etc.) CNNs take a different approach towards regularization: they take advantage of the hierarchical pattern in data and assemble patterns of increasing complexity using smaller and simpler patterns embossed in their filters. Therefore, on a scale of connectivity and complexity, CNNs are on the lower extreme.

Types of layers used : 1.Dense
                       2.Flatten
                       3.Dropout
                       4.Batch Normalization
                       5.Max-Pooling 2D
                       6.Convolutional 2D
Conclusion

To conclude the findings of the project, we figured out that FFNNs do not perform particularly well with transformed audio data. The maximum accuracy we were able to achieve was - 65.7%.

On the other hand, CNNs performed better. We achieved an accuracy of 87.15% on test data.

Conversion of audio data to MFCCs simplified the inputs to the model. When trying to perform the same using signal data or STFTs we discovered that the input size exploded. Due to limited resources we were unable to test the same.

Due to the temporal nature of the data, we believe that Recurrent Neural Networks (RNNs) or Long-Short Term Memory (LSTMs) may perform better.
