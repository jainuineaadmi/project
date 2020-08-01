Multi-model Emotion Recognition   
	Detail Solution Design & Architecture
Version 1.1
07/15/2020












1.	Introduction
Multimodal Emotion Recognition is a new discipline that aims to include text inputs, sound as well as video inputs. Recent studies have been exploring potential metrics to measure the coherence between emotions from different channels (Text, Audio, and Video). Hence, we have developed a multimodal emotion recognition web application built in Python Flask to analyze the emotions of the candidates appearing for the job. We are going to explore several categorical targets depending on the input considered. Table 1 gives a summary of all the categorical targets that we are going to evaluate.
 

Many psychology researchers believe that it is possible to exhibit 5 categories, or core factors, that determine one's personality. The acronym OCEAN (for openness, conscientiousness, extraversion, agreeableness, and neuroticism) is often referred to say, text model. We chose to use this precise model as it is nowadays the most popular in psychology: while the five dimensions don't capture the peculiarity of everyone's personality, it is the theoretical Framework most recognized by researchers and practitioners in this field.
Our aim is to develop an app that is able to provide real-time sentiment analysis with a visual user interface using Tensorow.js technology. Therefore, we have decided to separate two types of inputs:
1. Textual input, such as answers to questions that would be asked to a person from the platform. Example- Narrate a situation where you had shown leadership skills?
2. Video input from a live webcam or stored from an MP4 or WAV _le, from which we split the audio and the images.

2.	Technical Architecture

2.1.1	Technologies and Framework used
 We deployed a web app using Flask, and all the codes are written on the python programming language.

2.1.2	Data Sources
Text Data
For the text modal, we are using the Stream-of-consciousness dataset that was gathered in a study by Pennebaker and King [1999]. It consists of a total of 2,468 daily writing submissions from 34 psychology students (29 women and five men whose ages ranged from 18 to 67 with a mean of 26.4). The writing submissions were in the form of an unrated course assignment. For each assignment, students were expected to write a minimum of 20 minutes per day about a specific topic. The data was collected during a 2-week summer course between 1993to1996. Each student completed their daily writing for ten consecutive days. Students' personality scores were assessed by answering the Big Five Inventory (BFI) The BFI is a 44-item self-report questionnaire that provides a score for each of the five personality traits. Each item consists of short phrases and is rated using a 5-point scale that ranges from 1 (disagree strongly) to 5 (agree strongly). An instance in the data source consists of an ID, the actual essay, and five classification labels of the Big Five personality traits. Labels were originally in the form of either yes ('y') or no ('n') to indicate scoring high or low for a given trait.
Audio Data
For audio data sets, we are using the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). This database contains 7356 files (total size: 24.8 GB). The database contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions, and the song contains calm, happy, sad, angry, and fearful emotions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression. All conditions are available in three modality formats: Audio-only (16bit, 48kHz .wav), Audio-Video (720p H.264, AAC 48kHz, .mp4), and Video-only (no sound).

Video Datasets
For the video data sets, we are using the accessible FER2013 Kaggle Challenge data set. The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The data set remains quite challenging to use since there are empty pictures or wrongly classified images. Link- https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

2.1.3	Text mining for personality trait Classification

Emotion recognition through text is a challenging task that goes beyond conventional sentiment analysis: instead of merely detecting neutral, positive, or negative feelings from the text, the goal is to identify a set of emotions characterized by a higher granularity. For instance, feelings like anger or happiness could be included in the classification. As recognizing such emotions can turn out to be complicated even for the human eye, machine learning algorithms are likely to obtain mixed performances. It is important to note that nowadays, emotion recognition from facial expression tends to perform better than from textual interpretation. Indeed, many subtleties should be taken into account in order to achieve an accurate detection of human emotions through text, context-dependency being one of the most crucial. This is the reason why using advanced natural language processing is required to obtain the best performance possible.

Pipeline
•	Text data retrieving
•	Custom natural language pre-processing :
o	Tokenization of the document
o	Cleaning and standardization of formulations using regular expressions
o	Deletion of the punctuation
o	Lowercasing the tokens
o	Removal of predefined stopwords
o	Application of part-of-speech tags on the remaining tokens
o	Lemmatization of tokens using part-of-speech tags for more accuracy.
o	Padding the sequences of tokens of each document to constrain the shape of the input vectors.
•	300-dimension Word2Vec trainable embedding
•	Prediction using our trained model

Model

We have chosen a neural network architecture based on both one-dimensional convolutional neural networks and recurrent neural networks. The one-dimensional convolution layer plays a role comparable to feature extraction: it allows finding patterns in text data. The Long-Short Term Memory (LSTM) cell is then used in order to leverage on the sequential nature of natural language: unlike regular neural networks where inputs are assumed to be independent of each other, these architectures progressively accumulate and capture information through the sequences. LSTMs have the property of selectively remembering patterns for long durations of time. Our final model first includes three consecutive blocks consisting of the following four layers: one-dimensional convolution layer - max-pooling - spatial dropout - batch normalization. The numbers of convolution filters are respectively 128, 256, and 512 for each block, and kernel size is 8, max-pooling size is two, and the dropout rate is 0.3. Following the three blocks, we chose to stack 3 LSTM cells with 180 outputs each. Finally, a fully connected layer of 128 nodes is added before the last classification layer.

           
2.1.4	Audio processing for emotion recognition
Speech emotion recognition purpose is to automatically identify the emotional or physical state of a human being from his voice. The emotional state of a person hidden in his speech is an essential factor in human communication and interaction as it provides feedbacks in connection while not altering linguistic contents. The usual process for speech emotion recognition consists of three parts: signal processing, feature extraction, and classification. Signal processing applies an acoustic filter on original audio signals and splits it into meaningful units. The feature extraction is the sensitive point in speech emotion recognition because features need to both efficiently characterize the emotional content of a human speech and not depend on the lexical content or even the speaker. Finally, emotion classification will map the feature matrix to emotion labels.
            

Pipeline

The speech emotion recognition pipeline was built the following way:
•	Voice recording
•	Audio signal discretization
•	Log-mel-spectrogram extraction
•	Split spectrogram using a rolling window
•	Make a prediction using our pre-trained model
Model

The model we have chosen is a Time Distributed Convolutional Neural Network.

The main idea of a Time Distributed Convolutional Neural Network is to apply a rolling window (fixed size and time-step) all along the log-mel-spectrogram. Each of these windows will be the entry of a convolutional neural network, composed by four Local Feature Learning Blocks (LFLBs) and the output of each of these convolutional networks will be fed into a recurrent neural network composed by two cells LSTM (Long Short Term Memory) to learn the long-term contextual dependencies. Finally, a fully connected layer with softmax activation is used to predict the emotion detected in the voice.
 
To limit over-fitting, we tuned the model with :
•	Audio data augmentation
•	Early stopping
•	And kept the best model
2.1.5	Computer vision for emotion recognition
In the field of facial emotion recognition, most recent research papers focus on deep learning techniques, and more specifically, on Convolution Neural Network (CNN). 

Pipeline

The video processing pipeline was built the following way:
•	Launch the webcam
•	Identify the face by Histogram of Oriented Gradients
•	Zoom on the face
•	Dimension the face to 48 * 48 pixels
•	Make a prediction on the face using our pre-trained model
•	Also, identify the number of blinks on the facial landmarks on each picture
Model
The model we have chosen is an XCeption model since it outperformed the other approaches we developed so far. We tuned the model with:
•	Data augmentation
•	Early stopping
•	Decreasing the learning rate on the plateau
•	L2-Regularization
•	Class weight balancing
•	And kept the best model
The XCeption architecture is based on DepthWise Separable convolutions that allow to train much fewer parameters and therefore reduce training time.
          
When it comes to applying CNNs in real-life applications, being able to explain the results is a great challenge. We can indeed plot class activation maps, which display the pixels that have been activated by the last convolution layer. We notice how the pixels are being activated differently, depending on the emotion being labeled. The happiness seems to depend on the pixels linked to the eyes and mouth, whereas the sadness or the anger seems for example, to be more related to the eyebrows.
                              

2.1.6	Webapp
For the implementation of our models, we chose to create an open-source web application. It allows users to obtain in real-time a personalized assessment of their emotions or personality traits based on the analysis of a video, audio, or text extract sent directly via the platform. 

 
A page is dedicated to each communication channel (audio, video, text) and allows the user to be evaluated. A typical interview question is asked on each page, for instance: "Tell us about the last time you showed leadership." The audio/video extract (recorded via computer microphones/webcam) or text block can be retrieved once saved and processed by our algorithms (in the case of the text channel, the user can also upload a .pdf document that will be parsed by our tool).
 

 
 
Once the user has recorded or typed his answer, he is redirected to a summary page. In the case of the video interview, for example, this assessment allows him not only to know his "score" in each of the emotions identified by our model, but also the average score of the other candidates: in this way, he can reposition himself, and adjust his attitude at will. We believe that including a kind of benchmark in the analysis helps the user becoming aware of his or her position in relation to the average candidate. The text and video/audio summaries are slightly different: for the text interview summary, not only we chose to display the percentage score of identified personality traits for both the user and the other candidates, but also the most frequently used word in the answer. For the video and audio interview summaries, we displayed the perceived emotions scores of the user and the other candidates.
The following are the summary pages for both the text and video interviews.




 
 


2.1.7	How to use the web app locally?
To use the web app:
•	Clone the project locally. Github Link-
•	Go in the WebApp folder
•	Run `$ pip install -r requirements.txt. ``
•	Launch python main.py

2.1.8	Resources
We really thank to Maël Fabien, PhD scholar at EPFL Switzerland to help us customize this project as per our need.
3.	Factors Influencing Design

3.1	Assumptions and Dependencies
The assumption being the candidate has the sufficient resources such as good internet facility to access our platform. The proposed solution can classify candidates only on the pre-specified categories as mentioned in text, audio and Video Models above. 
3.2	Constraints
•	Web-app requires good front camera/web-cam to make predictions.
•	Fast Internet
•	Decent Microphone










