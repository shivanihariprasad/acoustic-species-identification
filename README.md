## Acoustic Species Identification

Birds can serve as effective indicators of changes in biodiversity due to their mobility and diverse habitat needs. The alterations in the species composition and bird population can reflect the progress or shortcomings of restoration projects. However, conventional bird biodiversity surveys relying on observer-based methods are often difficult and expensive to conduct over extensive areas. Alternatively, using audio recording devices combined with modern analytical techniques that employ machine learning can enable conservationists to study the correlation between restoration interventions and biodiversity in greater depth by sampling larger spatial scales with higher temporal resolution. Ultimately, the optimal objective is to create a pipeline capable of precisely detecting a wide range of species vocalizations within a specified location where audio equipment is installed.

Our team's MVP was to design and implement a solution for analyzing audio data to identify different bird species based on their distinct calls. By utilizing machine learning techniques, we aim to create a powerful model that can accurately recognize and predict the species of birds present in the audio recordings. We successfully accomplished this.

The proposed end product of this project is a robust and efficient multi-species classifier that can seamlessly process audio data and identify the bird species present in the recordings. Our solution will use melspectogram to represent features from the audio data, which will then be used to train the machine learning model.


## Methodology 

![Flow Diagram](images/flow_diagram.png)

PyHa: 
A tool designed to convert audio-based "weak" labels to "strong" moment-to-moment labels. We are using the TweetyNet model variant to identify bird calls in the audio clip.

This package is being developed and maintained by the [Engineers for Exploration Acoustic Species Identification Team](http://e4e.ucsd.edu/acoustic-species-identification) in collaboration with the [San Diego Zoo Wildlife Alliance](https://sandiegozoowildlifealliance.org/).