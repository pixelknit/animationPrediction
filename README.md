# Animation Prediction
Predict future animation frames based on previous extracted frames.  
The file animation_data.json is the data extracted from Blender, the original animaiton is a simple walk animation of a character, it has N frames, the data on this json file is for the transforms on each bone for each frame.  
The data is processed and normalized in prepare_data.py, then the 2 experimental model architectures are transformer_model.py and rnn_model.py, so far rhe RNN gives better results.  
Once the models are trained and saved, they are being used in future_frames.py to predict the sequence of future frames.  
The future_frames.py will reverse the normalization of the data and produce the predicted_future_frames.json file which can be loaded back in Blender and set the next keyframes for the bones with the predicted animation.  

# TODO:  
The main root node which is hips, has to be processed separately since is not giving good results, but overall the character is moving like it's walking.  

Initial test animation:  

![9-25-2024 (17-22-05)](https://github.com/user-attachments/assets/86c30985-a91c-4782-8676-df4acafcf3e4)  

Initial frame range of 245 frames:  

![Screenshot 2024-09-25 172446](https://github.com/user-attachments/assets/56cd40c2-b11c-4945-a583-c95be1c5ffbf)  

Applied next 10 predicted animation frames:  


![9-25-2024 (17-25-35)](https://github.com/user-attachments/assets/4db249e5-4f4d-49fb-b2f4-4d079fd2ad79)  


