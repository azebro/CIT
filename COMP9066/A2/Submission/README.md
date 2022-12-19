A neural chatbot using sequence to sequence model with attentional decoder. This is a fully functional chatbot.

This is based on Google Translate Tensorflow model https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)
Created by Chip Huyen



Usage for TRAINING:

step1: open the chatbot.ipynb in https://colab.research.google.com/


Step 2: update config.py file. open the file you will understand.
Change DATA_PATH to where you store your data

step 3: run all the file in sequential order till '!python chatbot.py --mode train'

step 4: once you think that the model loss is not changing much stop the training .

step 5: we will now train our model only on joey_data() so as to give our boa a personality. run the cell '!python chatbot.py --mode train_joey' and it will trind on friends dataset for joey character.


if you ave error sating 'ehs not equal to lhs' remove the last two ENC_VOCAB and DEC_VOCAB because our model checkpoints stored the data for whole dataset.

step 7: after removing save the file and run '!python chatbot.py --mode chat' block to chat with the bot.



usage for CHATTING AND FEEDBACK:

step 1: if you are directly running the bot without training extract the preprocessed.zip and store the preprocessed folder extracted from that zip where these .py files are. so that it will not process the data again direct infeer from the data provided.

step 2 :the pre trained weights for the model is available on my google drive link : https://drive.google.com/drive/folders/13BUTX_MXVoPS50rxCe9UFsqdQz-JoKTP?usp=sharing
you can see and import directly from that. you will only need the checkpoint file and the file with the largest number of itration eg : 'chatbot-12500.data-00000-of-00001' means its iteratioin number it 12,500 find the greatest one  and download the files with same name an store bt=oth of them is checkpoints folder.
now open the checkpoint file which doesnt have any extension as a text document (i used sublime text to open it), you will se some entres, now :
'model_checkpoint_path: "/content/drive/MyDrive/checkpoints/chatbot-10000"' replace this checkpoint path with yours and save(no need to rem=name others just rename the one on the first line.)

step 3: run all code blocks inside the chatbot.ipynb EXCEPT "!python3 data.py" and "!python chatbot.py --mode train" directly  run '!python chatbot.py --mode chat' block to chat with the bot. it will restore the last saved model checkpoint and start.
step 4: if you want to give feedback then simply type keyword 'feedback:' followed by correction if you dont like the answer.
for example: 
user>> where do you live.
bot>> <unk>.
user>> feedback:I live in New-York.

step 5 :press enter and continue chatting ,you can provide multiple feedback in single session.
step 6 :once you think you are done and want to close the chat simply press 'enter'. 
step 7 :after you end the session the bot will train the model for the feedbacks you provided.It will take time, be patient while it does that.


Notes:-

If mode is train, then you train the chatbot. By default, the model will restore the previously trained weights (if there is any) and continue training up on that.

If you want to start training from scratch, please delete all the checkpoints in the checkpoints folder.

If the mode is chat, you'll go into the interaction mode with the bot.

By default, all the conversations you have with the chatbot will be written into the file output_convo.txt in the processed folder. If you run this chatbot, I kindly ask you to send me the output_convo.txt so that I can improve the chatbot.