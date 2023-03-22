# ln -s ~/.local/bin/kaggle /usr/bin/kaggle
export KAGGLE_USERNAME=davidlainesv
export KAGGLE_KEY=4adf7ff84f13a31bb82d7941d01db38a
kaggle competitions download asl-signs
unzip asl-signs.zip
rm asl-signs.zip