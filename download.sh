#!/bin/bash

# download paper0018
wget -P database_data/ https://drive.google.com/file/d/1PBhkDUvGb9goTIzc_9HxCAtOB72KGw9K/view?usp=drive_link

# download satin0002
wget -P database_data/ https://drive.google.com/file/d/1tKWYCvvX073X8HIgc2kjC8_QoGUM5QVP/view?usp=drive_link

# download ceramic0014
wget -P database_data/ https://drive.google.com/file/d/1mCItNXJprGko8PGxUIXF5GHMozbrE-rC/view?usp=drive_link

# download metal0005
wget -P database_data/ https://drive.google.com/file/d/1uEQoDfjriP45v-uAYOL2JdKe2IA0wHSs/view?usp=drive_link

# download model
wget -P . https://drive.google.com/file/d/1px3Ij1B7GIESWhAAm0-MHOhwR6yjWaVB/view?usp=drive_link

# download device configuration
wget -P . https://drive.google.com/file/d/1dIqEQcImBUaTGfsy0SVb8S6Pjua5u317/view?usp=drive_link

unzip database_data/paper0018.zip -d database_data/
unzip database_data/satin0002.zip -d database_data/
unzip database_data/ceramic0014.zip -d database_data/
unzip database_data/metal0005.zip -d database_data/

unzip database_model.zip -d database_model/
unzip device_configuration.zip -d data_processing/device_configuration/
