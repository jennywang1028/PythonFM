#! /bin/bash

cd FM && python3 fm_server.py &
cd webserver && python server.py

if pgrep Python 
then
	killall Python
fi
