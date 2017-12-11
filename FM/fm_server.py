import zmq
from BatchGD import predict

def main():

	port = "5556"
	context = zmq.Context()
	socket = context.socket(zmq.REP)
	socket.bind("tcp://*:%s" % port)

	while True:
		msg = socket.recv_json()

		print ("Request received. " + str(msg))

		with open("../ml-100k/demo.data", "r") as f:
			movie_pred = predict(f)

		socket.send_json({"prediction":movie_pred})


if __name__ == '__main__':
	main()