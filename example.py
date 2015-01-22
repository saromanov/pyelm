from pyelm import ELM

def test_elm1():
	inp = [([1,0,0,1],[1,0]), ([0,1,0,0],[0,0]), ([1,1,1,1], [1,0]), ([1,0,0,0], [0,1])]
	elm = ELM(inp,7)
	elm.train(10)
	#elm.train(5, 10, [[0.74,0.88,0,145], [.2,1,0,.3], [1,0.74,1,1], [1,0,0,0]])


def test_elm2():
	X = np.random.randn(5,5)
	y = np.random.randint(2, size=(2,2))
	elm = ELM([X,y], 2)
	elm.train(50)


test_elm()
