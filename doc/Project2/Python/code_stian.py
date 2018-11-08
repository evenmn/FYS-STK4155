class MLP(Regression):
	
	def __init__(self,n_nodes,n_epochs,M,t_0,t_1,lamb=0,eta = 0.1,adaptive_learningrate = True):
		"""
		arguments:
		n_nodes - List containing the number of nodes in each layer
		M - batch size
		n_epochs - number of epochs
		t_0, t_1 - initial learning rate
		lamb - L2 penalization factor
		eta - static learning rate
		adaptive_learningrate - True if using adaptive learning rate. False if using static
		"""
		self.n_layers = len(n_nodes) #number of hidden layers
		self.n_nodes = n_nodes
		self.n_epochs = n_epochs
		self.M = M
		self.t_0 = t_0
		self.t_1 = t_1
		self.lamb = lamb
		self.eta = eta
		self.adaptive_learningrate = adaptive_learningrate


	def AF_purelin(self,hidden = True,c = 1):
		self.c = c
		def purelin(x):
			return(self.c*x)
		def purelin_derivative(x):
			return(self.c)
		if hidden == True:
			self.activation_function = np.vectorize(purelin)
			self.grad_act = np.vectorize(purelin_derivative)

		else:
			self.output_activation_function = np.vectorize(purelin)
			self.output_grad_act = np.vectorize(purelin_derivative)


	"""
	The following activation functions can be chosen for the hidden layer if hidden=True 
	and for the output layer if hidden=False
	"""
	def AF_pRELU(self,a, hidden=True):
		self.a = a
		def pRELU(x):
			if x <= 0:
				return(self.a*x)
			else:
				return(x)
		def pRELU_derivative(x):
			if x <= 0:
				return(self.a)
			else:
				return(1)
		if hidden == True:
			self.activation_function = np.vectorize(pRELU)
			self.grad_act = np.vectorize(pRELU_derivative)

		else:
			self.output_activation_function = np.vectorize(pRELU)
			self.output_grad_act = np.vectorize(pRELU_derivative)


	def AF_sigmoid(self, hidden = True):
		def sigmoid(x):
			return(1/(1 + np.exp(-x)))
		def sigmoid_derivative(x):
			return(self.activation_function(x)*(1 - self.activation_function(x)))
		if hidden == True:
			self.activation_function = sigmoid
			self.grad_act = sigmoid_derivative

		else:
			self.output_activation_function = sigmoid
			self.output_grad_act = sigmoid_derivative
		
	
	def AF_ELU(self,hidden = True, alpha = 0.1):
		self.alpha = alpha
		def ELU(x):
			if x < 0:
				return(self.alpha*(np.exp(x) - 1))
			else:
				return(x)
		def ELU_derivative(x):
			if x < 0:
				return(self.alpha*np.exp(x))
			else:
				return(1)
		if hidden == True:
			self.activation_function = np.vectorize(ELU)
			self.grad_act = np.vectorize(ELU_derivative)
		else:
			self.output_grad_act = np.vectorize(ELU_derivative)
			self.output_activation_function = np.vectorize(ELU)

	"""
	self.cost_ordinary() if regression problem
	self.cost_classification() if binary classification problem
	"""
	def cost_ordinary(self):
		def cost1(y_model):
			return(0.5*np2.sum((self.Y_batch.reshape(y_model.shape) - y_model)**2))
		self.cost_function = cost1
		def der_cost1(y_model):
			return(y_model - self.Y_batch.reshape(y_model.shape))
		self.grad_cost_y = der_cost1
		self.classification = False

	def cost_classification(self):
		def likelihood(y_model):
			return((y_model - self.Y_batch.reshape(y_model.shape))/(y_model*(1 - y_model)))
		self.grad_cost_y = likelihood
		self.classification = True
	
	def feed_forward(self):
		"""
		Feed forward algorithm
		"""
		self.z[0] = self.W[0]@self.X_batch.T + self.b[0]
		self.y[0] = self.activation_function(self.z[0])
		for i in range(1,self.n_layers):
			self.z[i] = self.W[i]@self.y[i-1] + self.b[i]
			self.y[i] = self.activation_function(self.z[i])

	def backpropagation(self):
		"""
		Back propagation algorithm
		"""
		self.delta[-1] = self.output_grad_act(self.z[-1])*self.grad_cost_y(self.y[-1])
		for i in range(len(self.delta)-2,-1,-1):
			self.delta[i] = self.W[i+1].T@self.delta[i+1]*self.grad_act(self.z[i])
		
		if self.adaptive_learningrate == False:
			self.W[-1] = self.W[-1] - self.eta*(self.delta[-1]@self.y[-2].T + self.lamb*self.W[-1])
			self.b[-1] = self.b[-1] - self.eta*np.sum(self.delta[-1],axis=1).reshape(len(self.b[-1]),1)
			for l in range(self.n_layers-2,0,-1):
				self.W[l] = self.W[l] - self.eta*(self.delta[l]@self.y[l-1].T + self.lamb*self.W[l])
				self.b[l] = self.b[l] - self.eta*np.sum(self.delta[l],axis=1).reshape(len(self.b[l]),1)
			self.W[0] = self.W[0] - self.eta*(self.delta[0]@self.X_batch + self.lamb*self.W[0])
			self.b[0] = self.b[0] - self.eta*np.sum(self.delta[0],axis=1).reshape(len(self.b[0]),1)

		else:
			self.W[-1] = self.W[-1] - (self.t_0/(self.epoch*self.m + self.i + self.t_1))*(self.delta[-1]@self.y[-2].T + self.lamb*self.W[-1])
			self.b[-1] = self.b[-1] - (self.t_0/(self.epoch*self.m + self.i + self.t_1))*np.sum(self.delta[-1],axis=1).reshape(len(self.b[-1]),1)
			for l in range(self.n_layers-2,0,-1):
				self.W[l] = self.W[l] - (self.t_0/(self.epoch*self.m+self.i + self.t_1))*(self.delta[l]@self.y[l-1].T + self.lamb*self.W[l])
				self.b[l] = self.b[l] - (self.t_0/(self.epoch*self.m+self.i + self.t_1))*np.sum(self.delta[l],axis=1).reshape(len(self.b[l]),1)
			self.W[0] = self.W[0] - (self.t_0/(self.epoch*self.m+self.i + self.t_1))*(self.delta[0]@self.X_batch + self.lamb*self.W[0])
			self.b[0] = self.b[0] - (self.t_0/(self.epoch*self.m+self.i + self.t_1))*np.sum(self.delta[0],axis=1).reshape(len(self.b[0]),1)
		
	def initialize(self,X):
		"""
		Initializes weights. Its done automatically in self.fit()
		"""
		X = X[0:self.M,:]
		W = []
		b = []
		z = []
		y = []
		delta = []
		W.append(np2.random.randn(self.n_nodes[0],len(X[0,:]))*np.sqrt(1/len(X[0,:])))
		b.append(np2.random.randn(self.n_nodes[0],1))
		z.append(np2.zeros((W[0]@X.T).shape))
		y.append(np2.zeros((W[0]@X.T).shape))
		delta.append(np2.zeros(y[0].shape))
		for i in range(1,self.n_layers):
			W.append(np2.random.randn(self.n_nodes[i],self.n_nodes[i-1])*np.sqrt(1/self.n_nodes[i-1]))
			b.append(np2.zeros((self.n_nodes[i],1)) + 0.1)
			z.append(np2.zeros((W[i]@y[i-1]).shape))
			y.append(np2.zeros(z[i].shape))
			delta.append(np2.zeros(y[i].shape))
		self.W = W
		self.b = b
		self.z = z
		self.y = y
		self.delta = delta


	def fit(self,X,Y):
		"""
		arguments:
		X - design matrix
		y - output targets
		"""
		self.Y = Y
		self.X = X
		m = int(np2.floor(len(self.Y)/self.M))
		self.m = m
		np2.random.seed(1)
		self.initialize(X)
		batch_i = np2.random.choice(m*self.M,size=(m,self.M),replace=False)
		for epoch in range(0,self.n_epochs):
			self.epoch = epoch
			for i in range(m):
				self.i = i
				k = np2.random.randint(m)
				self.X_batch = self.X[batch_i[k,:],:]
				self.Y_batch = self.Y[batch_i[k,:]]
				self.feed_forward()
				self.backpropagation()
		
		z = self.W[0]@self.X.T + self.b[0]
		y0 = self.activation_function(z)
		for i in range(1,self.n_layers-1):
			y1 = self.activation_function(self.W[i]@y0 + self.b[i])
			y0 = y1
		y1 = self.output_activation_function(self.W[-1]@y0 + self.b[-1])
		self.y_fi
