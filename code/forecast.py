import numpy as np
import tensorflow as tf
import utility2
import sys
import time

start_time = time.time()
print >> sys.stderr, 'Start time', start_time 
N = int(sys.argv[1])    # no. of users
folder = ''             # directory where twitter dataset is stored
if len(sys.argv)>2:
	folder = sys.argv[2]
	if not folder[-1]=='/':
		folder = folder + '/'
model_file = sys.argv[3]    # trained model location
THRESHOLD = 0.1
hidden_layer_size = N
input_size = N
target_size = 1

inv_map = utility2.get_id_map(folder+'good_id_to_old_id.json',N)
adj = utility2.get_adj(folder+'edgelist.txt', inv_map, N)
H = utility2.get_history(folder+'opinion.txt', inv_map, N)
G = utility2.make_adj_H(H,N)

# H = [x for x in H if int(x[0])<N]
# adj = [x[:N] for x in adj[:N]]

# X = utility2.make_opinions(folder+'new_opinion.txt',H,inv_map,N)
# X = [X[i][:N] for i in range(len(H)) if int(H[i][0]) < N]
X = utility2.get_m_H(H)
a,b,c = utility2.make_input(H[0],H[0],N,G)
user=[a]
dt=[b]
dm=[c]

for i in range(1, len(H)):
    a,b,c=utility2.make_input(H[i],H[i-1],N,G)
    user.append(a)
    dt.append(b)
    dm.append(c)

total_size = len(H)

pre_processed = time.time()
print >> sys.stderr, 'After pre-processing', pre_processed 
batch_size = int(0.9*len(H))
num_classes = 2
state_size = N
learning_rate = 0.1

"""
Placeholders
"""
U = tf.placeholder(tf.float32, [total_size,state_size], name='user')
T = tf.placeholder(tf.float32, [total_size,state_size], name='delta_t')
M = tf.placeholder(tf.float32, [total_size,state_size], name='delta_m')
ADJ = tf.placeholder(tf.float32, [state_size,state_size], name='adj_')
O = tf.placeholder(tf.float32, [total_size], name='o')
init_state = tf.zeros([state_size])


"""
Function to train the network
"""
with tf.variable_scope('rnn_cell', reuse=None):
    w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w9 = tf.get_variable('w9', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w10 = tf.get_variable('w10', [state_size], initializer=tf.random_uniform_initializer(0.1,1.0))
    w11 = tf.get_variable('w11', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    w12 = tf.get_variable('w12', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    w13 = tf.get_variable('w13', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    w14 = tf.get_variable('w14', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    w22 = tf.get_variable('w22', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    w23 = tf.get_variable('w23', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    w24 = tf.get_variable('w24', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    w25 = tf.get_variable('w25', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    w26 = tf.get_variable('w26', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    w27 = tf.get_variable('w27', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    W = tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))

    w1_x = tf.get_variable('w1_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w2_x = tf.get_variable('w2_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w3_x = tf.get_variable('w3_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w4_x = tf.get_variable('w4_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w5_x = tf.get_variable('w5_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w6_x = tf.get_variable('w6_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w7_x = tf.get_variable('w7_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w8_x = tf.get_variable('w8_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w9_x = tf.get_variable('w9_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w10_x = tf.get_variable('w10_x', [state_size], initializer=tf.random_uniform_initializer(0.1,1.0))
    w11_x = tf.get_variable('w11_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    w12_x = tf.get_variable('w12_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    w13_x = tf.get_variable('w13_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    w14_x = tf.get_variable('w14_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    w22_x = tf.get_variable('w22_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    w23_x = tf.get_variable('w23_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    w24_x = tf.get_variable('w24_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    w25_x = tf.get_variable('w25_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    w26_x = tf.get_variable('w26_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    w27_x = tf.get_variable('w27_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    W_x = tf.get_variable('W_x', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))


def rnn_cell(u,t,m,state,state_x,state2,state2_x,state3,state3_x,state4,state4_x): # modify weights
    with tf.variable_scope('rnn_cell', reuse=True):
        w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w9 = tf.get_variable('w9', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w10 = tf.get_variable('w10', [state_size], initializer=tf.random_uniform_initializer(0.1,1.0))
        w11 = tf.get_variable('w11', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        w12 = tf.get_variable('w12', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        w13 = tf.get_variable('w13', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        w14 = tf.get_variable('w14', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        w22 = tf.get_variable('w22', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        w23 = tf.get_variable('w23', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        w24 = tf.get_variable('w24', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        w25 = tf.get_variable('w25', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        w26 = tf.get_variable('w26', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        w27 = tf.get_variable('w27', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        W = tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))

        w1_x = tf.get_variable('w1_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w2_x = tf.get_variable('w2_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w3_x = tf.get_variable('w3_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w4_x = tf.get_variable('w4_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w5_x = tf.get_variable('w5_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w6_x = tf.get_variable('w6_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w7_x = tf.get_variable('w7_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w8_x = tf.get_variable('w8_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w9_x = tf.get_variable('w9_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w10_x = tf.get_variable('w10_x', [state_size], initializer=tf.random_uniform_initializer(0.1,1.0))
        w11_x = tf.get_variable('w11_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        w12_x = tf.get_variable('w12_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        w13_x = tf.get_variable('w13_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        w14_x = tf.get_variable('w14_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        w22_x = tf.get_variable('w22_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        w23_x = tf.get_variable('w23_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        w24_x = tf.get_variable('w24_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        w25_x = tf.get_variable('w25_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        w26_x = tf.get_variable('w26_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        w27_x = tf.get_variable('w27_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        W_x = tf.get_variable('W_x', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))

        sig1 = w3 * tf.sigmoid(w4 * (m - w5))
        sig2 = w6 * tf.sigmoid(-w7 * (m - w8))
        h = tf.sigmoid(w1 * tf.exp(-w2 * t) * state + tf.matmul([u], W)[0,:] * tf.matmul([u], ADJ)[0,:] * (sig1 - sig2))
        h2 = tf.tanh(w12 * h + w13 * state2 + w14)
        h3 = tf.tanh(w22 * h2 + w23 * state3 + w24) # w24 is bias term introdeced
        h4 = tf.tanh(w25 * h3 + w26 * state4 + w27)
        lamb = tf.exp(w9 + w10 * t + w11 * h4) # depends on last layer hidden state h3

        sig1_x = w3_x * tf.sigmoid(w4_x * (m - w5_x))
        sig2_x = w6_x * tf.sigmoid(-w7_x * (m - w8_x))
        h_x = tf.sigmoid(w1_x * tf.exp(-w2_x * t) * state_x + tf.matmul([u], W_x)[0,:] * tf.matmul([u], ADJ)[0,:] * (sig1_x - sig2_x))
        h2_x = tf.tanh(w12_x * h_x + w13_x * state2_x + w14_x) # w14_x is bias term introdeced
        h3_x = tf.tanh(w22_x * h2_x + w23_x * state3_x + w24_x) # w14_x is bias term introdeced
        h4_x = tf.tanh(w25_x * h3_x + w26_x * state4_x + w27_x)
        x = tf.tanh(w9_x + w11_x * h4_x)

        return h, lamb, h_x, x, h2, h2_x, h3, h3_x, h4, h4_x


"""
Adding rnn_cells to graph

This is a simplified version of the "rnn" function from Tensorflow's api. See:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py

state = init_state
state_x = init_state
st2 = init_state
st2_x = init_state
st3 = init_state
st3_x = init_state
st4 = init_state
st4_x = init_state
lambdas = []
lamb_states = []
opinions = []
op_states = []
lamb_low_states = []
op_low_states = []
lamb_high_states = []
op_high_states = []
lamb_highest_states = []
op_highest_states = []

for i in range(batch_size):
    state, lamb, state_x, x, st2, st2_x, st3, st3_x, st4, st4_x = rnn_cell(tf.gather(U,i),tf.gather(T,i),tf.gather(M,i),state,state_x,st2,st2_x,st3,st3_x,st4,st4_x)
    lambdas.append(lamb)
    lamb_states.append(state)
    opinions.append(x)
    op_states.append(state_x)
    lamb_low_states.append(st2)
    op_low_states.append(st2_x)
    lamb_high_states.append(st3)
    op_high_states.append(st3_x)
    lamb_highest_states.append(st4)
    op_highest_states.append(st4_x)

def get_integral(lamb_high_states):
    ret = 0.0
    with tf.variable_scope('rnn_cell', reuse=True):
        w9 = tf.get_variable('w9', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w10 = tf.get_variable('w10', [state_size], initializer=tf.random_uniform_initializer(0.1,1.0))
        w11 = tf.get_variable('w11', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        for i in range(batch_size-1):
            ret += tf.reduce_sum((tf.exp(w9+w10*tf.gather(T,i+1)+w11*lamb_high_states[i]) - tf.exp(w9+w11*lamb_high_states[i]))/w10)
    return ret

def log_lambda(lambdas):
    log_sum=0.0
    for i in range(batch_size):
        log_sum += tf.log(tf.reduce_sum(lambdas[i]*tf.gather(U,i)))
    return log_sum

def get_mse(opinions):
    val = 0.0
    for i in range(batch_size):
        val+= tf.reduce_sum(tf.pow(tf.reduce_sum(opinions[i]*tf.gather(U,i)) - tf.gather(O,i), 2))
    return val


total_loss = tf.Variable(tf.zeros([], dtype=np.float32), name='total_loss')
losses = get_integral(lamb_highest_states) - log_lambda(lambdas) + get_mse(opinions)
total_loss = losses
#tf.scalar_summary('total_loss', tf.reshape(total_loss,[]))
#merged = tf.merge_all_summaries()
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)
"""
saver = tf.train.Saver()


t_w1 = np.zeros((state_size),dtype=np.float32)
t_w2 = np.zeros((state_size),dtype=np.float32)
t_w3 = np.zeros((state_size),dtype=np.float32)
t_w4 = np.zeros((state_size),dtype=np.float32)
t_w5 = np.zeros((state_size),dtype=np.float32)
t_w6 = np.zeros((state_size),dtype=np.float32)
t_w7 = np.zeros((state_size),dtype=np.float32)
t_w8 = np.zeros((state_size),dtype=np.float32)
t_w9 = np.zeros((state_size),dtype=np.float32)
t_w10 = np.zeros((state_size),dtype=np.float32)
t_w11 = np.zeros((state_size),dtype=np.float32)
t_w12 = np.zeros((state_size),dtype=np.float32)
t_w13 = np.zeros((state_size),dtype=np.float32)
t_w14 = np.zeros((state_size),dtype=np.float32)
t_w22 = np.zeros((state_size),dtype=np.float32)
t_w23 = np.zeros((state_size),dtype=np.float32)
t_w24 = np.zeros((state_size),dtype=np.float32)
t_w25 = np.zeros((state_size),dtype=np.float32)
t_w26 = np.zeros((state_size),dtype=np.float32)
t_w27 = np.zeros((state_size),dtype=np.float32)
t_W = np.zeros((state_size,state_size),dtype=np.float32)

t_w1_x = np.zeros((state_size),dtype=np.float32)
t_w2_x = np.zeros((state_size),dtype=np.float32)
t_w3_x = np.zeros((state_size),dtype=np.float32)
t_w4_x = np.zeros((state_size),dtype=np.float32)
t_w5_x = np.zeros((state_size),dtype=np.float32)
t_w6_x = np.zeros((state_size),dtype=np.float32)
t_w7_x = np.zeros((state_size),dtype=np.float32)
t_w8_x = np.zeros((state_size),dtype=np.float32)
t_w9_x = np.zeros((state_size),dtype=np.float32)
t_w10_x = np.zeros((state_size),dtype=np.float32)
t_w11_x = np.zeros((state_size),dtype=np.float32)
t_w12_x = np.zeros((state_size),dtype=np.float32)
t_w13_x = np.zeros((state_size),dtype=np.float32)
t_w14_x = np.zeros((state_size),dtype=np.float32)
t_w22_x = np.zeros((state_size),dtype=np.float32)
t_w23_x = np.zeros((state_size),dtype=np.float32)
t_w24_x = np.zeros((state_size),dtype=np.float32)
t_w25_x = np.zeros((state_size),dtype=np.float32)
t_w26_x = np.zeros((state_size),dtype=np.float32)
t_w27_x = np.zeros((state_size),dtype=np.float32)
t_W_x = np.zeros((state_size,state_size),dtype=np.float32)

def train_network(num_epochs,state_size=N, verbose=True):
    with tf.Session() as sess:
        import os
        # if not os.path.isdir(folder+'logs_3_stacked'+str(N)):
        #     os.mkdir(folder+'logs_3_stacked'+str(N))
        #train_writer = tf.train.SummaryWriter(folder+'logs_3_stacked'+str(N), sess.graph)
        sess.run(tf.initialize_all_variables())
        training_losses = []
        if os.path.isfile(model_file):
            saver.restore(sess, model_file)
            training_state = np.zeros((state_size))
            t_w1, t_w2, t_w3, t_w4, t_w5, t_w6, t_w7, t_w8, t_w9, t_w10, t_w11, t_w12, t_w13, t_w14, t_w22, t_w23, t_w24, t_w25, t_w26, t_w27, t_W = sess.run(
                [w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w22,w23,w24,w25,w26,w27,W],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            t_w1_x, t_w2_x, t_w3_x, t_w4_x, t_w5_x, t_w6_x, t_w7_x, t_w8_x, t_w9_x, t_w10_x, t_w11_x, t_w12_x, t_w13_x, t_w14_x, t_w22_x, t_w23_x, t_w24_x, t_w25_x, t_w26_x, t_w27_x, t_W_x = sess.run(
                [w1_x,w2_x,w3_x,w4_x,w5_x,w6_x,w7_x,w8_x,w9_x,w10_x,w11_x,w12_x,w13_x,w14_x,w22_x,w23_x,w24_x,w25_x,w26_x,w27_x,W_x],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})

    return training_losses,t_w1, t_w2, t_w3, t_w4, t_w5, t_w6, t_w7, t_w8, t_w9, t_w10, t_w11, t_w12, t_w13, t_w14, t_w22, t_w23, t_w24, t_w25, t_w26, t_w27, t_W, t_w1_x, t_w2_x, t_w3_x, t_w4_x, t_w5_x, t_w6_x, t_w7_x, t_w8_x, t_w9_x, t_w10_x, t_w11_x, t_w12_x, t_w13_x, t_w14_x, t_w22_x, t_w23_x, t_w24_x, t_w25_x, t_w26_x, t_w27_x, t_W_x

training_losses,t_w1, t_w2, t_w3, t_w4, t_w5, t_w6, t_w7, t_w8, t_w9, t_w10, t_w11, t_w12, t_w13, t_w14, t_w22, t_w23, t_w24, t_w25, t_w26, t_w27, t_W, t_w1_x, t_w2_x, t_w3_x, t_w4_x, t_w5_x, t_w6_x, t_w7_x, t_w8_x, t_w9_x, t_w10_x, t_w11_x, t_w12_x, t_w13_x, t_w14_x, t_w22_x, t_w23_x, t_w24_x, t_w25_x, t_w26_x, t_w27_x, t_W_x = train_network(2000)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def rnn_cell_test(u,t,m,state,state_x,state2,state2_x,state3,state3_x,state4,state4_x):

    sig1 = t_w3 * sigmoid(t_w4 * (m - t_w5))
    sig2 = t_w6 * sigmoid(-t_w7 * (m - t_w8))
    h = sigmoid(t_w1 * np.exp(-t_w2 * t) * state + np.matmul([u], t_W)[0,:] * np.matmul([u], adj)[0,:] * (sig1 - sig2))
    h2 = np.tanh(t_w12 * h + t_w13 * state2 + t_w14)
    h3 = np.tanh(t_w22 * h2 + t_w23 * state3 + t_w24)
    h4 = np.tanh(t_w25 * h3 + t_w26 * state4 + t_w27)
    lamb = np.exp(t_w9 + t_w10 * t + t_w11 * h4)

    sig1_x = t_w3_x * sigmoid(t_w4_x * (m - t_w5_x))
    sig2_x = t_w6_x * sigmoid(-t_w7_x * (m - t_w8_x))
    h_x = sigmoid(t_w1_x * np.exp(-t_w2_x * t) * state_x + np.matmul([u], t_W_x)[0,:] * np.matmul([u], adj)[0,:] * (sig1_x - sig2_x))
    h2_x = np.tanh(t_w12_x * h_x + t_w13_x * state2_x + t_w14_x)
    h3_x = np.tanh(t_w22_x * h2_x + t_w23_x * state3_x + t_w24_x)
    h4_x = np.tanh(t_w25_x * h3_x + t_w26_x * state4_x + t_w27_x)
    x = np.tanh(t_w9_x + t_w11_x * h4_x)

    return h, lamb, h_x, x, h2, h2_x, h3, h3_x, h4, h4_x


def t_get_integral(t_lamb_high_states):
    ret = 0.0
    for i in range(batch_size,total_size-1):
        ret += np.add.reduce(np.exp(t_w9+t_w10*dt[i+1]+t_w11*t_lamb_high_states[i-batch_size]) - np.exp(t_w9+t_w11*t_lamb_high_states[i-batch_size]) / t_w10)
    return ret

def t_log_lambda(t_lambdas):
    log_sum=0.0
    for i in range(batch_size,total_size):
        log_sum += np.log(np.add.reduce(t_lambdas[i-batch_size]*user[i]))
    return log_sum

def t_get_mse(t_opinions):
    val = 0.0
    for i in range(batch_size,total_size):
        val+= np.power(np.add.reduce(t_opinions[i-batch_size]*user[i]) - X[i], 2)
    return val

def get_sent(u, G, tm):
    for i in G[u]:
        if i[1]>=tm:
            return i[0]
    return -2.5

def test_network():
    t_state = np.zeros((state_size))
    t_state_x = np.zeros((state_size))
    t_lambdas = []
    t_lamb_states = []
    t_opinions = []
    t_op_states = []
    t_st2 = np.zeros((state_size))
    t_st2_x = np.zeros((state_size))
    t_st3 = np.zeros((state_size))
    t_st3_x = np.zeros((state_size))
    t_st4 = np.zeros((state_size))
    t_st4_x = np.zeros((state_size))
    t_lamb_low_states = []
    t_op_low_states = []
    t_lamb_high_states = []
    t_op_high_states = []
    t_lamb_highest_states = []
    t_op_highest_states = []
    for i in range(batch_size,total_size):
        t_state, t_lamb, t_state_x, t_x, t_st2, t_st2_x, t_st3, t_st3_x, t_st4, t_st4_x = rnn_cell_test(user[i],dt[i],dm[i],t_state,t_state_x,t_st2,t_st2_x,t_st3,t_st3_x,t_st4,t_st4_x)
        t_lambdas.append(t_lamb)
        t_lamb_states.append(t_state)
        t_opinions.append(t_x)
        t_op_states.append(t_state_x)
        t_lamb_low_states.append(t_st2)
        t_op_low_states.append(t_st2_x)
        t_lamb_high_states.append(t_st3)
        t_op_high_states.append(t_st3_x)
        t_lamb_highest_states.append(t_st4)
        t_op_highest_states.append(t_st4_x)
    hours = range(11)
    t_last = H[batch_size-1][1]
    print t_last, '\n', H[total_size-1][1], '\n', H[total_size-1][1]-t_last
    #for i in range(batch_size,total_size):
        #hours.append((H[i][1]-t_last)/3600.0)
    mse_list = []
    ind = 0
    for i in hours:
        sum = 0.0
        tm = t_last + i * 3600
        while ind < len(t_opinions) and H[batch_size+ind][1] < tm:
            ind += 1
        num = 0
        for u in range(N):
            g = get_sent(u,G,tm)
            g1 = g
            if ind < len(t_opinions):
               g1 = t_opinions[ind][u]
            if g > -1:
               sum += (g - g1)**2
               num += 1
        if num > 0:
            mse_list.append(sum / float(num))
        else:
            mse_list.append(sum)
    pol_list = []
    ind = 0
    for i in hours:
        sum = 0.0
        tm = t_last + i * 3600
        while ind < len(t_opinions) and H[batch_size+ind][1] < tm:
            ind += 1
        num = 0
        for u in range(N):
            g = get_sent(u,G,tm)
            g1 = g
            if ind < len(t_opinions):
               g1 = t_opinions[ind][u]
            if g > -1 and (g-0.3) * (g1-0.3) < 0:
               sum += 1.0
            if g > -1:
               num += 1
        if num > 0:
            pol_list.append(sum / float(num))
        else:
            pol_list.append(sum)

    return hours, mse_list, pol_list

hours_, mse_list_, pol_list_ = test_network()
#print >> sys.stderr, "(MSE, SE, LAMBDA, TOTAL LOSS, POLARITY ERROR)", mse_loss_/float(total_size-batch_size), mse_loss_,lamb_loss_,test_loss_,pol_loss_
#print "(MSE, SE, LAMBDA, TOTAL LOSS, POLARITY ERROR)", mse_loss_/float(total_size-batch_size), mse_loss_,lamb_loss_,test_loss_,pol_loss_
print 'T = ', hours_, '\nMSE = ', mse_list_, '\nPolarity error = ',pol_list_
end_time = time.time()
print >> sys.stderr, 'Duration', end_time - start_time, 'seconds'
#print "MSE through time:", mlist_, '\nPolarity error through time:', plist_, '\nTime:', hours_
