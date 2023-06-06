#!/usr/bin/env python
# coding: utf-8
# In[1]:
#from IPython import get_ipython
#get_ipython().run_line_magic('env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID')
#get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=0')
#get_ipython().run_line_magic('env', 'XLA_PYTHON_CLIENT_PREALLOCATE=false')
"""
Example usage:

python -u main.py  --dataset_name omniglot --inner_steps 50 --num_clients 50 --algo pflego --rounds 5000 --client_lr 0.007 --server_lr 0.003 --scenario highpers --use_gpu True
#fedper = fedavg_personalized
python -u main.py  --dataset_name mnist --inner_steps 50 --num_clients 100 --algo fedavg_personalized --rounds 200 --client_lr 0.007 --server_lr 1.0 --scenario highpers --use_gpu True

python -u main.py  --dataset_name mnist --inner_steps 50 --num_clients 100 --algo fedavg_vanilla --rounds 200 --client_lr 0.007 --server_lr 1.0 --scenario highpers
"""

import argparse

parser = argparse.ArgumentParser( description='Federated Learning Process' )
parser.add_argument( "--inner_steps", type=int, help='Number of epochs', default=4 )
parser.add_argument( "--rounds", type=int, help="Number of rounds", default=200 )
parser.add_argument( "--dataset_name", type=str, help="Dataset to use e.x. cifar10, cifar100, mnist, fashion_mnist, emnist, miniimagenet", default='cifar10' )
parser.add_argument( "--algo", type=str, help="Algorithm to use fedavg, pflego", default='pflego' )
parser.add_argument( "--nn", type=str, help="Neural Network architecture", default='nn_default' )
parser.add_argument( "--num_clients", type=int, help="Number of clients", default='100' )
parser.add_argument( "--client_lr", type=float, help="Learning rate of client", default=0.01 )
parser.add_argument( "--server_lr", type=float, help="Learning rate of server", default=0.01 )
parser.add_argument( "--use_gpu", type=str, help="Learning rate of clients", default="False" )
parser.add_argument( "--scenario", type=str, help="Splitting scenario e.x. Round Robin", default="round_robin_medium" )
parser.add_argument( "--aggregate", type=str, help="Simple or Weighted Average", default="weighted" )
parser.add_argument( "--users_fraction", type=float, help="Fraction of users", default=0.2 )
parser.add_argument( "--optimizer", type=str, help="Optimizer sgd/adam, by default adam", default="adam" )
parser.add_argument( "--reset_mode", type=str, help="Reset client-speicific weights, by default empy, other modes: reset", default="" )
parser.add_argument( "--mu", type=float, help="Regularizer for theta, it is used only in FedProx", default=0.0 )
parser.add_argument( "--random_max_users_fraction", type=float, help="If the argument is positive, then we sample users_fraction up to random_max_fraction users", default=0.0 )
parser.add_argument( "--local_rep_steps", type=int, help="Number of epochs for training representation, only for FedRep and Ditto", default=0 )
parser.add_argument( "--put_on_cpu", type=str, help="Put jax data on cpu", default="False" )
parser.add_argument( "--ditto_global_lr", type=float, help="Learning rate for global weights of ditto", default=0.0 )
parser.add_argument( "--momentum", type=float, help="Momentum for client [FedRep only]", default=0.0 )

args = parser.parse_args()

rounds = args.rounds
inner_steps = args.inner_steps
dataset_name = args.dataset_name
algorithm = args.algo#pflego, fedavg_vanilla, fedavg_personalized, fedavg_vanilla_gradient
nn_architecture = args.nn
num_clients = args.num_clients
client_learning_rate = args.client_lr
server_learning_rate = args.server_lr
ditto_global_lr = args.ditto_global_lr
use_gpu = args.use_gpu
scenario = args.scenario#round_robin_low, round_robin_medium, round_robin_high
aggregate = args.aggregate
users_fraction = args.users_fraction
optimizer = args.optimizer
reset = args.reset_mode
mu = args.mu
original_mu = mu
random_max_users_fraction = args.random_max_users_fraction
local_rep_steps = args.local_rep_steps
put_on_cpu = args.put_on_cpu
momentum = args.momentum

if put_on_cpu == "False":
    put_on_cpu = False
else:
    put_on_cpu = True

C = int(num_clients * users_fraction )

if algorithm not in [ 'pflego', 'fedavg_vanilla', 'fedavg_personalized', 'fedavg_vanilla_gradient', 'fedrecon', 'fedrep', 'fedprox', 'feddane', 'ditto', 'fedbabu' ]:
    print( "Sorry not recognized algorithm" )
    exit(0)

if reset not in [ "", "reset" ]:
    print( "Sorry wrong reset mode" )
    exit(0)

if reset != "" and algorithm != "fedrecon":
    print( "Sorry only for fedrecon atm" )
    exit(0)

#import libraries
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if use_gpu == "False":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax.numpy as jnp
import numpy as np
import jax
#import optax
import os
print( "Jax devices" )
print( jax.devices() )

cpus = jax.devices("cpu")
print( "cpus" )
print( cpus )
#input()

from jax import grad
from jax import random
from jax.experimental import optimizers
#from jax.experimental import optix
from jax.experimental.optimizers import clip_grads
import tensorflow as tf
print( tf.__version__ )
tf.config.experimental.set_visible_devices([], "GPU")
#tf.compat.v1.enable_eager_execution()
tf.keras.backend.set_floatx('float64')
tf.executing_eagerly()
from copy import deepcopy
from jax import vmap # for auto-vectorizing functions
from functools import partial # for use with vmap
from jax import jit # for compiling functions for speedup
from jax.experimental import stax # neural network library
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, LeakyRelu, Flatten, Identity, LogSoftmax, BatchNorm # neural network layers
import matplotlib.pyplot as plt # visualization
from jax import jit # for compiling functions for speedup
from jax.tree_util import tree_multimap 
from jax.experimental.stax import elementwise
Redu_Mean = elementwise(jnp.mean, axis=[1,2] )
import tensorflow_datasets as tfds
from jax.experimental.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                   FanOut, Flatten, GeneralConv, Identity,
                                   MaxPool, Relu, LogSoftmax)
#end of import libraries

print( "Settings:")
print( "\tInner steps:", inner_steps )
print( "\tRounds:", rounds )
print( "\tDataset_name:", dataset_name )
print( "\nAlgorithm:", algorithm )
print( "\tArchitecture", nn_architecture )
print( "\tNumber of clients", num_clients )
print( "\tLearning rate", client_learning_rate )
print( "\tServer learning rate", server_learning_rate )
print( "\tUse Gpu", use_gpu )
print( "\tScenario", scenario )
print( "\tFract of clients", users_fraction, 'per round', C )
print( "\tGradient optimizer", optimizer )
print( "\tReset mode", reset )
print( "\tMu", mu )
print( "\tRandom_max_users_fraction", random_max_users_fraction )
print( "\tLocal rep steps[FedRep only]", local_rep_steps )
print( "\tMomentum[FedRep only]", momentum )

random_users_flag = False
if random_max_users_fraction > 0:
    if users_fraction > random_max_users_fraction:
        print( "users_fraction mu be <= random_max_users_fraction", users_fraction, random_max_fraction )
        exit(0)
    random_users_flag = True
    print( "Users will range" )

if optimizer not in ['adam','sgd']:
    print( "Wrong input for optimizer" )
    exit(0)

folder_struct = "../out"
if os.path.exists( folder_struct ) == False:
    os.mkdir(folder_struct)
    print( "Created folder", folder_struct )

folder_struct = "../out/results"
if os.path.exists(folder_struct) == False:
    os.mkdir(folder_struct)
    print( "Created folder", folder_struct )

folder_struct = "../out/results/" + dataset_name
if os.path.exists(folder_struct) == False:
    os.mkdir(folder_struct)
    print( "Created folder", folder_struct )
#create learning rate folder
folder_struct += "/" + str(client_learning_rate)
if os.path.exists(folder_struct) == False:
    os.mkdir(folder_struct)
    print( "Created folder", folder_struct )
#create algorithm folder
folder_struct += "/" + str(algorithm)
if os.path.exists(folder_struct) == False:
    os.mkdir(folder_struct)
    print( "Created folder", folder_struct )
#create architecture folder
folder_struct += "/" + str(nn_architecture)
if os.path.exists(folder_struct) == False:
    os.mkdir(folder_struct)
    print( "Created folder", folder_struct )
folder_struct += "/" + str(scenario)
if os.path.exists(folder_struct) == False:
    os.mkdir(folder_struct)
    print( "Created folder", folder_struct )
folder_struct += "/"

print( "Results will be saved at")
print( folder_struct )
#exit(0)

if scenario == "highpers":
    #each client have 2 categories
    K = 2
elif scenario == "mediumpers":
    #each client have 5 categories
    K = 5
elif scenario == "nopers":
    #each client have all categories
    K = 10
elif scenario == "divpers":
    #each client can have 2 up to 5  
    K = 5
else:
    print( "Wrong Input -- Abort")
    exit(0)

use_miniimagenet = False
use_mnist = False#True
use_cifar10 = False#True
use_cifar100 = False
use_emnist = False
use_omniglot = False
use_fashion_mnist = False

if dataset_name == 'cifar10':
    use_cifar10 = True
elif dataset_name == 'cifar100':
    use_cifar100 = True
elif dataset_name == 'mnist':
    use_mnist = True
elif dataset_name == 'fashion_mnist':
    use_fashion_mnist = True
elif dataset_name == 'emnist':
    use_emnist = True
    if scenario == "highpers":
        #each client have 2 categories
        K = 2
    elif scenario == "mediumpers":
        #each client have 31 categories
        K = 31
    elif scenario == "nopers":
        #each client have all categories
        K = 62
    elif scenario == "divpers":
        #each client can have 2 up to 5  
        K = 5
    else:
        print( "Wrong Input -- Abort")
        exit(0)
    print( "changed K to ", K )
elif dataset_name == 'miniimagenet':
    use_miniimagenet = True
elif dataset_name == 'omniglot':
    use_omniglot = True
else:
    print( "Sorry not recognizable dataset")
    exit(0)

Redu_Mean = elementwise(jnp.mean, axis=[1,2] )

if use_emnist:
    emnist_ds = tfds.load( "emnist", as_supervised=True, batch_size=-1 )
elif use_mnist:
    mnist_ds = tfds.load( "mnist", as_supervised=True, batch_size=-1 )
elif use_fashion_mnist:
    fashion_mnist_ds = tfds.load( "fashion_mnist", as_supervised=True, batch_size=-1 )
elif use_cifar10:
    cifar10_ds = tfds.load( "cifar10", as_supervised=True, batch_size=-1 )
elif use_cifar100:
    cifar100_ds = tfds.load("cifar100", as_supervised=True,  batch_size=-1 )#, shuffle_files=False )
elif use_omniglot:
    omniglot_ds = tfds.load( "omniglot", as_supervised=False )

#omniglot only
#Gathers 50 alphabets
def GatherOmniglot(ds):
    A = []
    for i in range(50):
        A.append( {} )

    def fill_list( ds, A ):
        for i in ds:
            alphabet = int( i['alphabet'].numpy() )
            alphabet_char_id = int( i['alphabet_char_id'].numpy() )
            if alphabet_char_id not in A[ alphabet ]:
                A[ alphabet ][ alphabet_char_id ] = []
            #print( 'belongs to alphabet', alphabet, 'and is character of order', alphabet_char_id )
            A[ alphabet ][ alphabet_char_id ].append( i['image'].numpy() )    

    fill_list( ds['train'], A )
    fill_list( ds['test'], A )
    
    return A

#Creates 50 users dictionaries
def SplitOmniglotIntoTrainTest( alphabets, rng ):
    
    train, test, cat_per_client = [], [], []
    
    for i in range( len(alphabets) ):#for each alphabet
        print( 'alphabet', i )
        user_train_dict, user_test_dict = { 'image': [], 'label': [] }, { 'image': [], 'label': [] }
        
        #print( 'user', i)
        keys = list( alphabets[i].keys() )#ordering in omniglot does not matter -- no missing categories -- could just be 0..len(alphabets[i])
        
        cat_per_client.append( [0] * len( keys ) )

        #print( keys )
        for j in range( len(keys) ):#for each category of alphabet[i]
            
            digit = keys[j]#true label
            
            #gather the existing 20 samples of digit
            #255 -uint8
            class_samples = tf.cast( tf.image.resize( tf.image.rgb_to_grayscale( 255-np.array( alphabets[ i ][ digit ] ) ), [28,28] ), dtype=tf.float64 )

            class_samples = np.array( class_samples )#alphabets[ i ][ digit ] )#needs np.array in order to immediately use [ permutation[...] ]
            
            augmented_data = np.concatenate( [ class_samples,
                        tf.image.rot90(class_samples,1).numpy(),
                        tf.image.rot90(class_samples,2).numpy(),
                        tf.image.rot90(class_samples,3).numpy() ], axis=0 )

            #print( 'maximum', augmented_data.max() )#correctly it reports 255.0
            #print( 'minimum', augmented_data.min() )#correctly it reports 0.0

            #split into 5 train - 15 test
            permutation = rng.permutation(20 * 4)#x4 rotations
            perm_train = augmented_data[ permutation[:15*4] ]
            perm_test = augmented_data[ permutation[15*4:] ]
            
            #append dictionaries for user_i
            user_train_dict['image'].extend( perm_train )
            user_test_dict['image'].extend( perm_test )
            
            user_train_dict['label'].extend( [digit] * 15 * 4 )#true class order
            user_test_dict['label'].extend( [ digit] * 5 * 4 )#true class order
            
        #add user_i dictionary
        train.append( user_train_dict )
        test.append( user_test_dict )
        
    for i in range( 50 ):
        with tf.device('/CPU:0'):
            train[i]['label'] = tf.Variable( train[i]['label'], dtype=tf.int64 )
            test[i]['label'] = tf.Variable( test[i]['label'], dtype=tf.int64 )

    #return dictionaries for 50 users
    return train, test, cat_per_client
#end of omniglot only

def split_ds_into_classes( ds, key='train', num_classes=10 ):
    images_per_class = []
    for i in range(num_classes):
        mask = ds[ key ][1].numpy() == i
        images_per_class.append( ds[ key ][0][ mask ] )
    return images_per_class
    
if use_mnist:
    ipc_train = split_ds_into_classes( mnist_ds, 'train' )
    ipc_test = split_ds_into_classes( mnist_ds, 'test' )
elif use_cifar10:
    ipc_train = split_ds_into_classes( cifar10_ds, 'train' )
    ipc_test = split_ds_into_classes( cifar10_ds, 'test' )
elif use_fashion_mnist:
    ipc_train = split_ds_into_classes( fashion_mnist_ds, 'train' )
    ipc_test = split_ds_into_classes( fashion_mnist_ds, 'test' )
elif use_emnist:
    ipc_train = split_ds_into_classes( emnist_ds, 'train', num_classes=62 )
    ipc_test = split_ds_into_classes( emnist_ds, 'test', num_classes=62 )
elif use_omniglot:
    pass#no need for omniglot

def random_number( a, b, rng ):
    return int( (b-a) * rng.random() + a )

def RoundRobin( sample_per_class, num_users=100, cat_per_client=None, mode='train', users_that_have_cat=None, rng=None, scenario='mediumpers' ):
    #Input
    #sampler per clas, e.x. spc[0] contains samples of class 0
    #number of users
    #Output: return clients

    spc = deepcopy(sample_per_class)

    if rng == None:
        rng = np.random.default_rng(1)

    tf_seed = random_number( 0, 1000, rng )
    tf.random.set_seed( tf_seed )

    #permute samples per category
    for c in range( len(spc) ):
        perm = tf.random.shuffle(tf.range(tf.shape( spc[c] )[0]))
        spc[c] = tf.gather( spc[c], perm, axis=0)

    M = len(spc)#number of classes

    if M != 10 and M != 62:
        print( "Error M can only be 10 or 62, here it is := ", M )
        exit(0)
    users_dict = [ { 'image': None, 'label': [] } for i in range(num_users) ]#create empty users
    print( scenario )
    #first assign categories per client
    if cat_per_client == None:
        if scenario == "highpers":
            #each client have 2
            cat_per_client = [ rng.choice( M, 2, replace=False ) for i in range( num_users ) ]
        elif scenario == "mediumpers":
            #each client have 5  
            #cat_per_client = [ rng.choice( M, 5, replace=False ) for i in range( num_users ) ]
            cat_per_client = [ rng.choice( M, M//2, replace=False ) for i in range( num_users ) ]
        elif scenario == "nopers":
            #each client have all categories
            cat_per_client = [ rng.choice( M, M, replace=False ) for i in range( num_users ) ]
        elif scenario == "divpers":
            #each client can have 2 up to 5  
            cat_per_client = [ rng.choice( M, max(2,1+rng.choice(5)), replace=False ) for i in range( num_users ) ]
        else:
            print( "UNKNOWN SCENARIO -- Wrong Input -- Abort")
            exit(0)

    if mode == 'train':
        appear = {}
        for i in cat_per_client:
            for j in i:
                if j not in appear:
                    appear[j] = 1
                    if len(appear) == M:
                        break

        #we want to make sure that all categories are going to appear
        if len(appear) != M:
            all_categories = rng.permutation(M)
            cnt = 0
            if len(all_categories) % 2 == 1:#if number of classes id odd then take 3 classes to transition to an even case
                cat_per_client[cnt] = np.array( all_categories[0:3] )
                cnt += 1
                for i in range( 3, M, 2 ):
                    cat_per_client[cnt] = np.array( [ all_categories[i], all_categories[i+1] ] )
                    cnt += 1
            else:
                for i in range( 0, M, 2 ):
                    cat_per_client[cnt] = np.array( [ all_categories[i], all_categories[i+1] ] )
                    cnt += 1

    if users_that_have_cat is None:
        #collect all users that have cat 0...M-1
        users_that_have_cat = [ [] for i in range(M) ]
        for i in range( num_users ):
            for j in cat_per_client[i]:
                users_that_have_cat[ j ].append( i )

    #iterate throuch each spc
    for category in range( len(spc) ):
        lo, hi = 0, len( spc[ category ] )#start, end
        #iterate through all users that have the specific category, until all samples are exhausted
        fl = False
        while fl == False:
            for user in users_that_have_cat[ category ]:
                if users_dict[user]['image'] is None:
                    users_dict[user]['image'] = [ spc[category][lo].numpy() ]#recently changed that part to .numpy, previously was EagerTensor#changed because we use tf.constant now. Maybe using jnp.array would speed up?
                else:
                    users_dict[user]['image'].append( spc[category][lo].numpy()  )# = np.concatenate( [users_dict[user]['image'], spc[category][lo] ], axis=2 )
                #print( type(users_dict[user]['label']), category )
                users_dict[user]['label'].append( category )
                lo += 1
                if lo == hi:
                    fl = True
                    break

    for i in range( num_users ):
        with tf.device('/CPU:0'):
            users_dict[i]['label'] = tf.Variable( users_dict[i]['label'], dtype=tf.int64 )

    return users_dict, cat_per_client, users_that_have_cat

if use_mnist:
    rng = np.random.default_rng(1)
    users_train_dict, cat_per_client, users_that_have_cat = RoundRobin( sample_per_class=ipc_train, num_users=num_clients, rng=rng, scenario=scenario )
    users_test_dict, _, _ = RoundRobin( sample_per_class=ipc_test, num_users=num_clients, cat_per_client=cat_per_client, mode='test', users_that_have_cat=users_that_have_cat, rng=rng, scenario=scenario )
elif use_fashion_mnist:
    rng = np.random.default_rng(1)
    users_train_dict, cat_per_client, users_that_have_cat = RoundRobin( sample_per_class=ipc_train, num_users=num_clients, rng=rng, scenario=scenario )
    users_test_dict, _, _ = RoundRobin( sample_per_class=ipc_test, num_users=num_clients, cat_per_client=cat_per_client, mode='test', users_that_have_cat=users_that_have_cat, rng=rng, scenario=scenario )
elif use_emnist:
    rng = np.random.default_rng(1)
    users_train_dict, cat_per_client, users_that_have_cat = RoundRobin( sample_per_class=ipc_train, num_users=num_clients, rng=rng, scenario=scenario )
    users_test_dict, _, _ = RoundRobin( sample_per_class=ipc_test, num_users=num_clients, cat_per_client=cat_per_client, mode='test', users_that_have_cat=users_that_have_cat, rng=rng, scenario=scenario )
elif use_cifar10:
    rng = np.random.default_rng(1)
    users_train_dict, cat_per_client, users_that_have_cat = RoundRobin( sample_per_class=ipc_train, num_users=num_clients, rng=rng, scenario=scenario )
    users_test_dict, _, _ = RoundRobin( sample_per_class=ipc_test, num_users=num_clients, cat_per_client=cat_per_client, mode='test', users_that_have_cat=users_that_have_cat, rng=rng, scenario=scenario )
elif use_omniglot:
    rng = np.random.default_rng(1)
    Alphabets = GatherOmniglot( omniglot_ds )
    users_train_dict, users_test_dict, cat_per_client = SplitOmniglotIntoTrainTest( Alphabets, rng )

dpc_train = users_train_dict
dpc_test = users_test_dict

import functools

def BuildMnistNetwork():
    return stax.serial(
        Flatten,
        Dense(200), 
        Relu,
    )

def BuildMnistNetworkVanilla():
    return stax.serial(
        Flatten,
        Dense(200),
        Relu,
        Dense( K )
    )

def BuildCifar10Network():
    return stax.serial(
        Conv(64, (5, 5), strides=(1,1), padding='SAME'),
        Relu,
        MaxPool( window_shape=(3,3), strides=(2,2), padding='SAME' ),
        
        Conv(64, (5, 5), strides=(1,1), padding='SAME'),
        Relu,
        MaxPool( window_shape=(3,3), strides=(2,2), padding='SAME' ),
        
        Flatten,
        
        Dense(384),
        Relu,
        Dense(192),
        Relu
    )

def BuildCifar10NetworkVanilla():
    return stax.serial(
        Conv(64, (5, 5), strides=(1,1), padding='SAME'),
        Relu,
        MaxPool( window_shape=(3,3), strides=(2,2), padding='SAME' ),
        
        Conv(64, (5, 5), strides=(1,1), padding='SAME'),
        Relu,
        MaxPool( window_shape=(3,3), strides=(2,2), padding='SAME' ),
        
        Flatten,
        
        Dense(384),
        Relu,
        Dense(192),
        Relu,
        Dense( K )
    )

#Redu_Mean = elementwise(jnp.mean, axis=[1,2] )
def BuildOmniglotNetwork():
    return stax.serial(
        Conv(64, (3, 3), strides=(1,1), padding='SAME'),
        Relu,
        MaxPool( window_shape=(2,2), strides=(2,2), padding='VALID' ),
    
        Conv(64, (3, 3), strides=(1,1), padding='SAME'),
        Relu,
        MaxPool( window_shape=(2,2), strides=(2,2), padding='VALID' ),
    
        Conv(64, (3, 3), strides=(1,1), padding='SAME'),
        Relu,
        MaxPool( window_shape=(2,2), strides=(2,2), padding='VALID' ),
    
        Conv(64, (3, 3), strides=(1,1), padding='SAME'),
        Relu,
        MaxPool( window_shape=(2,2), strides=(2,2), padding='VALID' ),

        Flatten,#64 units
    )

def BuildOmniglotNetworkVanilla( ):
    return stax.serial(
        Conv(64, (3, 3), strides=(1,1), padding='SAME'),
        Relu,
        MaxPool( window_shape=(2,2), strides=(2,2), padding='VALID' ),
    
        Conv(64, (3, 3), strides=(1,1), padding='SAME'),
        Relu,
        MaxPool( window_shape=(2,2), strides=(2,2), padding='VALID' ),
    
        Conv(64, (3, 3), strides=(1,1), padding='SAME'),
        Relu,
        MaxPool( window_shape=(2,2), strides=(2,2), padding='VALID' ),
    
        Conv(64, (3, 3), strides=(1,1), padding='SAME'),
        Relu,
        MaxPool( window_shape=(2,2), strides=(2,2), padding='VALID' ),

        Flatten,#64 units,
        Dense( 55 )
    )


cfg = {
}

def build_vgg9(cfg,batch_norm=False):
    pass

channels = 1
img_w, img_h = 28, 28

if use_cifar10:
    channels = 3
    img_w, img_h = 32, 32#32, 32
    if nn_architecture == 'vgg9':
        net_init, net_apply = build_vgg9( cfg['A'] )
        M = 512
    else:
        if algorithm in [ 'fedavg' ]:
            net_init, net_apply = BuildCifar10NetworkVanilla()
        else:
            net_init, net_apply = BuildCifar10Network()   #ResNet50()
        M = 192#84#256#4096 #OUTSHAPE (3, 61440)
    print( "Use Cifar10 set M = ", M )
elif use_mnist == True:
    channels = 1
    img_w, img_h = 28, 28
    # Use stax to set up network initialization and evaluation functions
    if algorithm in [ 'fedavg' ]:
        net_init, net_apply = BuildMnistNetworkVanilla()
    else:
        net_init, net_apply = BuildMnistNetwork()
    M = 200
    print( "Use Mnist set M = ", M )
elif use_fashion_mnist == True:
    channels = 1
    img_w, img_h = 28, 28
    # Use stax to set up network initialization and evaluation functions
    if algorithm in [ 'fedavg' ]:
        net_init, net_apply = BuildMnistNetworkVanilla()
    else:
        net_init, net_apply = BuildMnistNetwork()
    M = 200
    print( "Use Fashion Mnist set M = ", M )
elif use_emnist == True:
    channels = 1
    img_w, img_h = 28, 28
    # Use stax to set up network initialization and evaluation functions
    if algorithm in [ 'fedavg' ]:
        net_init, net_apply = BuildMnistNetworkVanilla()
    else:
        net_init, net_apply = BuildMnistNetwork()
    M = 200
    print( "Use EMnist set M = ", M )
elif use_omniglot == True:
    channels = 1
    img_w, img_h = 28, 28
    # Use stax to set up network initialization and evaluation functions
    if algorithm in [ 'fedavg' ]:
        net_init, net_apply = BuildOmniglotNetworkVanilla()#1623)
    else:
        net_init, net_apply = BuildOmniglotNetwork()
    M = 64
    print( "Use Omniglot set M = ", M )
elif use_miniimagenet:
    channels = 3
    img_w = img_h = 84

    num_filters = 32 
    # Use stax to set up network initialization and evaluation functions
    net_init, net_apply = stax.serial(
        Conv(num_filters, (3, 3), strides=(1,1), padding='SAME'),
          BatchNorm(),
        Relu,#moved was beforebatchnorm
        MaxPool( window_shape=(2,2), strides=(2,2), padding='VALID' ),
        
        Conv(num_filters, (3, 3), strides=(1,1), padding='SAME'),
        BatchNorm(),
        Relu,
        MaxPool( window_shape=(2,2), strides=(2,2), padding='VALID' ),
        
        Conv(num_filters, (3, 3), strides=(1,1), padding='SAME'),
        BatchNorm(),
        Relu,
        MaxPool( window_shape=(2,2), strides=(2,2), padding='VALID' ),
        
        Conv(num_filters, (3, 3), strides=(1,1), padding='SAME'),
        BatchNorm(),
        Relu,
        MaxPool( window_shape=(2,2), strides=(2,2), padding='VALID' ),
        
        Flatten,

    )
else:
    # Use stax to set up network initialization and evaluation functions
    net_init, net_apply = stax.serial(
        Conv(64, (3, 3), strides=(2,2), padding='SAME'), Relu,
        BatchNorm(),
        Conv(64, (3, 3), strides=(2,2), padding='SAME'), Relu,
        BatchNorm(),
        Conv(64, (3, 3), strides=(2,2), padding='SAME'), Relu,
        BatchNorm(),
        Conv(64, (3, 3), strides=(2,2), padding='SAME'), Relu,
        BatchNorm(),
        Redu_Mean,
        Dense(5),
    )

# Initialize parameters, not committing to a batch shape
rng = random.PRNGKey(0)
    
in_shape = (-1, img_w, img_h, channels)
out_shape, net_params = net_init(rng, in_shape)
print('OUTSHAPE', out_shape )

# Apply network to dummy inputs
inputs = np.zeros((32, img_w, img_h, channels))

if optimizer == "adam":
    opt_init, opt_update, get_params = optimizers.adam(step_size=server_learning_rate)
else:
    opt_init, opt_update, get_params = optimizers.sgd(step_size=server_learning_rate)#optimizers.adam(step_size=0.001)#optimizers.sgd(step_size=0.001)#optimizers.adam(step_size=1e-3)
opt_state = opt_init(net_params)

#test that predictions works fine
predictions2 = net_apply(net_params, inputs)

print( predictions2.shape )
#input()
#exit(0)
for i in net_params:
    for j in i:
        print( j.shape )#shape )


def cce_loss( logits, targets):
    # Computes ccee loss
    logits = stax.logsoftmax(logits )
    return -jnp.mean( jnp.sum( logits * targets,  axis=-1)  ) / 1.0

#pip install tensorflow-datasets
from copy import deepcopy

class MyDataset:
    
    def __init__(self, ds, par='omniglot', num_classes=None):
        self.m = {}
        self.par = par
        self.label_counter = {}
        self.lcnt = 0
        self.num_classes = num_classes
        with tf.device('/CPU:0'):
            self.ds = deepcopy(ds)
            #Resize,Normalize images,Convert to 64bit
            
            if par == 'omniglot':
                #print( type(self.ds['image'] ) )
                #print( type( self.ds['image'][0] ) )
                self.ds['image'] = jnp.array( self.ds['image'], dtype=jnp.float32 ) / 255.0
                #print( 'jnp maximum', self.ds['image'].max() )#correctly it reports 0.99999994
                #print( 'jnp minimum', self.ds['image'].min() )#correctly it reports 0.0
            elif par == 'miniimagent':
                self.ds['image'] = tf.image.convert_image_dtype( self.ds['image'], dtype=tf.float64 )# / 255.0
            elif par == 'mnist':
                self.ds['image'] = jnp.array( self.ds['image'], dtype=jnp.float64 ) / 255.0
                #print( 'jnp maximum', self.ds['image'].max() )#correctly it reports 0.99999994
                #print( 'jnp minimum', self.ds['image'].min() )#correctly it reports 0.0
            elif par == 'fashion_mnist':
                self.ds['image'] = jnp.array( self.ds['image'], dtype=jnp.float64 ) / 255.0
                #print( 'jnp maximum', self.ds['image'].max() )#correctly it reports 0.99999994
                #print( 'jnp minimum', self.ds['image'].min() )#correctly it reports 0.0
            elif par == 'emnist':
                self.ds['image'] = jnp.array( self.ds['image'], dtype=jnp.float64 ) / 255.0
                #print( 'jnp maximum', self.ds['image'].max() )#correctly it reports 0.99999994
                #print( 'jnp minimum', self.ds['image'].min() )#correctly it reports 0.0
            elif par == 'cifar10':
                self.ds['image'] = jnp.array( self.ds['image'], dtype=jnp.float64 ) / 255.0
                #print( 'jnp maximum', self.ds['image'].max() )#correctly it reports 0.99999994
                #print( 'jnp minimum', self.ds['image'].min() )#correctly it reports 0.0
                
            for label in sorted( self.ds['label'].numpy().tolist() ):
                if label not in self.label_counter:
                    self.label_counter[label] = self.lcnt
                    self.lcnt += 1
            #Build dictionary
            for pos, label in enumerate( self.ds['label'].numpy().tolist() ):
                #print( pos, label )
                if label not in self.m:
                    self.m[label] = [pos]
                else:
                    self.m[label].append(pos)
            #Store keys -- do not recalculate at sampling
            self.keys = list( self.m.keys() )   
            
    def GetData( self ):
        
        if self.num_classes is None:
          vectors = jnp.eye( len( self.label_counter ) )
        else:
          vectors = jnp.eye( self.num_classes )
        
        lbl = jnp.array( [ vectors[ self.label_counter[i] ] for i in self.ds['label'].numpy() ] )

        #print( 'number of classes', self.num_classes )#OK
        #print( 'first entry label', self.ds['label'].numpy()[0] )#OK
        #print( 'first entry one hot vector', lbl[0] )#OK

        return self.ds['image'], lbl#self.ds['label']

import pickle

def LoadDict( filename ):
    return pickle.load( open( filename, "rb") ) 

def fromDictToTf( dct ):
    with tf.device('/CPU:0'):
        b = np.array( dct['images'] )
        tf_images = tf.Variable(b)
        l = {}
        cnt = 0
        for i in set( dct['labels'] ):
            l[ i ] = cnt
            cnt += 1
        tf_lbl = tf.Variable([ l[i] for i in dct['labels'] ], dtype=tf.int64 )
    
    ret = { 'image': tf_images, 'label' : tf_lbl }
    return ret

if use_miniimagenet:
    train_dict = LoadDict( 'train_dict.pkl' )
    test_dict = LoadDict( 'test_dict.pkl' )

    #ds_test = tfds.load('miniimagenett', split='test', batch_size=-1 )#, as_supervised=True )
    ds_train = fromDictToTf( train_dict )
    ds_test = fromDictToTf( test_dict )
    train_sampler = MyDataset( ds_train, par='miniimagent' )#d1 )#s_train )
    test_sampler = MyDataset( ds_test, par='miniimagent' )#2 )

class Client:

    def __init__(self, seed, K, M, meta_train_data=None, meta_test_data=None, dataset_name='mnist'):
        self.key = jax.random.PRNGKey(seed)
        self.K, self.M = K, M
        self.W = jax.random.uniform( self.key, (self.K,self.M+1) )#define W uniform matrix of size KxM
        self.train_sampler = None#training sampler
        self.test_sampler = None#validation sampler
        
        self.visit_rng = np.random.default_rng(seed)#only for fedrecon
        self.first_visit_call = True#only for fedrecon

        self.rng = random.PRNGKey( seed )
        self.Lista = []
        channels = 1
        img_w, img_h = 28, 28

        if dataset_name == 'miniimagenet':
            channels = 3
            img_w = img_h = 84
            pass#add network
        elif dataset_name == 'mnist':
            channels = 1
            img_w, img_h = 28, 28
            if algorithm in [ 'fedavg' ]:
                self.net_init, self.net_apply = BuildMnistNetworkVanilla()
            else:
                self.net_init, self.net_apply = BuildMnistNetwork()
        elif dataset_name == 'fashion_mnist':
            channels = 1
            img_w, img_h = 28, 28
            if algorithm in [ 'fedavg' ]:
                self.net_init, self.net_apply = BuildMnistNetworkVanilla()
            else:
                self.net_init, self.net_apply = BuildMnistNetwork()
        elif dataset_name == 'emnist':
            channels = 1
            img_w, img_h = 28, 28
            if algorithm in [ 'fedavg' ]:
                self.net_init, self.net_apply = BuildMnistNetworkVanilla()
            else:
                self.net_init, self.net_apply = BuildMnistNetwork()

        elif dataset_name == 'cifar10':
            channels = 3
            img_w, img_h = 32, 32
            if nn_architecture == 'vgg9':
                self.net_init, self.net_apply = build_vgg9( cfg['A'] )
            else:
                if algorithm in [ 'fedavg' ]:
                    self.net_init, self.net_apply = BuildCifar10NetworkVanilla()
                else:
                    self.net_init, self.net_apply = BuildCifar10Network()    
        elif dataset_name == 'cifar100':
            channels = 3
            img_w, img_h = 32, 32
            if algorithm in [ 'fedavg' ]:
                self.net_init, self.net_apply = BuildCifar10NetworkVanilla()
            else:
                self.net_init, self.net_apply = BuildCifar10Network()
        elif dataset_name == 'omniglot':
            channels = 1
            img_w, img_h = 28, 28
            if algorithm in [ 'fedavg' ]:
                self.net_init, self.net_apply = BuildOmniglotNetworkVanilla()
            else:
                self.net_init, self.net_apply = BuildOmniglotNetwork()
            
        self.in_shape = (-1, img_w, img_h, channels)
        self.out_shape, self.net_params = self.net_init( self.rng, self.in_shape)
        
        if meta_train_data != None:
            #print( dataset_name )#(980, 3137)
            self.BuildTrainSampler( meta_train_data, dataset_name, self.K )
        if meta_test_data != None:
            self.BuildTestSampler( meta_test_data, dataset_name, self.K )
        
    def BuildTrainSampler( self, inp_dict, dataset_name='mnist', num_classes=None ):
        self.train_sampler = MyDataset( inp_dict, dataset_name, num_classes )
        
    def BuildTestSampler( self, inp_dict, dataset_name='mnist', num_classes=None ):
        self.test_sampler = MyDataset( inp_dict, dataset_name, num_classes )    
    
    def GetGlobalParams(self):
        self.theta = get_params(opt_state)#get (global)theta state

    def CalcPhi(self, inputs):
        self.phi = self.net_apply( self.theta, inputs )#calculcate phi
        self.phi = jnp.hstack( [self.phi, jnp.ones( ( self.phi.shape[0],1) ) ]  )

    def CalcLoss_precalc_phi(self, theta, W, outputs, phi):
        predicted = jnp.dot( W, phi.T ).T
        l = cce_loss( predicted, outputs )
        return l

    def CalcLoss(self, theta, W, inputs, outputs):
        phi = self.net_apply( theta, inputs )#calculcate phi
        if algorithm in [ 'fedavg_vanilla', 'fedavg_vanilla_gradient', 'fedprox', 'feddane', 'ditto' ]:
            predicted = phi
        else:
            phi = jnp.hstack( [phi, jnp.ones( ( phi.shape[0],1) ) ]  )
            predicted = jnp.dot( W, phi.T ).T
        l = cce_loss( predicted, outputs )
        return l
    
    def CalcPredictLoss(self, theta, W, inputs, outputs):
        phi = self.net_apply( theta, inputs )#calculcate phi

        if algorithm in [ 'fedavg_vanilla', 'fedavg_vanilla_gradient', 'fedprox', 'feddane', 'ditto' ]:
            predicted = phi
        else:
            phi = jnp.hstack( [phi, jnp.ones( ( phi.shape[0],1) ) ]  )
            predicted = jnp.dot( W, phi.T ).T
        l = cce_loss( predicted, outputs )
        return l, predicted

    def CalcGradient_precalc_phi_WrtW(self, outputs):
        #calculcate gradient wrt W -- use to perform sgd steps to W as explained in (c)
        g = grad( self.CalcLoss_precalc_phi, argnums=1 )( self.theta, self.W, outputs, self.phi )
        return g

    def CalcGradient_WrtW(self, inputs, outputs):
        #calculcate gradient wrt W -- use to perform sgd steps to W as explained in (c)
        g = grad( self.CalcLoss, argnums=1 )( self.theta, self.W, inputs, outputs )
        return g
    def CalcGradient_WrtTheta( self, inputs, outputs ):
        #calculcate gradient wrt theta -- use to perform sgd steps to global theta as explained in (d)
        g = grad( self.CalcLoss, argnums=0 )( self.theta, self.W, inputs, outputs )
        return g

    #feddane functions
    def CalcLossGold(self, theta, theta_orig, inputs, outputs, mu):
        phi = self.net_apply( theta, inputs )#calculcate phi
        predicted = phi
        theta_diff = prog_grad( theta, theta_orig, alpha=1.0 )
        
        product_fn = lambda p1, p2: ( jnp.sum( jax.lax.mul( p1, p2 ) ) )
        product = tree_multimap( product_fn, self.gold, theta_diff )
        inner_product = sum( jnp.sum(p) for p in jax.tree_leaves(product) )
        
        norm = 0.5 * mu * sum( jnp.sum(p**2) for p in jax.tree_leaves(theta_diff) )
        l = cce_loss( predicted, outputs ) + norm + inner_product
        return l

    def CalcPredictLossGold(self, theta, theta_orig, inputs, outputs, mu):
        phi = self.net_apply( theta, inputs )#calculcate phi
        predicted = phi
        theta_diff = prog_grad( theta, theta_orig, alpha=1.0 )
        
        product_fn = lambda p1, p2: ( jnp.sum( jax.lax.mul( p1, p2 ) ) )
        product = tree_multimap( product_fn, self.gold, theta_diff )
        inner_product = sum( jnp.sum(p) for p in jax.tree_leaves(product) )
        
        norm = 0.5 * mu * sum( jnp.sum(p**2) for p in jax.tree_leaves(theta_diff) )
        l = cce_loss( predicted, outputs ) + norm + inner_product
        return l, predicted

    def CalcGradientGold_WrtTheta( self, inputs, outputs, mu ):
        #calculcate gradient wrt theta -- use to perform sgd steps to global theta as explained in (d)
        g = grad( self.CalcLossGold, argnums=0 )( self.theta, self.theta_orig, inputs, outputs, mu )
        return g

    #fedprox functions
    def CalcLossReg(self, theta, theta_orig, inputs, outputs, mu):
        #print( 'LossReg mu', mu )
        phi = self.net_apply( theta, inputs )#calculcate phi
        #print( 'phi', phi )
        if algorithm in [ 'fedavg_vanilla', 'fedavg_vanilla_gradient', 'fedprox', 'feddane', 'ditto' ]:
            predicted = phi
        else:
            phi = jnp.hstack( [phi, jnp.ones( ( phi.shape[0],1) ) ]  )
            predicted = jnp.dot( W, phi.T ).T#compute class-probs line 9
        
        theta_diff = prog_grad( theta, theta_orig, alpha=1.0 )
        
        norm = 0.5 * mu * sum( jnp.sum(p**2) for p in jax.tree_leaves(theta_diff) )
        l = cce_loss( predicted, outputs ) + norm 
        return l
    
    def CalcPredictLossReg(self, theta, theta_orig, inputs, outputs, mu):
        phi = self.net_apply( theta, inputs )#calculcate phi

        if algorithm in [ 'fedavg_vanilla', 'fedavg_vanilla_gradient', 'fedprox', 'feddane', 'ditto' ]:
            predicted = phi
        else:
            phi = jnp.hstack( [phi, jnp.ones( ( phi.shape[0],1) ) ]  )
            predicted = jnp.dot( W, phi.T ).T#compute class-probs line 9
        
        theta_diff = prog_grad( theta, theta_orig, alpha=1.0 )
        
        norm = 0.5 * mu * sum( jnp.sum(p**2) for p in jax.tree_leaves(theta_diff) )
        
        l = cce_loss( predicted, outputs ) + norm 
        
        return l, predicted

    def CalcGradient_WrtThetaReg( self, inputs, outputs, mu ):
        #calculcate gradient wrt theta -- use to perform sgd steps to global theta as explained in (d)
        g = grad( self.CalcLossReg, argnums=0 )( self.theta, self.theta_orig, inputs, outputs, mu )
        return g
    #ditto
    def CalcGradient_WrtThetaGlobReg( self, inputs, outputs, mu, theta_orig=False ):
        #calculcate gradient wrt theta -- use to perform sgd steps to global theta as explained in (d)
        if theta_orig:
            g = grad( self.CalcLossReg, argnums=0 )( self.personalized_theta, self.theta_orig, inputs, outputs, mu )
        else:
            g = grad( self.CalcLossReg, argnums=0 )( self.personalized_theta, self.theta, inputs, outputs, mu )
        return g
    #end of fedprox functions

    def CalcJointGradientThetaW( self, inputs, outputs ):
        #calculcate gradient wrt theta -- use to perform sgd steps to global theta as explained in (d)
        g1, g2 = grad( self.CalcLoss, argnums=[0,1] )( self.theta, self.W, inputs, outputs )
        return g1, g2

    def ResetW(self):
        if self.first_visit_call:
            self.first_visit_call = False
            self.W = jax.random.uniform( self.key, (self.K,self.M+1) )#define W uniform matrix of size KxM
        else:
            self.W = jax.numpy.array( self.visit_rng.uniform( size=(self.K,self.M+1) ) )

print( "K at clients is", K )
##cat_per_client
#i,K,M
print( len(cat_per_client[0] ) )
print( "it has len" )

maxim_cat = max( [ len(cat_per_client[i]) for i in range( len(cat_per_client) ) ] )

if dataset_name == 'omniglot' and algorithm in [ 'fedavg_vanilla', 'fedavg_vanilla_gradient', 'fedprox', 'feddane', 'ditto', 'fedbabu' ]:
    client_k = []
    for i in range( len(dpc_train) ):
        client_k.append( [0] * 55  )
    print( 'will use maximum number of classes', 'maxim_cat := ', maxim_cat, 'client_k[0] :=', client_k[0], 'len of client_k := ', len(client_k) )
else:
    #default
    client_k = cat_per_client

#clients = [ Client(i, len(cat_per_client[i]), M, meta_train_data=dpc_train[i], meta_test_data=dpc_test[i], dataset_name=dataset_name ) for i in range( len(dpc_train ) )  ]
clients = [ Client(i, len( client_k[i] ), M, meta_train_data=dpc_train[i], meta_test_data=dpc_test[i], dataset_name=dataset_name ) for i in range( len(dpc_train ) )  ]

for i in range( len(clients) ):
    print( 'client ', i, 'has', len(clients[i].train_sampler.GetData()[0]), len(clients[i].test_sampler.GetData()[0]) )
    if dataset_name != 'omniglot':
        print( len(cat_per_client[i]), cat_per_client[i] )

def prog_constant( params, constant=1e-9 ): 
    inner_sgd_fn = lambda state: ( state + constant )
    return tree_multimap(inner_sgd_fn, params)

def prog_grad( params, grads, alpha=1.0 ): 
    inner_sgd_fn = lambda g, state: (state - alpha*g)
    return tree_multimap(inner_sgd_fn, grads, params)

def prog_scalar( params, scalar ): 
    inner_scalar_fn = lambda state: ( scalar * state )
    return tree_multimap(inner_scalar_fn, params)

def prog_weight_decay( params, scalar ):
    mask_fn = lambda p: jax.tree_util.tree_map(lambda x: (x) if x.ndim == 1 else (scalar*x), p)
    return mask_fn(params)

cce = tf.keras.losses.CategoricalCrossentropy( reduction=tf.keras.losses.Reduction.SUM )

###------------------Main algorithm------------------###

cnt = 0

Train_Losses, Test_Losses = [], []
Train_Accs, Test_Accs = [], []

clients_rng = np.random.default_rng(1)
random_max_clients_rng = np.random.default_rng(1)

if algorithm in ['feddane']:
    clients2_rng = np.random.default_rng(2)

if algorithm in ['fedbabu']:
    #all clients share the same [orthogonal] head
    u, s, vh = jnp.linalg.svd( clients[0].W, full_matrices=False )
    #u.shape, s. shape, vh.shape
    #((55, 55), (55,), (55, 65))
    clients[0].W = jnp.dot( u, vh )
    for i in range(1,num_clients):
        clients[i].W = deepcopy( clients[0].W )

for r in range( rounds ):
    #print( "round", r )

    if random_users_flag:
        C1 = int(num_clients * users_fraction )
        C2 = int(num_clients * random_max_users_fraction )
        C = random_number( C1, C2+1, random_max_clients_rng )

    if algorithm in ['feddane']:
        #calculate avg_gradient
        #sample C clients
        sampled_clients = clients2_rng.choice( num_clients, 2*C, replace=False  )
        sum_gradients = None
        nums, grads = [], []
        for i in range(C):
            c = sampled_clients[i]
            clients[ c ].GetGlobalParams()
            support_inp, support_labels = clients[c].train_sampler.GetData()
            query_inp, query_labels =  clients[c].test_sampler.GetData()
            client_grad = clients[c].CalcGradient_WrtTheta( support_inp, support_labels )
            nums.append( len(support_inp) )
            grads.append( client_grad )
        sum_nums = sum( nums )
        for i in range(C):
            if sum_gradients == None:
                sum_gradients = deepcopy( prog_scalar( grads[i], nums[i] / sum_nums ) )
            else:
                sum_gradients = prog_grad( sum_gradients, prog_scalar( grads[i], nums[i] / sum_nums ), -1.0 )
        #convert to avg --basically now is already in average form        
        average_gradients = sum_gradients

    #sample C clients
    sampled_clients = clients_rng.choice( num_clients, C, replace=False  )

    for i in range(C):
        clients[ sampled_clients[i] ].GetGlobalParams()
        if algorithm in ['fedprox', 'feddane','ditto']:
            clients[ sampled_clients[i] ].theta_orig = deepcopy( clients[ sampled_clients[i] ].theta )
        if algorithm in ['ditto']:
            if clients[ sampled_clients[i] ].personalized_theta == None:
                clients[ sampled_clients[i] ].personalized_theta = deepcopy( clients[ sampled_clients[i] ].theta )
                
    print( "Sampled")
    for i in sampled_clients:
        print( i, end=' ' )
    print()

    #for c in clients:
    #    c.GetGlobalParams()
         
    received_grads = []#receieved gradients from clients wrt Theta
    received_params = []

    round_train_loss = 0
    round_test_loss = 0
    round_acc = 0
    
    changes = []

    n_i_train, n_i_test, l_i_train, l_i_test = [], [], [], []

    round_train_sum_acc = 0
    round_test_sum_acc = 0

    orig_theta = deepcopy( clients[ sampled_clients[0] ].theta )
        
    for c_idx in range( C ):#len( clients ) ):
        
        c = sampled_clients[ c_idx ]

        support_inp, support_labels = clients[c].train_sampler.GetData()
        query_inp, query_labels =  clients[c].test_sampler.GetData()
        
        if algorithm in ['fedrecon']:
            if reset in [ 'reset' ]:
                clients[c].ResetW()#reset client-specific parameters to uniform values
            
        if algorithm in ['fedrep']:
            clients[c].fedrep_opt_init_W, clients[c].fedrep_opt_update_W, clients[c].fedrep_get_params_W = optimizers.momentum(step_size=client_learning_rate, mass=momentum)#optimizers.sgd(step_size=client_learning_rate)
            clients[c].fedrep_opt_init_theta, clients[c].fedrep_opt_update_theta, clients[c].fedrep_get_params_theta = optimizers.momentum(step_size=client_learning_rate, mass=momentum)#optimizers.sgd(step_size=client_learning_rate)
            clients[c].fedrep_cnt_W = 0
            clients[c].fedrep_cnt_theta = 0

            clients[c].fedrep_opt_state_W = clients[c].fedrep_opt_init_W( clients[c].W )
            clients[c].fedrep_opt_state_theta = clients[c].fedrep_opt_init_theta( clients[c].theta )
        elif algorithm in ['fedbabu']:
            clients[c].fedbabu_opt_init_theta, clients[c].fedbabu_opt_update_theta, clients[c].fedbabu_get_params_theta = optimizers.momentum(step_size=client_learning_rate, mass=0.9)#optimizers.sgd(step_size=client_learning_rate)
            clients[c].fedbabu_opt_state_theta = clients[c].fedbabu_opt_init_theta( clients[c].theta )
            clients[c].fedbabu_cnt_theta = 0
        elif algorithm in ['feddane']:
            client_grad = deepcopy( clients[c].CalcGradient_WrtTheta( support_inp, support_labels ) )
            clients[c].gold = prog_grad( average_gradients, client_grad, 1.0 )
            #print( clients[c].gold )
            #input()
        orig_W = deepcopy( clients[c].W )
        #train_client c at tasks
        inner_lista = []
        for L in range(inner_steps):
            if algorithm in [ 'fedavg_personalized']:
                #gW = clients[c].CalcGradient_WrtW( support_inp, support_labels )#gradient wrt W
                ###---pflego'
                #didn't update W yet, because W, theta have to be updated simultraneously, CalcGradient_WrtTheta internally uses the W of client
                #client_grad = clients[c].CalcGradient_WrtTheta( support_inp, support_labels )#gradient wrt Theta, fixed W
                client_grad, gW = clients[c].CalcJointGradientThetaW( support_inp, support_labels )
                ###---
                #now update
                clients[c].W = prog_grad( clients[c].W, gW, alpha=client_learning_rate )
                clients[c].theta = prog_grad( clients[c].theta, client_grad, alpha=client_learning_rate)
            #apply sgd step to W
            elif algorithm in [ 'pflego' ]:

                #if we are at the first step compute phi
                if L == 0:
                    clients[c].CalcPhi( support_inp )
                
                #if we are at the last step, compute Joint Gradient step, otherwise compute gradient w.r.t. W
                if L == inner_steps-1:
                    #store gradient for server
                    #gradient for server -- sum gradients per task and send client_grad back
                    client_grad, gW = clients[c].CalcJointGradientThetaW( support_inp, support_labels )
                    received_grads.append( (client_grad, len(support_inp) ) )
                    clients[c].W = prog_grad( clients[c].W, gW, alpha=server_learning_rate * num_clients / C )#, alpha=0.01 )
                else:
                    #gW = clients[c].CalcGradient_WrtW( support_inp, support_labels )#gradient wrt W
                    #gW = clients[c].CalcGradient_precalc_phi_WrtW( support_labels )#no need for input
                    #analytical calculation
                    predicted = jnp.dot( clients[c].W, clients[c].phi.T ).T
                    probabilities = jax.nn.softmax( predicted, axis=-1 )
                    #print( probabilities.shape, support_labels.shape, predicted.shape )#, 'gW', gW.shape )
                    gW = -jnp.dot( ( support_labels - probabilities ).T, clients[c].phi ) / support_labels.shape[0]
                    #print( 'error', jnp.sum(gW - gDirect) )
                    #input()
                    clients[c].W = prog_grad( clients[c].W, gW, alpha=client_learning_rate )#, alpha=0.01 )
            elif algorithm in [ 'fedrecon' ]:
                #if we are at the first step compute phi
                if L == 0:
                    clients[c].CalcPhi( support_inp )
                #update client-specific
                #gW = clients[c].CalcGradient_WrtW( support_inp, support_labels )#gradient wrt W
                #analytical computation
                predicted = jnp.dot( clients[c].W, clients[c].phi.T ).T
                probabilities = jax.nn.softmax( predicted, axis=-1 )
                #print( probabilities.shape, support_labels.shape, predicted.shape )#, 'gW', gW.shape )
                gW = -jnp.dot( ( support_labels - probabilities ).T, clients[c].phi ) / support_labels.shape[0]
                clients[c].W = prog_grad( clients[c].W, gW, alpha=client_learning_rate )#, alpha=0.01 )

                if L == inner_steps-1:#if you're at last update step, update local theta by 1 gradient step
                    client_grad = clients[c].CalcGradient_WrtTheta( support_inp, support_labels )#gradient wrt Theta, fixed W
                    received_grads.append( (client_grad, len(support_inp) ) )
            elif algorithm in [ 'fedavg_vanilla', 'fedavg_vanilla_gradient' ]:
                #print( 'update theta inner')#Just updates theta, there is no W
                client_grad = clients[c].CalcGradient_WrtTheta( support_inp, support_labels )

                #clients[c].theta <- clients.theta - client_learning_rate * client_grad
                clients[c].theta = prog_grad( clients[c].theta, client_grad, alpha=client_learning_rate)

                inner_train_loss, inner_train_predicted = clients[c].CalcPredictLoss( clients[c].theta, clients[c].W, support_inp, support_labels )#inputs, outputs):
                inner_lista.append( float  (inner_train_loss) )
            elif algorithm in [ 'fedprox' ]:
                full_gradient = deepcopy( clients[c].CalcGradient_WrtThetaReg( support_inp, support_labels, mu ) )

                clients[c].theta = deepcopy( prog_grad( clients[c].theta, full_gradient, alpha=client_learning_rate) )

                inner_train_loss, inner_train_predicted = clients[c].CalcPredictLossReg( clients[c].theta, clients[c].theta_orig, support_inp, support_labels, mu )#inputs, outputs):
                inner_lista.append( float(inner_train_loss) )
                
                #theta_diff = prog_constant( theta_diff )
            elif algorithm in [ 'feddane' ]:
                full_grad = deepcopy( clients[c].CalcGradientGold_WrtTheta( support_inp, support_labels, mu ) )

                #clients[c].theta = deepcopy( prog_grad( clients[c].theta, g_final, alpha=client_learning_rate) )
                clients[c].theta = deepcopy( prog_grad( clients[c].theta, full_grad, alpha=client_learning_rate) )
                
                inner_train_loss, inner_train_predicted = clients[c].CalcPredictLossGold( clients[c].theta, clients[c].theta_orig, support_inp, support_labels, mu )#inputs, outputs):
                inner_lista.append( float(inner_train_loss) )
            elif algorithm in ['ditto']:
                #the two updates below are independent
                client_grad = clients[c].CalcGradient_WrtTheta( support_inp, support_labels )#gradient wrt Theta, fixed 
                clients[c].theta = prog_grad( clients[c].theta, client_grad, alpha=ditto_global_lr )
                client_grad = clients[c].CalcGradient_WrtThetaGlobReg( support_inp, support_labels, mu, theta_orig=True )#gradient wrt Theta_Orig, fixed W
                clients[c].personalized_theta = prog_grad( clients[c].personalized_theta, client_grad, alpha=client_learning_rate )
            
                if L == inner_steps-1:
                    global_weights = get_params(opt_state)
                    received_params.append( prog_grad( clients[c].theta, global_weights, 1.0 )  )
                inner_train_loss, inner_train_predicted = clients[c].CalcPredictLoss( clients[c].personalized_theta, None, support_inp, support_labels )#inputs, outputs):
                inner_lista.append( float(inner_train_loss) )
            elif algorithm in ['fedbabu']:
                client_grad = clients[c].CalcGradient_WrtTheta( support_inp, support_labels )#gradient wrt Theta, fixed 
                #clients[c].theta = prog_grad( clients[c].theta, client_grad, alpha=client_learning_rate )
                
                clients[c].fedbabu_opt_state_theta = clients[c].fedbabu_opt_update_theta( clients[c].fedbabu_cnt_theta, client_grad, clients[c].fedbabu_opt_state_theta )
                clients[c].theta = clients[c].fedbabu_get_params_theta( clients[c].fedbabu_opt_state_theta )
                clients[c].fedbabu_cnt_theta += 1
                
                inner_train_loss, inner_train_predicted = clients[c].CalcPredictLoss( clients[c].theta, clients[c].W, support_inp, support_labels )#inputs, outputs):
                inner_lista.append( float  (inner_train_loss) )
            elif algorithm in [ 'fedrep' ]:

                if L < inner_steps-local_rep_steps:
                    #update client-specific
                    client_grad = clients[c].CalcGradient_WrtW( support_inp, support_labels )#gradient wrt Theta, fixed 
                    clients[c].fedrep_opt_state_W = clients[c].fedrep_opt_update_W( clients[c].fedrep_cnt_W, client_grad, clients[c].fedrep_opt_state_W )
                    clients[c].W = clients[c].fedrep_get_params_W( clients[c].fedrep_opt_state_W )
                    clients[c].fedrep_cnt_W += 1
                else:
                    #last local_rep_steps update only theta
                    client_grad = clients[c].CalcGradient_WrtTheta( support_inp, support_labels )#gradient wrt Theta, fixed W
                    clients[c].fedrep_opt_state_theta = clients[c].fedrep_opt_update_theta( clients[c].fedrep_cnt_theta, client_grad, clients[c].fedrep_opt_state_theta )
                    clients[c].theta = clients[c].fedrep_get_params_theta( clients[c].fedrep_opt_state_theta )
                    clients[c].fedrep_cnt_theta += 1

                inner_train_loss, inner_train_predicted = clients[c].CalcPredictLoss( clients[c].theta, clients[c].W, support_inp, support_labels )#inputs, outputs):
                inner_lista.append( float(inner_train_loss) )
        
        #+ maximum norm tou W |absolute value
        #add L inner_train_loss
        clients[c].Lista.extend( inner_lista )
        if c == sampled_clients[0]:
            print( 'First Sampled Client ', str(c), 'last 7 losses')
            for zz in clients[c].Lista[-7:]:
                print( zz )
            #np.save( "./clients/client" + str(c), clients[c].Lista )
            #input()./
        
        #changes.append( jnp.linalg.norm( prog_grad( orig_W, clients[c].W ) ) )
        #print( "c", c )
        if algorithm in ['ditto']:
            cl_test_loss, cl_test_predicted = clients[c].CalcPredictLoss(
                clients[c].personalized_theta, None, query_inp, query_labels )#inputs, outputs):
            cl_train_loss, cl_train_predicted = clients[c].CalcPredictLoss( clients[c].personalized_theta, None,
            support_inp, support_labels )#inputs, outputs)
        elif algorithm in ['feddane']:
            cl_test_loss, cl_test_predicted = clients[c].CalcPredictLossGold(
                clients[c].theta, clients[c].theta_orig, query_inp, query_labels, mu )#inputs, outputs):
            cl_train_loss, cl_train_predicted = clients[c].CalcPredictLossGold( clients[c].theta, clients[c].theta_orig,
            support_inp, support_labels, mu )#inputs, outputs):
        else:
            cl_test_loss, cl_test_predicted = clients[c].CalcPredictLoss(
                clients[c].theta, clients[c].W, query_inp, query_labels )#inputs, outputs):
            cl_train_loss, cl_train_predicted = clients[c].CalcPredictLoss( clients[c].theta, clients[c].W,
            support_inp, support_labels )#inputs, outputs):
        
        n_i_train.append( len(support_inp) )#or use shape[0]
        n_i_test.append( len(query_inp) )#or use shape[0]

        round_train_sum_acc += jnp.sum( stax.softmax( cl_train_predicted, axis=-1 ).argmax( axis=-1 ) == support_labels.argmax( axis=-1 ) )
        round_test_sum_acc += jnp.sum( stax.softmax( cl_test_predicted, axis=-1 ).argmax( axis=-1 ) == query_labels.argmax( axis=-1 ) )
        round_acc += jnp.sum( stax.softmax( cl_test_predicted, axis=-1 ).argmax( axis=-1 ) == query_labels.argmax( axis=-1 ) )

        l_i_train.append( cl_train_loss )#round_train_loss += cl_train_loss
        l_i_test.append( cl_test_loss )##round_test_loss += cl_test_loss
            
    #calculate train loss of round
    sn_train = sum(n_i_train)
    for ni, li in zip( n_i_train, l_i_train ):
        a_i = (ni/sn_train)
        round_train_loss += a_i * li
    #calculate test loss of round
    sn_test = sum(n_i_test)
    for ni, li in zip( n_i_test, l_i_test ):
        a_i = (ni/sn_test)
        round_test_loss += a_i * li

    #round_acc /= ( cl_test_predicted.shape[0] * len( clients ) )
    round_train_acc = round_train_sum_acc / ( sum( n_i_train ) )
    round_test_acc = round_test_sum_acc / ( sum( n_i_test ) )
    #print( round_train_loss)
    Train_Losses.append( round_train_loss )
    Test_Losses.append( round_test_loss )
    Train_Accs.append( round_train_acc )
    Test_Accs.append( round_test_acc )
    #"round_acc := ",round_acc,
    print( "round ", r, "train loss :=", round_train_loss, "test loss :=", round_test_loss, 'round_train_acc', round_train_acc, 'round_test_acc', round_test_acc )
    
    #update theta
    if algorithm in [ 'fedavg_vanilla', 'fedavg_personalized', 'fedprox', 'feddane', 'fedrep', 'fedbabu' ]:#, 'fedavg_vanilla_gradient' ]:
        sum_weights = None

        ####simple averaging
        """
        for i in range( len(clients) ):
            if i == 0:
                sum_weights = deepcopy( clients[i].theta )
            else:
                sum_weights = prog_grad( sum_weights, clients[i].theta, alpha=-1.0 )
        #convert to avg
        avg_weights = prog_scalar( sum_weights, 1.0/len(clients) )
        """
        ###end of simple averaging

        #print( 'sum', sum(n_i_train), sn_train )
        ###weighted average
        for i in range( C ):#len(clients) ):
            n_i = n_i_train[i]# / sn_train
            #print( 'n_i', n_i )
            if i == 0:
                sum_weights = deepcopy( prog_scalar( clients[ sampled_clients[ i ] ].theta, n_i / sn_train ) )
            else:
                sum_weights = prog_grad( sum_weights, prog_scalar(clients[ sampled_clients[ i ] ].theta, n_i / sn_train ), alpha=-1.0 )
        #convert to avg --basically now is already in average form
        avg_weights = deepcopy( sum_weights )#prog_scalar( sum_weights, 1.0/sn_train )

        #reinit params of global state
        opt_state = opt_init( avg_weights )
    elif algorithm in ['ditto']:
        sum_weights = None
        for i in range( C ):
            n_i = n_i_train[i]
            if i == 0:
                sum_weights = deepcopy( prog_scalar( received_params[i], n_i / sn_train ) )
            else:
                sum_weights = prog_grad( sum_weights, prog_scalar( received_params[i], n_i / sn_train ), alpha=-1.0 )
        
        global_weights = get_params(opt_state)
        global_weights = prog_grad( global_weights, sum_weights, -1.0 )

        #reinit params of global state
        opt_state = opt_init( global_weights )
    
    else:
        #Our algorithm
        #sum gradient then apply
        sum_grads = None 
        for cur_gradient, ni in received_grads:
            if sum_grads == None:
                sum_grads = prog_scalar( cur_gradient, ni / sn_train )#a_i * gradient
            else:
                sum_grads = prog_grad( sum_grads, prog_scalar( cur_gradient, ni / sn_train ), alpha=-1.0 )# += a_i * gradient

        if algorithm in ['pflego']:
            sum_grads = prog_scalar( sum_grads, num_clients / C )

        opt_state = opt_update( cnt, sum_grads, opt_state )
        cnt += 1

#Save Acc, Loss
np.save( folder_struct + "%s_InnerSteps%d_%dclients_Train_Accs_%s_arch_%s_%s_fraction_%s_serverlr_%s_M_%s_%s_mu%s_random_max_users_fraction%s_local_rep_steps%d_ditto_global_lr%s_momentum%s" % (dataset_name, inner_steps, len(clients), algorithm, nn_architecture, aggregate, users_fraction, server_learning_rate, M, reset, original_mu, random_max_users_fraction, local_rep_steps, ditto_global_lr, momentum ), Train_Accs )
np.save( folder_struct + "%s_InnerSteps%d_%dclients_Test_Accs_%s_arch_%s_%s_fraction_%s_serverlr_%s_M_%s_%s_mu%s_random_max_users_fraction%s_local_rep_steps%d_ditto_global_lr%s_momentum%s" % (dataset_name, inner_steps, len(clients), algorithm, nn_architecture, aggregate, users_fraction, server_learning_rate, M, reset, original_mu, random_max_users_fraction, local_rep_steps, ditto_global_lr, momentum ), Test_Accs )
np.save( folder_struct + "%s_InnerSteps%d_%dclients_TrainLosses_%s_arch_%s_%s_fraction_%s_serverlr_%s_M_%s_%s_mu%s_random_max_users_fraction%s_local_rep_steps%d_ditto_global_lr%s_momentum%s" % (dataset_name, inner_steps, len(clients), algorithm, nn_architecture, aggregate, users_fraction, server_learning_rate, M, reset, original_mu, random_max_users_fraction, local_rep_steps, ditto_global_lr, momentum ), Train_Losses )
np.save( folder_struct + "%s_InnerSteps%d_%dclients_TestLosses_%s_arch_%s_%s_fraction_%s_serverlr_%s_M_%s_%s_mu%s_random_max_users_fraction%s_local_rep_steps%d_ditto_global_lr%s_momentum%s" % (dataset_name, inner_steps, len(clients), algorithm, nn_architecture, aggregate, users_fraction, server_learning_rate, M, reset, original_mu, random_max_users_fraction, local_rep_steps, ditto_global_lr, momentum ), Test_Losses )
import numpy as onp
#theta = get_params(opt_state)
#onp.save( folder_struct + "%s_InnerSteps%d_%dclients_trained_params_%s_arch_%s_%s_fraction_%s_serverlr_%s_M_%s_%s_mu%s_random_max_users_fraction%s_local_rep_steps%d_ditto_global_lr%s_momentum%s" % (dataset_name, inner_steps, len(clients), algorithm, nn_architecture, aggregate, users_fraction, server_learning_rate, M, reset, original_mu, random_max_users_fraction, local_rep_steps, ditto_global_lr, momentum ), theta )
