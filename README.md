# DeepQLearning-Gomoku
A reinforcement AI playing Gomoku made by Jacques Payen, Charles Payen and Lucas Cabanac.

This project is composed of an AI model trainable through reinforcement, and a game environment to test and confront this AI. 
The goal of this project is to try to recreate the famous Alpha Zero AI of Google, but working for the Gomoku game instead of Go. We started this project to understand and experiment what are the differents elements needed to form a complex reinforcement AI.
It takes place in the form of a scolar project.<br> 

# Documentation

### Table of contents:
1. [Deep QLearning Model](#deep-learning)
2. [Monte Carlo Tree Search](#mcts)
3. [Trainning](#train)
4. [Test AI](#test)

<a id='deep-learning'></a>
Deep QLearning Model
------------

The goal of this model is to predict  from a board the probabilties of wining between all possibles moves.

### Model structure

To get the input we generate 3 boards representations from the board, each is boards flattened where each cell has a value of 1 or 0:<br>
The first get his cells values at 1 for player1 pawns from the oiginal board.<br>
The second get his cells values at 1 for player2 pawns from the oiginal board.<br>
The third get his cells values at 1 for empty cells from the oiginal board.<br>
For a Gomoku our input :<br>
Input Size = Board Size² * 3 = 20² * 3 = 1200<br>

The input is sent to a dense layer with a size of 10800 with rectified linear unit activaton.<br>
Then it goes through a second dense layer of size 400, to get the Q-Values.<br>
Finally a softmax is applied to get the probabilities between Q-Values.<br>

### Train function

We use a gradient descent to minimize our loss.<br>
For calculating the loss we tried two differents methods:<br>
1. Calculating a mean squared error between the Q-Values and the targets<br>
2. Calculating an absolute_difference between the probabilities of the Q-Value that was choose and the target<br>
Both gived us result but we keep the first method because we got better convergence over time.

<a id='mcts'></a>
Monte Carlo Tree Search
------------

We use the Monte Calo Tree Search (MCTS) to decide which move our AI will take during a game.

### Exploration policy

To navigate through games stats the MCTS follow a special policy.
If it get to a new game stat, we simulate it and collect the reward (1 = win / 0 = loose).
To explore we use the <a id='deep-learning'>deep learning model</a> to predict the probabilities of getting reward for the next stats.
At each step, to choose the next stats to simulate we select the node with the highter value. 
To calculate this value for each next stats we follow the probabilities of reward from the model, the number of time we went in, and the mean of reward we get from him. So while there isn't a child with reward, the policy will follow the prediction of our model to explore games stats. But when we start collecting rewards the policy will promote the paths with a good reward and avoid the paths with a bad reward.<br>

### Move selection

At the end of the MCTS exploration, it selects a move based on the data we get from the first games stats.
When we use the MCTS to play as best as possible we select the game stat we simulated the most.
When we use the MCTS for training the selection is made through a probability distribution with the number of time we simulate a game stat. It allows the AI to explore a larger variety of actions.

<a id='train'></a>
Training
------------

To train the model we store in a stack, transitions from game the AI is playing. Every 4 000 steps we feed the model with the transitions from the stack. We empty the stack and start collecting again.

### Transition sructure

A Transition is composed of 3 elements:<br>
1. The board before choosing a move<br>
2. The move choice<br>
3. A target value representing the reward get for this move<br>

The target is calculated from the reward we get at the end of the game passed through the transitions of this game.
Target t0 = 0.95 * Target t1 

### Methodes of training

First we trained the model with the <a id='mcts'>Monte Carlo Tree Search</a> against a random AI.
But the mcts is based on the model and wasn't really effective with an untrained model. Moreover it makes it slower to choose a move, so it takes too much time to make enought games to converge.<br>
Then we remove MCTS for training, and replace it for testing. The AI was only following the model prediction to select a move. 
We start to get result because with a better model the MCTS can predict critical move sooner (like when a player get 3 pawn in a row without any enemy pawn beside).<br>
After that we made the AI play against it-self to create more competitive games. But both used the same network so the training of the network was influenced by the network it-self.<br>
Finaly we separated the network on two, one for each player. The enemy player network doesn't train and doesn't get updateed with values from the AI network every 40 000 steps.

### How to train

You can train your own model by executing the file : train_neural.py

Or you can download our model already trained here: https://mega.nz/#F!y9wR3AJB!aUs_qsjzHPbldHk32oCc1g<br>
And put the 'save' directory at the root of the repository.

<a id='test'></a>
Test AI
------------

### Prerequisites

Before using the scripts make sure you have installed this following package:<br>
1.Python3<br>
2.Numpy <br>
3.Tensorflow<br> 
4.Tkinter 

### Usage

You only need to execute the scripts:<br>
1. 'train_nural.py' train the AI model against a copy of her-self.<br>
2. 'neural_vs_mtc.py' display the win rate for the last 10 game. the AI is against the MCTS without neural network to measure the effectiveness of the AI network.<br>
3. 'player_vs_ia.py' launches the environement to play against the AI.

