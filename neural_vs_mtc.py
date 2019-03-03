#!/usr/bin/env python3
from random import randint
import random
import numpy as np
import sys
import tensorflow as tf
import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

nb_cell = 400
nb_row = 20
nbGame = 100
size_buffer = 4000
update_qai_cycle = 40000

def nuProb():
    i = 0
    list = []
    while i != 400:
        list.append(0.5)
        i = i + 1
    return list

def convertBoard(board):
    size = nb_cell
    bBoard = []
    i = 0
    while i != size:
        if board[i] == 1:
            bBoard.append(1)
        else:
            bBoard.append(0)
        i = i + 1
    i = 0
    while i != size:
        if board[i] == 2:
            bBoard.append(1)
        else:
            bBoard.append(0)
        i = i + 1
    i = 0
    while i != size:
        if board[i] == 0:
            bBoard.append(1)
        else:
            bBoard.append(0)
        i = i + 1
    res = np.array(bBoard)
    return res

def getInvMove(board):
    size = nb_cell
    i = 0
    move = []
    while i != size:
        if board[i] == 0:
            move.append(0)
        else:
            move.append(1)
        i = i + 1
    return move

def isFull(board):
    for cell in board:
            if cell == 0:
                    return 0
    return 1

def getEBoard(old_board):
    board = np.copy(old_board)
    i = 0
    while i != len(board):
        if board[i] == 1:
            board[i] = 2
        elif board[i] == 2:
            board[i] = 1
        i = i + 1
    return board

def isWin(pos, board):
    slen = nb_row
    size = slen * slen
    wl = 5
    i = 0
    game = []
    while i != size:
        x = 0
        list = []
        while x != slen:
            list.append(board[i])
            x = x + 1
            i = i + 1
        game.append(np.copy(list))
    player = game[int(pos.y)][int(pos.x)]
    x = 0
    y = 0
    move = int(pos.x)
    i = int(pos.y)
    x = move
    y = i
    len = -1
    while x < slen and game[y][x] == player:
        x = x + 1
        len = len + 1
    x = move
    while x > -1 and game[y][x] == player:
        x = x - 1
        len = len + 1
    if len >= wl:
        return 1

    x = move
    y = i
    len = -1
    while y < slen and game[y][x] == player:
        y = y + 1
        len = len + 1
    y = i
    while y > -1 and game[y][x] == player:
        y = y - 1
        len = len + 1
    if len >= wl:
        return 1

    x = move
    y = i
    len = -1
    while y > - 1 and x > -1 and game[y][x] == player:
        y = y - 1
        len = len + 1
        x = x - 1
    y = i
    x = move
    while y < slen and x < slen and game[y][x] == player:
        y = y + 1
        len = len + 1
        x = x + 1
    if len >= wl:
        return 1

    x = move
    y = i
    len = -1
    while y > - 1 and x < slen and game[y][x] == player:
        y = y - 1
        len = len + 1
        x = x + 1
    y = i
    x = move
    while y < slen and x > -1 and game[y][x] == player:
        y = y + 1
        len = len + 1
        x = x - 1
    if len >= wl:
        return 1
    return 0

def is_intresting(x, y, board):
        if board[y * nb_row + x] != 0:
            return 0
        x_min = x - 1
        if x_min < 0:
            x_min = 0
        x_max = x + 1
        if x_max > 19:
            x_max = 19
        y_min = y - 1
        if y_min < 0:
            y_min = 0
        y_max = y + 1
        if y_max > 19:
            y_max = 19
        for i in range(y_min, y_max + 1):
            for j in range(x_min, x_max + 1):
                if board[i * nb_row + j] != 0:
             #       print("ok", i, j)
                    return 1
        return 0

class POS:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class NODE:
    def __init__(self, board, pos, depth, proba, qai, prob):
        self.qai = qai
        self.nb_try = 0
        self.sum_value = 0
        self.proba = proba
        self.ucb = 0.0001
        self.depth = depth
        self.nb_use = 0
        self.pos = pos
        self.child = []
        self.board = board.copy()
        self.printval = 0
        self.prob = prob
        if self.depth > -1:
            put_board(pos, depth % 2 + 1, self.board)
            self.simulate()

    def create_child(self, y, x, proba):
        pos = POS(x, y)
        new = NODE(self.board, pos, self.depth + 1, proba, self.qai, self.prob)
        self.child.append(new)

    def manager(self):
        v = 1
        r = 0
        n = 0
        u = 0
        i = 0
        #proba = []
        #total = 0
        for node in self.child:
            node.ucb = ((node.sum_value / node.nb_try) / 50 + node.proba) * (math.log(self.nb_use) / (1 + 2 * math.pow(node.nb_use, 2)))
            if i == 0 or (u < 0 and node.ucb < u) or (node.ucb > 0 and node.ucb > u) or (node.ucb == u and node.nb_try > n) :
                u = node.ucb
                n = node.nb_try
                r = i 
            i += 1
            #proba.append(node.ucb)
            #total += node.ucb
        #for p in range(len(self.child)):
        #    if proba[p] = proba[p] / total
        #r = np.random.choice(len(self.child), p=proba)
        return r

    def simulate(self):
        self.nb_try += 1
        w = isWin(self.pos, self.board)
        self.sum_value += w

    def expension(self):
        proba = 0.0
        if self.depth == -1 or (isWin(self.pos, self.board) == 0 and isFull(self.board) == 0):
            board = convertBoard(self.board)
            if self.depth % 2 == 1:
                board = getEBoard(board)
            if self.prob != 0:
                proba = self.qai.getProb(board, getInvMove(self.board))
            else:
                proba = nuProb()
            #proba = nuProb()
            #print(">>>", self.board)
            for i in range(nb_cell):
                if is_intresting(int(i % nb_row), int(i / nb_row), self.board) == 1:
                    self.create_child(int(i / nb_row), int(i % nb_row), proba[i])
        else:
            self.simulate()

    def update(self):
        t = 0
        v = 0
        for node in self.child:
            t += node.nb_try
            v += node.sum_value
        self.nb_try = t + 1
        self.sum_value = v * -1

    def select(self):
        self.nb_use += 1
        if len(self.child) == 0:
            #print("exprint(root.nb_try)p")
            self.expension()
        else:
            #print("manag")
            i = self.manager()
            self.child[i].select()
        if len(self.child) != 0:
            #print("update")
            self.update()

    def printv(self):
        for i in range(0, self.depth+2):
            sys.stdout.write('-')
        sys.stdout.flush()
        print(self.depth, self.nb_use, len(self.child), self.sum_value, self.nb_try, self.pos.y, self.pos.x, self.proba, self.ucb, self.printval)
        if self.depth < 0:
            for child in self.child:
                   child.printv()

def get_best_move(root):
    pos = []
    target = []
    sum = []
    mean = 0
    for child in root.child:
        if child.sum_value != 0:
            target.append((child.nb_use / (root.nb_use - 1) * child.proba * child.sum_value))
        else:
            target.append((child.nb_use / (root.nb_use - 1) * child.proba))
        #target.append(child.nb_use / (root.nb_use - 1))
        pos.append(child.pos)
    #r = np.random.choice(len(root.child), p=target)
    r = np.argmax(target)
    return pos[r]


def put_board(pos, player, board):
    v = pos.y * nb_row + pos.x
    board[int(v)] = player


def my_mcts(board, qai, start, i, mt, prob):
    if start == 0:
        start = 1
        return nb_cell / 2 + nb_row / 2
    pos = POS(-1, -1)
    root = NODE(board, pos, -1, 1, qai, prob)
    t = 0
    while root.nb_use < mt :
        #print(root.nb_try)
        root.select()
    pos = get_best_move(root)
    #if i >= nbGame :
        #print("OOOOOOOOOOOOOOOOOOOOO")
        #root.printv()
    return pos.y * nb_row + pos.x

def copy_model_parameters(sess, dqn1, dqn2):
    dqn1_params = [t for t in tf.trainable_variables() if t.name.startswith(dqn1.scope)]
    dqn1_params = sorted(dqn1_params, key=lambda v: v.name)
    dqn2_params = [t for t in tf.trainable_variables() if t.name.startswith(dqn2.scope)]
    dqn2_params = sorted(dqn2_params, key=lambda v: v.name)
    update_ops = []
    for dqn1_value, dqn2_value in zip(dqn1_params, dqn2_params):
        op = dqn2_value.assign(dqn1_value)
        update_ops.append(op)
    sess.run(update_ops)

class GOMOKU:
    def __init__(self, size, len, wl):
        self.size = size
        self.len = len
        self.wl = wl
        self.board = []
        self.game = []
        i = 0
        while i != self.size:
            self.board.append(0)
            i = i + 1
        i = 0
        list = []
        while i != self.len:
            list.append(0)
            i = i + 1
        i = 0
        while i != self.len:
            list = np.copy(list)
            self.game.append(list)
            i = i + 1

    def getEBoard(self):
        board = np.copy(self.board)
        i = 0
        while i != len(board):
            if board[i] == 1:
                board[i] = 2
            elif board[i] == 2:
                board[i] = 1
            i = i + 1
        return board

    def getBoard(self):
        return np.copy(self.board)

    def printBoard(self):
        i = 0
        while i != self.size :
            sys.stdout.write(str(self.board[i]))
            if (i + 1) % self.len == 0:
                sys.stdout.write('\n')
            i = i + 1
        sys.stdout.flush()

    def getMove(self):
        i = 0
        move = []
        while i != self.size:
            if self.board[i] == 0:
                move.append(i)
            i = i + 1
        return move

    def  getInvMove(self):
        i = 0
        move = []
        while i != self.size:
            if self.board[i] == 0:
                move.append(0)
            else:
                move.append(1)
            i = i + 1
        return move

    def getRandomMove(self):
        move = self.getMove()
        slen = len(move) - 1
        nb = randint(0, slen)
        return move[nb]

    def resetBoard(self):
        i = 0
        while i != self.size:
            self.board[i] = 0
            i = i + 1
        z = 0
        while z != self.len:
            a = 0
            while a != self.len:
                self.game[z][a] = 0
                a = a + 1
            z = z + 1

    def isWin(self, move, player):
        move = int(move)
        x = 0
        y = 0
        i = 0
        while move >= self.len:
            i = i + 1
            move = move - self.len
        x = move
        y = i
        len = -1
        while x < self.len and self.game[y][x] == player:
            x = x + 1
            len = len + 1
        x = move
        while x > -1 and self.game[y][x] == player:
            x = x - 1
            len = len + 1
        if len >= self.wl:
            return 1

        x = move
        y = i
        len = -1
        while y < self.len and self.game[y][x] == player:
            y = y + 1
            len = len + 1
        y = i
        while y > -1 and self.game[y][x] == player:
            y = y - 1
            len = len + 1
        if len >= self.wl:
            return 1

        x = move
        y = i
        len = -1
        while y > - 1 and x > -1 and self.game[y][x] == player :
            y = y - 1
            len = len + 1
            x = x - 1
        y = i
        x = move
        while y < self.len and x < self.len and self.game[y][x] == player :
            y = y + 1
            len = len + 1
            x = x + 1
        if len >= self.wl:
            return 1

        x = move
        y = i
        len = -1
        while y > - 1 and x < self.len and self.game[y][x] == player:
            y = y - 1
            len = len + 1
            x = x + 1
        y = i
        x = move
        while y < self.len and x > -1 and self.game[y][x] == player:
            y = y + 1
            len = len + 1
            x = x - 1
        if len >= self.wl:
            return 1
        return 0

    def placeMove(self, move, player):
        move = int(move)
        self.board[move] = player
        i = 0
        while move >= self.len:
            i = i + 1
            move = move - self.len
        self.game[i][move] = player

    def convertBoard(self):
        bBoard = []
        i = 0
        while i != self.size:
            if self.board[i] == 1:
                bBoard.append(1)
            else:
                bBoard.append(0)
            i = i + 1
        i = 0
        while i != self.size:
            if self.board[i] == 2:
                bBoard.append(1)
            else:
                bBoard.append(0)
            i = i + 1
        i = 0
        while i != self.size:
            if self.board[i] == 0:
                bBoard.append(1)
            else:
                bBoard.append(0)
            i = i + 1
        res = np.array(bBoard)
        return res

    def convertBoardEnnemy(self):
        bBoard = []
        i = 0
        while i != self.size:
            if self.board[i] == 2:
                bBoard.append(1)
            else:
                bBoard.append(0)
            i = i + 1
        i = 0
        while i != self.size:
            if self.board[i] == 1:
                bBoard.append(1)
            else:
                bBoard.append(0)
            i = i + 1
        i = 0
        while i != self.size:
            if self.board[i] == 0:
                bBoard.append(1)
            else:
                bBoard.append(0)
            i = i + 1
        res = np.array(bBoard)
        return res

    def poids(self, list):
        i = 0
        while i != 400:
            y = int(i / 20)
            x = i % 20
            if is_intresting(x, y, self.board) == 1:
                print(y, x, list[i])
            i = i + 1


class QAI:
    def __init__(self, scope):
        self.scope = scope
        self.lr = 0.01
        self.qvList = []
        self.probList = []
        self.moveList = []
        self.boardList = []
        ##tensorflow construction réseau de neuronnes
        with tf.variable_scope(scope):
            self.input = tf.placeholder(tf.float32, shape=(None, 1200), name='input')
            self.target = tf.placeholder(tf.float32, shape=(None, 400), name='target')
            self.layer = self.input
            self.layer = tf.layers.dense(self.input, 10800, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name=None)
            self.qv = tf.layers.dense(self.layer, 400, activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name='qv')
            self.prob = tf.nn.softmax(self.qv, name='prob')
            self.loss = tf.losses.mean_squared_error(predictions=self.qv, labels=self.target)
            self.train = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(self.loss, name='train')

    def setSession(self, session: tf.Session):
        self.sess = session

    def addMove(self, board, move):
        move = int(move)
        probas, qvs = self.sess.run([self.prob, self.qv], feed_dict={self.input: [board]})
        lqv = qvs[0]
        self.boardList.append(board)
        self.qvList.append(lqv)
        self.moveList.append(move)
        self.probList.append(lqv[move])

    def getProb(self, board, notPossible):
        probas, qvs = self.sess.run([self.prob, self.qv], feed_dict={self.input: [board]})
        lprob = probas[0]
        lqv = qvs[0]
        i = 0
        size = len(lprob)
        while (i != size):
            if notPossible[i] == 1:
                lprob[i] = - 2
            i = i + 1
        return lprob

    def optiFive(self, board, notPossible, game):
        probas, qvs = self.sess.run([self.prob, self.qv], feed_dict={self.input: [board]})
        lprob = probas[0]
        lqv = qvs[0]
        i = 0
        size = len(lprob)
        Rmove = []
        while (i != size):
            y = int(i / 20)
            x = i %20
            if notPossible[i] == 1:
                lprob[i] = - 20000
            elif is_intresting(x, y, game) == 0:
                lprob[i] = -10000
            else:
                Rmove.append(i)
            i = i + 1
        i = randint(0,9)
        if i < 2:
            return Rmove[randint(0, len(Rmove) -1)]
        return np.argmax(lprob)

    def optiMove(self, board, notPossible, game):
        probas, qvs = self.sess.run([self.prob, self.qv], feed_dict={self.input: [board]})
        lprob = probas[0]
        lqv = qvs[0]
        i = 0
        size = len(lprob)
        while (i != size):
            y = int(i / 20)
            x = i %20
            if is_intresting(x, y, game) == 0:
                lprob[i] = -1000
            if notPossible[i] == 1:
                lprob[i] = - 2000
            i = i + 1
        return np.argmax(lprob)

    def getMove(self, board, notPossible, r):
        self.boardList.append(board)
        probas, qvs = self.sess.run([self.prob, self.qv], feed_dict={self.input: [board]})
        lprob = probas[0]
        lqv = qvs[0]
        i = 0
        size = len(lprob)
        while (i != size):
            if notPossible[i] == 1:
                lprob[i] = - 1
            i = i + 1
        i = randint(1, 100)
        r = r * 100
        if i > r:
            move = np.argmax(lprob)
        else:
            move = randint(0, size - 1)
            while lprob[move] == - 1:
                move = randint(0, size - 1)
        self.qvList.append(lqv)
        self.moveList.append(move)
        self.probList.append(lqv[move])
        return move

    def getGameList(self, win, list1, list2, list3):
        if win == 1:
            win = 1.0
        elif win == 2:
            win = -1.0
        else:
            win = 0.0
        targetList = []
        i = 0
        self.probList.append(win)
        while i != len(self.moveList):
            target = self.qvList[i]
            target[self.moveList[i]] = 0.95 * self.probList[i+1]
            targetList.append(target)
            i = i + 1
        return list1 + self.boardList, list2 + targetList, list3 + self.moveList

    def trainNeural(self, boardList, targetList, moveList):
        self.sess.run([self.train], feed_dict={self.input: boardList, self.target: targetList})
        boardList = []
        targetList = []
        moveList = []
        return boardList, targetList, moveList

    def learn(self, win):
        if win == 1:
            win = 1.0
        elif win == 2:
            win = -1.0
        else:
            win = 0.0
        targetList = []
        i = 0
        self.probList.append(win)
        while i != len(self.moveList):
            target = self.qvList[i]
            target[self.moveList[i]] = 0.95 * self.probList[i+1]
            targetList.append(target)
            i = i + 1
        self.sess.run([self.train], feed_dict={self.input: self.boardList, self.target: targetList})

    def reset(self):
        self.qvList = []
        self.probList = []
        self.moveList = []
        self.boardList = []

    def ennemyMove(self, board, notPossible):
        probas, qvs = self.sess.run([self.prob, self.qv], feed_dict={self.input: [board]})
        lprob = probas[0]
        lqv = qvs[0]
        i = 0
        size = len(lprob)
        while (i != size):
            if notPossible[i] == 1:
                lprob[i] = - 10000
            i = i + 1
        move = np.argmax(lprob)
        return move

if __name__ == '__main__':
    qai = QAI(scope='qai')
    qai_copy = QAI(scope='qai_copy')
    game = GOMOKU(nb_cell, nb_row, 5)
    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    session.run(tf.global_variables_initializer())
    qai.setSession(session)
    qai_copy.setSession(session)

    path = os.getcwd()
    path = path + "/save/model.ckpt"
    saver = tf.train.Saver()
    saver.restore(session, path)

    i = 0
    move = 0
    boardList = []
    targetList = []
    moveList = []
    step = 0
    winrate = 0
    while i != nbGame:
        player = randint(1,2)
        win = 0
        start = 0
        while win == 0 and len(game.getMove()) != 0:
            if step % update_qai_cycle == 0:
                copy_model_parameters(session, qai, qai_copy)
                step = 0
            if player == 1:
                if start == 0:
                    move = randint(0,399)
                else:
                    move = my_mcts(game.getBoard(), qai, start, nbGame + 1, 50, 1)
                qai.addMove(game.convertBoard(), move)
                game.placeMove(move, player)
                win = game.isWin(move, player)
                if win == 1:
                    win = 1
                else:
                    player = 2
            else:
                if start == 0:
                    move = randint(0,399)
                else:
                    move = my_mcts(game.getEBoard(), qai, start, nbGame + 1, 50, 0)
                game.placeMove(move, player)
                win = game.isWin(move, player)
                if win == 1:
                    win = 2
                else:
                    player = 1
            start += 1
            step = step + 1
        if win == 1:
            winrate = winrate + 1
        qai.reset()
        game.resetBoard()
        i = i + 1
        print((winrate / i)*100)
