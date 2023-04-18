# CatanAI: Mastering Catan with Multi-Agent Reinforcement Learning

## Table of Contents

- [Introduction](#introduction)
- [Training Modes](#training-modes)
  - [Headless](#headless)
  - [GUI](#gui)
- [Agents](#agents)
  - [Randy (Random Agent)](#randy-random-agent)
  - [Adam (Basic DQN Agent)](#adam-basic-dqn-agent)
  - [Redmond (DQN Agent with a Love for Red Tiles)](#redmond-dqn-agent-with-a-love-for-red-tiles)

## Introduction

Catan, known better perhaps by its previous and longer name, Settlers of Catan, has been gracing the shelves of board game fans with its presence since 1995. It's widely considered to be one of the most popular "Eurogames" of all time. For good reason: the game has a random setup each time its played, so as well as being generally a very entertaining experience, it holds a lot of replayability that many other popular board games only wish they could possess. It is this pervasive randomness that makes Catan the ideal subject for framing one of machine learning's more visually attractive sub-fields: multi-agent reinforcement learning.

My work on an AI-driven Catan agent for this project, which I am creatively naming CatanAI\*, adds to a catalogue of similar work by machine learning researchers who have been studying Catan in this way since as early as 2004.

The goals of my project are as follows:

- Create a bespoke Catan game environment from scratch, using Python
- Add the ability for a virtual agent to play Catan
- Enable the agent to learn from its mistakes and master Catan, in an environment where similar agents are also competing for the same thing
- Allow Catan fans to play against my virtual agents

The main thing I want to get from this project is learning how small changes to reinforcement learning methods can have a big impact on efficacy of agent training, myself exploring the many intricacies of RL problems and multi-agent models.

I also love Catan, so any excuse to engage with it through this dissertation is very valuable.

\*because this was the working title of my project and it eventually just stuck.

## Training Modes

CatanAI benefits from the flexibility to tune the training process to the user's needs. However, there are two primary modes of training: headless and GUI.

### Headless

Headless training is the default mode of training. It is the most efficient way to train an agent, as it does not require the overhead of rendering the game state to the screen. There is minimal console output, and the only visual feedback is the occasional printout of the current state of the game.

### GUI

GUI training is the most visually appealing way to train an agent. It is also the slowest, as it requires the overhead of rendering the game state to the screen. With the GUI, one can see the game state being updated in real time, and can also see the agent's decision-making process as it plays the game. The game board is rendered using the PyGame library, and instead of relying on console output, much of the game state is displayed on the screen.

## Agents

There are few different agents that come packaged with CatanAI, each with its own unique quirks and strengths.

### Randy (Random Agent)

Randy is the most basic agent that comes packaged with CatanAI. It is a random agent, and as such, it is not very good at playing Catan. It is, however, a good baseline for comparison with other agents. Every step, Randy will choose a random action from the set of actions that are available to it. This means that Randy will often make mistakes, but it will also often make good decisions. It is a good agent to use for testing the game environment, as it is very easy to implement and is very easy to understand.

### Adam (Basic DQN Agent)

Adam learns with a DQN model, and is the primary agent I've used for training and testing as I've been developing CatanAI. It relies only on the game state as input, and is trained using a simple DQN model. The reward structure for Adam is also very simple, without much heuristic logic. It is a good agent to use for testing the game environment.

### Redmond (DQN Agent with a Love for Red Tiles)

Redmond is an early alternative agent that sadly doesn't get as much use as he deserves. He works very similar to Adam, except that he has a love for red-numbered tiles and gets a bonus for building adjacent to them.
