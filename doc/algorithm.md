# Algorithm

1) Initialize random robot configurations $P$.
2) Initialize parameters $\phi$ for reinforcement learning algorithm (PPO, SAC, DDPG, etc.).
3) Initialize parameters $\omega$ for hardware prediction.
4) Sample robots from $P$, create training batch $B$.
5) Train the reinforcement learning algorithms, store training data in $D$, update $\phi$. 
6) Sample training episodes from $D$, update $\omega$.