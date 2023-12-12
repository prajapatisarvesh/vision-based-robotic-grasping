import numpy as np
import torch
from collections import namedtuple

Batch = namedtuple(
    'Batch', ('states', 'actions', 'rewards', 'next_states', 'dones')
)


class ReplayMemory:
    def __init__(self, max_size, state_size):
        """Replay memory implemented as a circular buffer.

        Experiences will be removed in a FIFO manner after reaching maximum
        buffer size.

        Args:
            - max_size: Maximum size of the buffer.
            - state_size: Size of the state-space features for the environment.
        """
        self.max_size = max_size
        self.state_size = state_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # preallocating all the required memory, for speed concerns
        self.states = torch.empty((max_size, state_size)).to(self.device)
        self.actions = torch.empty((max_size, 1), dtype=torch.long).to(self.device)
        self.rewards = torch.empty((max_size, 1)).to(self.device)
        self.next_states = torch.empty((max_size, state_size)).to(self.device)
        self.dones = torch.empty((max_size, 1), dtype=torch.bool).to(self.device)

        # pointer to the current location in the circular buffer
        self.idx = 0
        # indicates number of transitions currently stored in the buffer
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer.

        :param state:  1-D np.ndarray of state-features.
        :param action:  integer action.
        :param reward:  float reward.
        :param next_state:  1-D np.ndarray of state-features.
        :param done:  boolean value indicating the end of an episode.
        """

        # YOUR CODE HERE:  store the input values into the appropriate
        # attributes, using the current buffer position `self.idx`

        self.states[self.idx] = torch.tensor(state).to(self.device)
        self.actions[self.idx] = torch.tensor(action).to(self.device)
        self.rewards[self.idx] = torch.tensor(reward).to(self.device)
        self.next_states[self.idx] = torch.tensor(next_state).to(self.device)
        self.dones[self.idx] = torch.tensor(done).to(self.device)

        # DO NOT EDIT
        # circulate the pointer to the next position
        self.idx = (self.idx + 1) % self.max_size
        # update the current buffer size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size) -> Batch:
        """Sample a batch of experiences.

        If the buffer contains less that `batch_size` transitions, sample all
        of them.

        :param batch_size:  Number of transitions to sample.
        :rtype: Batch
        """

        # YOUR CODE HERE:  randomly sample an appropriate number of
        # transitions *without replacement*.  If the buffer contains less than
        # `batch_size` transitions, return all of them.  The return type must
        # be a `Batch`.
        sample_indices = np.random.randint(low=0, high=self.size, size=batch_size)
        if self.size < batch_size:
            sample_indices = np.array(range(0, self.size))
        batch = Batch(self.states[sample_indices], self.actions[sample_indices], self.rewards[sample_indices], self.next_states[sample_indices], self.dones[sample_indices])

        return batch

    def populate(self, env, num_steps):
        """Populate this replay memory with `num_steps` from the random policy.

        :param env:  Openai Gym environment
        :param num_steps:  Number of steps to populate the
        """        
        # YOUR CODE HERE:  run a random policy for `num_steps` time-steps and
        # populate the replay memory with the resulting transitions.
        # Hint:  don't repeat code!  Use the self.add() method!
        state = env.reset()

        for i in range(num_steps):
          action = env.action_space.sample()
          try:  
              next_state, reward, done, _ = env.step(action)
          except Exception as e:
              next_state, reward, done, _,_ = env.step(action)
          self.add(state, action, reward, next_state, done)
          state = next_state
          if done:
            state = env.reset()
