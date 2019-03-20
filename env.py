import numpy as np

from collections import deque

class State:
    def __init__(self, state, params):
        self.state = state.copy()
        self.params = params.copy()

def preprocess(config, state, params):
    st = State(state, params)
    st.state = st.state.astype(np.float32)
    st.params = st.params.astype(np.float32)

    return st

class stacked_env:
    def __init__(self, config):
        self.config = config
        self.max_len = config.get('state_stack_size')

        self.stack = deque(maxlen=self.max_len)

    def append(self, state, params):
        st = preprocess(self.config, state, params)
        if len(self.stack) == 0:
            self.reset(st)
        else:
            self.stack.append(st)

    def current(self):
        states = [st.state for st in self.stack]
        params = [st.params for st in self.stack]

        states = np.stack(states, axis=0)[-1]
        params = np.stack(params, axis=0)[-1]

        return State(states, params)

    def reset(self, st):
        for _ in range(self.max_len):
            self.stack.append(st)

