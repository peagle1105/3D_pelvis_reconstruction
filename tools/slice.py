class Slice():
    def __init__(self, ctrl, state):
        self.ctrl = ctrl
        self.state = state
    
    def increment_slice(self):
        if self.state.data_loaded:
            new_index = min(self.state.slice_index + 1, self.state.slice_max)
            if new_index != self.state.slice_index:
                self.state.slice_index = new_index
    
    def decrement_slice(self):
        if self.state.data_loaded:
            new_index = max(self.state.slice_index - 1, self.state.slice_min)
            if new_index != self.state.slice_index:
                self.state.slice_index = new_index